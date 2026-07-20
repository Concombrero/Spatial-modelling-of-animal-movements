import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from script.reproduce_day_night.paths import basic_simulation_output_path
from script.reproduce_day_night.shared_config import (
    ONE_POPULATION_SIMULATION_CONFIG,
    activity_regimes_for_codes,
    apply_plot_typography,
    build_periodic_lighting_regime,
    build_periodic_lighting_regimes,
    resolve_experiment_config,
)
from script.reproduce_day_night.Solver import (
    DayNightModel1D,
    compute_spread_indicator,
    gaussian_initial_condition,
)


apply_plot_typography()


OUTPUT_DIRECTORY = basic_simulation_output_path()
OUTPUT_PATH = basic_simulation_output_path("sleep_pattern_spread.png")
EXPERIMENT_CONFIG = resolve_experiment_config(
    ONE_POPULATION_SIMULATION_CONFIG,
    "sleep_pattern_spread",
)
NUMBER_OF_POINTS = EXPERIMENT_CONFIG["number_of_points"]
NUMBER_OF_POPULATIONS = EXPERIMENT_CONFIG["number_of_populations"]
NUMBER_OF_CYCLES = EXPERIMENT_CONFIG["number_of_cycles"]
CYCLE_PERIOD = EXPERIMENT_CONFIG["cycle_period"]
TOTAL_TIME = NUMBER_OF_CYCLES * CYCLE_PERIOD
OBSERVATION_WINDOW = EXPERIMENT_CONFIG["observation_window"]
DT = EXPERIMENT_CONFIG["dt"]
COEFFICIENT_ATTRACTION = np.array(
    EXPERIMENT_CONFIG["coefficient_attraction"],
    dtype=float,
)
COEFFICIENT_DIFFUSION = np.array(
    EXPERIMENT_CONFIG["coefficient_diffusion"],
    dtype=float,
)
SIGHT_RADIUS = EXPERIMENT_CONFIG["sight_radius"]
SMELL_RADIUS = EXPERIMENT_CONFIG["smell_radius"]
SIGHT_WEIGHTS = EXPERIMENT_CONFIG["weights"]
MAX_WORKERS = EXPERIMENT_CONFIG["max_workers"]
DAY_START = EXPERIMENT_CONFIG["day_start"]
DEFAULT_SUNSET_VALUES = EXPERIMENT_CONFIG["sunset_values"]
ACTIVITY_REGIMES = activity_regimes_for_codes(EXPERIMENT_CONFIG["activity_codes"])


def build_lighting_regime(t_sunset):
    return build_periodic_lighting_regime(
        t_sunset,
        dt=DT,
        cycle_period=CYCLE_PERIOD,
        day_start=DAY_START,
    )


def build_lighting_regimes(sunset_values=DEFAULT_SUNSET_VALUES):
    return build_periodic_lighting_regimes(
        sunset_values,
        dt=DT,
        cycle_period=CYCLE_PERIOD,
        day_start=DAY_START,
    )


def build_solver(sight_weight, lighting_regime, activity_regime, number_of_points, dt):
    return DayNightModel1D(
        a_border=0.0,
        b_border=1.0,
        number_of_points=number_of_points,
        total_time=TOTAL_TIME,
        dt=dt,
        initial_condition=gaussian_initial_condition,
        coefficient_attraction=COEFFICIENT_ATTRACTION,
        coefficient_diffusion=COEFFICIENT_DIFFUSION,
        cycle_period=CYCLE_PERIOD,
        number_of_population=NUMBER_OF_POPULATIONS,
        day_start=lighting_regime["day_start"],
        day_end=lighting_regime["day_end"],
        time_input_mode="phase",
        activity_mode="always",
        activity_periods=activity_regime["periods"],
        sight_weight=sight_weight,
        sight_radius=SIGHT_RADIUS,
        smell_radius=SMELL_RADIUS,
    )


def compute_psi(model, observation_window=OBSERVATION_WINDOW, population_index=0):
    return compute_spread_indicator(
        model,
        observation_window,
        population_index=population_index,
    )


def run_single_case(
    sunset_index,
    regime_index,
    weight_index,
    lighting_regime,
    activity_regime,
    sight_weight,
    number_of_points,
    dt,
):
    model = build_solver(
        sight_weight,
        lighting_regime,
        activity_regime,
        number_of_points,
        dt,
    )
    model.solve()
    psi_value = compute_psi(model)
    return sunset_index, regime_index, weight_index, psi_value


def run_all_cases(
    lighting_regimes,
    activity_regimes,
    sight_weights,
    number_of_points,
    dt,
    max_workers=None,
):
    psi_grid = [
        [[None for _ in sight_weights] for _ in activity_regimes]
        for _ in lighting_regimes
    ]
    case_specs = [
        (
            sunset_index,
            regime_index,
            weight_index,
            lighting_regime,
            activity_regime,
            sight_weight,
        )
        for sunset_index, lighting_regime in enumerate(lighting_regimes)
        for regime_index, activity_regime in enumerate(activity_regimes)
        for weight_index, sight_weight in enumerate(sight_weights)
    ]

    if max_workers is None:
        max_workers = MAX_WORKERS

    if max_workers <= 1:
        for (
            sunset_index,
            regime_index,
            weight_index,
            lighting_regime,
            activity_regime,
            sight_weight,
        ) in case_specs:
            _, _, _, psi_value = run_single_case(
                sunset_index,
                regime_index,
                weight_index,
                lighting_regime,
                activity_regime,
                sight_weight,
                number_of_points,
                dt,
            )
            psi_grid[sunset_index][regime_index][weight_index] = psi_value
            print(
                f"Finished t_sunset={lighting_regime['display_sunset']:g}, {activity_regime['label']}, w={sight_weight:g}",
                flush=True,
            )

        return psi_grid

    with ProcessPoolExecutor(max_workers=min(max_workers, len(case_specs))) as executor:
        future_to_case = {
            executor.submit(
                run_single_case,
                sunset_index,
                regime_index,
                weight_index,
                lighting_regime,
                activity_regime,
                sight_weight,
                number_of_points,
                dt,
            ): (
                lighting_regime["display_sunset"],
                activity_regime["label"],
                sight_weight,
            )
            for (
                sunset_index,
                regime_index,
                weight_index,
                lighting_regime,
                activity_regime,
                sight_weight,
            ) in case_specs
        }

        for future in as_completed(future_to_case):
            sunset_value, activity_label, sight_weight = future_to_case[future]
            sunset_index, regime_index, weight_index, psi_value = future.result()
            psi_grid[sunset_index][regime_index][weight_index] = psi_value
            print(
                f"Finished t_sunset={sunset_value:g}, {activity_label}, w={sight_weight:g}",
                flush=True,
            )

    return psi_grid


def save_spread_plot(
    psi_grid,
    sight_weights,
    lighting_regimes,
    activity_regimes,
    output_path,
):
    figure, axes = plt.subplots(
        1,
        len(lighting_regimes),
        figsize=(4.2 * len(lighting_regimes), 4.8),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    axes = np.atleast_1d(axes)
    markers = ("o", "s", "^", "D", "v", "P")

    for axis, lighting_regime, psi_values_by_regime in zip(axes, lighting_regimes, psi_grid):
        for curve_index, (activity_regime, psi_values) in enumerate(
            zip(activity_regimes, psi_values_by_regime)
        ):
            axis.plot(
                sight_weights,
                psi_values,
                marker=activity_regime["marker"],
                linewidth=2.0,
                markersize=5.0,
                color=activity_regime["color"],
                label=activity_regime["label"],
            )

        axis.set_title(
            f"{lighting_regime['label']}\n$t_{{sunset}}={lighting_regime['display_sunset']:g}$"
        )
        axis.set_xlim(0.0, 1.0)
        axis.set_ylim(0.0, 1.0)
        axis.set_xticks(np.linspace(0.0, 1.0, 6))
        axis.set_yticks(np.linspace(0.0, 1.0, 6))
        axis.set_xlabel("w")
        axis.grid(True, alpha=0.3)

    axes[0].set_ylabel(r"$\Psi$")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.03))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, bbox_inches="tight", dpi=200)
    plt.close(figure)


def main():
    lighting_regimes = build_lighting_regimes()
    psi_grid = run_all_cases(
        lighting_regimes,
        ACTIVITY_REGIMES,
        SIGHT_WEIGHTS,
        NUMBER_OF_POINTS,
        DT,
    )
    save_spread_plot(
        psi_grid,
        SIGHT_WEIGHTS,
        lighting_regimes,
        ACTIVITY_REGIMES,
        OUTPUT_PATH,
    )
    print(f"Saved spread plot to {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

