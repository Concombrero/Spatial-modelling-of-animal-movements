import argparse
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
    resolve_experiment_config,
)
from script.reproduce_day_night.Solver import (
    DayNightModel1D,
    compute_spread_indicator,
    gaussian_initial_condition,
)


apply_plot_typography()


OUTPUT_DIRECTORY = basic_simulation_output_path()
OUTPUT_PATH = basic_simulation_output_path("spread_diurnal_vs_nocturnal.png")
EXPERIMENT_CONFIG = resolve_experiment_config(
    ONE_POPULATION_SIMULATION_CONFIG,
    "spread_diurnal_vs_nocturnal",
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
T_SUNSET = EXPERIMENT_CONFIG["t_sunset"]
ACTIVITY_REGIMES = activity_regimes_for_codes(EXPERIMENT_CONFIG["activity_codes"])


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compute the normalized spread indicator Psi for diurnal and "
            "nocturnal activity schedules at fixed t_sunset = 0.5."
        )
    )
    parser.add_argument(
        "--weights",
        nargs="+",
        type=float,
        default=list(SIGHT_WEIGHTS),
        help="Sight weights w to evaluate.",
    )
    parser.add_argument(
        "--number-of-points",
        type=int,
        default=NUMBER_OF_POINTS,
        help="Number of spatial grid points.",
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=DT,
        help="Output time step used by the solver.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=MAX_WORKERS,
        help="Maximum number of worker processes across all activity cases.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=OUTPUT_PATH,
        help="Path of the saved figure.",
    )
    return parser.parse_args()


def build_lighting_regime(t_sunset=T_SUNSET):
    return build_periodic_lighting_regime(
        t_sunset,
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
    return regime_index, weight_index, psi_value


def run_all_cases(
    lighting_regime,
    activity_regimes,
    sight_weights,
    number_of_points,
    dt,
    max_workers=None,
):
    psi_by_regime = {
        activity_regime["label"]: [None for _ in sight_weights]
        for activity_regime in activity_regimes
    }
    case_specs = [
        (
            regime_index,
            weight_index,
            activity_regime,
            sight_weight,
        )
        for regime_index, activity_regime in enumerate(activity_regimes)
        for weight_index, sight_weight in enumerate(sight_weights)
    ]

    if max_workers is None:
        max_workers = MAX_WORKERS

    if max_workers <= 1:
        for regime_index, weight_index, activity_regime, sight_weight in case_specs:
            _, _, psi_value = run_single_case(
                regime_index,
                weight_index,
                lighting_regime,
                activity_regime,
                sight_weight,
                number_of_points,
                dt,
            )
            psi_by_regime[activity_regime["label"]][weight_index] = psi_value
            print(
                f"Finished {activity_regime['label']}, t_sunset={lighting_regime['display_sunset']:g}, w={sight_weight:g}",
                flush=True,
            )

        return psi_by_regime

    with ProcessPoolExecutor(max_workers=min(max_workers, len(case_specs))) as executor:
        future_to_case = {
            executor.submit(
                run_single_case,
                regime_index,
                weight_index,
                lighting_regime,
                activity_regime,
                sight_weight,
                number_of_points,
                dt,
            ): (activity_regime["label"], sight_weight)
            for regime_index, weight_index, activity_regime, sight_weight in case_specs
        }

        for future in as_completed(future_to_case):
            activity_label, sight_weight = future_to_case[future]
            regime_index, weight_index, psi_value = future.result()
            psi_by_regime[activity_regimes[regime_index]["label"]][weight_index] = psi_value
            print(
                f"Finished {activity_label}, t_sunset={lighting_regime['display_sunset']:g}, w={sight_weight:g}",
                flush=True,
            )

    return psi_by_regime


def save_spread_plot(
    psi_by_regime,
    sight_weights,
    lighting_regime,
    activity_regimes,
    output_path,
):
    figure, axes = plt.subplots(
        1,
        len(activity_regimes),
        figsize=(4.4 * len(activity_regimes), 4.4),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    axes = np.atleast_1d(axes)

    for axis, activity_regime in zip(axes, activity_regimes):
        psi_values = psi_by_regime[activity_regime["label"]]
        axis.plot(
            sight_weights,
            psi_values,
            marker=activity_regime["marker"],
            linewidth=2.0,
            markersize=5.0,
            color=activity_regime["color"],
        )
        axis.set_title(
            f"{activity_regime['label']}\n$t_{{sunset}}={lighting_regime['display_sunset']:g}$"
        )
        axis.set_xlim(0.0, 1.0)
        axis.set_ylim(0.0, 1.0)
        axis.set_xticks(np.linspace(0.0, 1.0, 6))
        axis.set_yticks(np.linspace(0.0, 1.0, 6))
        axis.set_xlabel("w")
        axis.grid(True, alpha=0.3)

    axes[0].set_ylabel(r"$\Psi$")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, bbox_inches="tight", dpi=200)
    plt.close(figure)


def main():
    args = parse_args()
    sight_weights = tuple(float(weight) for weight in args.weights)

    if any(weight < 0.0 or weight > 1.0 for weight in sight_weights):
        raise ValueError("All weights must lie in the interval [0, 1].")

    if args.number_of_points < 2:
        raise ValueError("number_of_points must be at least 2.")

    if args.dt <= 0.0:
        raise ValueError("dt must be positive.")

    if args.max_workers < 1:
        raise ValueError("max_workers must be at least 1.")

    lighting_regime = build_lighting_regime()
    output_path = args.output_path.resolve()
    psi_by_regime = run_all_cases(
        lighting_regime,
        ACTIVITY_REGIMES,
        sight_weights,
        args.number_of_points,
        args.dt,
        max_workers=args.max_workers,
    )
    save_spread_plot(
        psi_by_regime,
        sight_weights,
        lighting_regime,
        ACTIVITY_REGIMES,
        output_path,
    )
    print(f"Saved spread plot to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
