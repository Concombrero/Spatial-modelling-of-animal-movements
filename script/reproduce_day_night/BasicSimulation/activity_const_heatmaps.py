import argparse
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
    apply_plot_typography,
    build_constant_activity_lighting_regimes,
    resolve_experiment_config,
)
from script.reproduce_day_night.Solver import (
    DayNightModel1D,
    gaussian_initial_condition,
)


apply_plot_typography()


OUTPUT_DIRECTORY = basic_simulation_output_path()
OUTPUT_PATH = basic_simulation_output_path("sight_weight_sunset_heatmaps.png")
EXPERIMENT_CONFIG = resolve_experiment_config(
    ONE_POPULATION_SIMULATION_CONFIG,
    "activity_const_heatmaps",
)
NUMBER_OF_POINTS = EXPERIMENT_CONFIG["number_of_points"]
NUMBER_OF_POPULATIONS = EXPERIMENT_CONFIG["number_of_populations"]
NUMBER_OF_CYCLES = EXPERIMENT_CONFIG["number_of_cycles"]
CYCLE_PERIOD = EXPERIMENT_CONFIG["cycle_period"]
TOTAL_TIME = NUMBER_OF_CYCLES * CYCLE_PERIOD
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
DEFAULT_SUNSET_VALUES = EXPERIMENT_CONFIG["sunset_values"]


def build_lighting_regimes(sunset_values=DEFAULT_SUNSET_VALUES):
    return build_constant_activity_lighting_regimes(
        sunset_values,
        total_time=TOTAL_TIME,
        cycle_period=CYCLE_PERIOD,
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Plot one density heatmap per sight weight and day-night regime for "
            "the always-active single-population model."
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
        "--output-path",
        type=Path,
        default=OUTPUT_PATH,
        help="Path of the saved figure.",
    )
    return parser.parse_args()


def build_solver(sight_weight, lighting_regime, number_of_points, dt):
    return DayNightModel1D(
        a_border=0.0,
        b_border=1.0,
        number_of_points=number_of_points,
        total_time=TOTAL_TIME,
        dt=dt,
        initial_condition=gaussian_initial_condition,
        coefficient_attraction=COEFFICIENT_ATTRACTION,
        coefficient_diffusion=COEFFICIENT_DIFFUSION,
        cycle_period=lighting_regime["cycle_period"],
        number_of_population=NUMBER_OF_POPULATIONS,
        day_start=lighting_regime["day_start"],
        day_end=lighting_regime["day_end"],
        time_input_mode="phase",
        activity_mode="always",
        sight_weight=sight_weight,
        sight_radius=SIGHT_RADIUS,
        smell_radius=SMELL_RADIUS,
    )


def run_case(sight_weight, lighting_regime, number_of_points, dt):
    model = build_solver(sight_weight, lighting_regime, number_of_points, dt)
    model.solve()
    return model


def run_all_cases(lighting_regimes, sight_weights, number_of_points, dt):
    models = []
    for lighting_regime in lighting_regimes:
        model_row = []
        for sight_weight in sight_weights:
            model = run_case(
                sight_weight,
                lighting_regime,
                number_of_points,
                dt,
            )
            model_row.append(model)
            print(
                (
                    f"Finished {lighting_regime['label']}, "
                    f"t_sunset={lighting_regime['display_sunset']:g}, "
                    f"w={sight_weight:g}"
                ),
                flush=True,
            )
        models.append(model_row)
    return models


def save_combined_heatmaps(models, sight_weights, lighting_regimes, output_path):
    figure, axes = plt.subplots(
        len(lighting_regimes),
        len(sight_weights),
        figsize=(4.0 * len(sight_weights), 3.6 * len(lighting_regimes)),
        squeeze=False,
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )

    vmin = min(float(np.min(model.U[:, :, 0])) for row in models for model in row)
    vmax = max(float(np.max(model.U[:, :, 0])) for row in models for model in row)
    image = None

    for row_index, (axes_row, model_row, lighting_regime) in enumerate(
        zip(axes, models, lighting_regimes)
    ):
        for column_index, (axis, model, sight_weight) in enumerate(
            zip(axes_row, model_row, sight_weights)
        ):
            image = axis.imshow(
                model.U[:, :, 0],
                origin="lower",
                aspect="auto",
                extent=[model.a_border, model.b_border, model.time[0], model.time[-1]],
                cmap="hot_r",
                vmin=vmin,
                vmax=vmax,
            )
            if row_index == 0:
                axis.set_title(f"w={sight_weight:g}")
            if row_index == len(lighting_regimes) - 1:
                axis.set_xlabel("x")
            if column_index == 0:
                axis.set_ylabel(
                    "t\n"
                    f"{lighting_regime['label']}\n"
                    f"$t_{{sunset}}={lighting_regime['display_sunset']:g}$"
                )
            if lighting_regime["show_transition_markers"]:
                model._add_transition_markers(axis, 0, show_legend=False)

    figure.colorbar(image, ax=axes, label="density")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)


def validate_args(args):
    if any(weight < 0.0 or weight > 1.0 for weight in args.weights):
        raise ValueError("All weights must lie in the interval [0, 1].")

    if args.number_of_points < 2:
        raise ValueError("number_of_points must be at least 2.")

    if args.dt <= 0.0:
        raise ValueError("dt must be positive.")


def main():
    args = parse_args()
    validate_args(args)
    sight_weights = tuple(float(weight) for weight in args.weights)
    lighting_regimes = build_lighting_regimes()
    output_path = args.output_path.resolve()
    models = run_all_cases(
        lighting_regimes,
        sight_weights,
        args.number_of_points,
        args.dt,
    )
    save_combined_heatmaps(models, sight_weights, lighting_regimes, output_path)
    print(f"Saved heatmap to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
