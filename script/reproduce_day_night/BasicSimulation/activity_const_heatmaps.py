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
from script.reproduce_day_night.Solver import (
    DEFAULT_SMELL_RADIUS,
    DEFAULT_SIGHT_RADIUS,
    DayNightModel1D,
    gaussian_initial_condition,
)


OUTPUT_DIRECTORY = basic_simulation_output_path()
OUTPUT_PATH = basic_simulation_output_path("sight_weight_sunset_heatmaps.png")
NUMBER_OF_POINTS = 256
NUMBER_OF_POPULATIONS = 1
NUMBER_OF_CYCLES = 3
CYCLE_PERIOD = 1.0
TOTAL_TIME = NUMBER_OF_CYCLES * CYCLE_PERIOD
DT = 0.01
COEFFICIENT_ATTRACTION = np.array([[0.2]])
COEFFICIENT_DIFFUSION = np.array([0.05])
SIGHT_WEIGHTS = (0.0, 0.5, 1.0)


def build_lighting_regimes():
    long_cycle_period = TOTAL_TIME + CYCLE_PERIOD
    return (
        {
            "label": "full day",
            "display_sunset": 1.0,
            "cycle_period": long_cycle_period,
            "day_start": 0.0,
            "day_end": TOTAL_TIME + 0.5 * CYCLE_PERIOD,
            "show_transition_markers": False,
        },
        {
            "label": "half day / half night",
            "display_sunset": 0.5,
            "cycle_period": CYCLE_PERIOD,
            "day_start": 0.0,
            "day_end": 0.5 * CYCLE_PERIOD,
            "show_transition_markers": True,
        },
        {
            "label": "full night",
            "display_sunset": 0.0,
            "cycle_period": long_cycle_period,
            "day_start": TOTAL_TIME + 0.25 * CYCLE_PERIOD,
            "day_end": TOTAL_TIME + 0.75 * CYCLE_PERIOD,
            "show_transition_markers": False,
        },
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
        sight_radius=DEFAULT_SIGHT_RADIUS,
        smell_radius=DEFAULT_SMELL_RADIUS,
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
