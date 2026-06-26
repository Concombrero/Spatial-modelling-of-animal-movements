import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from common_utils import (
    DEFAULT_SMELL_RADIUS,
    DEFAULT_SIGHT_RADIUS,
    gaussian_initial_condition,
)
from solver import DayNightModel1D


OUTPUT_DIRECTORY = Path(__file__).resolve().parent / "output"
OUTPUT_PATH = OUTPUT_DIRECTORY / "sleep_pattern_heatmaps.png"
NUMBER_OF_POINTS = 256
NUMBER_OF_POPULATIONS = 1
NUMBER_OF_CYCLES = 2
CYCLE_PERIOD = 1.0
TOTAL_TIME = NUMBER_OF_CYCLES * CYCLE_PERIOD
DT = 0.01
COEFFICIENT_ATTRACTION = np.array([[0.2]])
COEFFICIENT_DIFFUSION = np.array([0.05])
SIGHT_WEIGHTS = (0.0, 0.5, 1.0)
MAX_WORKERS = min(6, os.cpu_count() or 1)
DAY_START = 0.0
DAY_END = 0.5 * CYCLE_PERIOD
ACTIVITY_REGIMES = (
    {"label": "Diurnal", "periods": [(0.0, 0.5)]},
    {"label": "Nocturnal", "periods": [(0.5, 1.0)]},
    {"label": "Polyphasic 1", "periods": [(0.0, 0.25), (0.5, 0.75)]},
    {"label": "Polyphasic 2", "periods": [(0.25, 0.5), (0.75, 1.0)]},
    {"label": "Matutinal 1", "periods": [(0.0, 0.25), (0.75, 1.0)]},
    {"label": "Matutinal 2", "periods": [(0.25, 0.75)]},
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Plot one density heatmap per sight weight and activity schedule for "
            "the single-population sleep-pattern experiments."
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
        help="Maximum number of worker processes across all heatmap cases.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=OUTPUT_PATH,
        help="Path of the saved figure.",
    )
    return parser.parse_args()


def format_activity_periods(periods):
    return " + ".join(f"[{start:g}, {end:g}]" for start, end in periods)


def build_solver(sight_weight, activity_regime, number_of_points, dt):
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
        day_start=DAY_START,
        day_end=DAY_END,
        time_input_mode="phase",
        activity_mode="always",
        activity_periods=activity_regime["periods"],
        sight_weight=sight_weight,
        sight_radius=DEFAULT_SIGHT_RADIUS,
        smell_radius=DEFAULT_SMELL_RADIUS,
    )


def run_case(sight_weight, activity_regime, number_of_points, dt):
    model = build_solver(sight_weight, activity_regime, number_of_points, dt)
    model.solve()
    return model


def run_all_cases(
    activity_regimes,
    sight_weights,
    number_of_points,
    dt,
    max_workers=None,
):
    models = [[None for _ in sight_weights] for _ in activity_regimes]
    case_specs = [
        (row_index, column_index, sight_weight, activity_regime)
        for row_index, activity_regime in enumerate(activity_regimes)
        for column_index, sight_weight in enumerate(sight_weights)
    ]

    if max_workers is None:
        max_workers = MAX_WORKERS

    if max_workers <= 1:
        for row_index, column_index, sight_weight, activity_regime in case_specs:
            models[row_index][column_index] = run_case(
                sight_weight,
                activity_regime,
                number_of_points,
                dt,
            )
            print(f"Finished {activity_regime['label']}, w={sight_weight:g}", flush=True)
        return models

    with ProcessPoolExecutor(max_workers=min(max_workers, len(case_specs))) as executor:
        future_to_case = {
            executor.submit(
                run_case,
                sight_weight,
                activity_regime,
                number_of_points,
                dt,
            ): (
                row_index,
                column_index,
                sight_weight,
                activity_regime["label"],
            )
            for row_index, column_index, sight_weight, activity_regime in case_specs
        }

        for future in as_completed(future_to_case):
            row_index, column_index, sight_weight, label = future_to_case[future]
            models[row_index][column_index] = future.result()
            print(f"Finished {label}, w={sight_weight:g}", flush=True)

    return models


def save_combined_heatmaps(models, sight_weights, activity_regimes, output_path):
    figure, axes = plt.subplots(
        len(activity_regimes),
        len(sight_weights),
        figsize=(4.0 * len(sight_weights), 3.2 * len(activity_regimes)),
        squeeze=False,
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )

    vmin = min(float(np.min(model.U[:, :, 0])) for row in models for model in row)
    vmax = max(float(np.max(model.U[:, :, 0])) for row in models for model in row)
    image = None

    for row_index, (axes_row, model_row, activity_regime) in enumerate(
        zip(axes, models, activity_regimes)
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
            model._add_transition_markers(
                axis,
                0,
                show_legend=False,
                show_day_night_cycle=True,
                show_activity_period=True,
            )

            if row_index == 0:
                axis.set_title(f"w={sight_weight:g}")
            if row_index == len(activity_regimes) - 1:
                axis.set_xlabel("x")
            if column_index == 0:
                axis.set_ylabel(
                    "t\n"
                    f"{activity_regime['label']}\n"
                    f"{format_activity_periods(activity_regime['periods'])}"
                )

    figure.colorbar(image, ax=axes, label="density")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, bbox_inches="tight", dpi=200)
    plt.close(figure)


def validate_args(args):
    if any(weight < 0.0 or weight > 1.0 for weight in args.weights):
        raise ValueError("All weights must lie in the interval [0, 1].")

    if args.number_of_points < 2:
        raise ValueError("number_of_points must be at least 2.")

    if args.dt <= 0.0:
        raise ValueError("dt must be positive.")

    if args.max_workers < 1:
        raise ValueError("max_workers must be at least 1.")


def main():
    args = parse_args()
    validate_args(args)
    sight_weights = tuple(float(weight) for weight in args.weights)
    output_path = args.output_path.resolve()
    models = run_all_cases(
        ACTIVITY_REGIMES,
        sight_weights,
        args.number_of_points,
        args.dt,
        max_workers=args.max_workers,
    )
    save_combined_heatmaps(models, sight_weights, ACTIVITY_REGIMES, output_path)
    print(f"Saved heatmap to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())