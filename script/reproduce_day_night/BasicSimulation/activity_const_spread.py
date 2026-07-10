import argparse
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
from script.reproduce_day_night.Solver import (
    DEFAULT_SMELL_RADIUS,
    DEFAULT_SIGHT_RADIUS,
    DayNightModel1D,
    compute_spread_indicator,
    gaussian_initial_condition,
)


OUTPUT_DIRECTORY = basic_simulation_output_path()
OUTPUT_PATH = basic_simulation_output_path("sight_weight_sunset_spread.png")
NUMBER_OF_POINTS = 256
NUMBER_OF_POPULATIONS = 1
NUMBER_OF_CYCLES = 3
CYCLE_PERIOD = 1.0
TOTAL_TIME = NUMBER_OF_CYCLES * CYCLE_PERIOD
OBSERVATION_WINDOW = 1.0
DT = 0.01
COEFFICIENT_ATTRACTION = np.array([[0.2]])
COEFFICIENT_DIFFUSION = np.array([0.05])
SIGHT_WEIGHTS = tuple(np.round(np.linspace(0.0, 1.0, 11), 1))
DEFAULT_SUNSET_VALUES = (1.0, 0.5, 0.0)


def build_lighting_regime(t_sunset, total_time=TOTAL_TIME):
    t_sunset = float(t_sunset)
    if t_sunset < 0.0 or t_sunset > 1.0:
        raise ValueError("Each t_sunset value must lie in the interval [0, 1].")

    long_cycle_period = total_time + CYCLE_PERIOD
    if np.isclose(t_sunset, 1.0):
        return {
            "label": "full day",
            "display_sunset": 1.0,
            "cycle_period": long_cycle_period,
            "day_start": 0.0,
            "day_end": total_time + 0.5 * CYCLE_PERIOD,
        }

    if np.isclose(t_sunset, 0.0):
        return {
            "label": "full night",
            "display_sunset": 0.0,
            "cycle_period": long_cycle_period,
            "day_start": total_time + 0.25 * CYCLE_PERIOD,
            "day_end": total_time + 0.75 * CYCLE_PERIOD,
        }

    if np.isclose(t_sunset, 0.5):
        label = "half day / half night"
    else:
        label = "partial day"

    return {
        "label": label,
        "display_sunset": t_sunset,
        "cycle_period": CYCLE_PERIOD,
        "day_start": 0.0,
        "day_end": t_sunset * CYCLE_PERIOD,
    }


def build_lighting_regimes(sunset_values=DEFAULT_SUNSET_VALUES, total_time=TOTAL_TIME):
    return tuple(
        build_lighting_regime(t_sunset, total_time=total_time)
        for t_sunset in sunset_values
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compute the normalized spread indicator Omega and plot it against "
            "the sight weight for three day-night regimes."
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
        "--sunset-values",
        nargs="+",
        type=float,
        default=list(DEFAULT_SUNSET_VALUES),
        help="t_sunset values used to create one subplot per lighting regime.",
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
        default=3,
        help="Maximum number of worker processes across the three lighting regimes.",
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


def compute_omega(model, observation_window=OBSERVATION_WINDOW, population_index=0):
    return compute_spread_indicator(
        model,
        observation_window,
        population_index=population_index,
    )


def run_regime_cases(lighting_regime, sight_weights, number_of_points, dt):
    omega_values = []

    for sight_weight in sight_weights:
        model = build_solver(sight_weight, lighting_regime, number_of_points, dt)
        model.solve()
        omega_values.append(compute_omega(model))
        print(
            f"Finished t_sunset={lighting_regime['display_sunset']:g}, w={sight_weight:g}",
            flush=True,
        )

    return lighting_regime["display_sunset"], omega_values


def save_spread_plot(omega_by_regime, sight_weights, lighting_regimes, output_path):
    figure, axes = plt.subplots(
        1,
        len(lighting_regimes),
        figsize=(4.3 * len(lighting_regimes), 4.0),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    axes = np.atleast_1d(axes)

    for axis, lighting_regime in zip(axes, lighting_regimes):
        display_sunset = lighting_regime["display_sunset"]
        axis.plot(
            sight_weights,
            omega_by_regime[display_sunset],
            marker="o",
            linewidth=2.0,
            color="#1f77b4",
        )
        axis.set_title(
            f"{lighting_regime['label']}\n$t_{{sunset}}={display_sunset:g}$"
        )
        axis.set_xlim(0.0, 1.0)
        axis.set_ylim(0.0, 1.0)
        axis.set_xticks(np.linspace(0.0, 1.0, 6))
        axis.set_yticks(np.linspace(0.0, 1.0, 6))
        axis.set_xlabel("w")
        axis.grid(True, alpha=0.3)

    axes[0].set_ylabel(r"$\Omega$")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)


def main():
    args = parse_args()
    sight_weights = tuple(float(weight) for weight in args.weights)
    sunset_values = tuple(float(value) for value in args.sunset_values)
    if any(weight < 0.0 or weight > 1.0 for weight in sight_weights):
        raise ValueError("All weights must lie in the interval [0, 1].")

    if any(value < 0.0 or value > 1.0 for value in sunset_values):
        raise ValueError("All t_sunset values must lie in the interval [0, 1].")

    rounded_sunset_values = [round(value, 12) for value in sunset_values]
    if len(set(rounded_sunset_values)) != len(rounded_sunset_values):
        raise ValueError("t_sunset values must be distinct.")

    if args.number_of_points < 2:
        raise ValueError("number_of_points must be at least 2.")

    if args.dt <= 0.0:
        raise ValueError("dt must be positive.")

    if args.max_workers < 1:
        raise ValueError("max_workers must be at least 1.")

    lighting_regimes = build_lighting_regimes(sunset_values)
    output_path = args.output_path.resolve()

    if args.max_workers == 1:
        omega_by_regime = {
            display_sunset: omega_values
            for display_sunset, omega_values in (
                run_regime_cases(
                    lighting_regime,
                    sight_weights,
                    args.number_of_points,
                    args.dt,
                )
                for lighting_regime in lighting_regimes
            )
        }
    else:
        worker_count = min(args.max_workers, len(lighting_regimes))
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            futures = [
                executor.submit(
                    run_regime_cases,
                    lighting_regime,
                    sight_weights,
                    args.number_of_points,
                    args.dt,
                )
                for lighting_regime in lighting_regimes
            ]
            omega_by_regime = {}
            for future in as_completed(futures):
                display_sunset, omega_values = future.result()
                omega_by_regime[display_sunset] = omega_values

    save_spread_plot(omega_by_regime, sight_weights, lighting_regimes, output_path)
    print(f"Saved spread plot to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())