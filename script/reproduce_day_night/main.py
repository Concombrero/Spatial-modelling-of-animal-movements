from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from solver import DayNightModel1D


OUTPUT_ROOT = Path("./script/reproduce_day_night/output")
NUMBER_OF_POPULATIONS = 2
COEFFICIENT_ATTRACTION = np.array(
    [
        [0.12, -0.04],
        [0.06, 0.10],
    ]
)
COEFFICIENT_DIFFUSION = np.array([0.05, 0.03])


def gaussian_initial_condition(x, centers, width=0.10):
    x = np.asarray(x, dtype=float)
    dx = x[1] - x[0]
    length = (x[-1] - x[0]) + dx
    centers = np.atleast_1d(np.asarray(centers, dtype=float))

    profiles = []
    for center in centers:
        wrapped_distance = ((x - center + 0.5 * length) % length) - 0.5 * length
        profiles.append(np.exp(-0.5 * (wrapped_distance / width) ** 2))

    values = np.column_stack(profiles)
    masses = dx * np.sum(values, axis=0, keepdims=True)
    values = values / masses
    if values.shape[1] == 1:
        return values[:, 0]
    return values


def build_initial_condition(centers, width):
    def initial_condition(x):
        return gaussian_initial_condition(x, centers=centers, width=width)

    return initial_condition


def simulate_case(output_name, sight_weight):
    cycle_period = 1.0
    number_of_cycles = 4

    simulation = DayNightModel1D(
        a_border=0.0,
        b_border=1.0,
        number_of_points=128,
        total_time=number_of_cycles * cycle_period,
        dt=1.0e-3,
        initial_condition=build_initial_condition(
            centers=[0.25, 0.72],
            width=0.08,
        ),
        number_of_population=NUMBER_OF_POPULATIONS,
        coefficient_attraction=COEFFICIENT_ATTRACTION,
        coefficient_diffusion=COEFFICIENT_DIFFUSION,
        cycle_period=cycle_period,
        day_start=0.0,
        day_end=0.5,
        activity_start=0.05,
        activity_end=0.35,
        sight_weight=sight_weight,
        sight_radius=0.12,
        smell_radius=0.18,
    )

    simulation.solve()
    output_directory = OUTPUT_ROOT / output_name

    heatmap_figure, _ = simulation.plot_solution_heatmaps(
        save=True,
        save_path=output_directory / "solution_heatmaps.png",
    )
    animation_figure, _ = simulation.create_solution_gif(
        interval=20,
        fps=20,
        save=True,
        save_path=output_directory / "solution_animation.gif",
    )

    plt.close(heatmap_figure)
    plt.close(animation_figure)

    mass = simulation.get_mass()

    print(f"Saved outputs to {output_directory}")
    for population_index in range(NUMBER_OF_POPULATIONS):
        print(
            f"Population {population_index + 1} mass: "
            f"{mass[0, population_index]:.8f} -> {mass[-1, population_index]:.8f}"
        )


def simulate_w0():
    simulate_case("w_0", sight_weight=0.0)


def simulate_w1():
    simulate_case("w_1", sight_weight=1.0)


def main():
    simulate_w0()
    simulate_w1()


if __name__ == "__main__":
    raise SystemExit(main())