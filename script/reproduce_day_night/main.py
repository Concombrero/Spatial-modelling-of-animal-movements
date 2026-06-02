from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from solver import DayNightModel1D


def gaussian_initial_condition(x, center=0.30, width=0.10):
    x = np.asarray(x, dtype=float)
    dx = x[1] - x[0]
    length = (x[-1] - x[0]) + dx
    wrapped_distance = ((x - center + 0.5 * length) % length) - 0.5 * length
    values = np.exp(-0.5 * (wrapped_distance / width) ** 2)
    mass = dx * np.sum(values)
    return values / mass


def simulate_w0():
    simulation = DayNightModel1D(
        a_border=0.0,
        b_border=1.0,
        number_of_points=128,
        total_time=1.0,
        dt=1.0e-3,
        initial_condition=gaussian_initial_condition,
        coefficient_attraction=0.1,
        coefficient_diffusion=0.05,
        cycle_period=1.0,
        day_start=0.0,
        day_end=0.5,
        activity_mode="always",
        sight_weight=0.0,
        sight_radius=0.12,
        smell_radius=0.18,
    )

    simulation.solve()
    heatmap_figure, _ = simulation.plot_solution_heatmap(
        save=True,
        save_path=Path("./script/reproduce_day_night/output/w_0/solution_heatmap.png"),
    )
    animation_figure, _ = simulation.create_solution_gif(
        interval=20,
        fps=20,
        save=True,
        save_path=Path("./script/reproduce_day_night/output/w_0/solution_animation.gif"),
    )

    plt.close(heatmap_figure)
    plt.close(animation_figure)

    mass = simulation.get_mass()
    print("Saved w=0 outputs to script/reproduce_day_night/output/w_0")
    print(f"Initial mass: {mass[0]:.8f}")
    print(f"Final mass:   {mass[-1]:.8f}")


def simulate_w1():
    simulation = DayNightModel1D(
        a_border=0.0,
        b_border=1.0,
        number_of_points=128,
        total_time=1.0,
        dt=1.0e-3,
        initial_condition=gaussian_initial_condition,
        coefficient_attraction=0.1,
        coefficient_diffusion=0.05,
        cycle_period=1.0,
        day_start=0.0,
        day_end=0.5,
        activity_mode="always",
        sight_weight=1.0,
        sight_radius=0.12,
        smell_radius=0.18,
    )

    simulation.solve()
    heatmap_figure, _ = simulation.plot_solution_heatmap(
        save=True,
        save_path=Path("./script/reproduce_day_night/output/w_1/solution_heatmap.png"),
    )
    animation_figure, _ = simulation.create_solution_gif(
        interval=20,
        fps=20,
        save=True,
        save_path=Path("./script/reproduce_day_night/output/w_1/solution_animation.gif"),
    )

    plt.close(heatmap_figure)
    plt.close(animation_figure)

    mass = simulation.get_mass()
    print("Saved w=1 outputs to script/reproduce_day_night/output/w_1")
    print(f"Initial mass: {mass[0]:.8f}")
    print(f"Final mass:   {mass[-1]:.8f}")


def main():
    simulate_w0()
    simulate_w1()


if __name__ == "__main__":
    raise SystemExit(main())