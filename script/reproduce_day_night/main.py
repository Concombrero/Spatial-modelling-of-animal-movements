from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from solver import DayNightModel1D


OUTPUT_DIRECTORY = Path(__file__).resolve().parent / "output"
NUMBER_OF_POINTS = 256
NUMBER_OF_POPULATIONS = 1
NUMBER_OF_CYCLES = 2
CYCLE_PERIOD = 1.0
TOTAL_TIME = NUMBER_OF_CYCLES * CYCLE_PERIOD
DAY_START = 8.0
DAY_END = 22.0
COEFFICIENT_ATTRACTION = np.array([[0.1]])
COEFFICIENT_DIFFUSION = np.array([0.05])
CASES = (
	("pure_sight", 1.0),
	("pure_smell", 0.0),
)


def gaussian_initial_condition(x, center=0.35, width=0.08):
	x = np.asarray(x, dtype=float)
	dx = x[1] - x[0]
	length = (x[-1] - x[0]) + dx
	wrapped_distance = ((x - center + 0.5 * length) % length) - 0.5 * length
	values = np.exp(-0.5 * (wrapped_distance / width) ** 2)
	return values / (dx * np.sum(values))


def build_solver(sight_weight):
	return DayNightModel1D(
		a_border=0.0,
		b_border=1.0,
		number_of_points=NUMBER_OF_POINTS,
		total_time=TOTAL_TIME,
		dt=0.01,
		initial_condition=gaussian_initial_condition,
		coefficient_attraction=COEFFICIENT_ATTRACTION,
		coefficient_diffusion=COEFFICIENT_DIFFUSION,
		cycle_period=CYCLE_PERIOD,
		number_of_population=NUMBER_OF_POPULATIONS,
		day_start=DAY_START,
		day_end=DAY_END,
		time_input_mode="clock",
		clock_hours_per_cycle=24.0,
		activity_mode="always",
		sight_weight=sight_weight,
		sight_radius=0.10,
		smell_radius=0.15,
	)


def save_heatmap_without_title(model, output_path):
	figure, axis = model.plot_solution_heatmap()
	axis.set_title("")
	output_path.parent.mkdir(parents=True, exist_ok=True)
	figure.savefig(output_path, bbox_inches="tight")
	plt.close(figure)


def run_case(case_name, sight_weight):
	model = build_solver(sight_weight)
	model.solve()
	output_path = OUTPUT_DIRECTORY / f"{case_name}_heatmap.png"
	save_heatmap_without_title(model, output_path)
	return output_path


def main():
	output_paths = [run_case(case_name, sight_weight) for case_name, sight_weight in CASES]
	for output_path in output_paths:
		print(f"Saved heatmap to {output_path}")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
