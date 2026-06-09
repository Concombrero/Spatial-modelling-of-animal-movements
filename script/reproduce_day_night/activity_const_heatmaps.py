from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from solver import DayNightModel1D


OUTPUT_DIRECTORY = Path(__file__).resolve().parent / "output"
OUTPUT_PATH = OUTPUT_DIRECTORY / "sight_weight_sunset_heatmaps.png"
NUMBER_OF_POINTS = 256
NUMBER_OF_POPULATIONS = 1
NUMBER_OF_CYCLES = 3
CYCLE_PERIOD = 1.0
TOTAL_TIME = NUMBER_OF_CYCLES * CYCLE_PERIOD
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


def gaussian_initial_condition(x, center=0.5, width=0.08):
	x = np.asarray(x, dtype=float)
	dx = x[1] - x[0]
	length = (x[-1] - x[0]) + dx
	wrapped_distance = ((x - center + 0.5 * length) % length) - 0.5 * length
	values = np.exp(-0.5 * (wrapped_distance / width) ** 2)
	return values / (dx * np.sum(values))


def build_solver(sight_weight, lighting_regime):
	return DayNightModel1D(
		a_border=0.0,
		b_border=1.0,
		number_of_points=NUMBER_OF_POINTS,
		total_time=TOTAL_TIME,
		dt=0.01,
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
		sight_radius=0.1,
		smell_radius=0.2,
	)


def run_case(sight_weight, lighting_regime):
	model = build_solver(sight_weight, lighting_regime)
	model.solve()
	return model


def save_combined_heatmaps(models, sight_weights, lighting_regimes, output_path):
	figure, axes = plt.subplots(
		len(lighting_regimes),
		len(sight_weights),
		figsize=(4.0 * len(sight_weights), 3.6 * len(lighting_regimes)),
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


def main():
	lighting_regimes = build_lighting_regimes()
	models = [
		[run_case(sight_weight, lighting_regime) for sight_weight in SIGHT_WEIGHTS]
		for lighting_regime in lighting_regimes
	]
	save_combined_heatmaps(models, SIGHT_WEIGHTS, lighting_regimes, OUTPUT_PATH)
	print(f"Saved heatmap to {OUTPUT_PATH}")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
