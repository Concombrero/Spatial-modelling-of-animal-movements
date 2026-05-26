from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np


def plot_day_night_cycle(t0=0.0, t_day=14.0, T=24.0):
	if T <= 0:
		raise ValueError("T must be strictly positive.")
	if T != 24.0:
		raise ValueError("This plot is defined on a 24-hour cycle, so T must be 24.")

	t0 = t0 % T
	t_day = t_day % T
	day_duration = (t_day - t0) % T
	if day_duration == 0:
		raise ValueError("t0 and t_day must define a non-zero day interval.")

	fig, ax = plt.subplots(figsize=(6.2, 6.2), subplot_kw={"projection": "polar"})

	ax.set_theta_direction(-1)
	ax.set_theta_zero_location("N")

	inner_radius = 0.42
	ring_width = 0.38
	outer_radius = inner_radius + ring_width
	start_angle = 2 * np.pi * t0 / T
	day_angle = 2 * np.pi * day_duration / T
	night_angle = 2 * np.pi - day_angle

	ax.bar(
		x=start_angle + day_angle / 2,
		height=ring_width,
		width=day_angle,
		bottom=inner_radius,
		color="#f4c542",
		edgecolor="black",
		align="center",
	)
	ax.bar(
		x=start_angle + day_angle + night_angle / 2,
		height=ring_width,
		width=night_angle,
		bottom=inner_radius,
		color="#355c7d",
		edgecolor="black",
		align="center",
	)

	ax.plot(
		[start_angle, start_angle],
		[inner_radius, outer_radius],
		color="black",
		linestyle="--",
		linewidth=1.5,
	)
	ax.plot(
		[start_angle + day_angle, start_angle + day_angle],
		[inner_radius, outer_radius],
		color="black",
		linestyle="--",
		linewidth=1.5,
	)

	ax.text(
		start_angle + day_angle / 2,
		inner_radius + ring_width / 2,
		r"$I_{\mathrm{day}}$",
		ha="center",
		va="center",
		fontsize=13,
	)
	ax.text(
		start_angle + day_angle + night_angle / 2,
		inner_radius + ring_width / 2,
		r"$I_{\mathrm{night}}$",
		ha="center",
		va="center",
		fontsize=13,
		color="white",
	)
	ax.text(start_angle, outer_radius + 0.12, r"$t_0$", ha="center", va="center")
	ax.text(start_angle + day_angle, outer_radius + 0.12, r"$t_{\mathrm{day}}$", ha="center", va="center")

	hours = np.array([0, 4, 8, 12, 16, 20], dtype=float)
	ax.set_xticks(2 * np.pi * hours / T)
	ax.set_xticklabels([f"{int(hour)} h" for hour in hours])
	ax.set_yticks([])
	ax.set_ylim(0, outer_radius + 0.1)
	ax.grid(False)
	ax.spines["polar"].set_visible(False)
	ax.set_title("Day-night partition over a 24-hour cycle", va="bottom")

	fig.tight_layout()
	return fig, ax


def save_day_night_cycle(t0, t_day, savepath):
	fig, ax = plot_day_night_cycle(t0=t0, t_day=t_day)
	savepath = Path(savepath)
	savepath.parent.mkdir(parents=True, exist_ok=True)
	fig.savefig(savepath, bbox_inches="tight", dpi=300)

	return fig, ax


def main(argc, argv):
	if argc != 4:
		program_name = Path(argv[0]).name if argc > 0 else "plot_day_night_cycle.py"
		raise SystemExit(f"Usage: {program_name} <t0> <t_day> <savepath>")

	t0 = float(argv[1])
	t_day = float(argv[2])
	savepath = argv[3]
	_, _ = save_day_night_cycle(t0=t0, t_day=t_day, savepath=savepath)
	return 0


if __name__ == "__main__":
	raise SystemExit(main(len(sys.argv), sys.argv))
