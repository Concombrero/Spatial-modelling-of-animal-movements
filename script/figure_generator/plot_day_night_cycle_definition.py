import argparse
import math
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Wedge

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from script.reproduce_day_night.shared_config import PLOT_STYLE

DAY_COLOR = PLOT_STYLE["day_color"]
NIGHT_COLOR = PLOT_STYLE["night_color"]
EDGE_COLOR = PLOT_STYLE["axis_color"]
DEFAULT_T_SUNSET = 0.65
DEFAULT_NUMBER_OF_DAYS = 4
OUTPUT_PATH = (
    Path(__file__).resolve().parents[2]
    / "article"
    / "figures"
    / "day_night_cycle_definition.png"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Draw a circular day-night cycle with annotated variables."
    )
    parser.add_argument(
        "--t-sunset",
        type=float,
        default=DEFAULT_T_SUNSET,
        help="Daylight proportion t_sunset in the interval [0, 1].",
    )
    parser.add_argument(
        "--number-of-days",
        type=int,
        default=DEFAULT_NUMBER_OF_DAYS,
        help="Number of repeated days shown on the timeline subplot.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=OUTPUT_PATH,
        help="Saved figure path.",
    )
    return parser.parse_args()


def validate_inputs(t_sunset, number_of_days):
    if not 0.0 <= t_sunset <= 1.0:
        raise ValueError("t_sunset must lie in the interval [0, 1].")
    if number_of_days < 1:
        raise ValueError("number_of_days must be a positive integer.")


def phase_to_angle(phase):
    return 90.0 - 360.0 * phase


def polar_to_cartesian(radius, angle_degrees):
    angle_radians = math.radians(angle_degrees)
    return radius * math.cos(angle_radians), radius * math.sin(angle_radians)


def add_radial_line(axis, angle_degrees, inner_radius, outer_radius, style="-", alpha=1.0):
    x_start, y_start = polar_to_cartesian(inner_radius, angle_degrees)
    x_end, y_end = polar_to_cartesian(outer_radius, angle_degrees)
    axis.plot(
        [x_start, x_end],
        [y_start, y_end],
        linestyle=style,
        color=EDGE_COLOR,
        linewidth=1.5,
        alpha=alpha,
        zorder=4,
    )


def add_text_on_ray(axis, angle_degrees, radius, text, **kwargs):
    x_value, y_value = polar_to_cartesian(radius, angle_degrees)
    axis.text(x_value, y_value, text, **kwargs)


def add_sector_label(axis, angle_degrees, radius, text, color):
    add_text_on_ray(
        axis,
        angle_degrees,
        radius,
        text,
        ha="center",
        va="center",
        fontsize=13,
        color=color,
    )


def add_outside_interval_label(axis, angle_degrees, radius, text):
    add_text_on_ray(
        axis,
        angle_degrees,
        radius,
        text,
        ha="center",
        va="center",
        fontsize=12,
    )


def draw_cycle_subplot(axis, t_sunset):
    axis.set_aspect("equal")
    axis.axis("off")

    outer_radius = 1.0
    inner_radius = 0.56
    sunrise_angle = phase_to_angle(0.0)
    sunset_angle = phase_to_angle(t_sunset)
    day_mid_angle = phase_to_angle(0.5 * t_sunset)
    night_mid_phase = t_sunset + 0.5 * (1.0 - t_sunset)
    night_mid_angle = phase_to_angle(night_mid_phase)

    day_wedge = Wedge(
        center=(0.0, 0.0),
        r=outer_radius,
        theta1=sunset_angle,
        theta2=sunrise_angle,
        width=outer_radius - inner_radius,
        facecolor=DAY_COLOR,
        edgecolor=EDGE_COLOR,
        linewidth=1.6,
        zorder=2,
    )
    night_wedge = Wedge(
        center=(0.0, 0.0),
        r=outer_radius,
        theta1=sunrise_angle - 360.0,
        theta2=sunset_angle,
        width=outer_radius - inner_radius,
        facecolor=NIGHT_COLOR,
        edgecolor=EDGE_COLOR,
        linewidth=1.6,
        zorder=1,
    )
    center_disk = Circle(
        (0.0, 0.0),
        inner_radius,
        facecolor="white",
        edgecolor=EDGE_COLOR,
        linewidth=1.2,
        zorder=3,
    )

    axis.add_patch(night_wedge)
    axis.add_patch(day_wedge)
    axis.add_patch(center_disk)

    add_radial_line(axis, sunrise_angle, inner_radius, outer_radius + 0.1, alpha=0.75)
    add_radial_line(axis, sunset_angle, inner_radius, outer_radius + 0.1, style="--", alpha=0.85)

    add_sector_label(
        axis,
        day_mid_angle,
        0.78,
        r"$\mathcal{T}_{\mathrm{day}}$",
        EDGE_COLOR,
    )
    add_sector_label(
        axis,
        night_mid_angle,
        0.78,
        r"$\mathcal{T}_{\mathrm{night}}$",
        "white",
    )

    add_outside_interval_label(axis, day_mid_angle, 1.23, r"$[0,t_{\mathrm{sunset}})$")
    add_outside_interval_label(
        axis,
        night_mid_angle,
        1.3,
        r"$[t_{\mathrm{sunset}},1)$",
    )

    add_text_on_ray(
        axis,
        sunrise_angle,
        1.34,
        r"$t_{\mathrm{sunrise}} = 0$",
        ha="center",
        va="center",
        fontsize=12,
    )
    add_text_on_ray(
        axis,
        sunset_angle,
        1.36,
        r"$t_{\mathrm{sunset}}$",
        ha="center",
        va="center",
        fontsize=12,
    )
    

    axis.set_xlim(-1.55, 1.55)
    axis.set_ylim(-1.45, 1.45)


def draw_repeated_timeline(axis, t_sunset, number_of_days):
    timeline_y = 0.0
    line_width = 18

    for day_index in range(number_of_days):
        cycle_start = float(day_index)
        sunset_time = cycle_start + t_sunset
        cycle_end = cycle_start + 1.0

        if t_sunset > 0.0:
            axis.hlines(
                timeline_y,
                cycle_start,
                sunset_time,
                color=DAY_COLOR,
                linewidth=line_width,
                zorder=2,
            )
        if t_sunset < 1.0:
            axis.hlines(
                timeline_y,
                sunset_time,
                cycle_end,
                color=NIGHT_COLOR,
                linewidth=line_width,
                zorder=2,
            )

        axis.vlines(
            cycle_start,
            timeline_y - 0.18,
            timeline_y + 0.18,
            color=EDGE_COLOR,
            linewidth=1.0,
            alpha=0.55,
            zorder=3,
        )
        if 0.0 < t_sunset < 1.0:
            axis.vlines(
                sunset_time,
                timeline_y - 0.16,
                timeline_y + 0.16,
                color=EDGE_COLOR,
                linewidth=1.0,
                linestyle="--",
                alpha=0.8,
                zorder=3,
            )

    axis.vlines(
        float(number_of_days),
        timeline_y - 0.18,
        timeline_y + 0.18,
        color=EDGE_COLOR,
        linewidth=1.0,
        alpha=0.55,
        zorder=3,
    )

    axis.set_xlim(0.0, float(number_of_days))
    axis.set_ylim(-0.65, 0.72)
    axis.set_yticks([])
    axis.spines["left"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["top"].set_visible(False)
    axis.grid(axis="x", color=PLOT_STYLE["guide_color"], alpha=0.3, linestyle=":")
    axis.set_xlabel(r"Time $t$ (days)")
    axis.set_xticks([float(index) for index in range(number_of_days + 1)])
    axis.set_xticklabels([str(index) for index in range(number_of_days + 1)])

    if t_sunset >= 0.16:
        axis.text(
            0.5 * t_sunset,
            0.3,
            r"$\mathcal{T}_{\mathrm{day}}$",
            ha="center",
            va="bottom",
            fontsize=12,
        )
    if (1.0 - t_sunset) >= 0.16:
        axis.text(
            t_sunset + 0.5 * (1.0 - t_sunset),
            0.3,
            r"$\mathcal{T}_{\mathrm{night}}$",
            ha="center",
            va="bottom",
            fontsize=12,
            color=EDGE_COLOR,
        )

    if 0.0 < t_sunset < 1.0:
        axis.text(
            t_sunset,
            -0.35,
            r"$t_{\mathrm{sunset}}$",
            ha="center",
            va="top",
            fontsize=11,
        )


def create_cycle_figure(t_sunset, number_of_days):
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(12.2, 5.6),
        gridspec_kw={"width_ratios": [1.15, 1.85]},
        constrained_layout=True,
    )
    figure.patch.set_facecolor("white")
    draw_cycle_subplot(axes[0], t_sunset)
    draw_repeated_timeline(axes[1], t_sunset, number_of_days)
    return figure


def main():
    args = parse_args()
    validate_inputs(args.t_sunset, args.number_of_days)
    figure = create_cycle_figure(args.t_sunset, args.number_of_days)

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(figure)

    print(f"Saved day-night cycle illustration to {args.output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())