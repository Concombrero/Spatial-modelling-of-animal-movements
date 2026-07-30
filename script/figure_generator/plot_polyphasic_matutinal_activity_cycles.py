import argparse
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from script.reproduce_day_night.shared_config import (
    MATUTINAL_CODES,
    ONE_POPULATION_SIMULATION_CONFIG,
    POLYPHASIC_CODES,
    PLOT_STYLE,
    activity_regimes_for_codes,
    apply_plot_typography,
    describe_lighting_regime,
    display_activity_code,
    resolve_experiment_config,
)


apply_plot_typography()

EXPERIMENT_CONFIG = resolve_experiment_config(
    ONE_POPULATION_SIMULATION_CONFIG,
    "spread_polyphasic_matutinal",
)
INACTIVE_FACE_COLOR = PLOT_STYLE["inactive_face_color"]
INACTIVE_EDGE_COLOR = PLOT_STYLE["inactive_edge_color"]
AXIS_COLOR = PLOT_STYLE["axis_color"]
GUIDE_COLOR = PLOT_STYLE["guide_color"]
DEFAULT_SUNSET_VALUES = EXPERIMENT_CONFIG["sunset_values"]
ACTIVITY_REGIMES = activity_regimes_for_codes(POLYPHASIC_CODES + MATUTINAL_CODES)
OUTPUT_PATH = (
    Path(__file__).resolve().parents[2]
    / "article"
    / "figures"
    / "polyphasic_matutinal_activity_cycles.png"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Draw the activity cycles used in the polyphasic/matutinal "
            "day-night spread experiment."
        )
    )
    parser.add_argument(
        "--sunset-values",
        nargs="+",
        type=float,
        default=list(DEFAULT_SUNSET_VALUES),
        help="Daylight proportions t_sunset in the interval [0, 1].",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=OUTPUT_PATH,
        help="Saved figure path.",
    )
    return parser.parse_args()


def validate_inputs(sunset_values):
    if not sunset_values:
        raise ValueError("At least one t_sunset value must be provided.")

    if any(value < 0.0 or value > 1.0 for value in sunset_values):
        raise ValueError("Each t_sunset must lie in the interval [0, 1].")

    rounded_values = [round(value, 12) for value in sunset_values]
    if len(set(rounded_values)) != len(rounded_values):
        raise ValueError("t_sunset values must be distinct.")


def format_phase(value):
    return f"{value:.2f}".rstrip("0").rstrip(".")


def interpolate(start, end, fraction):
    return start + fraction * (end - start)


def lighting_label(t_sunset):
    label = describe_lighting_regime(float(t_sunset))
    return label[0].upper() + label[1:]


def normalise_intervals(intervals):
    cleaned = []

    for start, end in intervals:
        left = max(0.0, min(1.0, float(start)))
        right = max(0.0, min(1.0, float(end)))
        if right - left <= 1.0e-9:
            continue
        cleaned.append((left, right))

    if not cleaned:
        return []

    cleaned.sort()
    merged = [list(cleaned[0])]
    for start, end in cleaned[1:]:
        if start <= merged[-1][1] + 1.0e-9:
            merged[-1][1] = max(merged[-1][1], end)
            continue
        merged.append([start, end])

    return [(start, end) for start, end in merged]


def build_segments(active_intervals):
    segments = []
    cursor = 0.0

    for start, end in normalise_intervals(active_intervals):
        if start > cursor + 1.0e-9:
            segments.append((cursor, start, "inactive"))
        segments.append((start, end, "active"))
        cursor = end

    if cursor < 1.0 - 1.0e-9:
        segments.append((cursor, 1.0, "inactive"))

    return segments


def segment_style(state, regime):
    if state == "active":
        return regime["color"], regime["color"]
    return INACTIVE_FACE_COLOR, INACTIVE_EDGE_COLOR


def draw_segment_row(axis, y_center, segments, bar_height, regime):
    for start, end, state in segments:
        face_color, edge_color = segment_style(state, regime)
        axis.add_patch(
            Rectangle(
                (start, y_center - 0.5 * bar_height),
                end - start,
                bar_height,
                facecolor=face_color,
                edgecolor=edge_color,
                linewidth=1.6,
                zorder=3,
            )
        )


def draw_time_axis(axis, t_sunset, y_value):
    axis.annotate(
        "",
        xy=(1.04, y_value),
        xytext=(-0.02, y_value),
        arrowprops={"arrowstyle": "->", "linewidth": 1.5, "color": AXIS_COLOR},
        annotation_clip=False,
        zorder=4,
    )

    tick_positions = [0.0, 0.25, 0.5, 0.75, 1.0]
    for tick_x in tick_positions:
        axis.plot(
            [tick_x, tick_x],
            [y_value - 0.07, y_value + 0.07],
            color=AXIS_COLOR,
            linewidth=1.2,
            zorder=4,
        )

    axis.text(
        interpolate(0.0, t_sunset, 0.5),
        y_value + 0.18,
        r"$\mathcal{T}_{\mathrm{day}}$",
        ha="center",
        va="bottom",
        fontsize=12,
    )
    axis.text(
        interpolate(t_sunset, 1.0, 0.5),
        y_value + 0.18,
        r"$\mathcal{T}_{\mathrm{night}}$",
        ha="center",
        va="bottom",
        fontsize=12,
    )
    axis.text(
        t_sunset,
        y_value + 0.42,
        rf"$t_{{\mathrm{{sunset}}}}={format_phase(t_sunset)}$",
        ha="center",
        va="bottom",
        fontsize=11,
    )


def configure_panel(axis, t_sunset, show_row_labels):
    axis.set_facecolor("white")
    axis.axis("off")
    axis.set_xlim(-0.34 if show_row_labels else -0.02, 1.05)
    axis.set_ylim(0.2, 5.45)

    timeline_y = 4.7
    first_row_y = 3.85
    row_gap = 0.86
    bar_height = 0.44

    axis.plot(
        [t_sunset, t_sunset],
        [0.6, timeline_y - 0.1],
        color=GUIDE_COLOR,
        linewidth=1.1,
        linestyle="--",
        zorder=1,
    )
    draw_time_axis(axis, t_sunset, timeline_y)

    current_y = first_row_y
    for regime in ACTIVITY_REGIMES:
        draw_segment_row(
            axis,
            current_y,
            build_segments(regime["periods"]),
            bar_height,
            regime,
        )
        if show_row_labels:
            axis.text(
                -0.10,
                current_y,
                f"{display_activity_code(regime['code'])}  {regime['label']}",
                ha="right",
                va="center",
                fontsize=12,
                color=regime["color"],
            )
        current_y -= row_gap

    axis.set_title(lighting_label(t_sunset), pad=8)


def create_figure(sunset_values):
    figure, axes = plt.subplots(
        1,
        len(sunset_values),
        figsize=(4.2 * len(sunset_values), 5.7),
        sharey=True,
    )
    figure.patch.set_facecolor("white")
    axes = np.atleast_1d(axes)

    for index, (axis, t_sunset) in enumerate(zip(axes, sunset_values)):
        configure_panel(axis, t_sunset, show_row_labels=index == 0)
    figure.tight_layout()
    return figure


def main():
    args = parse_args()
    sunset_values = tuple(float(value) for value in args.sunset_values)
    validate_inputs(sunset_values)
    figure = create_figure(sunset_values)

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(figure)

    print(f"Saved activity-cycle figure to {args.output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())