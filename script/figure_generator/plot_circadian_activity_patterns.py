import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


ACTIVE_FACE_COLOR = "#F8EDE6"
ACTIVE_EDGE_COLOR = "#C77657"
INACTIVE_FACE_COLOR = "#EEF4FB"
INACTIVE_EDGE_COLOR = "#7C97B6"
AXIS_COLOR = "#111111"
GUIDE_COLOR = "#C5CCD5"
DEFAULT_T_SUNSET = 0.5
OUTPUT_PATH = (
    Path(__file__).resolve().parents[2]
    / "article"
    / "figures"
    / "circadian_activity_patterns.png"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Draw common circadian activity patterns against a day-night cycle."
        )
    )
    parser.add_argument(
        "--t-sunset",
        type=float,
        default=DEFAULT_T_SUNSET,
        help="Daylight proportion t_sunset in the interval [0, 1].",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=OUTPUT_PATH,
        help="Saved figure path.",
    )
    return parser.parse_args()


def validate_inputs(t_sunset):
    if not 0.0 <= t_sunset <= 1.0:
        raise ValueError("t_sunset must lie in the interval [0, 1].")


def interpolate(start, end, fraction):
    return start + fraction * (end - start)


def format_phase(value):
    return f"{value:.2f}".rstrip("0").rstrip(".")


def normalise_intervals(intervals):
    cleaned = []

    for start, end in intervals:
        left = max(0.0, min(1.0, float(start)))
        right = max(0.0, min(1.0, float(end)))
        if right - left <= 1e-9:
            continue
        cleaned.append((left, right))

    if not cleaned:
        return []

    cleaned.sort()
    merged = [list(cleaned[0])]
    for start, end in cleaned[1:]:
        if start <= merged[-1][1] + 1e-9:
            merged[-1][1] = max(merged[-1][1], end)
            continue
        merged.append([start, end])

    return [(start, end) for start, end in merged]


def build_segments(active_intervals):
    segments = []
    cursor = 0.0

    for start, end in normalise_intervals(active_intervals):
        if start > cursor + 1e-9:
            segments.append((cursor, start, "inactive"))
        segments.append((start, end, "active"))
        cursor = end

    if cursor < 1.0 - 1e-9:
        segments.append((cursor, 1.0, "inactive"))

    return segments


def build_pattern_groups(t_sunset):
    quarter_points = [0.0, 0.25, 0.5, 0.75, 1.0]

    return [
        {
            "label": "Diurnal",
            "tracks": [build_segments([(quarter_points[0], quarter_points[2])])],
        },
        {
            "label": "Nocturnal",
            "tracks": [build_segments([(quarter_points[2], quarter_points[4])])],
        },
        {
            "label": "Matutinal",
            "tracks": [
                build_segments(
                    [
                        (
                            quarter_points[1],
                            quarter_points[3],
                        )
                    ]
                ),
                build_segments(
                    [
                        (quarter_points[0], quarter_points[1]),
                        (quarter_points[3], quarter_points[4]),
                    ]
                ),
            ],
        },
        {
            "label": "Polyphasic",
            "tracks": [
                build_segments(
                    [
                        (quarter_points[0], quarter_points[1]),
                        (quarter_points[2], quarter_points[3]),
                    ]
                ),
                build_segments(
                    [
                        (quarter_points[1], quarter_points[2]),
                        (quarter_points[3], quarter_points[4]),
                    ]
                ),
            ],
        },
    ]


def segment_style(state):
    if state == "active":
        return ACTIVE_FACE_COLOR, ACTIVE_EDGE_COLOR
    return INACTIVE_FACE_COLOR, INACTIVE_EDGE_COLOR


def draw_segment_row(axis, y_center, segments, bar_height):
    for start, end, state in segments:
        face_color, edge_color = segment_style(state)
        axis.add_patch(
            Rectangle(
                (start, y_center - 0.5 * bar_height),
                end - start,
                bar_height,
                facecolor=face_color,
                edgecolor=edge_color,
                linewidth=1.8,
                zorder=3,
            )
        )


def draw_state_labels(axis, t_sunset, y_center):
    if t_sunset >= 0.16:
        axis.text(
            0.5 * t_sunset,
            y_center,
            r"$\mathcal{T}_{\mathrm{active}}$",
            ha="center",
            va="center",
            fontsize=11,
            color=ACTIVE_EDGE_COLOR,
            zorder=5,
        )

    if 1.0 - t_sunset >= 0.20:
        axis.text(
            t_sunset + 0.5 * (1.0 - t_sunset),
            y_center,
            r"$\mathcal{T}_{\mathrm{inactive}}$",
            ha="center",
            va="center",
            fontsize=11,
            color=INACTIVE_EDGE_COLOR,
            zorder=5,
        )


def draw_time_axis(axis, t_sunset, y_value):
    axis.annotate(
        "",
        xy=(1.04, y_value),
        xytext=(-0.02, y_value),
        arrowprops={"arrowstyle": "->", "linewidth": 1.6, "color": AXIS_COLOR},
        annotation_clip=False,
        zorder=4,
    )

    tick_positions = [0.0, 0.25, 0.5, 0.75, 1.0]
    for tick_x in tick_positions:
        axis.plot(
            [tick_x, tick_x],
            [y_value - 0.08, y_value + 0.08],
            color=AXIS_COLOR,
            linewidth=1.4,
            zorder=4,
        )

    axis.text(
        interpolate(0.0, t_sunset, 0.5),
        y_value + 0.22,
        r"$\mathcal{T}_{\mathrm{day}}$",
        ha="center",
        va="bottom",
        fontsize=14,
    )
    axis.text(
        interpolate(t_sunset, 1.0, 0.5),
        y_value + 0.22,
        r"$\mathcal{T}_{\mathrm{night}}$",
        ha="center",
        va="bottom",
        fontsize=14,
    )
    axis.text(
        t_sunset,
        y_value + 0.18,
        rf"$t_{{\mathrm{{sunset}}}}={format_phase(t_sunset)}$",
        ha="center",
        va="bottom",
        fontsize=12,
    )


def draw_group_bracket(axis, x_value, y_top, y_bottom):
    bracket_width = 0.025
    axis.plot(
        [x_value + bracket_width, x_value, x_value, x_value + bracket_width],
        [y_top, y_top, y_bottom, y_bottom],
        color=AXIS_COLOR,
        linewidth=1.3,
        clip_on=False,
        zorder=4,
    )


def create_figure(t_sunset):
    figure, axis = plt.subplots(figsize=(11.6, 6.3), constrained_layout=True)
    figure.patch.set_facecolor("white")
    axis.set_facecolor("white")
    axis.axis("off")
    axis.set_xlim(-0.34, 1.05)
    axis.set_ylim(0.2, 6.95)

    timeline_y = 6.35
    first_row_y = 5.35
    row_gap = 0.78
    group_gap = 0.24
    bar_height = 0.46

    axis.plot(
        [t_sunset, t_sunset],
        [0.9, timeline_y - 0.14],
        color=GUIDE_COLOR,
        linewidth=1.2,
        linestyle="--",
        zorder=1,
    )
    draw_time_axis(axis, t_sunset, timeline_y)

    current_y = first_row_y
    for group in build_pattern_groups(t_sunset):
        row_centres = []
        for row_index, segments in enumerate(group["tracks"]):
            row_centres.append(current_y)
            draw_segment_row(axis, current_y, segments, bar_height)
            if group["label"] == "Diurnal" and row_index == 0:
                draw_state_labels(axis, t_sunset, current_y)
            current_y -= row_gap

        axis.text(
            -0.16,
            sum(row_centres) / len(row_centres),
            group["label"],
            ha="right",
            va="center",
            fontsize=13,
        )

        if len(row_centres) > 1:
            draw_group_bracket(
                axis,
                -0.055,
                row_centres[0] + 0.5 * bar_height,
                row_centres[-1] - 0.5 * bar_height,
            )

        current_y -= group_gap

    return figure


def main():
    args = parse_args()
    validate_inputs(args.t_sunset)
    figure = create_figure(args.t_sunset)

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(figure)

    print(f"Saved circadian activity illustration to {args.output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
