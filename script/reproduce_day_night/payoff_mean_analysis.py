import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


CURRENT_DIRECTORY = Path(__file__).resolve().parent
if str(CURRENT_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIRECTORY))

from payoff_matrix import ACTIVITY_REGIMES


CASE_PAYOFF_FILENAME = "case_payoffs.csv"
RUN_CONFIG_FILENAME = "run_config.json"
ACTIVITY_ORDER = [regime["code"] for regime in ACTIVITY_REGIMES]
ACTIVITY_LABELS = {regime["code"]: regime["label"] for regime in ACTIVITY_REGIMES}
NUMERIC_PARAMETERS = {"w1", "w2"}
PARAMETER_ALIASES = {
    "w1": "w1",
    "w2": "w2",
    "cycle1": "cycle1",
    "cycle2": "cycle2",
    "circadian1": "cycle1",
    "circadian2": "cycle2",
    "circadian_cycle_1": "cycle1",
    "circadian_cycle_2": "cycle2",
    "circadian-cycle-1": "cycle1",
    "circadian-cycle-2": "cycle2",
}
PARAMETER_TITLES = {
    "w1": "Mean payoff as a function of w1",
    "w2": "Mean payoff as a function of w2",
    "cycle1": "Mean payoff as a function of circadian cycle 1",
    "cycle2": "Mean payoff as a function of circadian cycle 2",
}
PARAMETER_LABELS = {
    "w1": "w1",
    "w2": "w2",
    "cycle1": "Circadian cycle 1",
    "cycle2": "Circadian cycle 2",
}
PAYOFF_PLAYER_CHOICES = ("auto", "legacy", "prey", "predator")


def parse_bool(value):
    if isinstance(value, bool):
        return value

    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "f", "no", "n", "off"}:
        return False

    raise argparse.ArgumentTypeError(
        "Expected a boolean value such as true/false, yes/no, or 1/0."
    )


def normalize_parameter_name(value):
    normalized = value.strip().lower().replace(" ", "_")
    if normalized not in PARAMETER_ALIASES:
        available = ", ".join(sorted(PARAMETER_ALIASES))
        raise argparse.ArgumentTypeError(
            f"Unsupported x-axis parameter '{value}'. Choose from: {available}."
        )
    return PARAMETER_ALIASES[normalized]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Analyze one payoff output folder or a sweep of payoff folders and plot "
            "the mean payoff against w1, w2, circadian cycle 1, or circadian cycle 2."
        )
    )
    parser.add_argument(
        "--x-axis",
        required=True,
        type=normalize_parameter_name,
        help=(
            "Parameter used on the x axis. Supported values: w1, w2, cycle1, "
            "cycle2, circadian-cycle-1, circadian-cycle-2."
        ),
    )
    parser.add_argument(
        "--payoff-dir",
        required=True,
        type=Path,
        help=(
            "Folder containing payoff outputs. This can be a single run folder with "
            f"{CASE_PAYOFF_FILENAME} and {RUN_CONFIG_FILENAME}, or a parent folder "
            "containing several such runs."
        ),
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Path of the saved plot, for example output/mean_vs_w1.png.",
    )
    parser.add_argument(
        "--show-variance",
        default=False,
        type=parse_bool,
        help=(
            "Whether to add a second subplot showing the variance for each x value. "
            "Default: false."
        ),
    )
    parser.add_argument(
        "--payoff-player",
        choices=PAYOFF_PLAYER_CHOICES,
        default="auto",
        help=(
            "Which payoff column to average. 'legacy' uses the historical single "
            "payoff column, 'prey' uses prey_payoff, 'predator' uses predator_payoff. "
            "'auto' uses 'payoff' when it is the only available value column and "
            "otherwise requires an explicit prey/predator choice. Default: auto."
        ),
    )
    return parser.parse_args()


def is_run_directory(path):
    return (
        (path / CASE_PAYOFF_FILENAME).is_file()
        and (path / RUN_CONFIG_FILENAME).is_file()
    )

def discover_run_directories(payoff_dir):
    payoff_dir = Path(payoff_dir).expanduser().resolve()
    run_directories = set()

    if is_run_directory(payoff_dir):
        run_directories.add(payoff_dir)

    for config_path in payoff_dir.rglob(RUN_CONFIG_FILENAME):
        run_directory = config_path.parent.resolve()
        if is_run_directory(run_directory):
            run_directories.add(run_directory)

    return sorted(run_directories)


def load_run_weights(run_directory):
    config_path = run_directory / RUN_CONFIG_FILENAME
    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)

    weights = config.get("weights")
    if not isinstance(weights, (list, tuple)) or len(weights) != 2:
        raise ValueError(
            f"Expected two weights in {config_path}, found: {weights!r}."
        )

    return float(weights[0]), float(weights[1])


def resolve_payoff_column(case_payoff_path, fieldnames, payoff_player):
    available_fields = set(fieldnames or [])

    if payoff_player == "legacy":
        if "payoff" not in available_fields:
            raise ValueError(
                f"Column 'payoff' is not available in {case_payoff_path}."
            )
        return "payoff", "payoff"

    if payoff_player == "prey":
        if "prey_payoff" not in available_fields:
            raise ValueError(
                f"Column 'prey_payoff' is not available in {case_payoff_path}."
            )
        return "prey_payoff", "prey"

    if payoff_player == "predator":
        if "predator_payoff" not in available_fields:
            raise ValueError(
                f"Column 'predator_payoff' is not available in {case_payoff_path}."
            )
        return "predator_payoff", "predator"

    if "payoff" in available_fields:
        return "payoff", "payoff"

    if {"prey_payoff", "predator_payoff"}.issubset(available_fields):
        raise ValueError(
            "This run stores separate prey and predator payoffs. "
            "Choose --payoff-player prey or --payoff-player predator."
        )

    raise ValueError(
        f"Could not find a supported payoff column in {case_payoff_path}."
    )


def load_records(run_directory, payoff_player):
    w1, w2 = load_run_weights(run_directory)
    case_payoff_path = run_directory / CASE_PAYOFF_FILENAME
    records = []

    with case_payoff_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required_fields = {"prey", "predator"}
        missing_fields = required_fields.difference(reader.fieldnames or [])
        if missing_fields:
            missing_as_text = ", ".join(sorted(missing_fields))
            raise ValueError(
                f"Missing columns {missing_as_text} in {case_payoff_path}."
            )

        payoff_column, _ = resolve_payoff_column(
            case_payoff_path,
            reader.fieldnames,
            payoff_player,
        )

        for row in reader:
            records.append(
                {
                    "w1": w1,
                    "w2": w2,
                    "cycle1": row["prey"].strip(),
                    "cycle2": row["predator"].strip(),
                    "payoff": float(row[payoff_column]),
                }
            )

    return records


def collect_records(payoff_dir, payoff_player):
    run_directories = discover_run_directories(payoff_dir)
    if not run_directories:
        raise FileNotFoundError(
            "No payoff runs were found. The folder must contain at least one "
            f"{CASE_PAYOFF_FILENAME} and {RUN_CONFIG_FILENAME} pair."
        )

    records = []
    for run_directory in run_directories:
        records.extend(load_records(run_directory, payoff_player))

    if not records:
        raise ValueError("The selected payoff folder does not contain any payoff rows.")

    return run_directories, records


def compute_group_statistics(records, parameter):
    grouped_values = defaultdict(list)
    for record in records:
        grouped_values[record[parameter]].append(record["payoff"])

    if parameter in NUMERIC_PARAMETERS:
        ordered_keys = sorted(grouped_values)
    else:
        known_keys = [key for key in ACTIVITY_ORDER if key in grouped_values]
        unknown_keys = sorted(key for key in grouped_values if key not in ACTIVITY_LABELS)
        ordered_keys = known_keys + unknown_keys

    summary = []
    for key in ordered_keys:
        values = np.asarray(grouped_values[key], dtype=float)
        summary.append(
            {
                "x": key,
                "count": int(values.size),
                "mean": float(np.mean(values)),
                "variance": float(np.var(values)),
            }
        )

    return summary


def format_cycle_tick_label(code):
    label = ACTIVITY_LABELS.get(code)
    if label is None:
        return code
    return f"{code}\n{label}"


def plot_numeric_series(axis, x_values, y_values, y_label, color):
    axis.plot(x_values, y_values, marker="o", linewidth=2, color=color)
    axis.set_ylabel(y_label)
    axis.grid(True, axis="both", alpha=0.25)


def plot_categorical_series(axis, x_labels, y_values, y_label, color):
    positions = np.arange(len(x_labels), dtype=float)
    axis.bar(positions, y_values, color=color, edgecolor="black", alpha=0.85)
    axis.set_xticks(positions)
    axis.set_xticklabels(x_labels, rotation=15)
    axis.set_ylabel(y_label)
    axis.grid(True, axis="y", alpha=0.25)


def payoff_axis_label(payoff_player):
    if payoff_player == "prey":
        return "Mean prey payoff"
    if payoff_player == "predator":
        return "Mean predator payoff"
    return "Mean payoff"


def payoff_title(parameter, payoff_player):
    if payoff_player == "prey":
        return PARAMETER_TITLES[parameter].replace("Mean payoff", "Mean prey payoff")
    if payoff_player == "predator":
        return PARAMETER_TITLES[parameter].replace(
            "Mean payoff",
            "Mean predator payoff",
        )
    return PARAMETER_TITLES[parameter]


def save_plot(summary, parameter, output_path, show_variance, payoff_player):
    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    is_numeric = parameter in NUMERIC_PARAMETERS
    x_values = [item["x"] for item in summary]
    means = [item["mean"] for item in summary]
    variances = [item["variance"] for item in summary]
    mean_label = payoff_axis_label(payoff_player)

    if show_variance:
        figure, axes = plt.subplots(
            2,
            1,
            figsize=(10, 8),
            sharex=not is_numeric,
            constrained_layout=True,
        )
        mean_axis, variance_axis = axes
    else:
        figure, mean_axis = plt.subplots(figsize=(10, 5), constrained_layout=True)
        variance_axis = None

    if is_numeric:
        numeric_x_values = [float(value) for value in x_values]
        plot_numeric_series(mean_axis, numeric_x_values, means, mean_label, "tab:blue")
        mean_axis.set_xlabel(PARAMETER_LABELS[parameter])
        if variance_axis is not None:
            plot_numeric_series(
                variance_axis,
                numeric_x_values,
                variances,
                "Variance",
                "tab:orange",
            )
            variance_axis.set_xlabel(PARAMETER_LABELS[parameter])
    else:
        category_labels = [format_cycle_tick_label(value) for value in x_values]
        plot_categorical_series(mean_axis, category_labels, means, mean_label, "tab:blue")
        mean_axis.set_xlabel(PARAMETER_LABELS[parameter])
        if variance_axis is not None:
            plot_categorical_series(
                variance_axis,
                category_labels,
                variances,
                "Variance",
                "tab:orange",
            )
            variance_axis.set_xlabel(PARAMETER_LABELS[parameter])

    figure.suptitle(payoff_title(parameter, payoff_player))
    figure.savefig(output_path, dpi=300)
    plt.close(figure)


def print_summary(
    summary,
    parameter,
    run_directories,
    output_path,
    show_variance,
    payoff_player,
):
    print(f"Loaded {len(run_directories)} payoff run(s).")
    print(f"Grouped by: {PARAMETER_LABELS[parameter]}")
    if payoff_player != "auto":
        print(f"Payoff column: {payoff_player}")
    print(f"Variance subplot: {'on' if show_variance else 'off'}")
    print(f"Saved figure: {output_path}")
    print("value,count,mean,variance")
    for item in summary:
        x_value = item["x"]
        if parameter in NUMERIC_PARAMETERS:
            x_text = f"{float(x_value):g}"
        else:
            x_text = str(x_value)
        print(
            f"{x_text},{item['count']},{item['mean']:.10f},{item['variance']:.10f}"
        )


def main():
    args = parse_args()
    run_directories, records = collect_records(args.payoff_dir, args.payoff_player)
    summary = compute_group_statistics(records, args.x_axis)
    save_plot(
        summary,
        args.x_axis,
        args.output,
        args.show_variance,
        args.payoff_player,
    )
    print_summary(
        summary,
        args.x_axis,
        run_directories,
        args.output,
        args.show_variance,
        args.payoff_player,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())