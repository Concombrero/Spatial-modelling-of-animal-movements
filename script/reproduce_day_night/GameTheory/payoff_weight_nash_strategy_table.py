import argparse
import csv
import json
from pathlib import Path
import sys


if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))


DEFAULT_DETAILS_FILENAME = "nash_weight_details.json"
DEFAULT_OUTPUT_CSV_FILENAME = "nash_strategy_frequency_summary.csv"
DEFAULT_OUTPUT_MARKDOWN_FILENAME = "nash_strategy_frequency_summary.md"
PROBABILITY_ATOL = 1.0e-10
PERCENTAGE_DECIMALS = 6


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Load a weight-sweep Nash summary folder and count how often each "
            "strategy appears in the equilibrium support or as a leader for prey "
            "and predator."
        )
    )
    parser.add_argument(
        "source",
        type=Path,
        help=(
            "Path to a weight-sweep output folder or directly to its "
            f"{DEFAULT_DETAILS_FILENAME} file."
        ),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help=(
            "Optional CSV output path. Defaults to a file named "
            f"{DEFAULT_OUTPUT_CSV_FILENAME} next to the details JSON."
        ),
    )
    parser.add_argument(
        "--output-markdown",
        type=Path,
        default=None,
        help=(
            "Optional Markdown output path. Defaults to a file named "
            f"{DEFAULT_OUTPUT_MARKDOWN_FILENAME} next to the details JSON."
        ),
    )
    return parser.parse_args()


def resolve_details_path(source_path):
    candidate = Path(source_path).expanduser().resolve()
    if candidate.is_dir():
        details_path = candidate / DEFAULT_DETAILS_FILENAME
        if details_path.is_file():
            return details_path
        raise FileNotFoundError(
            f"Could not find {DEFAULT_DETAILS_FILENAME} inside {candidate}."
        )
    if candidate.is_file():
        return candidate
    raise FileNotFoundError(f"Could not find {candidate}.")


def load_details(details_path):
    with Path(details_path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    activity_codes = payload.get("activity_codes")
    pairs = payload.get("pairs")
    if not isinstance(activity_codes, list) or not all(
        isinstance(code, str) for code in activity_codes
    ):
        raise ValueError("Expected a JSON payload with an activity_codes string list.")
    if not isinstance(pairs, list):
        raise ValueError("Expected a JSON payload with a pairs list.")
    return activity_codes, pairs


def initialize_counts(activity_codes):
    counts = {}
    for strategy in activity_codes:
        counts[strategy] = {
            "prey_leader_count": 0,
            "prey_support_count": 0,
            "predator_leader_count": 0,
            "predator_support_count": 0,
        }
    return counts


def top_labels(strategy_probabilities, *, atol=PROBABILITY_ATOL):
    if not strategy_probabilities:
        return set()
    maximum = max(float(probability) for probability in strategy_probabilities.values())
    return {
        strategy
        for strategy, probability in strategy_probabilities.items()
        if maximum - float(probability) <= atol
    }


def update_player_counts(counts, strategy_probabilities, *, leader_key, support_key):
    leaders = top_labels(strategy_probabilities)
    for strategy in strategy_probabilities:
        counts[strategy][support_key] += 1
    for strategy in leaders:
        counts[strategy][leader_key] += 1


def summarize_pairs(activity_codes, pairs):
    counts = initialize_counts(activity_codes)
    equilibrium_instance_count = 0
    activity_code_set = set(activity_codes)

    for pair in pairs:
        equilibria = pair.get("equilibria")
        if not isinstance(equilibria, list):
            raise ValueError("Each pair must contain an equilibria list.")

        for equilibrium in equilibria:
            prey_strategy = equilibrium.get("prey_mixed_strategy")
            predator_strategy = equilibrium.get("predator_mixed_strategy")
            if not isinstance(prey_strategy, dict) or not isinstance(predator_strategy, dict):
                raise ValueError(
                    "Each equilibrium must contain prey_mixed_strategy and "
                    "predator_mixed_strategy mappings."
                )

            unknown_prey = set(prey_strategy) - activity_code_set
            unknown_predator = set(predator_strategy) - activity_code_set
            if unknown_prey or unknown_predator:
                unknown = ", ".join(sorted(unknown_prey | unknown_predator))
                raise ValueError(f"Encountered unknown strategy code(s): {unknown}.")

            equilibrium_instance_count += 1
            update_player_counts(
                counts,
                prey_strategy,
                leader_key="prey_leader_count",
                support_key="prey_support_count",
            )
            update_player_counts(
                counts,
                predator_strategy,
                leader_key="predator_leader_count",
                support_key="predator_support_count",
            )

    return counts, equilibrium_instance_count


def calculate_percentage(numerator, denominator):
    if denominator <= 0:
        return None
    return round(100.0 * float(numerator) / float(denominator), PERCENTAGE_DECIMALS)


def build_table_rows(activity_codes, counts, *, equilibrium_instance_count):
    rows = []
    total_role_instance_count = 2 * equilibrium_instance_count
    for strategy in activity_codes:
        strategy_counts = counts[strategy]
        prey_leader_count = strategy_counts["prey_leader_count"]
        prey_support_count = strategy_counts["prey_support_count"]
        predator_leader_count = strategy_counts["predator_leader_count"]
        predator_support_count = strategy_counts["predator_support_count"]
        total_leader_count = prey_leader_count + predator_leader_count
        total_support_count = prey_support_count + predator_support_count
        rows.append(
            {
                "strategy": strategy,
                "prey_leader_count": prey_leader_count,
                "prey_support_count": prey_support_count,
                "prey_support_percentage": calculate_percentage(
                    prey_support_count,
                    equilibrium_instance_count,
                ),
                "prey_percentage": calculate_percentage(
                    prey_leader_count,
                    prey_support_count,
                ),
                "predator_leader_count": predator_leader_count,
                "predator_support_count": predator_support_count,
                "predator_support_percentage": calculate_percentage(
                    predator_support_count,
                    equilibrium_instance_count,
                ),
                "predator_percentage": calculate_percentage(
                    predator_leader_count,
                    predator_support_count,
                ),
                "total_leader_count": total_leader_count,
                "total_support_count": total_support_count,
                "total_support_percentage": calculate_percentage(
                    total_support_count,
                    total_role_instance_count,
                ),
                "total_percentage": calculate_percentage(
                    total_leader_count,
                    total_support_count,
                ),
            }
        )
    return rows


def resolve_output_path(details_path, requested_path, default_filename):
    if requested_path is not None:
        return Path(requested_path).expanduser().resolve()
    return details_path.with_name(default_filename)


def save_csv(rows, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "strategy",
        "prey_leader_count",
        "prey_support_count",
        "prey_support_percentage",
        "prey_percentage",
        "predator_leader_count",
        "predator_support_count",
        "predator_support_percentage",
        "predator_percentage",
        "total_leader_count",
        "total_support_count",
        "total_support_percentage",
        "total_percentage",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            serialized_row = dict(row)
            for key in [
                "prey_support_percentage",
                "prey_percentage",
                "predator_support_percentage",
                "predator_percentage",
                "total_support_percentage",
                "total_percentage",
            ]:
                if serialized_row[key] is None:
                    serialized_row[key] = ""
            writer.writerow(serialized_row)


def format_percentage(value):
    if value is None:
        return "-"
    return f"{float(value):.{PERCENTAGE_DECIMALS}f}%"


def build_markdown_table(rows, *, weight_pair_count, equilibrium_instance_count):
    lines = [
        f"Processed {weight_pair_count} weight pair(s) and {equilibrium_instance_count} equilibrium instance(s).",
        "",
        "| Strategy | Prey leaders | Prey support | Prey support % | Prey leader % | Predator leaders | Predator support | Predator support % | Predator leader % | Total leaders | Total support | Total support % | Total leader % |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {strategy} | {prey_leader_count} | {prey_support_count} | {prey_support_percentage} | {prey_percentage} | "
            "{predator_leader_count} | {predator_support_count} | {predator_support_percentage} | {predator_percentage} | "
            "{total_leader_count} | {total_support_count} | {total_support_percentage} | {total_percentage} |".format(
                strategy=row["strategy"],
                prey_leader_count=row["prey_leader_count"],
                prey_support_count=row["prey_support_count"],
                prey_support_percentage=format_percentage(
                    row["prey_support_percentage"]
                ),
                prey_percentage=format_percentage(row["prey_percentage"]),
                predator_leader_count=row["predator_leader_count"],
                predator_support_count=row["predator_support_count"],
                predator_support_percentage=format_percentage(
                    row["predator_support_percentage"]
                ),
                predator_percentage=format_percentage(row["predator_percentage"]),
                total_leader_count=row["total_leader_count"],
                total_support_count=row["total_support_count"],
                total_support_percentage=format_percentage(
                    row["total_support_percentage"]
                ),
                total_percentage=format_percentage(row["total_percentage"]),
            )
        )
    return "\n".join(lines) + "\n"


def save_markdown(markdown_table, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(markdown_table, encoding="utf-8")


def validate_rows(rows):
    for row in rows:
        if row["prey_leader_count"] > row["prey_support_count"]:
            raise ValueError(
                f"Prey leader count exceeds support count for {row['strategy']}."
            )
        if row["predator_leader_count"] > row["predator_support_count"]:
            raise ValueError(
                f"Predator leader count exceeds support count for {row['strategy']}."
            )


def main():
    args = parse_args()
    details_path = resolve_details_path(args.source)
    activity_codes, pairs = load_details(details_path)
    counts, equilibrium_instance_count = summarize_pairs(activity_codes, pairs)
    rows = build_table_rows(
        activity_codes,
        counts,
        equilibrium_instance_count=equilibrium_instance_count,
    )
    validate_rows(rows)

    csv_output_path = resolve_output_path(
        details_path,
        args.output_csv,
        DEFAULT_OUTPUT_CSV_FILENAME,
    )
    markdown_output_path = resolve_output_path(
        details_path,
        args.output_markdown,
        DEFAULT_OUTPUT_MARKDOWN_FILENAME,
    )

    save_csv(rows, csv_output_path)
    markdown_table = build_markdown_table(
        rows,
        weight_pair_count=len(pairs),
        equilibrium_instance_count=equilibrium_instance_count,
    )
    save_markdown(markdown_table, markdown_output_path)

    print(markdown_table.rstrip())
    print()
    print(f"Saved CSV summary to {csv_output_path}")
    print(f"Saved Markdown summary to {markdown_output_path}")


if __name__ == "__main__":
    main()