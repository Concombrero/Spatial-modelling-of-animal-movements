import argparse
import json
from pathlib import Path

import numpy as np

from payoff_csv_utils import load_payoff_matrix_csv


DEFAULT_PAYOFF_MATRIX_PATH = Path(__file__).resolve().parent / "output/Pay-off/payoff_matrix.csv"
DEFAULT_OUTPUT_FILENAME = "payoff_minmax_maxmin.json"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Load a payoff matrix CSV, compute the prey minmax value and the "
            "predator maxmin value, and save the results as JSON next to the CSV."
        )
    )
    parser.add_argument(
        "--payoff-matrix",
        type=Path,
        default=DEFAULT_PAYOFF_MATRIX_PATH,
        help=(
            "Path to the payoff_matrix.csv file. Default: "
            f"{DEFAULT_PAYOFF_MATRIX_PATH}."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Optional JSON output path. Defaults to a file named "
            f"{DEFAULT_OUTPUT_FILENAME} in the same folder as the payoff matrix."
        ),
    )
    return parser.parse_args()


def resolve_output_path(payoff_matrix_path, output_path):
    if output_path is None:
        return Path(payoff_matrix_path).expanduser().resolve().with_name(
            DEFAULT_OUTPUT_FILENAME
        )
    return Path(output_path).expanduser().resolve()


def _strategies_matching_value(strategy_names, values, target_value, *, atol=1.0e-12):
    return [
        strategy_name
        for strategy_name, value in zip(strategy_names, values)
        if np.isclose(value, target_value, atol=atol, rtol=1.0e-9)
    ]


def compute_security_values(payoff_matrix_data):
    values = payoff_matrix_data.values

    prey_worst_case_values = np.max(values, axis=1)
    prey_minmax_value = float(np.min(prey_worst_case_values))
    prey_optimal_strategies = _strategies_matching_value(
        payoff_matrix_data.row_strategies,
        prey_worst_case_values,
        prey_minmax_value,
    )

    predator_guaranteed_values = np.min(values, axis=0)
    predator_maxmin_value = float(np.max(predator_guaranteed_values))
    predator_optimal_strategies = _strategies_matching_value(
        payoff_matrix_data.column_strategies,
        predator_guaranteed_values,
        predator_maxmin_value,
    )

    saddle_points = []
    if np.isclose(
        prey_minmax_value,
        predator_maxmin_value,
        atol=1.0e-12,
        rtol=1.0e-9,
    ):
        for row_index, prey_strategy in enumerate(payoff_matrix_data.row_strategies):
            if not np.isclose(
                prey_worst_case_values[row_index],
                prey_minmax_value,
                atol=1.0e-12,
                rtol=1.0e-9,
            ):
                continue

            for column_index, predator_strategy in enumerate(
                payoff_matrix_data.column_strategies
            ):
                candidate_value = values[row_index, column_index]
                if np.isclose(
                    predator_guaranteed_values[column_index],
                    predator_maxmin_value,
                    atol=1.0e-12,
                    rtol=1.0e-9,
                ) and np.isclose(
                    candidate_value,
                    prey_minmax_value,
                    atol=1.0e-12,
                    rtol=1.0e-9,
                ):
                    saddle_points.append(
                        {
                            "prey_strategy": prey_strategy,
                            "predator_strategy": predator_strategy,
                            "payoff": float(candidate_value),
                        }
                    )

    return {
        "payoff_matrix": str(payoff_matrix_data.source_path),
        "row_player": payoff_matrix_data.row_player_label,
        "column_player": payoff_matrix_data.column_player_label,
        "prey": {
            "objective": "minimize payoff",
            "row_worst_case_values": {
                strategy_name: float(value)
                for strategy_name, value in zip(
                    payoff_matrix_data.row_strategies,
                    prey_worst_case_values,
                )
            },
            "minmax_value": prey_minmax_value,
            "optimal_strategies": prey_optimal_strategies,
        },
        "predator": {
            "objective": "maximize payoff",
            "column_guaranteed_values": {
                strategy_name: float(value)
                for strategy_name, value in zip(
                    payoff_matrix_data.column_strategies,
                    predator_guaranteed_values,
                )
            },
            "maxmin_value": predator_maxmin_value,
            "optimal_strategies": predator_optimal_strategies,
        },
        "saddle_point_exists": bool(saddle_points),
        "saddle_points": saddle_points,
    }


def main():
    args = parse_args()
    payoff_matrix_data = load_payoff_matrix_csv(args.payoff_matrix)
    analysis = compute_security_values(payoff_matrix_data)

    output_path = resolve_output_path(payoff_matrix_data.source_path, args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(analysis, handle, indent=2)
        handle.write("\n")

    print(
        f"Saved minmax/maxmin analysis to {output_path}\n"
        f"Prey minmax value: {analysis['prey']['minmax_value']:.10f} "
        f"via {', '.join(analysis['prey']['optimal_strategies'])}\n"
        f"Predator maxmin value: {analysis['predator']['maxmin_value']:.10f} "
        f"via {', '.join(analysis['predator']['optimal_strategies'])}"
    )


if __name__ == "__main__":
    main()