import argparse
import json
from pathlib import Path
import sys

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from script.simulations.GameTheory.payoff_csv_utils import (
    load_payoff_game_data,
)
from script.simulations.paths import game_theory_payoff_output_path


DEFAULT_PAYOFF_MATRIX_PATH = game_theory_payoff_output_path("payoff_matrix.csv")
DEFAULT_OUTPUT_FILENAME = "payoff_minmax_maxmin.json"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Load a payoff matrix file or run directory, compute security values, "
            "and save the result as JSON next to the source."
        )
    )
    parser.add_argument(
        "--payoff-matrix",
        type=Path,
        default=DEFAULT_PAYOFF_MATRIX_PATH,
        help=(
            "Path to either payoff_matrix.csv or a payoff output directory. Default: "
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
        source_path = Path(payoff_matrix_path).expanduser().resolve()
        if source_path.is_dir():
            return source_path / DEFAULT_OUTPUT_FILENAME
        return source_path.with_name(DEFAULT_OUTPUT_FILENAME)
    return Path(output_path).expanduser().resolve()


def _strategies_matching_value(strategy_names, values, target_value, *, atol=1.0e-12):
    return [
        strategy_name
        for strategy_name, value in zip(strategy_names, values)
        if np.isclose(value, target_value, atol=atol, rtol=1.0e-9)
    ]


def compute_security_values(payoff_game_data):
    prey_values = payoff_game_data.prey_values
    predator_values = payoff_game_data.predator_values

    if payoff_game_data.is_zero_sum:
        predator_worst_case_values = np.max(predator_values, axis=1)
        prey_minmax_value = float(np.min(predator_worst_case_values))
        prey_optimal_strategies = _strategies_matching_value(
            payoff_game_data.row_strategies,
            predator_worst_case_values,
            prey_minmax_value,
        )

        predator_guaranteed_values = np.min(predator_values, axis=0)
        predator_maxmin_value = float(np.max(predator_guaranteed_values))
        predator_optimal_strategies = _strategies_matching_value(
            payoff_game_data.column_strategies,
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
            for row_index, prey_strategy in enumerate(payoff_game_data.row_strategies):
                if not np.isclose(
                    predator_worst_case_values[row_index],
                    prey_minmax_value,
                    atol=1.0e-12,
                    rtol=1.0e-9,
                ):
                    continue

                for column_index, predator_strategy in enumerate(
                    payoff_game_data.column_strategies
                ):
                    candidate_value = predator_values[row_index, column_index]
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

        result = {
            "payoff_mode": payoff_game_data.payoff_mode,
            "prey_payoff_matrix": str(payoff_game_data.prey_source_path),
            "predator_payoff_matrix": str(payoff_game_data.predator_source_path),
            "payoff_matrix": str(payoff_game_data.predator_source_path),
            "row_player": payoff_game_data.row_player_label,
            "column_player": payoff_game_data.column_player_label,
            "prey": {
                "objective": "minimize payoff",
                "row_worst_case_values": {
                    strategy_name: float(value)
                    for strategy_name, value in zip(
                        payoff_game_data.row_strategies,
                        predator_worst_case_values,
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
                        payoff_game_data.column_strategies,
                        predator_guaranteed_values,
                    )
                },
                "maxmin_value": predator_maxmin_value,
                "optimal_strategies": predator_optimal_strategies,
            },
            "saddle_point_exists": bool(saddle_points),
            "saddle_points": saddle_points,
        }
        return result

    prey_guaranteed_values = np.min(prey_values, axis=1)
    prey_maximin_value = float(np.max(prey_guaranteed_values))
    prey_optimal_strategies = _strategies_matching_value(
        payoff_game_data.row_strategies,
        prey_guaranteed_values,
        prey_maximin_value,
    )

    predator_guaranteed_values = np.min(predator_values, axis=0)
    predator_maximin_value = float(np.max(predator_guaranteed_values))
    predator_optimal_strategies = _strategies_matching_value(
        payoff_game_data.column_strategies,
        predator_guaranteed_values,
        predator_maximin_value,
    )

    return {
        "payoff_mode": payoff_game_data.payoff_mode,
        "prey_payoff_matrix": str(payoff_game_data.prey_source_path),
        "predator_payoff_matrix": str(payoff_game_data.predator_source_path),
        "row_player": payoff_game_data.row_player_label,
        "column_player": payoff_game_data.column_player_label,
        "analysis_type": "general_sum_security_values",
        "prey": {
            "objective": "maximize payoff",
            "row_guaranteed_values": {
                strategy_name: float(value)
                for strategy_name, value in zip(
                    payoff_game_data.row_strategies,
                    prey_guaranteed_values,
                )
            },
            "maximin_value": prey_maximin_value,
            "optimal_strategies": prey_optimal_strategies,
        },
        "predator": {
            "objective": "maximize payoff",
            "column_guaranteed_values": {
                strategy_name: float(value)
                for strategy_name, value in zip(
                    payoff_game_data.column_strategies,
                    predator_guaranteed_values,
                )
            },
            "maximin_value": predator_maximin_value,
            "optimal_strategies": predator_optimal_strategies,
        },
    }


def main():
    args = parse_args()
    payoff_game_data = load_payoff_game_data(args.payoff_matrix)
    analysis = compute_security_values(payoff_game_data)

    output_path = resolve_output_path(args.payoff_matrix, args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(analysis, handle, indent=2)
        handle.write("\n")

    if payoff_game_data.is_zero_sum:
        print(
            f"Saved minmax/maxmin analysis to {output_path}\n"
            f"Prey minmax value: {analysis['prey']['minmax_value']:.10f} "
            f"via {', '.join(analysis['prey']['optimal_strategies'])}\n"
            f"Predator maxmin value: {analysis['predator']['maxmin_value']:.10f} "
            f"via {', '.join(analysis['predator']['optimal_strategies'])}"
        )
    else:
        print(
            f"Saved security-value analysis to {output_path}\n"
            f"Prey maximin value: {analysis['prey']['maximin_value']:.10f} "
            f"via {', '.join(analysis['prey']['optimal_strategies'])}\n"
            f"Predator maximin value: {analysis['predator']['maximin_value']:.10f} "
            f"via {', '.join(analysis['predator']['optimal_strategies'])}"
        )


if __name__ == "__main__":
    main()