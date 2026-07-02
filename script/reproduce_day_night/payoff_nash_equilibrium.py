import argparse
import json
import warnings
from pathlib import Path

import numpy as np

from payoff_csv_utils import load_payoff_matrix_csv


DEFAULT_PAYOFF_MATRIX_PATH = Path(__file__).resolve().parent / "output/Pay-off/payoff_matrix.csv"
DEFAULT_OUTPUT_FILENAME = "payoff_nash_equilibrium.json"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Load a payoff matrix CSV, compute Nash equilibria with Nashpy, and "
            "save the results as JSON next to the CSV."
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


def _normalize_probability_vector(values, *, atol=1.0e-12):
    array = np.asarray(values, dtype=float)
    clipped = np.where(np.abs(array) <= atol, 0.0, array)
    clipped = np.clip(clipped, 0.0, None)
    total = float(np.sum(clipped))
    if total <= atol:
        raise ValueError("Received an empty mixed strategy from Nashpy.")
    return clipped / total


def _strategy_as_mapping(strategy_names, probabilities, *, atol=1.0e-12):
    normalized = _normalize_probability_vector(probabilities, atol=atol)
    return {
        strategy_name: float(probability)
        for strategy_name, probability in zip(strategy_names, normalized)
        if probability > atol
    }


def _equilibrium_key(prey_probabilities, predator_probabilities):
    rounded_prey = tuple(np.round(prey_probabilities, decimals=12))
    rounded_predator = tuple(np.round(predator_probabilities, decimals=12))
    return rounded_prey, rounded_predator


def compute_nash_equilibria(payoff_matrix_data):
    import nashpy as nash

    predator_payoff_matrix = payoff_matrix_data.values
    prey_payoff_matrix = -predator_payoff_matrix
    game = nash.Game(prey_payoff_matrix, predator_payoff_matrix)

    unique_equilibria = []
    seen_keys = set()
    warning_messages = []

    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        raw_equilibria = list(game.support_enumeration())

    for warning in caught_warnings:
        warning_messages.append(str(warning.message))

    for prey_mixed_strategy, predator_mixed_strategy in raw_equilibria:
        normalized_prey = _normalize_probability_vector(prey_mixed_strategy)
        normalized_predator = _normalize_probability_vector(predator_mixed_strategy)
        equilibrium_key = _equilibrium_key(normalized_prey, normalized_predator)
        if equilibrium_key in seen_keys:
            continue

        seen_keys.add(equilibrium_key)
        predator_expected_payoff = float(
            normalized_prey @ predator_payoff_matrix @ normalized_predator
        )
        unique_equilibria.append(
            {
                "prey_mixed_strategy": _strategy_as_mapping(
                    payoff_matrix_data.row_strategies,
                    normalized_prey,
                ),
                "predator_mixed_strategy": _strategy_as_mapping(
                    payoff_matrix_data.column_strategies,
                    normalized_predator,
                ),
                "prey_expected_payoff": float(-predator_expected_payoff),
                "predator_expected_payoff": predator_expected_payoff,
            }
        )

    if not unique_equilibria:
        raise ValueError("Nashpy did not return any Nash equilibrium for this matrix.")

    return {
        "payoff_matrix": str(payoff_matrix_data.source_path),
        "row_player": payoff_matrix_data.row_player_label,
        "column_player": payoff_matrix_data.column_player_label,
        "row_player_objective": "maximize negative payoff",
        "column_player_objective": "maximize payoff",
        "equilibrium_count": len(unique_equilibria),
        "equilibria": unique_equilibria,
        "warnings": warning_messages,
    }


def main():
    args = parse_args()
    payoff_matrix_data = load_payoff_matrix_csv(args.payoff_matrix)
    equilibria = compute_nash_equilibria(payoff_matrix_data)

    output_path = resolve_output_path(payoff_matrix_data.source_path, args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(equilibria, handle, indent=2)
        handle.write("\n")

    first_equilibrium = equilibria["equilibria"][0]
    print(
        f"Saved Nash equilibrium analysis to {output_path}\n"
        f"Computed {equilibria['equilibrium_count']} equilibrium result(s).\n"
        f"First predator expected payoff: {first_equilibrium['predator_expected_payoff']:.10f}"
    )


if __name__ == "__main__":
    main()