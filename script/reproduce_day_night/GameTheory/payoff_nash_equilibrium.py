import argparse
import json
from pathlib import Path
import sys

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from script.reproduce_day_night.GameTheory.payoff_csv_utils import (
    load_payoff_game_data,
)
from script.reproduce_day_night.GameTheory.payoff_replicator_analysis import (
    MissingDependencyError,
    compute_nash_equilibria as compute_nash_equilibria_from_matrices,
    require_nashpy,
)
from script.reproduce_day_night.paths import game_theory_payoff_output_path


DEFAULT_PAYOFF_MATRIX_PATH = game_theory_payoff_output_path("payoff_matrix.csv")
DEFAULT_OUTPUT_FILENAME = "payoff_nash_equilibrium.json"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Load a payoff matrix file or run directory, compute Nash equilibria "
            "with Nashpy, and save the results as JSON next to the source."
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


def compute_nash_equilibria(payoff_game_data):
    raw_equilibria = compute_nash_equilibria_from_matrices(
        payoff_game_data.prey_values,
        payoff_game_data.predator_values,
    )
    if not raw_equilibria:
        raise ValueError("Nashpy did not return any Nash equilibrium for this matrix.")

    unique_equilibria = []
    for equilibrium in raw_equilibria:
        normalized_prey = _normalize_probability_vector(equilibrium["prey_strategy"])
        normalized_predator = _normalize_probability_vector(
            equilibrium["predator_strategy"]
        )
        unique_equilibria.append(
            {
                "algorithm": equilibrium["algorithm"],
                "prey_mixed_strategy": _strategy_as_mapping(
                    payoff_game_data.row_strategies,
                    normalized_prey,
                ),
                "predator_mixed_strategy": _strategy_as_mapping(
                    payoff_game_data.column_strategies,
                    normalized_predator,
                ),
                "prey_expected_payoff": float(equilibrium["prey_payoff"]),
                "predator_expected_payoff": float(equilibrium["predator_payoff"]),
            }
        )

    first_predator_path = str(payoff_game_data.predator_source_path)
    first_prey_path = str(payoff_game_data.prey_source_path)

    result = {
        "payoff_mode": payoff_game_data.payoff_mode,
        "prey_payoff_matrix": first_prey_path,
        "predator_payoff_matrix": first_predator_path,
        "row_player": payoff_game_data.row_player_label,
        "column_player": payoff_game_data.column_player_label,
        "row_player_objective": (
            "maximize negative payoff"
            if payoff_game_data.is_zero_sum
            else "maximize payoff"
        ),
        "column_player_objective": "maximize payoff",
        "equilibrium_count": len(unique_equilibria),
        "equilibria": unique_equilibria,
        "warnings": [],
    }

    if payoff_game_data.is_zero_sum:
        result["payoff_matrix"] = first_predator_path

    return result


def main():
    try:
        args = parse_args()
        require_nashpy()
        payoff_game_data = load_payoff_game_data(args.payoff_matrix)
        equilibria = compute_nash_equilibria(payoff_game_data)

        output_path = resolve_output_path(args.payoff_matrix, args.output)
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
    except MissingDependencyError as error:
        raise SystemExit(str(error)) from error


if __name__ == "__main__":
    main()