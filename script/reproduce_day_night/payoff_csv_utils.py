import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class PayoffMatrixData:
    row_player_label: str
    column_player_label: str
    row_strategies: tuple[str, ...]
    column_strategies: tuple[str, ...]
    values: np.ndarray
    source_path: Path


def _parse_axis_labels(header_cell):
    normalized = header_cell.strip()
    if "/" in normalized:
        parts = [part.strip() for part in normalized.split("/", maxsplit=1)]
        if len(parts) == 2 and all(parts):
            return parts[0], parts[1]
    return "Row Player", "Column Player"


def load_payoff_matrix_csv(payoff_matrix_path):
    source_path = Path(payoff_matrix_path).expanduser().resolve()
    if not source_path.is_file():
        raise FileNotFoundError(f"Payoff matrix CSV not found: {source_path}")

    with source_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        header = next(reader, None)
        if header is None or len(header) < 2:
            raise ValueError(
                "Expected a header row with one label column followed by at least "
                "one strategy column."
            )

        row_player_label, column_player_label = _parse_axis_labels(header[0])
        column_strategies = tuple(cell.strip() for cell in header[1:])
        if not all(column_strategies):
            raise ValueError("Found an empty strategy name in the header row.")

        row_strategies = []
        matrix_rows = []

        for row_index, row in enumerate(reader, start=2):
            if not row or not any(cell.strip() for cell in row):
                continue

            expected_length = len(column_strategies) + 1
            if len(row) != expected_length:
                raise ValueError(
                    f"Row {row_index} in {source_path} has {len(row)} columns; "
                    f"expected {expected_length}."
                )

            strategy_name = row[0].strip()
            if not strategy_name:
                raise ValueError(f"Missing row strategy name on row {row_index}.")

            try:
                payoff_values = [float(cell) for cell in row[1:]]
            except ValueError as error:
                raise ValueError(
                    f"Non-numeric payoff value on row {row_index} in {source_path}."
                ) from error

            row_strategies.append(strategy_name)
            matrix_rows.append(payoff_values)

    if not matrix_rows:
        raise ValueError(f"No payoff rows were found in {source_path}.")

    values = np.asarray(matrix_rows, dtype=float)
    return PayoffMatrixData(
        row_player_label=row_player_label,
        column_player_label=column_player_label,
        row_strategies=tuple(row_strategies),
        column_strategies=column_strategies,
        values=values,
        source_path=source_path,
    )