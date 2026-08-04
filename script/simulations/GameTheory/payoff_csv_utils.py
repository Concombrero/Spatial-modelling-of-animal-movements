import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


RUN_CONFIG_FILENAME = "run_config.json"
LEGACY_PAYOFF_MATRIX_FILENAME = "payoff_matrix.csv"
PREY_PAYOFF_MATRIX_FILENAME = "payoff_matrix_prey.csv"
PREDATOR_PAYOFF_MATRIX_FILENAME = "payoff_matrix_predator.csv"
OVERLAP_PAYOFF_MODE = "overlap"
POPULATION_INTEGRAL_PAYOFF_MODE = "population-integral"
NET_GROWTH_PAYOFF_MODE = "net-growth"


@dataclass(frozen=True)
class PayoffMatrixData:
    row_player_label: str
    column_player_label: str
    row_strategies: tuple[str, ...]
    column_strategies: tuple[str, ...]
    values: np.ndarray
    source_path: Path


@dataclass(frozen=True)
class PayoffGameData:
    row_player_label: str
    column_player_label: str
    row_strategies: tuple[str, ...]
    column_strategies: tuple[str, ...]
    prey_values: np.ndarray
    predator_values: np.ndarray
    prey_source_path: Path
    predator_source_path: Path
    payoff_mode: str

    @property
    def is_zero_sum(self):
        return np.allclose(
            self.prey_values,
            -self.predator_values,
            atol=1.0e-12,
            rtol=1.0e-9,
        )


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


def _load_payoff_mode(run_directory):
    config_path = Path(run_directory).expanduser().resolve() / RUN_CONFIG_FILENAME
    if not config_path.is_file():
        return None

    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)

    payoff_mode = config.get("payoff_mode")
    return payoff_mode if isinstance(payoff_mode, str) else None


def _validate_matching_matrix_layout(prey_matrix_data, predator_matrix_data):
    if prey_matrix_data.row_player_label != predator_matrix_data.row_player_label:
        raise ValueError(
            "Prey and predator payoff matrices disagree on the row-player label."
        )
    if prey_matrix_data.column_player_label != predator_matrix_data.column_player_label:
        raise ValueError(
            "Prey and predator payoff matrices disagree on the column-player label."
        )
    if prey_matrix_data.row_strategies != predator_matrix_data.row_strategies:
        raise ValueError(
            "Prey and predator payoff matrices disagree on the row strategies."
        )
    if prey_matrix_data.column_strategies != predator_matrix_data.column_strategies:
        raise ValueError(
            "Prey and predator payoff matrices disagree on the column strategies."
        )
    if prey_matrix_data.values.shape != predator_matrix_data.values.shape:
        raise ValueError(
            "Prey and predator payoff matrices disagree on the matrix shape."
        )


def _build_zero_sum_game_data(matrix_data):
    return PayoffGameData(
        row_player_label=matrix_data.row_player_label,
        column_player_label=matrix_data.column_player_label,
        row_strategies=matrix_data.row_strategies,
        column_strategies=matrix_data.column_strategies,
        prey_values=-matrix_data.values,
        predator_values=matrix_data.values,
        prey_source_path=matrix_data.source_path,
        predator_source_path=matrix_data.source_path,
        payoff_mode=OVERLAP_PAYOFF_MODE,
    )


def load_payoff_game_data(payoff_source):
    source_path = Path(payoff_source).expanduser().resolve()

    if source_path.is_file():
        return _build_zero_sum_game_data(load_payoff_matrix_csv(source_path))

    if not source_path.is_dir():
        raise FileNotFoundError(f"Payoff source not found: {source_path}")

    payoff_mode = _load_payoff_mode(source_path)
    legacy_matrix_path = source_path / LEGACY_PAYOFF_MATRIX_FILENAME
    prey_matrix_path = source_path / PREY_PAYOFF_MATRIX_FILENAME
    predator_matrix_path = source_path / PREDATOR_PAYOFF_MATRIX_FILENAME

    if payoff_mode == OVERLAP_PAYOFF_MODE and legacy_matrix_path.is_file():
        return _build_zero_sum_game_data(load_payoff_matrix_csv(legacy_matrix_path))

    if payoff_mode in {
        POPULATION_INTEGRAL_PAYOFF_MODE,
        NET_GROWTH_PAYOFF_MODE,
    }:
        if not prey_matrix_path.is_file() or not predator_matrix_path.is_file():
            raise FileNotFoundError(
                "Expected payoff_matrix_prey.csv and payoff_matrix_predator.csv "
                f"in {source_path} for payoff_mode={payoff_mode!r}."
            )

    if prey_matrix_path.is_file() and predator_matrix_path.is_file():
        prey_matrix_data = load_payoff_matrix_csv(prey_matrix_path)
        predator_matrix_data = load_payoff_matrix_csv(predator_matrix_path)
        _validate_matching_matrix_layout(prey_matrix_data, predator_matrix_data)
        resolved_mode = payoff_mode
        if resolved_mode is None:
            if np.allclose(
                prey_matrix_data.values,
                -predator_matrix_data.values,
                atol=1.0e-12,
                rtol=1.0e-9,
            ):
                resolved_mode = OVERLAP_PAYOFF_MODE
            else:
                resolved_mode = POPULATION_INTEGRAL_PAYOFF_MODE

        return PayoffGameData(
            row_player_label=prey_matrix_data.row_player_label,
            column_player_label=prey_matrix_data.column_player_label,
            row_strategies=prey_matrix_data.row_strategies,
            column_strategies=prey_matrix_data.column_strategies,
            prey_values=prey_matrix_data.values,
            predator_values=predator_matrix_data.values,
            prey_source_path=prey_matrix_data.source_path,
            predator_source_path=predator_matrix_data.source_path,
            payoff_mode=resolved_mode,
        )

    if legacy_matrix_path.is_file():
        return _build_zero_sum_game_data(load_payoff_matrix_csv(legacy_matrix_path))

    raise FileNotFoundError(
        "Could not find a payoff matrix source in "
        f"{source_path}. Expected {LEGACY_PAYOFF_MATRIX_FILENAME} or the pair "
        f"{PREY_PAYOFF_MATRIX_FILENAME}/{PREDATOR_PAYOFF_MATRIX_FILENAME}."
    )