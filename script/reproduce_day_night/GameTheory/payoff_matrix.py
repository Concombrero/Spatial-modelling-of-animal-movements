import argparse
import csv
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from script.reproduce_day_night.paths import game_theory_payoff_output_path
from script.reproduce_day_night.Solver import DayNightModel1D


OUTPUT_DIRECTORY = game_theory_payoff_output_path()
CSV_OUTPUT_PATH = game_theory_payoff_output_path("payoff_matrix.csv")
PREY_CSV_OUTPUT_PATH = game_theory_payoff_output_path("payoff_matrix_prey.csv")
PREDATOR_CSV_OUTPUT_PATH = game_theory_payoff_output_path(
    "payoff_matrix_predator.csv"
)
CASE_PAYOFF_OUTPUT_PATH = game_theory_payoff_output_path("case_payoffs.csv")
RUN_CONFIG_OUTPUT_PATH = game_theory_payoff_output_path("run_config.json")
HEATMAP_OUTPUT_PATH = game_theory_payoff_output_path("payoff_matrix.png")
PREY_HEATMAP_OUTPUT_PATH = game_theory_payoff_output_path("payoff_matrix_prey.png")
PREDATOR_HEATMAP_OUTPUT_PATH = game_theory_payoff_output_path(
    "payoff_matrix_predator.png"
)
POPULATION_HEATMAP_OUTPUT_DIRECTORY = game_theory_payoff_output_path(
    "population_heatmaps"
)
NUMBER_OF_POINTS = 128
NUMBER_OF_POPULATIONS = 2
NUMBER_OF_CYCLES = 4
CYCLE_PERIOD = 1.0
TOTAL_TIME = NUMBER_OF_CYCLES * CYCLE_PERIOD
OBSERVATION_WINDOW = 2 * CYCLE_PERIOD
DT = 0.1
DAY_START = 0.0
DEFAULT_T_SUNSET = 0.5
DEFAULT_WEIGHTS = (0.5, 0.5)
DEFAULT_SIGHT_RADIUS = 0.1
DEFAULT_SMELL_RADIUS = 0.2
DEFAULT_INITIAL_CENTERS = (0.25, 0.70)
DEFAULT_INITIAL_WIDTH = 0.1
DEFAULT_ATTRACTION = (
    (0.1, -0.2),
    (0.2, 0.1),
)
DEFAULT_DIFFUSION = (0.04, 0.04)
DEFAULT_REACTION_RATES = {
    "prey_growth": 0.2,
    "predator_decay": 0.1,
    "predation_rate": 0.15,
    "conversion_rate": 0.08,
}
MAX_WORKERS = min(16, os.cpu_count() or 1)
OVERLAP_PAYOFF_MODE = "overlap"
POPULATION_INTEGRAL_PAYOFF_MODE = "population-integral"
PAYOFF_MODE_CHOICES = (
    OVERLAP_PAYOFF_MODE,
    POPULATION_INTEGRAL_PAYOFF_MODE,
)
ACTIVITY_REGIMES = (
    {"code": "D", "label": "Diurnal", "periods": [(0.0, 0.5)]},
    {"code": "N", "label": "Nocturnal", "periods": [(0.5, 1.0)]},
    {
        "code": "P1",
        "label": "Polyphasic 1",
        "periods": [(0.0, 0.25), (0.5, 0.75)],
    },
    {
        "code": "P2",
        "label": "Polyphasic 2",
        "periods": [(0.25, 0.5), (0.75, 1.0)],
    },
    {
        "code": "M1",
        "label": "Matutinal 1",
        "periods": [(0.0, 0.25), (0.75, 1.0)],
    },
    {"code": "M2", "label": "Matutinal 2", "periods": [(0.25, 0.75)]},
)
ACTIVITY_REGIME_CODES = tuple(regime["code"] for regime in ACTIVITY_REGIMES)


def _coerce_population_parameter(values, *, parameter_name):
    array = np.asarray(values, dtype=float)
    if array.shape == ():
        return float(array)

    flattened = tuple(float(value) for value in np.ravel(array))
    if len(flattened) != NUMBER_OF_POPULATIONS:
        raise ValueError(
            f"{parameter_name} must be a scalar or have length {NUMBER_OF_POPULATIONS}."
        )

    return flattened


def build_config(
    *,
    t_sunset=DEFAULT_T_SUNSET,
    weights=DEFAULT_WEIGHTS,
    sight_radius=DEFAULT_SIGHT_RADIUS,
    smell_radius=DEFAULT_SMELL_RADIUS,
    number_of_points=NUMBER_OF_POINTS,
    dt=DT,
    number_of_cycles=NUMBER_OF_CYCLES,
    observation_window=OBSERVATION_WINDOW,
    initial_centers=DEFAULT_INITIAL_CENTERS,
    initial_width=DEFAULT_INITIAL_WIDTH,
    diffusion=DEFAULT_DIFFUSION,
    attraction=DEFAULT_ATTRACTION,
    reaction_rates=None,
):
    if reaction_rates is None:
        reaction_rates = DEFAULT_REACTION_RATES

    return {
        "t_sunset": float(t_sunset),
        "weights": tuple(float(weight) for weight in weights),
        "sight_radius": _coerce_population_parameter(
            sight_radius,
            parameter_name="sight_radius",
        ),
        "smell_radius": _coerce_population_parameter(
            smell_radius,
            parameter_name="smell_radius",
        ),
        "number_of_points": int(number_of_points),
        "dt": float(dt),
        "number_of_cycles": int(number_of_cycles),
        "observation_window": float(observation_window),
        "initial_centers": tuple(float(center) for center in initial_centers),
        "initial_width": float(initial_width),
        "diffusion": tuple(float(value) for value in diffusion),
        "attraction": tuple(
            tuple(float(value) for value in row) for row in attraction
        ),
        "reaction_rates": {
            "prey_growth": float(reaction_rates["prey_growth"]),
            "predator_decay": float(reaction_rates["predator_decay"]),
            "predation_rate": float(reaction_rates["predation_rate"]),
            "conversion_rate": float(reaction_rates["conversion_rate"]),
        },
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compute the 6x6 prey-predator payoff matrix defined in the paper "
            "using either the legacy overlap energy or population-specific "
            "final-window integrals."
        )
    )
    parser.add_argument(
        "--t-sunset",
        type=float,
        default=DEFAULT_T_SUNSET,
        help="Daylight proportion t_sunset in [0, 1]. Default: 0.5.",
    )
    parser.add_argument(
        "--weights",
        nargs=2,
        type=float,
        metavar=("W1", "W2"),
        default=list(DEFAULT_WEIGHTS),
        help=(
            "Sight weights for the prey and predator. Default: 0.5 0.5."
        ),
    )
    parser.add_argument(
        "--sight-radius",
        type=float,
        default=None,
        help=(
            "Shared sight-radius shorthand applied to prey and predator unless "
            "overridden by a population-specific flag. Default when no override "
            f"is provided: {DEFAULT_SIGHT_RADIUS:g}."
        ),
    )
    parser.add_argument(
        "--prey-sight-radius",
        type=float,
        default=None,
        help=(
            "Sight radius used by the prey. Defaults to --sight-radius or "
            f"{DEFAULT_SIGHT_RADIUS:g}."
        ),
    )
    parser.add_argument(
        "--predator-sight-radius",
        type=float,
        default=None,
        help=(
            "Sight radius used by the predator. Defaults to --sight-radius or "
            f"{DEFAULT_SIGHT_RADIUS:g}."
        ),
    )
    parser.add_argument(
        "--smell-radius",
        type=float,
        default=None,
        help=(
            "Shared smell-radius shorthand applied to prey and predator unless "
            "overridden by a population-specific flag. Default when no override "
            f"is provided: {DEFAULT_SMELL_RADIUS:g}."
        ),
    )
    parser.add_argument(
        "--prey-smell-radius",
        type=float,
        default=None,
        help=(
            "Smell radius used by the prey. Defaults to --smell-radius or "
            f"{DEFAULT_SMELL_RADIUS:g}."
        ),
    )
    parser.add_argument(
        "--predator-smell-radius",
        type=float,
        default=None,
        help=(
            "Smell radius used by the predator. Defaults to --smell-radius or "
            f"{DEFAULT_SMELL_RADIUS:g}."
        ),
    )
    parser.add_argument(
        "--number-of-points",
        type=int,
        default=NUMBER_OF_POINTS,
        help="Number of spatial grid points. Default: 64.",
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=DT,
        help="Stored output timestep. Default: 0.1.",
    )
    parser.add_argument(
        "--number-of-cycles",
        type=int,
        default=NUMBER_OF_CYCLES,
        help="Number of daily cycles to simulate. Default: 4.",
    )
    parser.add_argument(
        "--observation-window",
        type=float,
        default=OBSERVATION_WINDOW,
        help="Final-time window used in the overlap integral. Default: 1.0.",
    )
    parser.add_argument(
        "--payoff-mode",
        choices=PAYOFF_MODE_CHOICES,
        default=OVERLAP_PAYOFF_MODE,
        help=(
            "Payoff functional used over the final observation window. "
            "'overlap' keeps the legacy sqrt(u1*u2) overlap payoff. "
            "'population-integral' uses the raw u1 integral for the prey payoff "
            "and the raw u2 integral for the predator payoff. Default: overlap."
        ),
    )
    parser.add_argument(
        "--prey-growth",
        type=float,
        default=DEFAULT_REACTION_RATES["prey_growth"],
        help=(
            f"Lotka-Volterra prey growth rate r1. Default: "
            f"{DEFAULT_REACTION_RATES['prey_growth']:g}."
        ),
    )
    parser.add_argument(
        "--predator-decay",
        type=float,
        default=DEFAULT_REACTION_RATES["predator_decay"],
        help=(
            f"Lotka-Volterra predator decay rate r2. Default: "
            f"{DEFAULT_REACTION_RATES['predator_decay']:g}."
        ),
    )
    parser.add_argument(
        "--predation-rate",
        type=float,
        default=DEFAULT_REACTION_RATES["predation_rate"],
        help=f"Predation rate a. Default: {DEFAULT_REACTION_RATES['predation_rate']:g}.",
    )
    parser.add_argument(
        "--conversion-rate",
        type=float,
        default=DEFAULT_REACTION_RATES["conversion_rate"],
        help=(
            f"Predator conversion rate b. Default: "
            f"{DEFAULT_REACTION_RATES['conversion_rate']:g}."
        ),
    )
    parser.add_argument(
        "--chi11",
        type=float,
        default=DEFAULT_ATTRACTION[0][0],
        help=f"Prey self-attraction coefficient. Default: {DEFAULT_ATTRACTION[0][0]:g}.",
    )
    parser.add_argument(
        "--chi12",
        type=float,
        default=DEFAULT_ATTRACTION[0][1],
        help=f"Prey response to predator. Default: {DEFAULT_ATTRACTION[0][1]:g}.",
    )
    parser.add_argument(
        "--chi21",
        type=float,
        default=DEFAULT_ATTRACTION[1][0],
        help=f"Predator response to prey. Default: {DEFAULT_ATTRACTION[1][0]:g}.",
    )
    parser.add_argument(
        "--chi22",
        type=float,
        default=DEFAULT_ATTRACTION[1][1],
        help=f"Predator self-attraction coefficient. Default: {DEFAULT_ATTRACTION[1][1]:g}.",
    )
    parser.add_argument(
        "--diffusion",
        nargs=2,
        type=float,
        metavar=("D1", "D2"),
        default=list(DEFAULT_DIFFUSION),
        help=(
            f"Diffusion coefficients for prey and predator. Default: "
            f"{DEFAULT_DIFFUSION[0]:g} {DEFAULT_DIFFUSION[1]:g}."
        ),
    )
    parser.add_argument(
        "--initial-centers",
        nargs=2,
        type=float,
        metavar=("X1", "X2"),
        default=list(DEFAULT_INITIAL_CENTERS),
        help="Initial Gaussian centers for prey and predator. Default: 0.25 0.70.",
    )
    parser.add_argument(
        "--initial-width",
        type=float,
        default=DEFAULT_INITIAL_WIDTH,
        help=(
            f"Shared Gaussian width for both populations. Default: "
            f"{DEFAULT_INITIAL_WIDTH:g}."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIRECTORY,
        help="Directory where the CSV and figures are saved.",
    )
    parser.add_argument(
        "--strategy-codes",
        type=str,
        help=(
            "Optional comma-separated subset of activity codes to include in the "
            "payoff matrix, for example D,N,P1,M1. Default: all six codes."
        ),
    )
    parser.add_argument(
        "--heatmap-prey",
        choices=ACTIVITY_REGIME_CODES,
        help=(
            "Prey activity code filter for the saved population heatmaps. "
            "Use together with --heatmap-predator to save only one activity pair. "
            "When both are omitted, the script saves heatmaps for every matrix entry."
        ),
    )
    parser.add_argument(
        "--heatmap-predator",
        choices=ACTIVITY_REGIME_CODES,
        help=(
            "Predator activity code filter for the saved population heatmaps. "
            "Use together with --heatmap-prey."
        ),
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=MAX_WORKERS,
        help="Number of parallel worker processes. Default: min(16, cpu_count).",
    )
    return parser.parse_args()


def resolve_population_parameter_pair(
    shared_value,
    prey_value,
    predator_value,
    *,
    default_value,
):
    if prey_value is None:
        prey_value = shared_value if shared_value is not None else default_value

    if predator_value is None:
        predator_value = shared_value if shared_value is not None else default_value

    return (float(prey_value), float(predator_value))


def resolve_activity_regimes(strategy_codes_text):
    if strategy_codes_text is None:
        return ACTIVITY_REGIMES

    regime_by_code = {regime["code"]: regime for regime in ACTIVITY_REGIMES}
    selected_regimes = []
    seen_codes = set()
    for raw_code in str(strategy_codes_text).split(","):
        strategy_code = raw_code.strip()
        if not strategy_code:
            continue
        if strategy_code not in regime_by_code:
            raise ValueError(f"Unknown strategy code: {strategy_code}")
        if strategy_code in seen_codes:
            continue
        selected_regimes.append(regime_by_code[strategy_code])
        seen_codes.add(strategy_code)

    if not selected_regimes:
        raise ValueError("--strategy-codes must include at least one valid code.")

    return tuple(selected_regimes)


def build_lighting_regime(t_sunset, dt):
    t_sunset = float(t_sunset)
    if t_sunset < 0.0 or t_sunset > 1.0:
        raise ValueError("t_sunset must lie in the interval [0, 1].")

    epsilon = min(max(0.5 * float(dt), 1.0e-6), 0.25 * CYCLE_PERIOD)
    effective_sunset = min(max(t_sunset, epsilon), 1.0 - epsilon)
    return {
        "display_sunset": t_sunset,
        "day_start": DAY_START,
        "day_end": effective_sunset * CYCLE_PERIOD,
    }


def build_initial_condition(centers, width):
    centers = tuple(float(center) for center in centers)
    width = float(width)

    def initial_condition(x):
        x = np.asarray(x, dtype=float)
        dx = x[1] - x[0]
        length = (x[-1] - x[0]) + dx
        profiles = []

        for center in centers:
            wrapped_distance = ((x - center + 0.5 * length) % length) - 0.5 * length
            profiles.append(np.exp(-0.5 * (wrapped_distance / width) ** 2))

        values = np.column_stack(profiles)
        masses = dx * np.sum(values, axis=0, keepdims=True)
        return values / masses

    return initial_condition


def build_reaction_term(reaction_rates):
    prey_growth = float(reaction_rates["prey_growth"])
    predator_decay = float(reaction_rates["predator_decay"])
    predation_rate = float(reaction_rates["predation_rate"])
    conversion_rate = float(reaction_rates["conversion_rate"])

    def reaction_term(population, time, model):
        u1 = population[:, 0]
        u2 = population[:, 1]
        active_1 = float(model.is_active(time, population_index=0))
        active_2 = float(model.is_active(time, population_index=1))
        return np.column_stack(
            (
                active_1 * prey_growth * u1 - active_2 * predation_rate * u1 * u2,
                -predator_decay * u2 + active_2 * conversion_rate * u1 * u2,
            )
        )

    return reaction_term


def build_config(
    *,
    t_sunset,
    weights,
    sight_radius,
    smell_radius,
    number_of_points,
    dt,
    number_of_cycles,
    observation_window,
    payoff_mode=OVERLAP_PAYOFF_MODE,
    initial_centers,
    initial_width,
    diffusion,
    attraction,
    reaction_rates,
):
    return {
        "t_sunset": float(t_sunset),
        "weights": tuple(float(weight) for weight in weights),
        "sight_radius": _coerce_population_parameter(
            sight_radius,
            parameter_name="sight_radius",
        ),
        "smell_radius": _coerce_population_parameter(
            smell_radius,
            parameter_name="smell_radius",
        ),
        "number_of_points": int(number_of_points),
        "dt": float(dt),
        "number_of_cycles": int(number_of_cycles),
        "observation_window": float(observation_window),
        "payoff_mode": str(payoff_mode),
        "initial_centers": tuple(float(center) for center in initial_centers),
        "initial_width": float(initial_width),
        "diffusion": tuple(float(value) for value in diffusion),
        "attraction": tuple(
            tuple(float(value) for value in row) for row in attraction
        ),
        "reaction_rates": {
            key: float(value) for key, value in reaction_rates.items()
        },
    }


def build_config_from_args(args):
    sight_radius = resolve_population_parameter_pair(
        args.sight_radius,
        args.prey_sight_radius,
        args.predator_sight_radius,
        default_value=DEFAULT_SIGHT_RADIUS,
    )
    smell_radius = resolve_population_parameter_pair(
        args.smell_radius,
        args.prey_smell_radius,
        args.predator_smell_radius,
        default_value=DEFAULT_SMELL_RADIUS,
    )

    return build_config(
        t_sunset=args.t_sunset,
        weights=args.weights,
        sight_radius=sight_radius,
        smell_radius=smell_radius,
        number_of_points=args.number_of_points,
        dt=args.dt,
        number_of_cycles=args.number_of_cycles,
        observation_window=args.observation_window,
        initial_centers=args.initial_centers,
        initial_width=args.initial_width,
        diffusion=args.diffusion,
        attraction=(
            (args.chi11, args.chi12),
            (args.chi21, args.chi22),
        ),
        reaction_rates={
            "prey_growth": args.prey_growth,
            "predator_decay": args.predator_decay,
            "predation_rate": args.predation_rate,
            "conversion_rate": args.conversion_rate,
        },
    )


def build_solver(prey_regime, predator_regime, config):
    attraction = np.asarray(config["attraction"], dtype=float)
    diffusion = np.asarray(config["diffusion"], dtype=float)
    weights = np.asarray(config["weights"], dtype=float)
    lighting_regime = build_lighting_regime(config["t_sunset"], config["dt"])
    total_time = config["number_of_cycles"] * CYCLE_PERIOD

    return DayNightModel1D(
        a_border=0.0,
        b_border=1.0,
        number_of_points=config["number_of_points"],
        total_time=total_time,
        dt=config["dt"],
        initial_condition=build_initial_condition(
            config["initial_centers"],
            config["initial_width"],
        ),
        coefficient_attraction=attraction,
        coefficient_diffusion=diffusion,
        cycle_period=CYCLE_PERIOD,
        number_of_population=NUMBER_OF_POPULATIONS,
        day_start=lighting_regime["day_start"],
        day_end=lighting_regime["day_end"],
        time_input_mode="phase",
        activity_mode="always",
        activity_periods=[prey_regime["periods"], predator_regime["periods"]],
        sight_weight=weights,
        sight_radius=config["sight_radius"],
        smell_radius=config["smell_radius"],
        reaction_term=build_reaction_term(config["reaction_rates"]),
    )


def solve_case(prey_regime, predator_regime, config):
    model = build_solver(prey_regime, predator_regime, config)
    model.solve()

    payoff_mode = config.get("payoff_mode", OVERLAP_PAYOFF_MODE)
    normalisation = config["observation_window"] / CYCLE_PERIOD

    if payoff_mode == OVERLAP_PAYOFF_MODE:
        raw_predator_payoff = model.get_overlap_energy(
            population_indices=(0, 1),
            observation_window=config["observation_window"],
        )
        predator_payoff = float(raw_predator_payoff / normalisation)
        prey_payoff = -predator_payoff
        return model, prey_payoff, predator_payoff

    if payoff_mode == POPULATION_INTEGRAL_PAYOFF_MODE:
        prey_payoff = float(
            model.get_population_window_integral(
                0,
                observation_window=config["observation_window"],
            )
            / normalisation
        )
        predator_payoff = float(
            model.get_population_window_integral(
                1,
                observation_window=config["observation_window"],
            )
            / normalisation
        )
        return model, prey_payoff, predator_payoff

    raise ValueError(f"Unsupported payoff_mode: {payoff_mode!r}")


def build_population_heatmap_output_path(output_directory, prey_regime, predator_regime):
    return output_directory / (
        f"prey_{prey_regime['code']}_predator_{predator_regime['code']}.png"
    )


def _normalise_json_data(value):
    if isinstance(value, dict):
        return {key: _normalise_json_data(value[key]) for key in sorted(value)}
    if isinstance(value, (list, tuple)):
        return [_normalise_json_data(item) for item in value]
    return value


def build_run_config_snapshot(config):
    return _normalise_json_data(config)


def load_run_config_snapshot(output_path):
    output_path = Path(output_path)
    if not output_path.exists():
        return None

    with output_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_run_config_snapshot(output_path, config):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(build_run_config_snapshot(config), handle, indent=2, sort_keys=True)


def is_run_config_compatible(config, output_path):
    saved_snapshot = load_run_config_snapshot(output_path)
    if saved_snapshot is None:
        return False

    return saved_snapshot == build_run_config_snapshot(config)


def should_save_population_heatmaps(
    prey_regime,
    predator_regime,
    prey_code=None,
    predator_code=None,
):
    if prey_code is None and predator_code is None:
        return True

    return (
        prey_regime["code"] == prey_code
        and predator_regime["code"] == predator_code
    )


def format_case_payoff_summary(prey_payoff, predator_payoff, payoff_mode):
    if payoff_mode == OVERLAP_PAYOFF_MODE:
        return f"E={predator_payoff:.6f}"

    return (
        f"prey={prey_payoff:.6f}, "
        f"predator={predator_payoff:.6f}"
    )


def format_population_heatmap_payoff_summary(prey_payoff, predator_payoff, payoff_mode):
    if payoff_mode == OVERLAP_PAYOFF_MODE:
        return f"$\\mathcal{{E}}={predator_payoff:.3f}$"

    return (
        f"prey payoff={prey_payoff:.3f}, "
        f"predator payoff={predator_payoff:.3f}"
    )


def run_single_case(
    prey_index,
    predator_index,
    prey_regime,
    predator_regime,
    config,
    population_heatmap_output_directory=None,
    heatmap_prey_code=None,
    heatmap_predator_code=None,
):
    model, prey_payoff, predator_payoff = solve_case(
        prey_regime,
        predator_regime,
        config,
    )

    if population_heatmap_output_directory is not None and should_save_population_heatmaps(
        prey_regime,
        predator_regime,
        prey_code=heatmap_prey_code,
        predator_code=heatmap_predator_code,
    ):
        save_population_heatmaps(
            model,
            prey_regime,
            predator_regime,
            build_population_heatmap_output_path(
                population_heatmap_output_directory,
                prey_regime,
                predator_regime,
            ),
            config["t_sunset"],
            config["weights"],
            prey_payoff,
            predator_payoff,
            config.get("payoff_mode", OVERLAP_PAYOFF_MODE),
        )

    return prey_index, predator_index, float(prey_payoff), float(predator_payoff)


def build_empty_payoff_matrices(activity_regimes):
    shape = (len(activity_regimes), len(activity_regimes))
    return {
        "prey": np.full(shape, np.nan, dtype=float),
        "predator": np.full(shape, np.nan, dtype=float),
    }


def load_case_payoff_matrices(activity_regimes, output_path, payoff_mode):
    matrices = build_empty_payoff_matrices(activity_regimes)
    output_path = Path(output_path)
    if not output_path.exists():
        return matrices

    regime_index_by_code = {
        regime["code"]: index for index, regime in enumerate(activity_regimes)
    }

    with output_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            prey_code = row.get("prey")
            predator_code = row.get("predator")
            if prey_code not in regime_index_by_code or predator_code not in regime_index_by_code:
                continue

            if payoff_mode == OVERLAP_PAYOFF_MODE:
                payoff_text = row.get("payoff")
                prey_text = row.get("prey_payoff")
                predator_text = row.get("predator_payoff")
                if payoff_text not in {None, ""}:
                    predator_payoff = float(payoff_text)
                    prey_payoff = -predator_payoff
                elif prey_text not in {None, ""} and predator_text not in {None, ""}:
                    prey_payoff = float(prey_text)
                    predator_payoff = float(predator_text)
                else:
                    continue
            else:
                prey_text = row.get("prey_payoff")
                predator_text = row.get("predator_payoff")
                if prey_text in {None, ""} or predator_text in {None, ""}:
                    continue
                prey_payoff = float(prey_text)
                predator_payoff = float(predator_text)

            row_index = regime_index_by_code[prey_code]
            column_index = regime_index_by_code[predator_code]
            matrices["prey"][row_index, column_index] = prey_payoff
            matrices["predator"][row_index, column_index] = predator_payoff

    return matrices


def load_case_payoff_matrix(activity_regimes, output_path):
    return load_case_payoff_matrices(
        activity_regimes,
        output_path,
        OVERLAP_PAYOFF_MODE,
    )["predator"]


def initialise_case_payoff_output(output_path, payoff_mode):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        return

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        if payoff_mode == OVERLAP_PAYOFF_MODE:
            writer.writerow(["prey", "predator", "payoff", "prey_payoff", "predator_payoff"])
        else:
            writer.writerow(["prey", "predator", "prey_payoff", "predator_payoff"])


def append_case_payoff(
    output_path,
    prey_code,
    predator_code,
    prey_payoff,
    predator_payoff,
    payoff_mode,
):
    initialise_case_payoff_output(output_path, payoff_mode)
    with Path(output_path).open("a", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        if payoff_mode == OVERLAP_PAYOFF_MODE:
            writer.writerow(
                [
                    prey_code,
                    predator_code,
                    f"{float(predator_payoff):.10f}",
                    f"{float(prey_payoff):.10f}",
                    f"{float(predator_payoff):.10f}",
                ]
            )
        else:
            writer.writerow(
                [
                    prey_code,
                    predator_code,
                    f"{float(prey_payoff):.10f}",
                    f"{float(predator_payoff):.10f}",
                ]
            )


def run_all_cases(
    activity_regimes,
    config,
    max_workers,
    population_heatmap_output_directory=None,
    heatmap_prey_code=None,
    heatmap_predator_code=None,
    case_payoff_output_path=None,
    echo=True,
):
    payoff_mode = config.get("payoff_mode", OVERLAP_PAYOFF_MODE)
    if case_payoff_output_path is None:
        payoff_matrices = build_empty_payoff_matrices(activity_regimes)
    else:
        payoff_matrices = load_case_payoff_matrices(
            activity_regimes,
            case_payoff_output_path,
            payoff_mode,
        )

    completed_mask = np.isfinite(payoff_matrices["prey"]) & np.isfinite(
        payoff_matrices["predator"]
    )

    case_specs = [
        (prey_index, predator_index, prey_regime, predator_regime)
        for prey_index, prey_regime in enumerate(activity_regimes)
        for predator_index, predator_regime in enumerate(activity_regimes)
        if not completed_mask[prey_index, predator_index]
    ]

    if case_payoff_output_path is not None:
        initialise_case_payoff_output(case_payoff_output_path, payoff_mode)

    if not case_specs:
        return payoff_matrices

    total_case_count = len(case_specs)

    if max_workers <= 1:
        if echo:
            print(
                f"  Solving {total_case_count} payoff case(s) with 1 worker.",
                flush=True,
            )

        for completed_case_count, (
            prey_index,
            predator_index,
            prey_regime,
            predator_regime,
        ) in enumerate(case_specs, start=1):
            _, _, prey_payoff, predator_payoff = run_single_case(
                prey_index,
                predator_index,
                prey_regime,
                predator_regime,
                config,
                population_heatmap_output_directory,
                heatmap_prey_code,
                heatmap_predator_code,
            )
            payoff_matrices["prey"][prey_index, predator_index] = prey_payoff
            payoff_matrices["predator"][prey_index, predator_index] = predator_payoff
            if case_payoff_output_path is not None:
                append_case_payoff(
                    case_payoff_output_path,
                    prey_regime["code"],
                    predator_regime["code"],
                    prey_payoff,
                    predator_payoff,
                    payoff_mode,
                )
            if echo:
                print(
                    f"  Finished case {completed_case_count}/{total_case_count}: "
                    f"prey={prey_regime['code']}, predator={predator_regime['code']}: "
                    f"{format_case_payoff_summary(prey_payoff, predator_payoff, payoff_mode)}",
                    flush=True,
                )
        return payoff_matrices

    worker_count = min(max_workers, len(case_specs))
    if echo:
        print(
            f"  Solving {total_case_count} payoff case(s) with {worker_count} worker(s).",
            flush=True,
        )

    executor = ProcessPoolExecutor(max_workers=worker_count)
    interrupted = False
    try:
        future_to_case = {
            executor.submit(
                run_single_case,
                prey_index,
                predator_index,
                prey_regime,
                predator_regime,
                config,
                population_heatmap_output_directory,
                heatmap_prey_code,
                heatmap_predator_code,
            ): (prey_regime["code"], predator_regime["code"])
            for prey_index, predator_index, prey_regime, predator_regime in case_specs
        }

        for completed_case_count, future in enumerate(as_completed(future_to_case), start=1):
            prey_code, predator_code = future_to_case[future]
            prey_index, predator_index, prey_payoff, predator_payoff = future.result()
            payoff_matrices["prey"][prey_index, predator_index] = prey_payoff
            payoff_matrices["predator"][prey_index, predator_index] = predator_payoff
            if case_payoff_output_path is not None:
                append_case_payoff(
                    case_payoff_output_path,
                    prey_code,
                    predator_code,
                    prey_payoff,
                    predator_payoff,
                    payoff_mode,
                )
            if echo:
                print(
                    f"  Finished case {completed_case_count}/{total_case_count}: "
                    f"prey={prey_code}, predator={predator_code}: "
                    f"{format_case_payoff_summary(prey_payoff, predator_payoff, payoff_mode)}",
                    flush=True,
                )
    except KeyboardInterrupt:
        interrupted = True
        executor.shutdown(wait=False, cancel_futures=True)
        if echo:
            print(
                "\n  Interrupted while computing payoff cases; cancelled remaining workers.",
                flush=True,
            )
        raise
    finally:
        if not interrupted:
            executor.shutdown(wait=True)

    return payoff_matrices


def save_payoff_csv(matrix, activity_regimes, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Prey / Predator"] + [regime["code"] for regime in activity_regimes])
        for prey_regime, row in zip(activity_regimes, matrix):
            writer.writerow([prey_regime["code"]] + [f"{value:.10f}" for value in row])


def save_payoff_heatmap(
    matrix,
    activity_regimes,
    output_path,
    t_sunset,
    weights,
    *,
    title,
    colorbar_label,
):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    labels = [regime["code"] for regime in activity_regimes]

    figure, axis = plt.subplots(figsize=(7.2, 6.1), constrained_layout=True)
    image = axis.imshow(matrix, cmap="viridis")
    axis.set_xticks(np.arange(len(labels)), labels=labels)
    axis.set_yticks(np.arange(len(labels)), labels=labels)
    axis.set_xlabel("Predator activity pattern")
    axis.set_ylabel("Prey activity pattern")
    axis.set_title(
        f"{title}\n"
        f"$t_{{sunset}}={t_sunset:g}$, $w_1={weights[0]:g}$, $w_2={weights[1]:g}$"
    )

    value_span = float(np.max(matrix) - np.min(matrix))
    threshold = float(np.min(matrix) + 0.5 * value_span)
    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            value = matrix[row_index, column_index]
            text_color = "white" if value >= threshold else "black"
            axis.text(
                column_index,
                row_index,
                f"{value:.3f}",
                ha="center",
                va="center",
                color=text_color,
                fontsize=9,
            )

    figure.colorbar(image, ax=axis, label=colorbar_label)
    figure.savefig(output_path, bbox_inches="tight", dpi=200)
    plt.close(figure)


def save_population_heatmaps(
    model,
    prey_regime,
    predator_regime,
    output_path,
    t_sunset,
    weights,
    prey_payoff,
    predator_payoff,
    payoff_mode,
):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    figure, axes = plt.subplots(
        1,
        NUMBER_OF_POPULATIONS,
        figsize=(10.0, 4.2),
        squeeze=False,
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    flat_axes = axes.ravel()
    extent = [model.a_border, model.b_border, model.time[0], model.time[-1]]
    vmin = float(np.min(model.U))
    vmax = float(np.max(model.U))
    image = None
    titles = (
        ("u1 (prey)", prey_regime),
        ("u2 (predator)", predator_regime),
    )

    for population_index, (axis, (population_label, regime)) in enumerate(
        zip(flat_axes, titles)
    ):
        image = axis.imshow(
            model.U[:, :, population_index],
            origin="lower",
            aspect="auto",
            extent=extent,
            cmap="hot_r",
            vmin=vmin,
            vmax=vmax,
        )
        axis.set_title(f"{population_label}\n{regime['label']} [{regime['code']}]")
        axis.set_xlabel("x")
        if population_index == 0:
            axis.set_ylabel("t")
        model._add_transition_markers(
            axis,
            population_index,
            show_legend=population_index == 0,
            show_day_night_cycle=True,
            show_activity_period=True,
        )

    figure.suptitle(
        "Predator-prey population heatmaps\n"
        f"prey={prey_regime['code']}, predator={predator_regime['code']}, "
        f"{format_population_heatmap_payoff_summary(prey_payoff, predator_payoff, payoff_mode)}, "
        f"$t_{{sunset}}={t_sunset:g}$, "
        f"$w_1={weights[0]:g}$, $w_2={weights[1]:g}$"
    )
    figure.colorbar(image, ax=flat_axes, label="density")
    figure.savefig(output_path, bbox_inches="tight", dpi=200)
    plt.close(figure)


def format_matrix_for_console(matrix, activity_regimes):
    labels = [regime["code"] for regime in activity_regimes]
    header = ["Prey/Pred"] + labels
    rows = [header]
    for label, values in zip(labels, matrix):
        rows.append([label] + [f"{value:.4f}" for value in values])

    column_widths = [max(len(row[column_index]) for row in rows) for column_index in range(len(header))]
    return "\n".join(
        "  ".join(
            value.rjust(column_widths[column_index])
            for column_index, value in enumerate(row)
        )
        for row in rows
    )


def validate_config(
    config,
    max_workers,
    activity_regimes,
    heatmap_prey_code=None,
    heatmap_predator_code=None,
):
    if config.get("payoff_mode") not in PAYOFF_MODE_CHOICES:
        raise ValueError(
            "payoff_mode must be one of: "
            + ", ".join(PAYOFF_MODE_CHOICES)
            + "."
        )

    if config["number_of_points"] < 2:
        raise ValueError("number_of_points must be at least 2.")

    if config["dt"] <= 0.0:
        raise ValueError("dt must be positive.")

    if config["number_of_cycles"] < 1:
        raise ValueError("number_of_cycles must be at least 1.")

    if config["observation_window"] <= 0.0:
        raise ValueError("observation_window must be positive.")

    if max_workers < 1:
        raise ValueError("max_workers must be at least 1.")

    if (heatmap_prey_code is None) != (heatmap_predator_code is None):
        raise ValueError(
            "--heatmap-prey and --heatmap-predator must be provided together."
        )

    selected_codes = {regime["code"] for regime in activity_regimes}

    if heatmap_prey_code is not None and heatmap_prey_code not in ACTIVITY_REGIME_CODES:
        raise ValueError(f"Unknown prey activity code: {heatmap_prey_code}")

    if (
        heatmap_predator_code is not None
        and heatmap_predator_code not in ACTIVITY_REGIME_CODES
    ):
        raise ValueError(f"Unknown predator activity code: {heatmap_predator_code}")

    if heatmap_prey_code is not None and heatmap_prey_code not in selected_codes:
        raise ValueError(
            f"Heatmap prey code {heatmap_prey_code} is not included in --strategy-codes."
        )

    if heatmap_predator_code is not None and heatmap_predator_code not in selected_codes:
        raise ValueError(
            "Heatmap predator code "
            f"{heatmap_predator_code} is not included in --strategy-codes."
        )

    if any(weight < 0.0 or weight > 1.0 for weight in config["weights"]):
        raise ValueError("Each sight weight must lie in [0, 1].")

    sight_radius = np.atleast_1d(np.asarray(config["sight_radius"], dtype=float))
    smell_radius = np.atleast_1d(np.asarray(config["smell_radius"], dtype=float))

    if sight_radius.size not in {1, NUMBER_OF_POPULATIONS}:
        raise ValueError(
            f"sight_radius must be a scalar or have length {NUMBER_OF_POPULATIONS}."
        )

    if smell_radius.size not in {1, NUMBER_OF_POPULATIONS}:
        raise ValueError(
            f"smell_radius must be a scalar or have length {NUMBER_OF_POPULATIONS}."
        )

    if not np.all(np.isfinite(sight_radius)) or np.any(sight_radius <= 0.0):
        raise ValueError("Each sight_radius must be positive and finite.")

    if not np.all(np.isfinite(smell_radius)) or np.any(smell_radius <= 0.0):
        raise ValueError("Each smell_radius must be positive and finite.")

    if config["initial_width"] <= 0.0:
        raise ValueError("initial_width must be positive.")

    build_lighting_regime(config["t_sunset"], config["dt"])


def build_config_from_args(args):
    sight_radius = resolve_population_parameter_pair(
        args.sight_radius,
        args.prey_sight_radius,
        args.predator_sight_radius,
        default_value=DEFAULT_SIGHT_RADIUS,
    )
    smell_radius = resolve_population_parameter_pair(
        args.smell_radius,
        args.prey_smell_radius,
        args.predator_smell_radius,
        default_value=DEFAULT_SMELL_RADIUS,
    )

    config = build_config(
        t_sunset=args.t_sunset,
        weights=args.weights,
        sight_radius=sight_radius,
        smell_radius=smell_radius,
        number_of_points=args.number_of_points,
        dt=args.dt,
        number_of_cycles=args.number_of_cycles,
        observation_window=args.observation_window,
        payoff_mode=args.payoff_mode,
        initial_centers=args.initial_centers,
        initial_width=args.initial_width,
        diffusion=args.diffusion,
        attraction=(
            (args.chi11, args.chi12),
            (args.chi21, args.chi22),
        ),
        reaction_rates={
            "prey_growth": args.prey_growth,
            "predator_decay": args.predator_decay,
            "predation_rate": args.predation_rate,
            "conversion_rate": args.conversion_rate,
        },
    )
    selected_activity_regimes = resolve_activity_regimes(args.strategy_codes)
    config["strategy_codes"] = tuple(regime["code"] for regime in selected_activity_regimes)
    return config


def build_output_paths(output_dir):
    output_dir = Path(output_dir)
    return {
        "output_dir": output_dir,
        "csv_output_path": output_dir / CSV_OUTPUT_PATH.name,
        "prey_csv_output_path": output_dir / PREY_CSV_OUTPUT_PATH.name,
        "predator_csv_output_path": output_dir / PREDATOR_CSV_OUTPUT_PATH.name,
        "case_payoff_output_path": output_dir / CASE_PAYOFF_OUTPUT_PATH.name,
        "run_config_output_path": output_dir / RUN_CONFIG_OUTPUT_PATH.name,
        "heatmap_output_path": output_dir / HEATMAP_OUTPUT_PATH.name,
        "prey_heatmap_output_path": output_dir / PREY_HEATMAP_OUTPUT_PATH.name,
        "predator_heatmap_output_path": output_dir / PREDATOR_HEATMAP_OUTPUT_PATH.name,
        "population_heatmap_output_directory": (
            output_dir / POPULATION_HEATMAP_OUTPUT_DIRECTORY.name
        ),
    }


def prepare_output_directory_for_run(output_paths, config, *, echo):
    run_config_output_path = output_paths["run_config_output_path"]
    case_payoff_output_path = output_paths["case_payoff_output_path"]
    saved_snapshot = load_run_config_snapshot(run_config_output_path)
    current_snapshot = build_run_config_snapshot(config)

    if saved_snapshot == current_snapshot:
        return

    if echo and case_payoff_output_path.exists():
        if saved_snapshot is None:
            print(
                f"Existing case data in {output_paths['output_dir']} has no config snapshot; restarting this folder from scratch.",
                flush=True,
            )
        else:
            print(
                f"Existing case data in {output_paths['output_dir']} was generated with a different configuration; restarting this folder from scratch.",
                flush=True,
            )

    if case_payoff_output_path.exists():
        case_payoff_output_path.unlink()

    save_run_config_snapshot(run_config_output_path, config)


def run_payoff_experiment(
    *,
    output_dir,
    config,
    max_workers,
    activity_regimes=ACTIVITY_REGIMES,
    heatmap_prey_code=None,
    heatmap_predator_code=None,
    echo=True,
):
    validate_config(
        config,
        max_workers,
        activity_regimes,
        heatmap_prey_code=heatmap_prey_code,
        heatmap_predator_code=heatmap_predator_code,
    )
    payoff_mode = config.get("payoff_mode", OVERLAP_PAYOFF_MODE)
    output_paths = build_output_paths(output_dir)
    prepare_output_directory_for_run(output_paths, config, echo=echo)
    completed_case_matrices = load_case_payoff_matrices(
        activity_regimes,
        output_paths["case_payoff_output_path"],
        payoff_mode,
    )
    completed_case_count = int(
        np.count_nonzero(
            np.isfinite(completed_case_matrices["prey"])
            & np.isfinite(completed_case_matrices["predator"])
        )
    )
    if echo and completed_case_count > 0:
        print(
            f"Resuming from {completed_case_count} completed case(s) in "
            f"{output_paths['case_payoff_output_path']}",
            flush=True,
        )

    payoff_matrices = run_all_cases(
        activity_regimes,
        config,
        max_workers,
        population_heatmap_output_directory=output_paths[
            "population_heatmap_output_directory"
        ],
        heatmap_prey_code=heatmap_prey_code,
        heatmap_predator_code=heatmap_predator_code,
        case_payoff_output_path=output_paths["case_payoff_output_path"],
        echo=echo,
    )
    if payoff_mode == OVERLAP_PAYOFF_MODE:
        save_payoff_csv(
            payoff_matrices["predator"],
            activity_regimes,
            output_paths["csv_output_path"],
        )
        save_payoff_heatmap(
            payoff_matrices["predator"],
            activity_regimes,
            output_paths["heatmap_output_path"],
            config["t_sunset"],
            config["weights"],
            title="Predator-prey payoff matrix",
            colorbar_label=r"$\mathcal{E}$",
        )
    else:
        save_payoff_csv(
            payoff_matrices["prey"],
            activity_regimes,
            output_paths["prey_csv_output_path"],
        )
        save_payoff_csv(
            payoff_matrices["predator"],
            activity_regimes,
            output_paths["predator_csv_output_path"],
        )
        save_payoff_heatmap(
            payoff_matrices["prey"],
            activity_regimes,
            output_paths["prey_heatmap_output_path"],
            config["t_sunset"],
            config["weights"],
            title="Prey payoff matrix",
            colorbar_label="prey payoff",
        )
        save_payoff_heatmap(
            payoff_matrices["predator"],
            activity_regimes,
            output_paths["predator_heatmap_output_path"],
            config["t_sunset"],
            config["weights"],
            title="Predator payoff matrix",
            colorbar_label="predator payoff",
        )

    saved_heatmap_count = (
        1 if heatmap_prey_code is not None else len(activity_regimes) ** 2
    )
    if echo:
        if payoff_mode == OVERLAP_PAYOFF_MODE:
            print("\nComputed payoff matrix:")
            print(format_matrix_for_console(payoff_matrices["predator"], activity_regimes))
            print(f"\nSaved payoff matrix CSV to {output_paths['csv_output_path']}")
            print(
                f"Saved payoff matrix heatmap to {output_paths['heatmap_output_path']}"
            )
        else:
            print("\nComputed prey payoff matrix:")
            print(format_matrix_for_console(payoff_matrices["prey"], activity_regimes))
            print("\nComputed predator payoff matrix:")
            print(format_matrix_for_console(payoff_matrices["predator"], activity_regimes))
            print(
                f"\nSaved prey payoff matrix CSV to {output_paths['prey_csv_output_path']}"
            )
            print(
                f"Saved predator payoff matrix CSV to {output_paths['predator_csv_output_path']}"
            )
            print(
                f"Saved prey payoff heatmap to {output_paths['prey_heatmap_output_path']}"
            )
            print(
                f"Saved predator payoff heatmap to {output_paths['predator_heatmap_output_path']}"
            )
        print(
            f"Saved {saved_heatmap_count} population heatmap file(s) to "
            f"{output_paths['population_heatmap_output_directory']}"
        )

    return {
        "matrix": payoff_matrices["predator"],
        "prey_matrix": payoff_matrices["prey"],
        "predator_matrix": payoff_matrices["predator"],
        "saved_heatmap_count": saved_heatmap_count,
        **output_paths,
    }


def main():
    args = parse_args()
    config = build_config_from_args(args)
    activity_regimes = resolve_activity_regimes(args.strategy_codes)
    run_payoff_experiment(
        output_dir=args.output_dir,
        config=config,
        max_workers=args.max_workers,
        activity_regimes=activity_regimes,
        heatmap_prey_code=args.heatmap_prey,
        heatmap_predator_code=args.heatmap_predator,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())