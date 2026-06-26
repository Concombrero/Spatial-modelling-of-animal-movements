import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


CURRENT_DIRECTORY = Path(__file__).resolve().parent
if str(CURRENT_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIRECTORY))

from payoff_matrix import (
    ACTIVITY_REGIMES,
    CYCLE_PERIOD,
    DEFAULT_ATTRACTION,
    DEFAULT_DIFFUSION,
    DEFAULT_INITIAL_CENTERS,
    DEFAULT_INITIAL_WIDTH,
    DEFAULT_REACTION_RATES,
    DEFAULT_SIGHT_RADIUS,
    DEFAULT_SMELL_RADIUS,
    build_lighting_regime,
)
from solver import DayNightModel1D


OUTPUT_DIRECTORY = Path(__file__).resolve().parent / "output/evolutionary_game"
ROUND_PAYOFF_OUTPUT_PATH = OUTPUT_DIRECTORY / "round_payoffs.csv"
DISTRIBUTION_HISTORY_OUTPUT_PATH = OUTPUT_DIRECTORY / "distribution_history.csv"
SELECTION_EVENT_OUTPUT_PATH = OUTPUT_DIRECTORY / "selection_events.csv"
RUN_CONFIG_OUTPUT_PATH = OUTPUT_DIRECTORY / "run_config.json"
SHARE_PLOT_OUTPUT_PATH = OUTPUT_DIRECTORY / "strategy_shares.png"

DEFAULT_NUMBER_OF_POINTS = 64
DEFAULT_DT = 0.1
DEFAULT_NUMBER_OF_CYCLES = 4
DEFAULT_ROUNDS = 8
DEFAULT_SELECTION_EVENTS = 2
DEFAULT_SELECTION_PERCENTAGE = 10.0
MINIMUM_SHARE_FRACTION = 0.01
DEFAULT_PREY_TOTAL_MASS = 1.0
DEFAULT_PREDATOR_TOTAL_MASS = 1.0
NUMBER_OF_SPECIES = 2


def _coerce_species_parameter(values, *, parameter_name):
    array = np.asarray(values, dtype=float)
    if array.shape == ():
        return float(array)

    flattened = tuple(float(value) for value in np.ravel(array))
    if len(flattened) != NUMBER_OF_SPECIES:
        raise ValueError(
            f"{parameter_name} must be a scalar or have length {NUMBER_OF_SPECIES}."
        )

    return flattened


def build_config(
    *,
    w1=0.5,
    w2=0.5,
    t_sunset=0.5,
    number_of_points=DEFAULT_NUMBER_OF_POINTS,
    dt=DEFAULT_DT,
    number_of_cycles=DEFAULT_NUMBER_OF_CYCLES,
    rounds=DEFAULT_ROUNDS,
    selection_events=DEFAULT_SELECTION_EVENTS,
    selection_percentage=DEFAULT_SELECTION_PERCENTAGE,
    prey_total_mass=DEFAULT_PREY_TOTAL_MASS,
    predator_total_mass=DEFAULT_PREDATOR_TOTAL_MASS,
    sight_radius=DEFAULT_SIGHT_RADIUS,
    smell_radius=DEFAULT_SMELL_RADIUS,
    initial_centers=DEFAULT_INITIAL_CENTERS,
    initial_width=DEFAULT_INITIAL_WIDTH,
    diffusion=DEFAULT_DIFFUSION,
    attraction=DEFAULT_ATTRACTION,
    reaction_rates=None,
):
    if reaction_rates is None:
        reaction_rates = DEFAULT_REACTION_RATES

    return {
        "weights": (float(w1), float(w2)),
        "payoff_definition": "weighted overlap from model.get_overlap_energy()",
        "prey_objective": "minimize payoff",
        "predator_objective": "maximize payoff",
        "minimum_share_percentage": 100.0 * MINIMUM_SHARE_FRACTION,
        "selection_update": (
            "Each selection step transfers a fixed percentage of the losing "
            "regime's current share to the winning regime while keeping every "
            "regime above the minimum share floor."
        ),
        "t_sunset": float(t_sunset),
        "number_of_points": int(number_of_points),
        "dt": float(dt),
        "number_of_cycles": int(number_of_cycles),
        "rounds": int(rounds),
        "selection_events": int(selection_events),
        "selection_percentage": float(selection_percentage),
        "prey_total_mass": float(prey_total_mass),
        "predator_total_mass": float(predator_total_mass),
        "sight_radius": _coerce_species_parameter(
            sight_radius,
            parameter_name="sight_radius",
        ),
        "smell_radius": _coerce_species_parameter(
            smell_radius,
            parameter_name="smell_radius",
        ),
        "initial_centers": tuple(float(value) for value in initial_centers),
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
            "Run an evolutionary predator-prey simulation where prey and predators "
            "share one species-level model but are split into circadian subgroups. "
            "Each ecological round runs the PDE solver for a fixed time, computes "
            "per-subgroup overlap payoffs with model.get_overlap_energy(), and then "
            "redistributes population share from the worst subgroup to the best "
            "one. Prey minimize overlap while predators maximize it."
        )
    )
    parser.add_argument(
        "--w1",
        type=float,
        default=0.5,
        help="Fixed sight weight for every prey subgroup. Default: 0.5.",
    )
    parser.add_argument(
        "--w2",
        type=float,
        default=0.5,
        help="Fixed sight weight for every predator subgroup. Default: 0.5.",
    )
    parser.add_argument(
        "--t-sunset",
        type=float,
        default=0.5,
        help="Daylight proportion t_sunset in [0, 1]. Default: 0.5.",
    )
    parser.add_argument(
        "--number-of-points",
        type=int,
        default=DEFAULT_NUMBER_OF_POINTS,
        help=f"Number of spatial grid points. Default: {DEFAULT_NUMBER_OF_POINTS}.",
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=DEFAULT_DT,
        help=f"Stored output timestep for each round. Default: {DEFAULT_DT:g}.",
    )
    parser.add_argument(
        "--number-of-cycles",
        type=int,
        default=DEFAULT_NUMBER_OF_CYCLES,
        help=(
            "Number of day-night cycles in each ecological round. "
            f"Default: {DEFAULT_NUMBER_OF_CYCLES}."
        ),
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=DEFAULT_ROUNDS,
        help=f"Number of evolutionary rounds. Default: {DEFAULT_ROUNDS}.",
    )
    parser.add_argument(
        "--selection-events",
        type=int,
        default=DEFAULT_SELECTION_EVENTS,
        help=(
            "How many percentage-transfer selection steps to apply per round and "
            "species. "
            f"Default: {DEFAULT_SELECTION_EVENTS}."
        ),
    )
    parser.add_argument(
        "--selection-percentage",
        type=float,
        default=DEFAULT_SELECTION_PERCENTAGE,
        help=(
            "Percentage of the losing regime's current share transferred to the "
            "winning regime during each selection step. Default: "
            f"{DEFAULT_SELECTION_PERCENTAGE:g}."
        ),
    )
    parser.add_argument(
        "--prey-total-mass",
        type=float,
        default=DEFAULT_PREY_TOTAL_MASS,
        help=(
            "Total initial prey mass shared across all prey circadian subgroups. "
            f"Default: {DEFAULT_PREY_TOTAL_MASS:g}."
        ),
    )
    parser.add_argument(
        "--predator-total-mass",
        type=float,
        default=DEFAULT_PREDATOR_TOTAL_MASS,
        help=(
            "Total initial predator mass shared across all predator circadian "
            f"subgroups. Default: {DEFAULT_PREDATOR_TOTAL_MASS:g}."
        ),
    )
    parser.add_argument(
        "--sight-radius",
        type=float,
        default=DEFAULT_SIGHT_RADIUS,
        help=(
            "Shared sight radius for every subgroup. "
            f"Default: {DEFAULT_SIGHT_RADIUS:g}."
        ),
    )
    parser.add_argument(
        "--smell-radius",
        type=float,
        default=DEFAULT_SMELL_RADIUS,
        help=(
            "Shared smell radius for every subgroup. "
            f"Default: {DEFAULT_SMELL_RADIUS:g}."
        ),
    )
    parser.add_argument(
        "--prey-growth",
        type=float,
        default=DEFAULT_REACTION_RATES["prey_growth"],
        help=(
            "Lotka-Volterra prey growth rate r1. Default: "
            f"{DEFAULT_REACTION_RATES['prey_growth']:g}."
        ),
    )
    parser.add_argument(
        "--predator-decay",
        type=float,
        default=DEFAULT_REACTION_RATES["predator_decay"],
        help=(
            "Lotka-Volterra predator decay rate r2. Default: "
            f"{DEFAULT_REACTION_RATES['predator_decay']:g}."
        ),
    )
    parser.add_argument(
        "--predation-rate",
        type=float,
        default=DEFAULT_REACTION_RATES["predation_rate"],
        help=(
            "Lotka-Volterra predation rate a. Default: "
            f"{DEFAULT_REACTION_RATES['predation_rate']:g}."
        ),
    )
    parser.add_argument(
        "--conversion-rate",
        type=float,
        default=DEFAULT_REACTION_RATES["conversion_rate"],
        help=(
            "Lotka-Volterra predator conversion rate b. Default: "
            f"{DEFAULT_REACTION_RATES['conversion_rate']:g}."
        ),
    )
    parser.add_argument(
        "--chi11",
        type=float,
        default=DEFAULT_ATTRACTION[0][0],
        help=f"Prey-prey attraction coefficient. Default: {DEFAULT_ATTRACTION[0][0]:g}.",
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
        help=f"Predator-predator attraction coefficient. Default: {DEFAULT_ATTRACTION[1][1]:g}.",
    )
    parser.add_argument(
        "--diffusion",
        nargs=2,
        type=float,
        metavar=("D1", "D2"),
        default=list(DEFAULT_DIFFUSION),
        help=(
            "Diffusion coefficients shared by all prey and predator subgroups. "
            f"Default: {DEFAULT_DIFFUSION[0]:g} {DEFAULT_DIFFUSION[1]:g}."
        ),
    )
    parser.add_argument(
        "--initial-centers",
        nargs=2,
        type=float,
        metavar=("X1", "X2"),
        default=list(DEFAULT_INITIAL_CENTERS),
        help=(
            "Initial Gaussian centers for prey and predator species. "
            f"Default: {DEFAULT_INITIAL_CENTERS[0]:g} {DEFAULT_INITIAL_CENTERS[1]:g}."
        ),
    )
    parser.add_argument(
        "--initial-width",
        type=float,
        default=DEFAULT_INITIAL_WIDTH,
        help=(
            "Shared Gaussian width used to split each species across its circadian "
            f"subgroups. Default: {DEFAULT_INITIAL_WIDTH:g}."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIRECTORY,
        help="Directory where CSV, JSON, and figures are saved.",
    )
    return parser.parse_args()


def build_config_from_args(args):
    return build_config(
        w1=args.w1,
        w2=args.w2,
        t_sunset=args.t_sunset,
        number_of_points=args.number_of_points,
        dt=args.dt,
        number_of_cycles=args.number_of_cycles,
        rounds=args.rounds,
        selection_events=args.selection_events,
        selection_percentage=args.selection_percentage,
        prey_total_mass=args.prey_total_mass,
        predator_total_mass=args.predator_total_mass,
        sight_radius=args.sight_radius,
        smell_radius=args.smell_radius,
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


def validate_config(config):
    if config["number_of_points"] < 2:
        raise ValueError("number_of_points must be at least 2.")

    if config["dt"] <= 0.0:
        raise ValueError("dt must be positive.")

    if config["number_of_cycles"] < 1:
        raise ValueError("number_of_cycles must be at least 1.")

    if config["rounds"] < 1:
        raise ValueError("rounds must be at least 1.")

    if config["selection_events"] < 0:
        raise ValueError("selection_events must be non-negative.")

    if config["selection_percentage"] < 0.0 or config["selection_percentage"] > 100.0:
        raise ValueError("selection_percentage must lie in [0, 100].")

    if config["prey_total_mass"] <= 0.0 or config["predator_total_mass"] <= 0.0:
        raise ValueError("Total prey and predator masses must be positive.")

    if any(weight < 0.0 or weight > 1.0 for weight in config["weights"]):
        raise ValueError("w1 and w2 must lie in [0, 1].")

    sight_radius = np.atleast_1d(np.asarray(config["sight_radius"], dtype=float))
    smell_radius = np.atleast_1d(np.asarray(config["smell_radius"], dtype=float))

    if sight_radius.size not in {1, NUMBER_OF_SPECIES}:
        raise ValueError(
            f"sight_radius must be a scalar or have length {NUMBER_OF_SPECIES}."
        )

    if smell_radius.size not in {1, NUMBER_OF_SPECIES}:
        raise ValueError(
            f"smell_radius must be a scalar or have length {NUMBER_OF_SPECIES}."
        )

    if not np.all(np.isfinite(sight_radius)) or np.any(sight_radius <= 0.0):
        raise ValueError("Each sight_radius must be positive and finite.")

    if not np.all(np.isfinite(smell_radius)) or np.any(smell_radius <= 0.0):
        raise ValueError("Each smell_radius must be positive and finite.")

    if config["initial_width"] <= 0.0:
        raise ValueError("initial_width must be positive.")

    build_lighting_regime(config["t_sunset"], config["dt"])


def counts_to_shares(counts):
    counts = np.asarray(counts, dtype=float)
    total_count = float(np.sum(counts))
    if total_count <= 0.0:
        raise ValueError("At least one subgroup share must be positive.")
    return counts / total_count


def build_equal_shares(regime_count):
    return np.full(int(regime_count), 1.0 / float(regime_count), dtype=float)


def build_periodic_profile(x, center, width):
    x = np.asarray(x, dtype=float)
    dx = float(x[1] - x[0])
    length = float((x[-1] - x[0]) + dx)
    wrapped_distance = ((x - float(center) + 0.5 * length) % length) - 0.5 * length
    profile = np.exp(-0.5 * (wrapped_distance / float(width)) ** 2)
    profile_mass = dx * float(np.sum(profile))
    if profile_mass <= 0.0:
        raise ValueError("Initial profile mass must be positive.")
    return profile / profile_mass


def build_weighted_initial_state(x, prey_shares, predator_shares, config):
    regime_count = len(ACTIVITY_REGIMES)
    prey_profile = build_periodic_profile(
        x,
        config["initial_centers"][0],
        config["initial_width"],
    )
    predator_profile = build_periodic_profile(
        x,
        config["initial_centers"][1],
        config["initial_width"],
    )

    prey_masses = config["prey_total_mass"] * counts_to_shares(prey_shares)
    predator_masses = config["predator_total_mass"] * counts_to_shares(predator_shares)

    initial_state = np.zeros((x.size, 2 * regime_count), dtype=float)
    initial_state[:, :regime_count] = prey_profile[:, np.newaxis] * prey_masses[np.newaxis, :]
    initial_state[:, regime_count:] = (
        predator_profile[:, np.newaxis] * predator_masses[np.newaxis, :]
    )
    return initial_state


def build_activity_periods():
    prey_periods = [list(regime["periods"]) for regime in ACTIVITY_REGIMES]
    predator_periods = [list(regime["periods"]) for regime in ACTIVITY_REGIMES]
    return prey_periods + predator_periods


def build_species_attraction_matrix(regime_count, attraction):
    attraction = np.asarray(attraction, dtype=float)
    prey_prey = np.full((regime_count, regime_count), attraction[0, 0], dtype=float)
    prey_predator = np.full(
        (regime_count, regime_count),
        attraction[0, 1],
        dtype=float,
    )
    predator_prey = np.full(
        (regime_count, regime_count),
        attraction[1, 0],
        dtype=float,
    )
    predator_predator = np.full(
        (regime_count, regime_count),
        attraction[1, 1],
        dtype=float,
    )
    return np.block(
        [
            [prey_prey, prey_predator],
            [predator_prey, predator_predator],
        ]
    )


def build_species_diffusion_vector(regime_count, diffusion):
    diffusion = np.asarray(diffusion, dtype=float)
    return np.concatenate(
        (
            np.full(regime_count, diffusion[0], dtype=float),
            np.full(regime_count, diffusion[1], dtype=float),
        )
    )


def build_species_weight_vector(regime_count, weights):
    weights = np.asarray(weights, dtype=float)
    return np.concatenate(
        (
            np.full(regime_count, weights[0], dtype=float),
            np.full(regime_count, weights[1], dtype=float),
        )
    )


def build_species_parameter_vector(regime_count, parameter_values):
    parameter_values = np.atleast_1d(np.asarray(parameter_values, dtype=float))
    if parameter_values.size == 1:
        return float(parameter_values[0])

    return np.concatenate(
        (
            np.full(regime_count, parameter_values[0], dtype=float),
            np.full(regime_count, parameter_values[1], dtype=float),
        )
    )


def build_reaction_term(reaction_rates, regime_count):
    prey_growth = float(reaction_rates["prey_growth"])
    predator_decay = float(reaction_rates["predator_decay"])
    predation_rate = float(reaction_rates["predation_rate"])
    conversion_rate = float(reaction_rates["conversion_rate"])

    def reaction_term(population, time, model):
        activity_mask = model.get_activity_mask(time).astype(float)

        prey = population[:, :regime_count]
        predator = population[:, regime_count:]

        prey_activity = activity_mask[:regime_count][np.newaxis, :]
        predator_activity = activity_mask[regime_count:][np.newaxis, :]

        total_prey_density = np.sum(prey, axis=1, keepdims=True)
        active_predator_density = np.sum(
            predator * predator_activity,
            axis=1,
            keepdims=True,
        )

        prey_source = (
            prey_growth * prey_activity * prey
            - predation_rate * prey * active_predator_density
        )
        predator_source = (
            -predator_decay * predator
            + conversion_rate * predator_activity * predator * total_prey_density
        )

        return np.concatenate((prey_source, predator_source), axis=1)

    return reaction_term


def build_model(config, prey_shares, predator_shares):
    regime_count = len(ACTIVITY_REGIMES)
    lighting_regime = build_lighting_regime(config["t_sunset"], config["dt"])

    def placeholder_initial_condition(x):
        x = np.asarray(x)
        return np.ones((x.size, 2 * regime_count), dtype=float)

    model = DayNightModel1D(
        a_border=0.0,
        b_border=1.0,
        number_of_points=config["number_of_points"],
        total_time=config["number_of_cycles"] * CYCLE_PERIOD,
        dt=config["dt"],
        initial_condition=placeholder_initial_condition,
        coefficient_attraction=build_species_attraction_matrix(
            regime_count,
            config["attraction"],
        ),
        coefficient_diffusion=build_species_diffusion_vector(
            regime_count,
            config["diffusion"],
        ),
        cycle_period=CYCLE_PERIOD,
        number_of_population=2 * regime_count,
        day_start=lighting_regime["day_start"],
        day_end=lighting_regime["day_end"],
        time_input_mode="phase",
        activity_mode="always",
        activity_periods=build_activity_periods(),
        sight_weight=build_species_weight_vector(regime_count, config["weights"]),
        sight_radius=build_species_parameter_vector(
            regime_count,
            config["sight_radius"],
        ),
        smell_radius=build_species_parameter_vector(
            regime_count,
            config["smell_radius"],
        ),
        reaction_term=build_reaction_term(config["reaction_rates"], regime_count),
    )

    weighted_initial_state = build_weighted_initial_state(
        model.x,
        prey_shares,
        predator_shares,
        config,
    )
    model.U[0, :, :] = weighted_initial_state
    model.U_fourier[0, :, :] = np.fft.fft(weighted_initial_state, axis=0)
    model._solution_computed = False
    return model


def compute_overlap_payoffs(model, prey_shares, predator_shares):
    regime_count = len(ACTIVITY_REGIMES)
    prey_weights = counts_to_shares(prey_shares)
    predator_weights = counts_to_shares(predator_shares)
    pairwise_overlap = np.zeros((regime_count, regime_count), dtype=float)

    for prey_index in range(regime_count):
        for predator_index in range(regime_count):
            pairwise_overlap[prey_index, predator_index] = model.get_overlap_energy(
                population_indices=(prey_index, regime_count + predator_index)
            )

    prey_payoffs = pairwise_overlap @ predator_weights
    predator_payoffs = pairwise_overlap.T @ prey_weights
    return pairwise_overlap, prey_payoffs, predator_payoffs


def apply_selection(
    shares,
    payoffs,
    selection_events,
    selection_percentage,
    *,
    maximize,
):
    updated_shares = counts_to_shares(shares)
    payoffs = np.asarray(payoffs, dtype=float)
    finite_indices = [
        index for index in range(updated_shares.size) if np.isfinite(payoffs[index])
    ]
    if maximize:
        loser_order = sorted(finite_indices, key=lambda index: (payoffs[index], index))
        winner_order = sorted(finite_indices, key=lambda index: (-payoffs[index], index))
    else:
        loser_order = sorted(finite_indices, key=lambda index: (-payoffs[index], index))
        winner_order = sorted(finite_indices, key=lambda index: (payoffs[index], index))

    event_records = []
    transfer_fraction = 0.01 * float(selection_percentage)
    minimum_share = 1.0e-14
    minimum_allowed_share = MINIMUM_SHARE_FRACTION
    for event_index in range(selection_events):
        loser_index = next(
            (
                index
                for index in loser_order
                if updated_shares[index] > (minimum_allowed_share + minimum_share)
            ),
            None,
        )
        winner_index = next(
            (
                index
                for index in winner_order
                if index != loser_index
            ),
            None,
        )

        if loser_index is None or winner_index is None:
            break

        available_share = float(updated_shares[loser_index] - minimum_allowed_share)
        transferred_share = min(transfer_fraction, available_share)
        if transferred_share <= minimum_share:
            break

        updated_shares[loser_index] -= transferred_share
        updated_shares[winner_index] += transferred_share
        event_records.append(
            {
                "event": event_index + 1,
                "loser_index": loser_index,
                "winner_index": winner_index,
                "loser_payoff": float(payoffs[loser_index]),
                "winner_payoff": float(payoffs[winner_index]),
                "transferred_share": float(transferred_share),
            }
        )

    return counts_to_shares(updated_shares), event_records


def build_round_payoff_rows(
    round_index,
    species,
    shares,
    initial_masses,
    final_masses,
    payoffs,
):
    rows = []
    shares = counts_to_shares(shares)
    for regime_index, regime in enumerate(ACTIVITY_REGIMES):
        rows.append(
            {
                "round": int(round_index),
                "species": species,
                "regime": regime["code"],
                "share_before": float(shares[regime_index]),
                "percentage_before": float(100.0 * shares[regime_index]),
                "initial_mass": float(initial_masses[regime_index]),
                "final_mass": float(final_masses[regime_index]),
                "mass_change": float(final_masses[regime_index] - initial_masses[regime_index]),
                "payoff": float(payoffs[regime_index])
                if np.isfinite(payoffs[regime_index])
                else "",
            }
        )
    return rows


def build_selection_rows(round_index, species, event_records):
    rows = []
    for record in event_records:
        rows.append(
            {
                "round": int(round_index),
                "species": species,
                "event": int(record["event"]),
                "source_regime": ACTIVITY_REGIMES[record["loser_index"]]["code"],
                "target_regime": ACTIVITY_REGIMES[record["winner_index"]]["code"],
                "source_payoff": float(record["loser_payoff"]),
                "target_payoff": float(record["winner_payoff"]),
                "transferred_share": float(record["transferred_share"]),
                "transferred_percentage": float(100.0 * record["transferred_share"]),
            }
        )
    return rows


def append_distribution_rows(history_rows, generation, species, shares):
    shares = counts_to_shares(shares)
    for regime_index, regime in enumerate(ACTIVITY_REGIMES):
        history_rows.append(
            {
                "generation": int(generation),
                "species": species,
                "regime": regime["code"],
                "share": float(shares[regime_index]),
                "percentage": float(100.0 * shares[regime_index]),
            }
        )


def simulate_round(config, prey_shares, predator_shares):
    model = build_model(config, prey_shares, predator_shares)
    model.solve()
    masses = model.get_mass()

    regime_count = len(ACTIVITY_REGIMES)
    initial_masses = masses[0, :]
    final_masses = masses[-1, :]

    prey_initial_masses = initial_masses[:regime_count]
    predator_initial_masses = initial_masses[regime_count:]
    prey_final_masses = final_masses[:regime_count]
    predator_final_masses = final_masses[regime_count:]

    pairwise_overlap, prey_payoffs, predator_payoffs = compute_overlap_payoffs(
        model,
        prey_shares,
        predator_shares,
    )

    return {
        "pairwise_overlap": pairwise_overlap,
        "prey_initial_masses": prey_initial_masses,
        "predator_initial_masses": predator_initial_masses,
        "prey_final_masses": prey_final_masses,
        "predator_final_masses": predator_final_masses,
        "prey_payoffs": prey_payoffs,
        "predator_payoffs": predator_payoffs,
    }


def write_csv(output_path, fieldnames, rows):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_run_config(output_path, config):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2, sort_keys=True)


def save_share_plot(history_rows, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    generations = sorted({row["generation"] for row in history_rows})
    row_lookup = {
        (row["species"], row["regime"], row["generation"]): row["share"]
        for row in history_rows
    }

    figure, axes = plt.subplots(
        1,
        2,
        figsize=(12.0, 4.8),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )

    colors = plt.get_cmap("tab20")(np.linspace(0.05, 0.85, len(ACTIVITY_REGIMES)))

    for axis, species in zip(axes, ("prey", "predator")):
        share_series = []
        labels = []
        for regime in ACTIVITY_REGIMES:
            regime_code = regime["code"]
            share_series.append(
                [
                row_lookup[(species, regime_code, generation)]
                for generation in generations
                ]
            )
            labels.append(regime_code)

        stacks = axis.stackplot(
            generations,
            *share_series,
            labels=labels,
            colors=colors,
            alpha=0.75,
            linewidth=0.8,
            edgecolor="#1f4aa8",
        )

        axis.set_title(f"{species.capitalize()} circadian shares")
        axis.set_xlabel("Generation")
        axis.grid(alpha=0.22, linewidth=0.6)
        axis.set_xlim(generations[0], generations[-1])

    axes[0].set_ylabel("Population share")
    axes[0].set_ylim(0.0, 1.0)
    axes[1].legend(title="Regime", loc="center left", bbox_to_anchor=(1.02, 0.5))

    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def format_share_summary(shares):
    shares = counts_to_shares(shares)
    return ", ".join(
        f"{regime['code']}={100.0 * share:.1f}%"
        for regime, share in zip(ACTIVITY_REGIMES, shares)
    )


def format_best_summary(payoffs, *, maximize):
    payoffs = np.asarray(payoffs, dtype=float)
    finite_indices = [index for index in range(payoffs.size) if np.isfinite(payoffs[index])]
    if not finite_indices:
        return "none"

    if maximize:
        best_index = max(finite_indices, key=lambda index: (payoffs[index], -index))
        worst_index = min(finite_indices, key=lambda index: (payoffs[index], index))
    else:
        best_index = min(finite_indices, key=lambda index: (payoffs[index], index))
        worst_index = max(finite_indices, key=lambda index: (payoffs[index], -index))

    return (
        f"best={ACTIVITY_REGIMES[best_index]['code']} ({payoffs[best_index]:.3f}), "
        f"worst={ACTIVITY_REGIMES[worst_index]['code']} ({payoffs[worst_index]:.3f})"
    )


def build_output_paths(output_dir):
    output_dir = Path(output_dir)
    return {
        "output_dir": output_dir,
        "round_payoffs": output_dir / ROUND_PAYOFF_OUTPUT_PATH.name,
        "distribution_history": output_dir / DISTRIBUTION_HISTORY_OUTPUT_PATH.name,
        "selection_events": output_dir / SELECTION_EVENT_OUTPUT_PATH.name,
        "run_config": output_dir / RUN_CONFIG_OUTPUT_PATH.name,
        "share_plot": output_dir / SHARE_PLOT_OUTPUT_PATH.name,
    }


def resolve_output_selection(save_outputs=None):
    output_selection = {
        "round_payoffs": True,
        "distribution_history": True,
        "selection_events": True,
        "run_config": True,
        "share_plot": True,
    }
    if save_outputs is None:
        return output_selection

    unknown_keys = set(save_outputs) - set(output_selection)
    if unknown_keys:
        unknown_key_list = ", ".join(sorted(unknown_keys))
        raise ValueError(f"Unknown output selection keys: {unknown_key_list}")

    output_selection.update(
        {key: bool(value) for key, value in save_outputs.items()}
    )
    return output_selection


def run_evolutionary_game(config, output_dir, *, echo=True, save_outputs=None):
    validate_config(config)
    output_paths = build_output_paths(output_dir)
    output_selection = resolve_output_selection(save_outputs)

    regime_count = len(ACTIVITY_REGIMES)
    prey_shares = build_equal_shares(regime_count)
    predator_shares = build_equal_shares(regime_count)

    round_rows = []
    distribution_history_rows = []
    selection_rows = []

    append_distribution_rows(distribution_history_rows, 0, "prey", prey_shares)
    append_distribution_rows(distribution_history_rows, 0, "predator", predator_shares)

    for round_index in range(1, config["rounds"] + 1):
        round_result = simulate_round(config, prey_shares, predator_shares)

        round_rows.extend(
            build_round_payoff_rows(
                round_index,
                "prey",
                prey_shares,
                round_result["prey_initial_masses"],
                round_result["prey_final_masses"],
                round_result["prey_payoffs"],
            )
        )
        round_rows.extend(
            build_round_payoff_rows(
                round_index,
                "predator",
                predator_shares,
                round_result["predator_initial_masses"],
                round_result["predator_final_masses"],
                round_result["predator_payoffs"],
            )
        )

        prey_shares, prey_events = apply_selection(
            prey_shares,
            round_result["prey_payoffs"],
            config["selection_events"],
            config["selection_percentage"],
            maximize=False,
        )
        predator_shares, predator_events = apply_selection(
            predator_shares,
            round_result["predator_payoffs"],
            config["selection_events"],
            config["selection_percentage"],
            maximize=True,
        )

        selection_rows.extend(build_selection_rows(round_index, "prey", prey_events))
        selection_rows.extend(
            build_selection_rows(round_index, "predator", predator_events)
        )

        append_distribution_rows(
            distribution_history_rows,
            round_index,
            "prey",
            prey_shares,
        )
        append_distribution_rows(
            distribution_history_rows,
            round_index,
            "predator",
            predator_shares,
        )

        if echo:
            print(
                f"Round {round_index}: prey {format_best_summary(round_result['prey_payoffs'], maximize=False)}; "
                f"next shares {format_share_summary(prey_shares)}",
                flush=True,
            )
            print(
                f"Round {round_index}: predator {format_best_summary(round_result['predator_payoffs'], maximize=True)}; "
                f"next shares {format_share_summary(predator_shares)}",
                flush=True,
            )

    if output_selection["round_payoffs"]:
        write_csv(
            output_paths["round_payoffs"],
            [
                "round",
                "species",
                "regime",
                "share_before",
                "percentage_before",
                "initial_mass",
                "final_mass",
                "mass_change",
                "payoff",
            ],
            round_rows,
        )
    if output_selection["distribution_history"]:
        write_csv(
            output_paths["distribution_history"],
            ["generation", "species", "regime", "share", "percentage"],
            distribution_history_rows,
        )
    if output_selection["selection_events"]:
        write_csv(
            output_paths["selection_events"],
            [
                "round",
                "species",
                "event",
                "source_regime",
                "target_regime",
                "source_payoff",
                "target_payoff",
                "transferred_share",
                "transferred_percentage",
            ],
            selection_rows,
        )
    if output_selection["run_config"]:
        save_run_config(output_paths["run_config"], config)
    if output_selection["share_plot"]:
        save_share_plot(distribution_history_rows, output_paths["share_plot"])

    if echo:
        if output_selection["round_payoffs"]:
            print(
                f"Saved round payoffs to {output_paths['round_payoffs']}",
                flush=True,
            )
        if output_selection["distribution_history"]:
            print(
                f"Saved distribution history to {output_paths['distribution_history']}",
                flush=True,
            )
        if output_selection["selection_events"]:
            print(
                f"Saved selection events to {output_paths['selection_events']}",
                flush=True,
            )
        if output_selection["run_config"]:
            print(
                f"Saved run config to {output_paths['run_config']}",
                flush=True,
            )
        if output_selection["share_plot"]:
            print(
                f"Saved share plot to {output_paths['share_plot']}",
                flush=True,
            )

    return {
        "final_prey_shares": prey_shares,
        "final_predator_shares": predator_shares,
        "saved_outputs": output_selection,
        **output_paths,
    }


def main():
    args = parse_args()
    config = build_config_from_args(args)
    run_evolutionary_game(config, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())