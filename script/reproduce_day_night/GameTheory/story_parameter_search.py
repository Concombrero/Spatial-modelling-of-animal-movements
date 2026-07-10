import argparse
import itertools
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from script.reproduce_day_night.GameTheory.payoff_matrix import (
    ACTIVITY_REGIMES,
    CYCLE_PERIOD,
    DEFAULT_ATTRACTION,
    DEFAULT_DIFFUSION,
    DEFAULT_INITIAL_CENTERS,
    DEFAULT_INITIAL_WIDTH,
    DEFAULT_REACTION_RATES,
    build_config as build_payoff_config,
    build_lighting_regime,
    run_all_cases,
)
from script.reproduce_day_night.GameTheory.payoff_replicator_analysis import (
    simulate_replicator_dynamics,
)
from script.reproduce_day_night.paths import story_search_output_path
from script.reproduce_day_night.Solver import DayNightModel1D


FRAGMENTED_LABELS = ("P1", "P2", "M1", "M2")
TWILIGHT_LABELS = ("P1", "M1")
MINIMUM_SHARE_FRACTION = 0.01
DEFAULT_ROUNDS = 8
DEFAULT_SELECTION_EVENTS = 2
DEFAULT_SELECTION_PERCENTAGE = 10.0
DEFAULT_PREY_TOTAL_MASS = 1.0
DEFAULT_PREDATOR_TOTAL_MASS = 1.0
DEFAULT_STRATEGY_CODES = tuple(regime["code"] for regime in ACTIVITY_REGIMES)


@dataclass(frozen=True)
class SearchResult:
    story: str
    score: float
    w1: float
    w2: float
    t_sunset: float
    prey_sight_radius: float
    prey_smell_radius: float
    predator_sight_radius: float
    predator_smell_radius: float
    prey_tail_top: str
    predator_tail_top: str
    prey_tail_shares: dict[str, float]
    predator_tail_shares: dict[str, float]
    prey_final_shares: dict[str, float]
    predator_final_shares: dict[str, float]
    predator_diurnal_peak: float
    predator_diurnal_peak_time: float
    prey_nocturnal_switch_time: float | None
    predator_fragmented_switch_time: float | None
    predator_fragmented_label: str


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Search for parameter regimes whose evolutionary trajectories match "
            "ecological story templates. By default this runs the tractable "
            "payoff-matrix replicator; the multi-regime evolutionary game is "
            "still available for one-off probes."
        )
    )
    parser.add_argument(
        "--story",
        choices=("story1", "story2"),
        required=True,
        help="Story template to score against.",
    )
    parser.add_argument(
        "--w1-values",
        nargs="+",
        type=float,
        required=True,
        help="Prey sight-weight values.",
    )
    parser.add_argument(
        "--w2-values",
        nargs="+",
        type=float,
        required=True,
        help="Predator sight-weight values.",
    )
    parser.add_argument(
        "--sunset-values",
        nargs="+",
        type=float,
        required=True,
        help="Daylight durations t_sunset.",
    )
    parser.add_argument(
        "--prey-sight-values",
        nargs="+",
        type=float,
        required=True,
        help="Prey sight-radius values.",
    )
    parser.add_argument(
        "--prey-smell-values",
        nargs="+",
        type=float,
        required=True,
        help="Prey smell-radius values.",
    )
    parser.add_argument(
        "--predator-sight-values",
        nargs="+",
        type=float,
        required=True,
        help="Predator sight-radius values.",
    )
    parser.add_argument(
        "--predator-smell-values",
        nargs="+",
        type=float,
        required=True,
        help="Predator smell-radius values.",
    )
    parser.add_argument(
        "--strategy-codes",
        nargs="+",
        default=list(DEFAULT_STRATEGY_CODES),
        help=(
            "Subset of circadian strategy codes used during the search. Default: "
            + " ".join(DEFAULT_STRATEGY_CODES)
        ),
    )
    parser.add_argument(
        "--simulation-mode",
        choices=("evolutionary-game", "payoff-replicator"),
        default="payoff-replicator",
        help=(
            "Simulation engine used for scoring. Default: payoff-replicator."
        ),
    )
    parser.add_argument(
        "--time-span",
        type=float,
        default=400.0,
        help=(
            "Replicator final time for scoring when using payoff-replicator mode. "
            "Default: 400."
        ),
    )
    parser.add_argument(
        "--time-steps",
        type=int,
        default=8000,
        help=(
            "Replicator sample count for scoring when using payoff-replicator "
            "mode. Default: 8000."
        ),
    )
    parser.add_argument(
        "--number-of-points",
        type=int,
        default=64,
        help="Spatial grid size used during the coarse search. Default: 64.",
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=0.1,
        help="Stored PDE timestep. Default: 0.1.",
    )
    parser.add_argument(
        "--number-of-cycles",
        type=int,
        default=4,
        help="Number of day-night cycles per payoff run. Default: 4.",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=DEFAULT_ROUNDS,
        help=(
            "Number of ecological selection rounds when using evolutionary-game "
            f"mode. Default: {DEFAULT_ROUNDS}."
        ),
    )
    parser.add_argument(
        "--selection-events",
        type=int,
        default=DEFAULT_SELECTION_EVENTS,
        help=(
            "How many selection transfers to apply per round and species in "
            "evolutionary-game mode. "
            f"Default: {DEFAULT_SELECTION_EVENTS}."
        ),
    )
    parser.add_argument(
        "--selection-percentage",
        type=float,
        default=DEFAULT_SELECTION_PERCENTAGE,
        help=(
            "Percentage of the losing regime's current share transferred to the "
            "winner at each selection event in evolutionary-game mode. "
            f"Default: {DEFAULT_SELECTION_PERCENTAGE:g}."
        ),
    )
    parser.add_argument(
        "--prey-total-mass",
        type=float,
        default=DEFAULT_PREY_TOTAL_MASS,
        help=(
            "Total prey mass split across circadian regimes in evolutionary-game "
            f"mode. Default: {DEFAULT_PREY_TOTAL_MASS:g}."
        ),
    )
    parser.add_argument(
        "--predator-total-mass",
        type=float,
        default=DEFAULT_PREDATOR_TOTAL_MASS,
        help=(
            "Total predator mass split across circadian regimes in evolutionary-"
            f"game mode. Default: {DEFAULT_PREDATOR_TOTAL_MASS:g}."
        ),
    )
    parser.add_argument(
        "--observation-window",
        type=float,
        default=2.0,
        help=(
            "Final overlap window used for payoff-replicator mode. Default: 2.0."
        ),
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Parallel workers for each payoff-matrix solve. Default: 4.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="How many top-ranked cases to print and save. Default: 10.",
    )
    parser.add_argument(
        "--tail-fraction",
        type=float,
        default=0.15,
        help="Final-time fraction used to define the converged tail behavior. Default: 0.15.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        help=(
            "Optional path where the ranked results are written as JSON. "
            "Defaults to GameTheory/output/story_search/<story>_ranked_results.json."
        ),
    )
    return parser.parse_args()


def get_label_maps():
    labels = tuple(regime["code"] for regime in ACTIVITY_REGIMES)
    index_by_label = {label: index for index, label in enumerate(labels)}
    return labels, index_by_label


def resolve_activity_regimes(strategy_codes):
    regime_by_code = {regime["code"]: regime for regime in ACTIVITY_REGIMES}
    resolved_regimes = []
    seen_codes = set()
    for strategy_code in strategy_codes:
        normalized_code = str(strategy_code)
        if normalized_code not in regime_by_code:
            raise ValueError(f"Unknown strategy code: {normalized_code}")
        if normalized_code in seen_codes:
            continue
        resolved_regimes.append(regime_by_code[normalized_code])
        seen_codes.add(normalized_code)
    if not resolved_regimes:
        raise ValueError("At least one strategy code must be selected.")
    return tuple(resolved_regimes)


def build_history_maps(labels, prey_history, predator_history):
    canonical_codes = tuple(regime["code"] for regime in ACTIVITY_REGIMES)
    zero_template = np.zeros(prey_history.shape[1], dtype=float)
    prey_map = {label: zero_template.copy() for label in canonical_codes}
    predator_map = {label: zero_template.copy() for label in canonical_codes}
    for index, label in enumerate(labels):
        prey_map[label] = np.asarray(prey_history[index], dtype=float)
        predator_map[label] = np.asarray(predator_history[index], dtype=float)
    return prey_map, predator_map


def tail_mean(history_map, tail_count):
    return {
        label: float(np.mean(values[-tail_count:]))
        for label, values in history_map.items()
    }


def final_shares(history_map):
    return {label: float(values[-1]) for label, values in history_map.items()}


def top_label(share_map):
    return max(share_map, key=share_map.__getitem__)


def last_switch_time(history, labels, target_label, time_grid):
    if target_label not in labels:
        return None
    target_index = labels.index(target_label)
    top_indices = np.argmax(history, axis=0)
    switch_indices = np.flatnonzero(
        (top_indices[1:] == target_index) & (top_indices[:-1] != target_index)
    )
    if switch_indices.size == 0:
        if int(top_indices[-1]) == target_index:
            return float(time_grid[0])
        return None
    return float(time_grid[int(switch_indices[-1] + 1)])


def safe_time_order(*times):
    filtered_times = [time for time in times if time is not None]
    if len(filtered_times) != len(times):
        return False
    return all(first < second for first, second in zip(filtered_times, filtered_times[1:]))


def score_story1(
    time_grid,
    labels,
    prey_history,
    predator_history,
    tail_fraction,
):
    tail_count = max(2, int(round(tail_fraction * len(time_grid))))
    prey_map, predator_map = build_history_maps(labels, prey_history, predator_history)
    prey_tail = tail_mean(prey_map, tail_count)
    predator_tail = tail_mean(predator_map, tail_count)
    prey_final = final_shares(prey_map)
    predator_final = final_shares(predator_map)

    predator_d_values = predator_map["D"]
    predator_diurnal_peak_index = int(np.argmax(predator_d_values))
    predator_diurnal_peak = float(predator_d_values[predator_diurnal_peak_index])
    predator_diurnal_peak_time = float(time_grid[predator_diurnal_peak_index])

    predator_fragmented_label = max(
        FRAGMENTED_LABELS,
        key=lambda label: predator_tail[label],
    )
    prey_nocturnal_switch_time = last_switch_time(
        prey_history,
        labels,
        "N",
        time_grid,
    )
    predator_fragmented_switch_time = last_switch_time(
        predator_history,
        labels,
        predator_fragmented_label,
        time_grid,
    )

    prey_tail_top = top_label(prey_tail)
    predator_tail_top = top_label(predator_tail)
    prey_tail_nocturnal = prey_tail["N"]
    prey_tail_diurnal = prey_tail["D"]
    predator_tail_diurnal = predator_tail["D"]
    predator_tail_fragmented = predator_tail[predator_fragmented_label]

    ordering_bonus = 1.0 if safe_time_order(
        predator_diurnal_peak_time,
        prey_nocturnal_switch_time,
        predator_fragmented_switch_time,
    ) else -1.5
    score = (
        4.0 * predator_diurnal_peak
        + 5.0 * prey_tail_nocturnal
        + 4.0 * predator_tail_fragmented
        - 4.0 * prey_tail_diurnal
        - 4.0 * predator_tail_diurnal
        + ordering_bonus
    )
    if prey_tail_top != "N":
        score -= 2.5
    if predator_tail_top == "D":
        score -= 2.5
    if predator_diurnal_peak < 0.30:
        score -= 2.0
    if prey_tail_nocturnal < 0.20:
        score -= 2.0
    if predator_tail_fragmented < 0.20:
        score -= 2.0

    return SearchResult(
        story="story1",
        score=float(score),
        w1=0.0,
        w2=0.0,
        t_sunset=0.0,
        prey_sight_radius=0.0,
        prey_smell_radius=0.0,
        predator_sight_radius=0.0,
        predator_smell_radius=0.0,
        prey_tail_top=prey_tail_top,
        predator_tail_top=predator_tail_top,
        prey_tail_shares=prey_tail,
        predator_tail_shares=predator_tail,
        prey_final_shares=prey_final,
        predator_final_shares=predator_final,
        predator_diurnal_peak=predator_diurnal_peak,
        predator_diurnal_peak_time=predator_diurnal_peak_time,
        prey_nocturnal_switch_time=prey_nocturnal_switch_time,
        predator_fragmented_switch_time=predator_fragmented_switch_time,
        predator_fragmented_label=predator_fragmented_label,
    )


def score_story2(
    time_grid,
    labels,
    prey_history,
    predator_history,
    tail_fraction,
):
    tail_count = max(2, int(round(tail_fraction * len(time_grid))))
    prey_map, predator_map = build_history_maps(labels, prey_history, predator_history)
    prey_tail = tail_mean(prey_map, tail_count)
    predator_tail = tail_mean(predator_map, tail_count)
    prey_final = final_shares(prey_map)
    predator_final = final_shares(predator_map)

    predator_d_values = predator_map["D"]
    predator_diurnal_peak_index = int(np.argmax(predator_d_values))
    predator_diurnal_peak = float(predator_d_values[predator_diurnal_peak_index])
    predator_diurnal_peak_time = float(time_grid[predator_diurnal_peak_index])

    twilight_label = max(TWILIGHT_LABELS, key=lambda label: prey_tail[label])
    predator_fragmented_label = max(
        FRAGMENTED_LABELS,
        key=lambda label: predator_tail[label],
    )
    prey_tail_top = top_label(prey_tail)
    predator_tail_top = top_label(predator_tail)
    prey_diurnal_nocturnal = prey_tail["D"] + prey_tail["N"]
    predator_diurnal_nocturnal = predator_tail["D"] + predator_tail["N"]
    twilight_refuge_strength = prey_tail[twilight_label]
    predator_fragmented_strength = predator_tail[predator_fragmented_label]

    score = (
        6.0 * twilight_refuge_strength
        + 3.0 * predator_fragmented_strength
        - 5.0 * prey_diurnal_nocturnal
        - 3.0 * predator_diurnal_nocturnal
    )
    if prey_tail_top not in TWILIGHT_LABELS:
        score -= 2.5
    if prey_diurnal_nocturnal > 0.20:
        score -= 2.0
    if predator_diurnal_nocturnal > 0.20:
        score -= 1.5

    return SearchResult(
        story="story2",
        score=float(score),
        w1=0.0,
        w2=0.0,
        t_sunset=0.0,
        prey_sight_radius=0.0,
        prey_smell_radius=0.0,
        predator_sight_radius=0.0,
        predator_smell_radius=0.0,
        prey_tail_top=prey_tail_top,
        predator_tail_top=predator_tail_top,
        prey_tail_shares=prey_tail,
        predator_tail_shares=predator_tail,
        prey_final_shares=prey_final,
        predator_final_shares=predator_final,
        predator_diurnal_peak=predator_diurnal_peak,
        predator_diurnal_peak_time=predator_diurnal_peak_time,
        prey_nocturnal_switch_time=last_switch_time(prey_history, labels, "N", time_grid),
        predator_fragmented_switch_time=last_switch_time(
            predator_history,
            labels,
            predator_fragmented_label,
            time_grid,
        ),
        predator_fragmented_label=predator_fragmented_label,
    )


def attach_parameters(result, case):
    return SearchResult(
        story=result.story,
        score=result.score,
        w1=float(case["w1"]),
        w2=float(case["w2"]),
        t_sunset=float(case["t_sunset"]),
        prey_sight_radius=float(case["prey_sight_radius"]),
        prey_smell_radius=float(case["prey_smell_radius"]),
        predator_sight_radius=float(case["predator_sight_radius"]),
        predator_smell_radius=float(case["predator_smell_radius"]),
        prey_tail_top=result.prey_tail_top,
        predator_tail_top=result.predator_tail_top,
        prey_tail_shares=result.prey_tail_shares,
        predator_tail_shares=result.predator_tail_shares,
        prey_final_shares=result.prey_final_shares,
        predator_final_shares=result.predator_final_shares,
        predator_diurnal_peak=result.predator_diurnal_peak,
        predator_diurnal_peak_time=result.predator_diurnal_peak_time,
        prey_nocturnal_switch_time=result.prey_nocturnal_switch_time,
        predator_fragmented_switch_time=result.predator_fragmented_switch_time,
        predator_fragmented_label=result.predator_fragmented_label,
    )


def build_cases(args):
    parameter_product = itertools.product(
        args.w1_values,
        args.w2_values,
        args.sunset_values,
        args.prey_sight_values,
        args.prey_smell_values,
        args.predator_sight_values,
        args.predator_smell_values,
    )
    for (
        w1,
        w2,
        t_sunset,
        prey_sight_radius,
        prey_smell_radius,
        predator_sight_radius,
        predator_smell_radius,
    ) in parameter_product:
        yield {
            "w1": float(w1),
            "w2": float(w2),
            "t_sunset": float(t_sunset),
            "prey_sight_radius": float(prey_sight_radius),
            "prey_smell_radius": float(prey_smell_radius),
            "predator_sight_radius": float(predator_sight_radius),
            "predator_smell_radius": float(predator_smell_radius),
        }


def counts_to_shares(shares):
    shares = np.asarray(shares, dtype=float)
    total_share = float(np.sum(shares))
    if total_share <= 0.0:
        raise ValueError("At least one regime share must be positive.")
    return shares / total_share


def build_equal_shares(regime_count):
    return np.full(regime_count, 1.0 / float(regime_count), dtype=float)


def build_periodic_profile(x, center, width):
    x = np.asarray(x, dtype=float)
    dx = float(x[1] - x[0])
    domain_length = float((x[-1] - x[0]) + dx)
    wrapped_distance = ((x - float(center) + 0.5 * domain_length) % domain_length) - 0.5 * domain_length
    profile = np.exp(-0.5 * (wrapped_distance / float(width)) ** 2)
    profile_mass = dx * float(np.sum(profile))
    if profile_mass <= 0.0:
        raise ValueError("Initial profile mass must be positive.")
    return profile / profile_mass


def build_weighted_initial_state(
    x,
    prey_shares,
    predator_shares,
    *,
    regime_count,
    initial_centers,
    initial_width,
    prey_total_mass,
    predator_total_mass,
):
    prey_profile = build_periodic_profile(x, initial_centers[0], initial_width)
    predator_profile = build_periodic_profile(x, initial_centers[1], initial_width)

    prey_masses = float(prey_total_mass) * counts_to_shares(prey_shares)
    predator_masses = float(predator_total_mass) * counts_to_shares(predator_shares)

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


def build_activity_periods_for_regimes(activity_regimes):
    prey_periods = [list(regime["periods"]) for regime in activity_regimes]
    predator_periods = [list(regime["periods"]) for regime in activity_regimes]
    return prey_periods + predator_periods


def build_species_attraction_matrix(regime_count, attraction):
    attraction = np.asarray(attraction, dtype=float)
    prey_prey = np.full((regime_count, regime_count), attraction[0, 0], dtype=float)
    prey_predator = np.full((regime_count, regime_count), attraction[0, 1], dtype=float)
    predator_prey = np.full((regime_count, regime_count), attraction[1, 0], dtype=float)
    predator_predator = np.full((regime_count, regime_count), attraction[1, 1], dtype=float)
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
    if parameter_values.size != 2:
        raise ValueError("Species parameter values must be scalar or a prey/predator pair.")
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


def build_evolutionary_game_model(case, args, prey_shares, predator_shares, activity_regimes):
    regime_count = len(activity_regimes)
    lighting_regime = build_lighting_regime(case["t_sunset"], args.dt)

    def placeholder_initial_condition(x):
        x = np.asarray(x, dtype=float)
        return np.ones((x.size, 2 * regime_count), dtype=float)

    model = DayNightModel1D(
        a_border=0.0,
        b_border=1.0,
        number_of_points=args.number_of_points,
        total_time=args.number_of_cycles * CYCLE_PERIOD,
        dt=args.dt,
        initial_condition=placeholder_initial_condition,
        coefficient_attraction=build_species_attraction_matrix(
            regime_count,
            DEFAULT_ATTRACTION,
        ),
        coefficient_diffusion=build_species_diffusion_vector(
            regime_count,
            DEFAULT_DIFFUSION,
        ),
        cycle_period=CYCLE_PERIOD,
        number_of_population=2 * regime_count,
        day_start=lighting_regime["day_start"],
        day_end=lighting_regime["day_end"],
        time_input_mode="phase",
        activity_mode="always",
        activity_periods=build_activity_periods_for_regimes(activity_regimes),
        sight_weight=build_species_weight_vector(
            regime_count,
            (case["w1"], case["w2"]),
        ),
        sight_radius=build_species_parameter_vector(
            regime_count,
            (case["prey_sight_radius"], case["predator_sight_radius"]),
        ),
        smell_radius=build_species_parameter_vector(
            regime_count,
            (case["prey_smell_radius"], case["predator_smell_radius"]),
        ),
        reaction_term=build_reaction_term(DEFAULT_REACTION_RATES, regime_count),
    )
    reset_model_state(model, prey_shares, predator_shares, args)
    return model


def reset_model_state(model, prey_shares, predator_shares, args):
    regime_count = prey_shares.size
    initial_state = build_weighted_initial_state(
        model.x,
        prey_shares,
        predator_shares,
        regime_count=regime_count,
        initial_centers=DEFAULT_INITIAL_CENTERS,
        initial_width=DEFAULT_INITIAL_WIDTH,
        prey_total_mass=args.prey_total_mass,
        predator_total_mass=args.predator_total_mass,
    )
    model.U[0, :, :] = initial_state
    model.U_fourier[0, :, :] = np.fft.fft(initial_state, axis=0)
    model._solution_computed = False


def compute_overlap_payoffs(model, prey_shares, predator_shares, activity_regimes):
    regime_count = len(activity_regimes)
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
    return prey_payoffs, predator_payoffs


def apply_selection(shares, payoffs, selection_events, selection_percentage, *, maximize):
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

    transfer_ratio = 0.01 * float(selection_percentage)
    minimum_allowed_share = MINIMUM_SHARE_FRACTION
    tolerance = 1.0e-14

    for _ in range(int(selection_events)):
        loser_index = next(
            (
                index
                for index in loser_order
                if updated_shares[index] > (minimum_allowed_share + tolerance)
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
        transferred_share = min(
            transfer_ratio * float(updated_shares[loser_index]),
            available_share,
        )
        if transferred_share <= tolerance:
            break

        updated_shares[loser_index] -= transferred_share
        updated_shares[winner_index] += transferred_share

    return counts_to_shares(updated_shares)


def simulate_evolutionary_game_case(args, case, activity_regimes):
    regime_count = len(activity_regimes)
    prey_shares = build_equal_shares(regime_count)
    predator_shares = build_equal_shares(regime_count)
    prey_history = [prey_shares.copy()]
    predator_history = [predator_shares.copy()]

    model = build_evolutionary_game_model(
        case,
        args,
        prey_shares,
        predator_shares,
        activity_regimes,
    )

    for _ in range(args.rounds):
        reset_model_state(model, prey_shares, predator_shares, args)
        model.solve()
        prey_payoffs, predator_payoffs = compute_overlap_payoffs(
            model,
            prey_shares,
            predator_shares,
            activity_regimes,
        )
        prey_shares = apply_selection(
            prey_shares,
            prey_payoffs,
            args.selection_events,
            args.selection_percentage,
            maximize=False,
        )
        predator_shares = apply_selection(
            predator_shares,
            predator_payoffs,
            args.selection_events,
            args.selection_percentage,
            maximize=True,
        )
        prey_history.append(prey_shares.copy())
        predator_history.append(predator_shares.copy())

    time_grid = np.arange(args.rounds + 1, dtype=float)
    return (
        time_grid,
        np.asarray(prey_history, dtype=float).T,
        np.asarray(predator_history, dtype=float).T,
    )


def evaluate_payoff_replicator_case(args, case, labels, activity_regimes):
    config = build_payoff_config(
        t_sunset=case["t_sunset"],
        weights=(case["w1"], case["w2"]),
        sight_radius=(case["prey_sight_radius"], case["predator_sight_radius"]),
        smell_radius=(case["prey_smell_radius"], case["predator_smell_radius"]),
        number_of_points=args.number_of_points,
        dt=args.dt,
        number_of_cycles=args.number_of_cycles,
        observation_window=args.observation_window,
        initial_centers=DEFAULT_INITIAL_CENTERS,
        initial_width=DEFAULT_INITIAL_WIDTH,
        diffusion=DEFAULT_DIFFUSION,
        attraction=DEFAULT_ATTRACTION,
        reaction_rates=DEFAULT_REACTION_RATES,
    )
    payoff_result = run_all_cases(
        activity_regimes,
        config,
        args.max_workers,
        echo=False,
    )
    if isinstance(payoff_result, dict):
        time_grid, prey_history, predator_history = simulate_replicator_dynamics(
            payoff_result["prey"],
            args.time_span,
            args.time_steps,
            payoff_result["predator"],
        )
    else:
        time_grid, prey_history, predator_history = simulate_replicator_dynamics(
            payoff_result,
            args.time_span,
            args.time_steps,
        )

    if args.story == "story1":
        base_result = score_story1(
            time_grid,
            labels,
            prey_history,
            predator_history,
            args.tail_fraction,
        )
    else:
        base_result = score_story2(
            time_grid,
            labels,
            prey_history,
            predator_history,
            args.tail_fraction,
        )

    return attach_parameters(base_result, case)


def evaluate_case(args, case, labels):
    activity_regimes = resolve_activity_regimes(args.strategy_codes)
    if args.simulation_mode == "payoff-replicator":
        return evaluate_payoff_replicator_case(args, case, labels, activity_regimes)

    time_grid, prey_history, predator_history = simulate_evolutionary_game_case(
        args,
        case,
        activity_regimes,
    )
    if args.story == "story1":
        base_result = score_story1(
            time_grid,
            labels,
            prey_history,
            predator_history,
            args.tail_fraction,
        )
    else:
        base_result = score_story2(
            time_grid,
            labels,
            prey_history,
            predator_history,
            args.tail_fraction,
        )
    return attach_parameters(base_result, case)


def print_result_summary(rank, result):
    prey_focus = result.prey_tail_shares.get("N", 0.0) if result.story == "story1" else max(
        result.prey_tail_shares.get("P1", 0.0),
        result.prey_tail_shares.get("M1", 0.0),
    )
    print(
        f"{rank:2d}. score={result.score: .4f} "
        f"w1={result.w1:.2f} w2={result.w2:.2f} t_sunset={result.t_sunset:.2f} "
        f"Rs=({result.prey_sight_radius:.2f},{result.predator_sight_radius:.2f}) "
        f"Rm=({result.prey_smell_radius:.2f},{result.predator_smell_radius:.2f}) "
        f"prey_tail={result.prey_tail_top} predator_tail={result.predator_tail_top} "
        f"focus={prey_focus:.3f} pred_D_peak={result.predator_diurnal_peak:.3f} "
        f"pred_frag={result.predator_fragmented_label}:{result.predator_tail_shares[result.predator_fragmented_label]:.3f}"
    )


def main():
    args = parse_args()
    if args.output_json is None:
        args.output_json = story_search_output_path(
            f"{args.story}_ranked_results.json"
        )
    activity_regimes = resolve_activity_regimes(args.strategy_codes)
    labels = tuple(regime["code"] for regime in activity_regimes)
    cases = list(build_cases(args))
    results = []
    total_case_count = len(cases)

    for case_index, case in enumerate(cases, start=1):
        print(
            f"[{case_index}/{total_case_count}] "
            f"w1={case['w1']:.2f}, w2={case['w2']:.2f}, t_sunset={case['t_sunset']:.2f}, "
            f"prey_r=({case['prey_sight_radius']:.2f},{case['prey_smell_radius']:.2f}), "
            f"pred_r=({case['predator_sight_radius']:.2f},{case['predator_smell_radius']:.2f}), "
            f"mode={args.simulation_mode}",
            flush=True,
        )
        result = evaluate_case(args, case, labels)
        results.append(result)
        print(
            f"    score={result.score:.4f}, prey_tail={result.prey_tail_top}, "
            f"pred_tail={result.predator_tail_top}, pred_D_peak={result.predator_diurnal_peak:.3f}",
            flush=True,
        )

    ranked_results = sorted(results, key=lambda result: result.score, reverse=True)
    print()
    print(f"Top {min(args.top_k, len(ranked_results))} {args.story} candidates:")
    for rank, result in enumerate(ranked_results[: args.top_k], start=1):
        print_result_summary(rank, result)

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with args.output_json.open("w", encoding="utf-8") as handle:
            json.dump([asdict(result) for result in ranked_results], handle, indent=2)
        print()
        print(f"Saved ranked results to {args.output_json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())