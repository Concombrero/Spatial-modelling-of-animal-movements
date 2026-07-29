import argparse
import csv
import json
from pathlib import Path
import shlex
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch
import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from script.reproduce_day_night.GameTheory.payoff_matrix import (
    DEFAULT_ATTRACTION,
    DEFAULT_DIFFUSION,
    DEFAULT_INITIAL_CENTERS,
    DEFAULT_INITIAL_WIDTH,
    DEFAULT_REACTION_RATES,
    DEFAULT_SIGHT_RADIUS,
    DEFAULT_SMELL_RADIUS,
    DEFAULT_T_SUNSET,
    MAX_WORKERS,
    NET_GROWTH_PAYOFF_MODE,
    OVERLAP_PAYOFF_MODE,
    PAYOFF_MODE_CHOICES,
    build_config_from_run_config_payload,
    build_output_paths,
    build_config,
    prepare_output_directory_for_run,
    resolve_activity_regimes,
    resolve_population_parameter_pair,
    run_all_cases,
    save_payoff_csv,
    save_payoff_heatmap,
)
from script.reproduce_day_night.GameTheory.payoff_replicator_analysis import (
    MissingDependencyError,
    compute_nash_equilibria,
    require_nashpy,
)
from script.reproduce_day_night.paths import ensure_directory, game_theory_output_path
from script.reproduce_day_night.shared_config import (
    ACTIVITY_COLORS,
    ACTIVITY_LABELS,
    PLOT_STYLE,
    TWO_POPULATION_SIMULATION_CONFIG,
    apply_plot_typography,
)


apply_plot_typography()


DEFAULT_OUTPUT_DIRECTORY = game_theory_output_path("weight_nash_heatmap")
DEFAULT_SUMMARY_FILENAME = "nash_weight_summary.csv"
DEFAULT_DETAILS_FILENAME = "nash_weight_details.json"
DEFAULT_COMPONENTS_FIGURE_FILENAME = "nash_consensus_components.png"
DEFAULT_DIAGNOSTICS_FIGURE_FILENAME = "nash_equilibrium_diagnostics.png"
DEFAULT_CONFIG_FILENAME = "run_config.json"
TWO_POPULATION_BASE_CONFIG = TWO_POPULATION_SIMULATION_CONFIG["base"]
TWO_POPULATION_ANALYSIS_CONFIG = TWO_POPULATION_SIMULATION_CONFIG["analysis"]
DEFAULT_NUMBER_OF_POINTS = TWO_POPULATION_BASE_CONFIG["number_of_points"]
DEFAULT_DT = TWO_POPULATION_BASE_CONFIG["dt"]
DEFAULT_NUMBER_OF_CYCLES = TWO_POPULATION_BASE_CONFIG["number_of_cycles"]
DEFAULT_OBSERVATION_WINDOW = TWO_POPULATION_BASE_CONFIG["observation_window"]
DEFAULT_WEIGHT_SWEEP_PAYOFF_MODE = TWO_POPULATION_ANALYSIS_CONFIG[
    "weight_sweep_payoff_mode"
]
DEFAULT_WEIGHT_VALUES = TWO_POPULATION_ANALYSIS_CONFIG["weight_sweep_values"]
WEIGHT_RUNS_DIRECTORY_NAME = "weight_runs"
REMAINING_PIPELINE_SCRIPT_FILENAME = "run_remaining_pipeline.sh"
CONSENSUS_LABEL = "Multi"
CONSENSUS_COLOUR = PLOT_STYLE["consensus_color"]
PROBABILITY_ATOL = 1.0e-10


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Sweep the sight-weight plane (w1, w2), compute the payoff matrix for "
            "each pair, compute the mixed Nash equilibrium set, and save article-"
            "oriented summary figures for the resulting equilibrium structure."
        )
    )
    parser.add_argument(
        "--run-config",
        type=Path,
        help=(
            "Load the saved sweep definition from an existing run_config.json "
            "file. When provided, the sweep grid and simulation parameters are "
            "read from that file while --output-dir and --max-workers still apply."
        ),
    )
    parser.add_argument(
        "--w1-values",
        nargs="+",
        type=float,
        help="Prey sight-weight grid. Default: 0 0.1 ... 1.",
    )
    parser.add_argument(
        "--w2-values",
        nargs="+",
        type=float,
        help="Predator sight-weight grid. Default: 0 0.1 ... 1.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Directory where the CSV summary, JSON details, figures, and run "
            "config are saved. Default: the directory containing --run-config "
            f"when provided, otherwise {DEFAULT_OUTPUT_DIRECTORY}."
        ),
    )
    parser.add_argument(
        "--strategy-codes",
        type=str,
        help=(
            "Optional comma-separated subset of activity codes, for example "
            "D,N,P1,M1. Default: all six codes."
        ),
    )
    parser.add_argument(
        "--payoff-mode",
        choices=PAYOFF_MODE_CHOICES,
        default=DEFAULT_WEIGHT_SWEEP_PAYOFF_MODE,
        help=(
            "Payoff functional used to build each matrix. Default: "
            f"{DEFAULT_WEIGHT_SWEEP_PAYOFF_MODE}."
        ),
    )
    parser.add_argument(
        "--t-sunset",
        type=float,
        default=DEFAULT_T_SUNSET,
        help="Daylight proportion t_sunset in [0, 1]. Default: 0.5.",
    )
    parser.add_argument(
        "--sight-radius",
        type=float,
        default=None,
        help=(
            "Shared sight-radius shorthand applied to prey and predator unless "
            "overridden by a population-specific flag."
        ),
    )
    parser.add_argument(
        "--prey-sight-radius",
        type=float,
        default=None,
        help="Sight radius used by the prey.",
    )
    parser.add_argument(
        "--predator-sight-radius",
        type=float,
        default=None,
        help="Sight radius used by the predator.",
    )
    parser.add_argument(
        "--smell-radius",
        type=float,
        default=None,
        help=(
            "Shared smell-radius shorthand applied to prey and predator unless "
            "overridden by a population-specific flag."
        ),
    )
    parser.add_argument(
        "--prey-smell-radius",
        type=float,
        default=None,
        help="Smell radius used by the prey.",
    )
    parser.add_argument(
        "--predator-smell-radius",
        type=float,
        default=None,
        help="Smell radius used by the predator.",
    )
    parser.add_argument(
        "--number-of-points",
        type=int,
        default=DEFAULT_NUMBER_OF_POINTS,
        help=(
            "Number of spatial grid points. Default: "
            f"{DEFAULT_NUMBER_OF_POINTS}."
        ),
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=DEFAULT_DT,
        help=f"Stored output timestep. Default: {DEFAULT_DT:g}.",
    )
    parser.add_argument(
        "--number-of-cycles",
        type=int,
        default=DEFAULT_NUMBER_OF_CYCLES,
        help=(
            "Number of daily cycles to simulate. Default: "
            f"{DEFAULT_NUMBER_OF_CYCLES}."
        ),
    )
    parser.add_argument(
        "--observation-window",
        type=float,
        default=DEFAULT_OBSERVATION_WINDOW,
        help=(
            "Final-time window used to evaluate the payoff functional. "
            f"Default: {DEFAULT_OBSERVATION_WINDOW:g}."
        ),
    )
    parser.add_argument(
        "--prey-growth",
        type=float,
        default=DEFAULT_REACTION_RATES["prey_growth"],
        help="Lotka-Volterra prey growth rate.",
    )
    parser.add_argument(
        "--predator-decay",
        type=float,
        default=DEFAULT_REACTION_RATES["predator_decay"],
        help="Lotka-Volterra predator decay rate.",
    )
    parser.add_argument(
        "--predation-rate",
        type=float,
        default=DEFAULT_REACTION_RATES["predation_rate"],
        help="Predation rate.",
    )
    parser.add_argument(
        "--conversion-rate",
        type=float,
        default=DEFAULT_REACTION_RATES["conversion_rate"],
        help="Predator conversion rate.",
    )
    parser.add_argument(
        "--chi11",
        type=float,
        default=DEFAULT_ATTRACTION[0][0],
        help="Prey self-attraction coefficient.",
    )
    parser.add_argument(
        "--chi12",
        type=float,
        default=DEFAULT_ATTRACTION[0][1],
        help="Prey response to predator.",
    )
    parser.add_argument(
        "--chi21",
        type=float,
        default=DEFAULT_ATTRACTION[1][0],
        help="Predator response to prey.",
    )
    parser.add_argument(
        "--chi22",
        type=float,
        default=DEFAULT_ATTRACTION[1][1],
        help="Predator self-attraction coefficient.",
    )
    parser.add_argument(
        "--diffusion",
        nargs=2,
        type=float,
        metavar=("D1", "D2"),
        default=list(DEFAULT_DIFFUSION),
        help="Diffusion coefficients for prey and predator.",
    )
    parser.add_argument(
        "--initial-centers",
        nargs=2,
        type=float,
        metavar=("X1", "X2"),
        default=list(DEFAULT_INITIAL_CENTERS),
        help="Initial Gaussian centers for prey and predator.",
    )
    parser.add_argument(
        "--initial-width",
        type=float,
        default=DEFAULT_INITIAL_WIDTH,
        help="Shared Gaussian width for both populations.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=MAX_WORKERS,
        help="Parallel workers used for each payoff-matrix solve.",
    )
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help=(
            "Regenerate the summary figures from the existing "
            f"{DEFAULT_SUMMARY_FILENAME} in --output-dir without recomputing "
            "the weight sweep."
        ),
    )
    return parser.parse_args()


def resolve_weight_values(raw_values, *, option_name):
    if raw_values is None:
        return DEFAULT_WEIGHT_VALUES

    resolved_values = []
    seen_values = set()
    for raw_value in raw_values:
        value = round(float(raw_value), 10)
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{option_name} values must lie in [0, 1].")
        if value in seen_values:
            continue
        seen_values.add(value)
        resolved_values.append(value)

    if not resolved_values:
        raise ValueError(f"{option_name} must include at least one value.")

    return tuple(resolved_values)


def validate_args(args):
    if args.max_workers < 1:
        raise ValueError("--max-workers must be at least 1.")


def build_base_config(args):
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
        weights=(0.0, 0.0),
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
        attraction=((args.chi11, args.chi12), (args.chi21, args.chi22)),
        reaction_rates={
            "prey_growth": args.prey_growth,
            "predator_decay": args.predator_decay,
            "predation_rate": args.predation_rate,
            "conversion_rate": args.conversion_rate,
        },
    )


def load_saved_weight_nash_run_config(config_path):
    config_path = Path(config_path).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"run_config.json not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {config_path}.")

    if "payoff_config" not in payload:
        raise ValueError(
            f"{config_path} looks like a payoff_matrix run_config.json. Replay it "
            "with script.reproduce_day_night.GameTheory.payoff_matrix or "
            "run_payoff_pipeline.sh instead."
        )

    missing_keys = [
        key for key in ("w1_values", "w2_values", "activity_codes", "payoff_config")
        if key not in payload
    ]
    if missing_keys:
        missing_text = ", ".join(missing_keys)
        raise ValueError(
            f"{config_path} is missing required run_config keys: {missing_text}."
        )

    activity_codes = payload["activity_codes"]
    if isinstance(activity_codes, str):
        activity_codes_text = activity_codes
    elif isinstance(activity_codes, (list, tuple)):
        activity_codes_text = ",".join(str(code) for code in activity_codes)
    else:
        raise ValueError(
            f"Expected 'activity_codes' in {config_path} to be a string or list."
        )

    return {
        "w1_values": resolve_weight_values(
            payload["w1_values"],
            option_name=f"{config_path.name}:w1_values",
        ),
        "w2_values": resolve_weight_values(
            payload["w2_values"],
            option_name=f"{config_path.name}:w2_values",
        ),
        "activity_regimes": resolve_activity_regimes(activity_codes_text),
        "base_config": build_config_from_run_config_payload(
            payload["payoff_config"],
            source_path=config_path,
        ),
    }


def resolve_weight_sweep_configuration(args):
    run_config_path = args.run_config
    if run_config_path is None and args.plot_only:
        inferred_run_config_path = resolve_output_dir(args) / DEFAULT_CONFIG_FILENAME
        if inferred_run_config_path.is_file():
            run_config_path = inferred_run_config_path

    if run_config_path is None:
        return {
            "w1_values": resolve_weight_values(args.w1_values, option_name="--w1-values"),
            "w2_values": resolve_weight_values(args.w2_values, option_name="--w2-values"),
            "activity_regimes": resolve_activity_regimes(args.strategy_codes),
            "base_config": build_base_config(args),
        }

    return load_saved_weight_nash_run_config(run_config_path)


def resolve_output_dir(args):
    if args.output_dir is not None:
        return args.output_dir.expanduser().resolve()

    if args.run_config is not None:
        return args.run_config.expanduser().resolve().parent

    return DEFAULT_OUTPUT_DIRECTORY.expanduser().resolve()


def format_weight_slug(value):
    return f"{float(value):.10g}".replace("-", "m").replace(".", "p")


def build_weight_run_directory(output_dir, w1, w2):
    return (
        Path(output_dir)
        / WEIGHT_RUNS_DIRECTORY_NAME
        / f"w1_{format_weight_slug(w1)}__w2_{format_weight_slug(w2)}"
    )


def save_pair_payoff_outputs(output_paths, activity_regimes, config, payoff_matrices):
    payoff_mode = config.get("payoff_mode", NET_GROWTH_PAYOFF_MODE)
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
        return

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


def build_remaining_pipeline_command_lines(run_directory, config, max_workers):
    repo_root = Path(__file__).resolve().parents[3]
    pipeline_script = Path(__file__).resolve().with_name("run_payoff_pipeline.sh")
    weights = tuple(float(value) for value in config["weights"])
    sight_radius = tuple(float(value) for value in np.atleast_1d(config["sight_radius"]))
    smell_radius = tuple(float(value) for value in np.atleast_1d(config["smell_radius"]))
    diffusion = tuple(float(value) for value in config["diffusion"])
    attraction = tuple(tuple(float(value) for value in row) for row in config["attraction"])
    initial_centers = tuple(float(value) for value in config["initial_centers"])
    reaction_rates = config["reaction_rates"]
    strategy_codes = ",".join(config.get("strategy_codes", ()))

    command_lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        'OUTPUT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)',
        f"REPO_ROOT={shlex.quote(str(repo_root))}",
        f"PIPELINE_SCRIPT={shlex.quote(str(pipeline_script))}",
        "",
        "cd -- \"$REPO_ROOT\"",
        "",
        "# This reuses the saved case_payoffs.csv in this folder when the payoff",
        "# configuration matches, so finished payoff cases are not recomputed.",
        "bash \"$PIPELINE_SCRIPT\" \\",
        "  --output-dir \"$OUTPUT_DIR\" \\",
        "  --python .venv/bin/python \\",
        f"  --payoff-mode {shlex.quote(str(config['payoff_mode']))} \\",
        f"  --t-sunset {config['t_sunset']:g} \\",
        f"  --weights {weights[0]:g} {weights[1]:g} \\",
        f"  --prey-sight-radius {sight_radius[0]:g} \\",
        f"  --predator-sight-radius {sight_radius[-1]:g} \\",
        f"  --prey-smell-radius {smell_radius[0]:g} \\",
        f"  --predator-smell-radius {smell_radius[-1]:g} \\",
        f"  --number-of-points {int(config['number_of_points'])} \\",
        f"  --dt {config['dt']:g} \\",
        f"  --number-of-cycles {int(config['number_of_cycles'])} \\",
        f"  --observation-window {config['observation_window']:g} \\",
        f"  --prey-growth {reaction_rates['prey_growth']:g} \\",
        f"  --predator-decay {reaction_rates['predator_decay']:g} \\",
        f"  --predation-rate {reaction_rates['predation_rate']:g} \\",
        f"  --conversion-rate {reaction_rates['conversion_rate']:g} \\",
        f"  --chi11 {attraction[0][0]:g} \\",
        f"  --chi12 {attraction[0][1]:g} \\",
        f"  --chi21 {attraction[1][0]:g} \\",
        f"  --chi22 {attraction[1][1]:g} \\",
        f"  --diffusion {diffusion[0]:g} {diffusion[1]:g} \\",
        f"  --initial-centers {initial_centers[0]:g} {initial_centers[1]:g} \\",
        f"  --initial-width {config['initial_width']:g} \\",
        f"  --strategy-codes {shlex.quote(strategy_codes)} \\",
        f"  --max-workers {int(max_workers)}",
    ]
    return command_lines


def write_remaining_pipeline_script(run_directory, config, max_workers):
    output_path = Path(run_directory) / REMAINING_PIPELINE_SCRIPT_FILENAME
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "\n".join(build_remaining_pipeline_command_lines(run_directory, config, max_workers))
        + "\n",
        encoding="utf-8",
    )
    output_path.chmod(0o755)
    return output_path


def probability_mapping(labels, probabilities, *, atol=PROBABILITY_ATOL):
    return {
        label: float(probability)
        for label, probability in zip(labels, probabilities)
        if float(probability) > atol
    }


def support_size(probabilities, *, atol=PROBABILITY_ATOL):
    return int(np.count_nonzero(np.asarray(probabilities, dtype=float) > atol))


def normalized_entropy(probabilities, *, atol=PROBABILITY_ATOL):
    vector = np.asarray(probabilities, dtype=float)
    active = vector[vector > atol]
    if active.size <= 1:
        return 0.0
    entropy = -float(np.sum(active * np.log(active)))
    return float(entropy / np.log(len(vector)))


def top_labels(labels, probabilities, *, atol=PROBABILITY_ATOL):
    vector = np.asarray(probabilities, dtype=float)
    maximum = float(np.max(vector))
    return {
        label
        for label, probability in zip(labels, vector)
        if maximum - float(probability) <= atol
    }


def summarize_player(labels, equilibria, *, strategy_key, label_to_index):
    leader_sets = [top_labels(labels, equilibrium[strategy_key]) for equilibrium in equilibria]
    consensus_label = None
    if leader_sets and all(len(leader_set) == 1 for leader_set in leader_sets):
        candidate = next(iter(leader_sets[0]))
        if all(next(iter(leader_set)) == candidate for leader_set in leader_sets[1:]):
            consensus_label = candidate

    support_sizes = [support_size(equilibrium[strategy_key]) for equilibrium in equilibria]
    entropies = [normalized_entropy(equilibrium[strategy_key]) for equilibrium in equilibria]
    if consensus_label is None:
        consensus_probability_mean = None
        consensus_probability_min = None
        consensus_probability_max = None
    else:
        label_index = label_to_index[consensus_label]
        consensus_probabilities = [
            float(equilibrium[strategy_key][label_index]) for equilibrium in equilibria
        ]
        consensus_probability_mean = float(np.mean(consensus_probabilities))
        consensus_probability_min = float(np.min(consensus_probabilities))
        consensus_probability_max = float(np.max(consensus_probabilities))

    return {
        "consensus_label": consensus_label,
        "support_size_min": int(min(support_sizes)),
        "support_size_max": int(max(support_sizes)),
        "normalized_entropy_mean": float(np.mean(entropies)),
        "normalized_entropy_min": float(np.min(entropies)),
        "normalized_entropy_max": float(np.max(entropies)),
        "consensus_probability_mean": consensus_probability_mean,
        "consensus_probability_min": consensus_probability_min,
        "consensus_probability_max": consensus_probability_max,
    }


def summarize_cell(labels, equilibria):
    label_to_index = {label: index for index, label in enumerate(labels)}
    prey_summary = summarize_player(
        labels,
        equilibria,
        strategy_key="prey_strategy",
        label_to_index=label_to_index,
    )
    predator_summary = summarize_player(
        labels,
        equilibria,
        strategy_key="predator_strategy",
        label_to_index=label_to_index,
    )
    prey_payoffs = [float(equilibrium["prey_payoff"]) for equilibrium in equilibria]
    predator_payoffs = [float(equilibrium["predator_payoff"]) for equilibrium in equilibria]
    return {
        "prey": prey_summary,
        "predator": predator_summary,
        "equilibrium_count": len(equilibria),
        "prey_expected_payoff_mean": float(np.mean(prey_payoffs)),
        "prey_expected_payoff_min": float(np.min(prey_payoffs)),
        "prey_expected_payoff_max": float(np.max(prey_payoffs)),
        "predator_expected_payoff_mean": float(np.mean(predator_payoffs)),
        "predator_expected_payoff_min": float(np.min(predator_payoffs)),
        "predator_expected_payoff_max": float(np.max(predator_payoffs)),
    }


def resolve_support_size_display_value(minimum_size, maximum_size):
    if int(minimum_size) == int(maximum_size):
        return float(minimum_size)
    return np.nan


def build_equilibrium_details(labels, equilibria):
    details = []
    for equilibrium in equilibria:
        details.append(
            {
                "algorithm": equilibrium["algorithm"],
                "prey_mixed_strategy": probability_mapping(
                    labels,
                    equilibrium["prey_strategy"],
                ),
                "predator_mixed_strategy": probability_mapping(
                    labels,
                    equilibrium["predator_strategy"],
                ),
                "prey_expected_payoff": float(equilibrium["prey_payoff"]),
                "predator_expected_payoff": float(equilibrium["predator_payoff"]),
            }
        )
    return details


def compute_nash_weight_summary(
    activity_regimes,
    base_config,
    w1_values,
    w2_values,
    *,
    output_dir,
    max_workers,
):
    labels = [regime["code"] for regime in activity_regimes]
    component_labels = labels + [CONSENSUS_LABEL]
    component_index = {label: index for index, label in enumerate(component_labels)}

    prey_component_grid = np.zeros((len(w1_values), len(w2_values)), dtype=int)
    predator_component_grid = np.zeros((len(w1_values), len(w2_values)), dtype=int)
    prey_probability_grid = np.full((len(w1_values), len(w2_values)), np.nan, dtype=float)
    predator_probability_grid = np.full(
        (len(w1_values), len(w2_values)),
        np.nan,
        dtype=float,
    )
    prey_support_size_grid = np.full(
        (len(w1_values), len(w2_values)),
        np.nan,
        dtype=float,
    )
    predator_support_size_grid = np.full(
        (len(w1_values), len(w2_values)),
        np.nan,
        dtype=float,
    )
    equilibrium_count_grid = np.zeros((len(w1_values), len(w2_values)), dtype=float)
    summary_rows = []
    detail_rows = []
    total_pair_count = len(w1_values) * len(w2_values)

    print(
        f"Computing Nash heatmaps for {total_pair_count} weight pair(s) with "
        f"strategies {', '.join(labels)}.",
        flush=True,
    )

    processed_pair_count = 0
    for row_index, w1 in enumerate(w1_values):
        for column_index, w2 in enumerate(w2_values):
            processed_pair_count += 1
            print(
                f"  Starting pair {processed_pair_count}/{total_pair_count}: "
                f"w1={w1:g}, w2={w2:g}",
                flush=True,
            )

            config = dict(base_config)
            config["weights"] = (w1, w2)
            config["strategy_codes"] = tuple(labels)
            pair_output_dir = build_weight_run_directory(output_dir, w1, w2)
            output_paths = build_output_paths(pair_output_dir)
            prepare_output_directory_for_run(output_paths, config, echo=False)
            payoff_matrices = run_all_cases(
                activity_regimes,
                config,
                max_workers=max_workers,
                case_payoff_output_path=output_paths["case_payoff_output_path"],
                echo=True,
            )
            save_pair_payoff_outputs(
                output_paths,
                activity_regimes,
                config,
                payoff_matrices,
            )
            remaining_pipeline_script = write_remaining_pipeline_script(
                pair_output_dir,
                config,
                max_workers,
            )
            equilibria = compute_nash_equilibria(
                payoff_matrices["prey"],
                payoff_matrices["predator"],
            )
            cell_summary = summarize_cell(labels, equilibria)
            prey_consensus_label = cell_summary["prey"]["consensus_label"]
            predator_consensus_label = cell_summary["predator"]["consensus_label"]
            prey_display_label = (
                CONSENSUS_LABEL if prey_consensus_label is None else prey_consensus_label
            )
            predator_display_label = (
                CONSENSUS_LABEL
                if predator_consensus_label is None
                else predator_consensus_label
            )

            prey_component_grid[row_index, column_index] = component_index[
                prey_display_label
            ]
            predator_component_grid[row_index, column_index] = component_index[
                predator_display_label
            ]
            if cell_summary["prey"]["consensus_probability_mean"] is not None:
                prey_probability_grid[row_index, column_index] = cell_summary["prey"][
                    "consensus_probability_mean"
                ]
            if cell_summary["predator"]["consensus_probability_mean"] is not None:
                predator_probability_grid[row_index, column_index] = cell_summary[
                    "predator"
                ]["consensus_probability_mean"]
            prey_support_size_grid[row_index, column_index] = (
                resolve_support_size_display_value(
                    cell_summary["prey"]["support_size_min"],
                    cell_summary["prey"]["support_size_max"],
                )
            )
            predator_support_size_grid[row_index, column_index] = (
                resolve_support_size_display_value(
                    cell_summary["predator"]["support_size_min"],
                    cell_summary["predator"]["support_size_max"],
                )
            )
            equilibrium_count_grid[row_index, column_index] = float(
                cell_summary["equilibrium_count"]
            )

            summary_rows.append(
                {
                    "w1": w1,
                    "w2": w2,
                    "run_directory": str(pair_output_dir),
                    "remaining_pipeline_script": str(remaining_pipeline_script),
                    "equilibrium_count": cell_summary["equilibrium_count"],
                    "prey_consensus_leader": prey_display_label,
                    "prey_consensus_probability_mean": cell_summary["prey"][
                        "consensus_probability_mean"
                    ],
                    "prey_consensus_probability_min": cell_summary["prey"][
                        "consensus_probability_min"
                    ],
                    "prey_consensus_probability_max": cell_summary["prey"][
                        "consensus_probability_max"
                    ],
                    "prey_support_size_min": cell_summary["prey"][
                        "support_size_min"
                    ],
                    "prey_support_size_max": cell_summary["prey"][
                        "support_size_max"
                    ],
                    "prey_normalized_entropy_mean": cell_summary["prey"][
                        "normalized_entropy_mean"
                    ],
                    "predator_consensus_leader": predator_display_label,
                    "predator_consensus_probability_mean": cell_summary["predator"][
                        "consensus_probability_mean"
                    ],
                    "predator_consensus_probability_min": cell_summary["predator"][
                        "consensus_probability_min"
                    ],
                    "predator_consensus_probability_max": cell_summary["predator"][
                        "consensus_probability_max"
                    ],
                    "predator_support_size_min": cell_summary["predator"][
                        "support_size_min"
                    ],
                    "predator_support_size_max": cell_summary["predator"][
                        "support_size_max"
                    ],
                    "predator_normalized_entropy_mean": cell_summary["predator"][
                        "normalized_entropy_mean"
                    ],
                    "prey_expected_payoff_mean": cell_summary[
                        "prey_expected_payoff_mean"
                    ],
                    "prey_expected_payoff_min": cell_summary[
                        "prey_expected_payoff_min"
                    ],
                    "prey_expected_payoff_max": cell_summary[
                        "prey_expected_payoff_max"
                    ],
                    "predator_expected_payoff_mean": cell_summary[
                        "predator_expected_payoff_mean"
                    ],
                    "predator_expected_payoff_min": cell_summary[
                        "predator_expected_payoff_min"
                    ],
                    "predator_expected_payoff_max": cell_summary[
                        "predator_expected_payoff_max"
                    ],
                }
            )
            detail_rows.append(
                {
                    "w1": w1,
                    "w2": w2,
                    "run_directory": str(pair_output_dir),
                    "remaining_pipeline_script": str(remaining_pipeline_script),
                    "equilibrium_count": cell_summary["equilibrium_count"],
                    "prey_consensus_leader": prey_display_label,
                    "predator_consensus_leader": predator_display_label,
                    "equilibria": build_equilibrium_details(labels, equilibria),
                }
            )

            print(
                f"  [{processed_pair_count}/{total_pair_count}] "
                f"w1={w1:g}, w2={w2:g}: "
                f"prey={prey_display_label}, predator={predator_display_label}, "
                f"equilibria={cell_summary['equilibrium_count']}",
                flush=True,
            )

    return {
        "summary_rows": summary_rows,
        "detail_rows": detail_rows,
        "prey_component_grid": prey_component_grid,
        "predator_component_grid": predator_component_grid,
        "prey_probability_grid": prey_probability_grid,
        "predator_probability_grid": predator_probability_grid,
        "prey_support_size_grid": prey_support_size_grid,
        "predator_support_size_grid": predator_support_size_grid,
        "equilibrium_count_grid": equilibrium_count_grid,
        "component_labels": component_labels,
    }


def format_optional_float(value):
    if value is None:
        return ""
    return f"{float(value):.10f}"


def save_summary_csv(summary_rows, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "w1",
        "w2",
        "run_directory",
        "remaining_pipeline_script",
        "equilibrium_count",
        "prey_consensus_leader",
        "prey_consensus_probability_mean",
        "prey_consensus_probability_min",
        "prey_consensus_probability_max",
        "prey_support_size_min",
        "prey_support_size_max",
        "prey_normalized_entropy_mean",
        "predator_consensus_leader",
        "predator_consensus_probability_mean",
        "predator_consensus_probability_min",
        "predator_consensus_probability_max",
        "predator_support_size_min",
        "predator_support_size_max",
        "predator_normalized_entropy_mean",
        "prey_expected_payoff_mean",
        "prey_expected_payoff_min",
        "prey_expected_payoff_max",
        "predator_expected_payoff_mean",
        "predator_expected_payoff_min",
        "predator_expected_payoff_max",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(
                {
                    "w1": f"{row['w1']:.10g}",
                    "w2": f"{row['w2']:.10g}",
                    "run_directory": row["run_directory"],
                    "remaining_pipeline_script": row["remaining_pipeline_script"],
                    "equilibrium_count": int(row["equilibrium_count"]),
                    "prey_consensus_leader": row["prey_consensus_leader"],
                    "prey_consensus_probability_mean": format_optional_float(
                        row["prey_consensus_probability_mean"]
                    ),
                    "prey_consensus_probability_min": format_optional_float(
                        row["prey_consensus_probability_min"]
                    ),
                    "prey_consensus_probability_max": format_optional_float(
                        row["prey_consensus_probability_max"]
                    ),
                    "prey_support_size_min": int(row["prey_support_size_min"]),
                    "prey_support_size_max": int(row["prey_support_size_max"]),
                    "prey_normalized_entropy_mean": (
                        f"{row['prey_normalized_entropy_mean']:.10f}"
                    ),
                    "predator_consensus_leader": row["predator_consensus_leader"],
                    "predator_consensus_probability_mean": format_optional_float(
                        row["predator_consensus_probability_mean"]
                    ),
                    "predator_consensus_probability_min": format_optional_float(
                        row["predator_consensus_probability_min"]
                    ),
                    "predator_consensus_probability_max": format_optional_float(
                        row["predator_consensus_probability_max"]
                    ),
                    "predator_support_size_min": int(
                        row["predator_support_size_min"]
                    ),
                    "predator_support_size_max": int(
                        row["predator_support_size_max"]
                    ),
                    "predator_normalized_entropy_mean": (
                        f"{row['predator_normalized_entropy_mean']:.10f}"
                    ),
                    "prey_expected_payoff_mean": (
                        f"{row['prey_expected_payoff_mean']:.10f}"
                    ),
                    "prey_expected_payoff_min": (
                        f"{row['prey_expected_payoff_min']:.10f}"
                    ),
                    "prey_expected_payoff_max": (
                        f"{row['prey_expected_payoff_max']:.10f}"
                    ),
                    "predator_expected_payoff_mean": (
                        f"{row['predator_expected_payoff_mean']:.10f}"
                    ),
                    "predator_expected_payoff_min": (
                        f"{row['predator_expected_payoff_min']:.10f}"
                    ),
                    "predator_expected_payoff_max": (
                        f"{row['predator_expected_payoff_max']:.10f}"
                    ),
                }
            )


def parse_optional_float(value):
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        return None
    return float(text)


def parse_required_float(value, *, field_name):
    parsed_value = parse_optional_float(value)
    if parsed_value is None:
        raise ValueError(f"Missing required float field: {field_name}.")
    return parsed_value


def parse_required_int(value, *, field_name):
    parsed_value = parse_optional_float(value)
    if parsed_value is None:
        raise ValueError(f"Missing required integer field: {field_name}.")
    return int(parsed_value)


def load_summary_csv(summary_path):
    summary_path = Path(summary_path).expanduser().resolve()
    if not summary_path.is_file():
        raise FileNotFoundError(f"Nash summary CSV not found: {summary_path}")

    with summary_path.open("r", newline="", encoding="utf-8") as handle:
        rows = [dict(row) for row in csv.DictReader(handle)]

    if not rows:
        raise ValueError(f"Nash summary CSV is empty: {summary_path}")

    return rows


def build_plot_grids_from_summary_rows(
    summary_rows,
    component_labels,
    w1_values,
    w2_values,
):
    component_index = {label: index for index, label in enumerate(component_labels)}
    w1_index = {round(float(value), 10): index for index, value in enumerate(w1_values)}
    w2_index = {round(float(value), 10): index for index, value in enumerate(w2_values)}

    prey_component_grid = np.full(
        (len(w1_values), len(w2_values)),
        component_index[CONSENSUS_LABEL],
        dtype=int,
    )
    predator_component_grid = np.full(
        (len(w1_values), len(w2_values)),
        component_index[CONSENSUS_LABEL],
        dtype=int,
    )
    prey_probability_grid = np.full((len(w1_values), len(w2_values)), np.nan, dtype=float)
    predator_probability_grid = np.full(
        (len(w1_values), len(w2_values)),
        np.nan,
        dtype=float,
    )
    prey_support_size_grid = np.full(
        (len(w1_values), len(w2_values)),
        np.nan,
        dtype=float,
    )
    predator_support_size_grid = np.full(
        (len(w1_values), len(w2_values)),
        np.nan,
        dtype=float,
    )
    equilibrium_count_grid = np.full((len(w1_values), len(w2_values)), np.nan, dtype=float)
    filled_mask = np.zeros((len(w1_values), len(w2_values)), dtype=bool)

    for row in summary_rows:
        w1 = round(parse_required_float(row.get("w1"), field_name="w1"), 10)
        w2 = round(parse_required_float(row.get("w2"), field_name="w2"), 10)
        if w1 not in w1_index or w2 not in w2_index:
            raise ValueError(
                f"Summary row has weights outside the configured grid: w1={w1:g}, w2={w2:g}."
            )

        row_index = w1_index[w1]
        column_index = w2_index[w2]
        if filled_mask[row_index, column_index]:
            raise ValueError(
                f"Duplicate summary row for w1={w1:g}, w2={w2:g}."
            )

        prey_label = str(row.get("prey_consensus_leader", "")).strip() or CONSENSUS_LABEL
        predator_label = (
            str(row.get("predator_consensus_leader", "")).strip() or CONSENSUS_LABEL
        )
        if prey_label not in component_index:
            raise ValueError(f"Unknown prey consensus leader in summary: {prey_label}")
        if predator_label not in component_index:
            raise ValueError(
                f"Unknown predator consensus leader in summary: {predator_label}"
            )

        prey_component_grid[row_index, column_index] = component_index[prey_label]
        predator_component_grid[row_index, column_index] = component_index[predator_label]

        prey_probability = parse_optional_float(row.get("prey_consensus_probability_mean"))
        if prey_probability is not None:
            prey_probability_grid[row_index, column_index] = prey_probability
        predator_probability = parse_optional_float(
            row.get("predator_consensus_probability_mean")
        )
        if predator_probability is not None:
            predator_probability_grid[row_index, column_index] = predator_probability

        prey_support_size_grid[row_index, column_index] = resolve_support_size_display_value(
            parse_required_int(row.get("prey_support_size_min"), field_name="prey_support_size_min"),
            parse_required_int(row.get("prey_support_size_max"), field_name="prey_support_size_max"),
        )
        predator_support_size_grid[row_index, column_index] = resolve_support_size_display_value(
            parse_required_int(
                row.get("predator_support_size_min"),
                field_name="predator_support_size_min",
            ),
            parse_required_int(
                row.get("predator_support_size_max"),
                field_name="predator_support_size_max",
            ),
        )
        equilibrium_count_grid[row_index, column_index] = float(
            parse_required_int(row.get("equilibrium_count"), field_name="equilibrium_count")
        )
        filled_mask[row_index, column_index] = True

    if not np.all(filled_mask):
        missing_row_index, missing_column_index = np.argwhere(~filled_mask)[0]
        raise ValueError(
            "Summary CSV is missing a weight pair for the configured grid: "
            f"w1={w1_values[missing_row_index]:g}, w2={w2_values[missing_column_index]:g}."
        )

    return {
        "prey_component_grid": prey_component_grid,
        "predator_component_grid": predator_component_grid,
        "prey_probability_grid": prey_probability_grid,
        "predator_probability_grid": predator_probability_grid,
        "prey_support_size_grid": prey_support_size_grid,
        "predator_support_size_grid": predator_support_size_grid,
        "equilibrium_count_grid": equilibrium_count_grid,
    }


def save_detail_json(
    output_path,
    *,
    w1_values,
    w2_values,
    activity_regimes,
    detail_rows,
):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "w1_values": list(w1_values),
        "w2_values": list(w2_values),
        "activity_codes": [regime["code"] for regime in activity_regimes],
        "pairs": detail_rows,
    }
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def save_run_config(
    output_path,
    *,
    base_config,
    w1_values,
    w2_values,
    activity_regimes,
):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "w1_values": list(w1_values),
        "w2_values": list(w2_values),
        "activity_codes": [regime["code"] for regime in activity_regimes],
        "weight_runs_directory": WEIGHT_RUNS_DIRECTORY_NAME,
        "remaining_pipeline_script_name": REMAINING_PIPELINE_SCRIPT_FILENAME,
        "equilibrium_summary_definition": (
            "consensus leading component of the mixed Nash equilibrium set; "
            "Multi indicates either several equilibria with different leaders or "
            "an equilibrium whose top probability is tied"
        ),
        "payoff_config": base_config,
    }
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def style_weight_axes(axis, w1_values, w2_values):
    axis.set_xlabel("$w_2$")
    axis.set_xticks(np.arange(len(w2_values)))
    axis.set_xticklabels([f"{value:g}" for value in w2_values], rotation=45, ha="right")
    axis.set_yticks(np.arange(len(w1_values)))
    axis.set_yticklabels([f"{value:g}" for value in w1_values])
    axis.set_xticks(np.arange(-0.5, len(w2_values), 1), minor=True)
    axis.set_yticks(np.arange(-0.5, len(w1_values), 1), minor=True)
    axis.grid(which="minor", color="white", linewidth=0.8, alpha=0.6)
    axis.tick_params(which="minor", bottom=False, left=False)


def save_consensus_component_heatmap(
    prey_component_grid,
    predator_component_grid,
    prey_probability_grid,
    predator_probability_grid,
    prey_support_size_grid,
    predator_support_size_grid,
    component_labels,
    w1_values,
    w2_values,
    output_path,
    *,
    payoff_mode,
    t_sunset,
):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    colours = [
        ACTIVITY_COLORS[label] if label != CONSENSUS_LABEL else CONSENSUS_COLOUR
        for label in component_labels
    ]
    colour_map = ListedColormap(colours)
    colour_bounds = np.arange(len(component_labels) + 1) - 0.5
    colour_norm = BoundaryNorm(colour_bounds, colour_map.N)
    displayed_labels = list(component_labels)
    consensus_index = component_labels.index(CONSENSUS_LABEL)
    if not np.any(prey_component_grid == consensus_index) and not np.any(
        predator_component_grid == consensus_index
    ):
        displayed_labels = [
            label for label in displayed_labels if label != CONSENSUS_LABEL
        ]

    legend_handles = []
    for label in displayed_labels:
        if label == CONSENSUS_LABEL:
            legend_text = CONSENSUS_LABEL
        else:
            legend_text = f"{label} ({ACTIVITY_LABELS[label]})"
        legend_handles.append(
            Patch(
                facecolor=(
                    ACTIVITY_COLORS[label]
                    if label != CONSENSUS_LABEL
                    else CONSENSUS_COLOUR
                ),
                edgecolor="black",
                label=legend_text,
            )
        )

    probability_cmap = plt.get_cmap("viridis").copy()
    probability_cmap.set_bad("#f3f3f3")
    support_cmap = plt.get_cmap("magma").copy()
    support_cmap.set_bad("#f3f3f3")
    combined_support_values = np.concatenate(
        (
            np.ravel(prey_support_size_grid),
            np.ravel(predator_support_size_grid),
        )
    )
    valid_support_values = combined_support_values[~np.isnan(combined_support_values)]
    has_support_values = valid_support_values.size > 0
    support_ticks = None
    support_norm = None
    if has_support_values:
        support_min = int(np.min(valid_support_values))
        support_max = int(np.max(valid_support_values))
        support_ticks = np.arange(support_min, support_max + 1, dtype=float)
        support_bounds = np.arange(support_min - 0.5, support_max + 1.5, 1.0)
        support_norm = BoundaryNorm(support_bounds, support_cmap.N)

    figure_height = max(8.8, 0.82 * len(w1_values))
    figure = plt.figure(
        figsize=(15.8, figure_height),
        constrained_layout=True,
    )
    grid_spec = figure.add_gridspec(
        3,
        3,
        height_ratios=(1.0, 1.0, 0.42),
    )
    axes = np.empty((2, 3), dtype=object)
    axes[0, 0] = figure.add_subplot(grid_spec[0, 0])
    for column_index in range(1, 3):
        axes[0, column_index] = figure.add_subplot(
            grid_spec[0, column_index],
            sharex=axes[0, 0],
            sharey=axes[0, 0],
        )
    for column_index in range(3):
        axes[1, column_index] = figure.add_subplot(
            grid_spec[1, column_index],
            sharex=axes[0, 0],
            sharey=axes[0, 0],
        )
    legend_axes = [figure.add_subplot(grid_spec[2, column_index]) for column_index in range(3)]
    for column_index, title in enumerate(
        (
            "Consensus leader",
            "Leader proportion",
            "Number of strategies\nin Nash equilibrium",
        )
    ):
        axes[0, column_index].set_title(title)

    row_definitions = (
        (
            "Prey",
            prey_component_grid,
            prey_probability_grid,
            prey_support_size_grid,
        ),
        (
            "Predator",
            predator_component_grid,
            predator_probability_grid,
            predator_support_size_grid,
        ),
    )

    probability_image = None
    support_image = None
    for row_index, (row_label, component_grid, probability_grid, support_size_grid) in enumerate(
        row_definitions
    ):
        consensus_axis = axes[row_index, 0]
        probability_axis = axes[row_index, 1]
        support_axis = axes[row_index, 2]

        consensus_axis.imshow(
            component_grid,
            cmap=colour_map,
            norm=colour_norm,
            origin="lower",
            aspect="auto",
            interpolation="nearest",
        )
        probability_image = probability_axis.imshow(
            np.ma.masked_invalid(probability_grid),
            origin="lower",
            aspect="auto",
            interpolation="nearest",
            cmap=probability_cmap,
            vmin=0.0,
            vmax=1.0,
        )
        support_image = support_axis.imshow(
            np.ma.masked_invalid(support_size_grid),
            origin="lower",
            aspect="auto",
            interpolation="nearest",
            cmap=support_cmap,
            norm=support_norm,
        )

        for column_index, axis in enumerate(axes[row_index]):
            style_weight_axes(axis, w1_values, w2_values)
            if row_index == 0:
                axis.set_xlabel("")
                axis.tick_params(labelbottom=False)
            if column_index > 0:
                axis.tick_params(labelleft=False)

        consensus_axis.set_ylabel(f"{row_label}\n$w_1$")

    legend_axes[0].axis("off")
    legend_axes[0].legend(
        handles=legend_handles,
        loc="center",
        ncol=2,
        frameon=False,
        fontsize=12,
        title="Consensus leader",
        title_fontsize=15,
        columnspacing=1.4,
        handletextpad=0.7,
        handlelength=1.8,
        borderpad=0.8,
        labelspacing=0.7,
    )
    if probability_image is not None:
        legend_axes[1].axis("off")
        probability_colorbar_axis = legend_axes[1].inset_axes([0.12, 0.28, 0.76, 0.5])
        figure.colorbar(
            probability_image,
            cax=probability_colorbar_axis,
            orientation="horizontal",
            label="Probability",
        )
    if has_support_values and support_image is not None:
        legend_axes[2].axis("off")
        support_colorbar_axis = legend_axes[2].inset_axes([0.12, 0.28, 0.76, 0.5])
        figure.colorbar(
            support_image,
            cax=support_colorbar_axis,
            orientation="horizontal",
            label="Support size",
            ticks=support_ticks.tolist(),
        )
    figure.savefig(output_path, bbox_inches="tight", dpi=200)
    plt.close(figure)


def save_diagnostic_heatmap(
    prey_probability_grid,
    predator_probability_grid,
    equilibrium_count_grid,
    w1_values,
    w2_values,
    output_path,
    *,
    payoff_mode,
    t_sunset,
):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure_height = max(4.8, 0.55 * len(w1_values))
    figure, axes = plt.subplots(
        1,
        3,
        figsize=(16.0, figure_height),
        sharey=True,
        constrained_layout=True,
    )

    probability_cmap = plt.get_cmap("viridis").copy()
    probability_cmap.set_bad("#f3f3f3")
    masked_prey_probability = np.ma.masked_invalid(prey_probability_grid)
    masked_predator_probability = np.ma.masked_invalid(predator_probability_grid)

    prey_image = axes[0].imshow(
        masked_prey_probability,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        cmap=probability_cmap,
        vmin=0.0,
        vmax=1.0,
    )
    axes[0].set_title("Prey leader probability")

    predator_image = axes[1].imshow(
        masked_predator_probability,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        cmap=probability_cmap,
        vmin=0.0,
        vmax=1.0,
    )
    axes[1].set_title("Predator leader probability")

    count_image = axes[2].imshow(
        equilibrium_count_grid,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        cmap="magma",
    )
    axes[2].set_title("Equilibrium count")

    for axis in axes:
        style_weight_axes(axis, w1_values, w2_values)
    axes[0].set_ylabel("$w_1$")

    figure.colorbar(prey_image, ax=axes[0], shrink=0.9, pad=0.02, label="Probability")
    figure.colorbar(
        predator_image,
        ax=axes[1],
        shrink=0.9,
        pad=0.02,
        label="Probability",
    )
    figure.colorbar(
        count_image,
        ax=axes[2],
        shrink=0.9,
        pad=0.02,
        label="Count",
    )
    figure.suptitle(
        "Mixed-Nash diagnostics across the weight plane\n"
        f"payoff mode={payoff_mode}, $t_{{sunset}}={t_sunset:g}$"
    )
    figure.text(
        0.5,
        0.01,
        "Blank probability cells indicate that the equilibrium set has no unique leader for that player.",
        ha="center",
        fontsize=9,
    )
    figure.savefig(output_path, bbox_inches="tight", dpi=200)
    plt.close(figure)


def main():
    try:
        args = parse_args()
        validate_args(args)
        require_nashpy()

        resolved_config = resolve_weight_sweep_configuration(args)
        w1_values = resolved_config["w1_values"]
        w2_values = resolved_config["w2_values"]
        activity_regimes = resolved_config["activity_regimes"]
        base_config = resolved_config["base_config"]
        output_dir = ensure_directory(resolve_output_dir(args))
        payoff_mode = base_config["payoff_mode"]
        t_sunset = base_config["t_sunset"]

        summary_output_path = output_dir / DEFAULT_SUMMARY_FILENAME
        details_output_path = output_dir / DEFAULT_DETAILS_FILENAME
        components_output_path = output_dir / DEFAULT_COMPONENTS_FIGURE_FILENAME
        diagnostics_output_path = output_dir / DEFAULT_DIAGNOSTICS_FIGURE_FILENAME
        config_output_path = output_dir / DEFAULT_CONFIG_FILENAME

        if args.plot_only:
            result = build_plot_grids_from_summary_rows(
                load_summary_csv(summary_output_path),
                [regime["code"] for regime in activity_regimes] + [CONSENSUS_LABEL],
                w1_values,
                w2_values,
            )
            result["component_labels"] = [
                regime["code"] for regime in activity_regimes
            ] + [CONSENSUS_LABEL]
        else:
            result = compute_nash_weight_summary(
                activity_regimes,
                base_config,
                w1_values,
                w2_values,
                output_dir=output_dir,
                max_workers=args.max_workers,
            )

            save_summary_csv(result["summary_rows"], summary_output_path)
            save_detail_json(
                details_output_path,
                w1_values=w1_values,
                w2_values=w2_values,
                activity_regimes=activity_regimes,
                detail_rows=result["detail_rows"],
            )
            save_run_config(
                config_output_path,
                base_config=base_config,
                w1_values=w1_values,
                w2_values=w2_values,
                activity_regimes=activity_regimes,
            )
        save_consensus_component_heatmap(
            result["prey_component_grid"],
            result["predator_component_grid"],
            result["prey_probability_grid"],
            result["predator_probability_grid"],
            result["prey_support_size_grid"],
            result["predator_support_size_grid"],
            result["component_labels"],
            w1_values,
            w2_values,
            components_output_path,
            payoff_mode=payoff_mode,
            t_sunset=t_sunset,
        )
        save_diagnostic_heatmap(
            result["prey_probability_grid"],
            result["predator_probability_grid"],
            result["equilibrium_count_grid"],
            w1_values,
            w2_values,
            diagnostics_output_path,
            payoff_mode=payoff_mode,
            t_sunset=t_sunset,
        )

        if args.plot_only:
            print(
                f"Regenerated consensus component figure at {components_output_path}\n"
                f"Regenerated diagnostic figure at {diagnostics_output_path}\n"
                f"Used saved summary CSV at {summary_output_path}"
            )
        else:
            print(
                f"Saved Nash summary CSV to {summary_output_path}\n"
                f"Saved Nash detail JSON to {details_output_path}\n"
                f"Saved consensus component figure to {components_output_path}\n"
                f"Saved diagnostic figure to {diagnostics_output_path}\n"
                f"Saved run config to {config_output_path}"
            )
    except MissingDependencyError as error:
        raise SystemExit(str(error)) from error


if __name__ == "__main__":
    main()