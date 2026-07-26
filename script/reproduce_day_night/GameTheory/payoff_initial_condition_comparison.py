import argparse
import json
from dataclasses import dataclass
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from script.reproduce_day_night.GameTheory.payoff_matrix import (
    DEFAULT_ATTRACTION,
    DEFAULT_DIFFUSION,
    DEFAULT_INITIAL_CENTERS,
    DEFAULT_INITIAL_CONDITION_PERTURBATION_AMPLITUDE,
    DEFAULT_INITIAL_CONDITION_PERTURBATION_LENGTH,
    DEFAULT_INITIAL_CONDITION_PERTURBATION_SEED,
    DEFAULT_INITIAL_WIDTH,
    DEFAULT_REACTION_RATES,
    DEFAULT_SIGHT_RADIUS,
    DEFAULT_SMELL_RADIUS,
    DEFAULT_T_SUNSET,
    DEFAULT_WEIGHTS,
    DT,
    HOMOGENEOUS_INITIAL_CONDITION_PROFILE,
    MAX_WORKERS,
    NUMBER_OF_CYCLES,
    NUMBER_OF_POINTS,
    OBSERVATION_WINDOW,
    PERTURBED_HOMOGENEOUS_INITIAL_CONDITION_PROFILE,
    POPULATION_INTEGRAL_PAYOFF_MODE,
    build_config_from_run_config_payload,
    build_config,
    build_initial_condition,
    resolve_activity_regimes,
    run_payoff_experiment,
)
from script.reproduce_day_night.paths import ensure_directory, game_theory_output_path
from script.reproduce_day_night.shared_config import apply_plot_typography


apply_plot_typography()


DEFAULT_OUTPUT_PATH = (
    Path(__file__).resolve().parents[3]
    / "article"
    / "figures"
    / "payoff_initial_condition_comparison.png"
)
DEFAULT_CACHE_DIRECTORY = game_theory_output_path("initial_condition_payoff_matrices")
DEFAULT_COMPARISON_NUMBER_OF_CYCLES = NUMBER_OF_CYCLES
DEFAULT_PROFILE_POINTS = 512
PREY_LINE_COLOR = "#3B3B3B"
PREDATOR_LINE_COLOR = "#D1495B"
PREY_LINE_STYLE = "--"
PREDATOR_LINE_STYLE = "-"


@dataclass(frozen=True)
class InitialConditionCase:
    slug: str
    label: str
    profile: str
    centers: tuple[float, float]
    width: float


INITIAL_CONDITION_CASES = {
    "homogeneous": InitialConditionCase(
        slug="homogeneous",
        label="Spatially homogeneous",
        profile=HOMOGENEOUS_INITIAL_CONDITION_PROFILE,
        centers=DEFAULT_INITIAL_CENTERS,
        width=DEFAULT_INITIAL_WIDTH,
    ),
    "perturbed-homogeneous": InitialConditionCase(
        slug="perturbed-homogeneous",
        label="Perturbed homogeneous",
        profile=PERTURBED_HOMOGENEOUS_INITIAL_CONDITION_PROFILE,
        centers=DEFAULT_INITIAL_CENTERS,
        width=DEFAULT_INITIAL_WIDTH,
    ),
    "overlapping-gaussian": InitialConditionCase(
        slug="overlapping-gaussian",
        label="Overlapping Gaussian pulses",
        profile="gaussian",
        centers=(0.45, 0.55),
        width=DEFAULT_INITIAL_WIDTH,
    ),
    "disjoint-gaussian": InitialConditionCase(
        slug="disjoint-gaussian",
        label="Disjoint Gaussian pulses",
        profile="gaussian",
        centers=DEFAULT_INITIAL_CENTERS,
        width=DEFAULT_INITIAL_WIDTH,
    ),
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compare population-integral prey and predator payoff matrices at "
            "$w_1=w_2$ across several spatial initial conditions. Each row of "
            "the saved figure shows one initial condition together with the "
            "corresponding prey and predator payoff matrices."
        )
    )
    parser.add_argument(
        "--run-config",
        type=Path,
        help=(
            "Load shared two-population parameters from an existing "
            "payoff_matrix or payoff_weight_nash_heatmap run_config.json file. "
            "The comparison still uses --weight for the displayed fixed-weight "
            "slice and keeps the built-in initial-condition families."
        ),
    )
    parser.add_argument(
        "--conditions",
        nargs="+",
        choices=tuple(INITIAL_CONDITION_CASES),
        default=list(INITIAL_CONDITION_CASES),
        help="Initial-condition families to include. Default: all four documented cases.",
    )
    parser.add_argument(
        "--weight",
        type=float,
        default=DEFAULT_WEIGHTS[0],
        help="Shared sight weight used for both populations. Default: 0.5.",
    )
    parser.add_argument(
        "--t-sunset",
        type=float,
        default=DEFAULT_T_SUNSET,
        help="Daylight proportion t_sunset in [0, 1]. Default: 0.5.",
    )
    parser.add_argument(
        "--number-of-points",
        type=int,
        default=NUMBER_OF_POINTS,
        help="Number of spatial grid points used by the solver. Default: 128.",
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=DT,
        help="Stored output time step used by the solver. Default: 0.1.",
    )
    parser.add_argument(
        "--number-of-cycles",
        type=int,
        default=DEFAULT_COMPARISON_NUMBER_OF_CYCLES,
        help=(
            "Number of daily cycles to simulate before evaluating the payoff. "
            f"Default: {DEFAULT_COMPARISON_NUMBER_OF_CYCLES}."
        ),
    )
    parser.add_argument(
        "--observation-window",
        type=float,
        default=OBSERVATION_WINDOW,
        help="Final-time window used by the population-integral payoff. Default: 2.0.",
    )
    parser.add_argument(
        "--diffusion",
        nargs=2,
        type=float,
        metavar=("D1", "D2"),
        default=list(DEFAULT_DIFFUSION),
        help="Diffusion coefficients for prey and predator. Default: 0.04 0.04.",
    )
    parser.add_argument(
        "--initial-width",
        type=float,
        default=DEFAULT_INITIAL_WIDTH,
        help="Gaussian width used by the overlapping and disjoint pulse cases. Default: 0.1.",
    )
    parser.add_argument(
        "--profile-points",
        type=int,
        default=DEFAULT_PROFILE_POINTS,
        help="Number of points used to draw the initial-condition curves. Default: 512.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=DEFAULT_CACHE_DIRECTORY,
        help=(
            "Directory where the per-condition payoff CSVs and heatmaps are cached. "
            "Default: GameTheory/output/initial_condition_payoff_matrices/."
        ),
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Path of the saved comparison figure.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=MAX_WORKERS,
        help="Maximum number of worker processes used per condition.",
    )
    return parser.parse_args()


def resolve_activity_regimes_from_run_config_payload(payload, *, source_path):
    activity_codes = payload.get("activity_codes")
    if activity_codes is None:
        activity_codes = payload.get("strategy_codes")

    if activity_codes is None and isinstance(payload.get("payoff_config"), dict):
        activity_codes = payload["payoff_config"].get("strategy_codes")

    if activity_codes is None:
        return resolve_activity_regimes(None)

    if isinstance(activity_codes, str):
        activity_codes_text = activity_codes
    elif isinstance(activity_codes, (list, tuple)):
        activity_codes_text = ",".join(str(code) for code in activity_codes)
    else:
        raise ValueError(
            f"Expected strategy codes in {source_path} to be a string or list."
        )

    return resolve_activity_regimes(activity_codes_text)


def load_shared_run_configuration(config_path):
    config_path = Path(config_path).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"run_config.json not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {config_path}.")

    config_payload = payload
    if "payoff_config" in payload:
        config_payload = dict(payload["payoff_config"])
        if "strategy_codes" not in config_payload and "activity_codes" in payload:
            config_payload["strategy_codes"] = payload["activity_codes"]

    shared_base_config = build_config_from_run_config_payload(
        config_payload,
        source_path=config_path,
    )
    activity_regimes = resolve_activity_regimes_from_run_config_payload(
        payload,
        source_path=config_path,
    )
    return shared_base_config, activity_regimes


def build_case_config(case, args, activity_regimes, *, shared_base_config=None):
    if shared_base_config is None:
        t_sunset = args.t_sunset
        sight_radius = (DEFAULT_SIGHT_RADIUS, DEFAULT_SIGHT_RADIUS)
        smell_radius = (DEFAULT_SMELL_RADIUS, DEFAULT_SMELL_RADIUS)
        number_of_points = args.number_of_points
        dt = args.dt
        number_of_cycles = args.number_of_cycles
        observation_window = args.observation_window
        payoff_mode = POPULATION_INTEGRAL_PAYOFF_MODE
        initial_width = args.initial_width
        diffusion = args.diffusion
        attraction = DEFAULT_ATTRACTION
        reaction_rates = DEFAULT_REACTION_RATES
    else:
        t_sunset = shared_base_config["t_sunset"]
        sight_radius = shared_base_config["sight_radius"]
        smell_radius = shared_base_config["smell_radius"]
        number_of_points = shared_base_config["number_of_points"]
        dt = shared_base_config["dt"]
        number_of_cycles = shared_base_config["number_of_cycles"]
        observation_window = shared_base_config["observation_window"]
        payoff_mode = shared_base_config["payoff_mode"]
        initial_width = shared_base_config["initial_width"]
        diffusion = shared_base_config["diffusion"]
        attraction = shared_base_config["attraction"]
        reaction_rates = shared_base_config["reaction_rates"]

    config = build_config(
        t_sunset=t_sunset,
        weights=(args.weight, args.weight),
        sight_radius=sight_radius,
        smell_radius=smell_radius,
        number_of_points=number_of_points,
        dt=dt,
        number_of_cycles=number_of_cycles,
        observation_window=observation_window,
        payoff_mode=payoff_mode,
        initial_centers=case.centers,
        initial_width=initial_width,
        initial_condition_profile=case.profile,
        initial_condition_perturbation_amplitude=(
            DEFAULT_INITIAL_CONDITION_PERTURBATION_AMPLITUDE
        ),
        initial_condition_perturbation_length=(
            DEFAULT_INITIAL_CONDITION_PERTURBATION_LENGTH
        ),
        initial_condition_perturbation_seed=DEFAULT_INITIAL_CONDITION_PERTURBATION_SEED,
        diffusion=diffusion,
        attraction=attraction,
        reaction_rates=reaction_rates,
    )
    config["strategy_codes"] = tuple(regime["code"] for regime in activity_regimes)
    return config


def sample_initial_condition_profiles(config, number_of_points):
    x = np.linspace(0.0, 1.0, int(number_of_points), endpoint=False)
    values = build_initial_condition(
        config["initial_centers"],
        config["initial_width"],
        profile=config.get("initial_condition_profile", "gaussian"),
        perturbation_amplitude=config.get(
            "initial_condition_perturbation_amplitude",
            DEFAULT_INITIAL_CONDITION_PERTURBATION_AMPLITUDE,
        ),
        perturbation_length=config.get(
            "initial_condition_perturbation_length",
            DEFAULT_INITIAL_CONDITION_PERTURBATION_LENGTH,
        ),
        perturbation_seed=config.get(
            "initial_condition_perturbation_seed",
            DEFAULT_INITIAL_CONDITION_PERTURBATION_SEED,
        ),
    )(x)
    return x, values


def _annotate_heatmap(axis, matrix, threshold):
    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            value = float(matrix[row_index, column_index])
            text_color = "white" if value >= threshold else "black"
            axis.text(
                column_index,
                row_index,
                f"{value:.2f}",
                ha="center",
                va="center",
                color=text_color,
                fontsize=8,
            )


def render_comparison_figure(results, activity_regimes, output_path, weight, t_sunset):
    labels = [regime["code"] for regime in activity_regimes]
    row_count = len(results)
    figure, axes = plt.subplots(
        row_count,
        3,
        figsize=(14.5, 3.8 * row_count),
        squeeze=False,
        constrained_layout=True,
    )

    prey_min = min(float(np.min(result["prey_matrix"])) for result in results)
    prey_max = max(float(np.max(result["prey_matrix"])) for result in results)
    predator_min = min(float(np.min(result["predator_matrix"])) for result in results)
    predator_max = max(float(np.max(result["predator_matrix"])) for result in results)

    prey_image = None
    predator_image = None

    for row_index, result in enumerate(results):
        condition_axis, prey_axis, predator_axis = axes[row_index]
        x = result["x"]
        values = result["initial_profiles"]

        condition_axis.plot(
            x,
            values[:, 0],
            PREY_LINE_STYLE,
            color=PREY_LINE_COLOR,
            linewidth=2.7,
            label="Prey",
        )
        condition_axis.plot(
            x,
            values[:, 1],
            PREDATOR_LINE_STYLE,
            color=PREDATOR_LINE_COLOR,
            linewidth=2.7,
            label="Predator",
        )
        condition_axis.set_xlim(0.0, 1.0)
        condition_axis.set_xlabel("x")
        condition_axis.set_ylabel("density")
        condition_axis.set_title(result["case"].label)
        if row_index == 0:
            condition_axis.legend(frameon=False, loc="upper right")

        prey_matrix = result["prey_matrix"]
        prey_image = prey_axis.imshow(
            prey_matrix,
            cmap="viridis",
            vmin=prey_min,
            vmax=prey_max,
        )
        prey_axis.set_xticks(np.arange(len(labels)), labels=labels)
        prey_axis.set_yticks(np.arange(len(labels)), labels=labels)
        prey_axis.set_xlabel("Predator strategy")
        prey_axis.set_ylabel("Prey strategy")
        if row_index == 0:
            prey_axis.set_title("Prey payoff matrix")
        _annotate_heatmap(prey_axis, prey_matrix, prey_min + 0.5 * (prey_max - prey_min))

        predator_matrix = result["predator_matrix"]
        predator_image = predator_axis.imshow(
            predator_matrix,
            cmap="viridis",
            vmin=predator_min,
            vmax=predator_max,
        )
        predator_axis.set_xticks(np.arange(len(labels)), labels=labels)
        predator_axis.set_yticks(np.arange(len(labels)), labels=labels)
        predator_axis.set_xlabel("Predator strategy")
        predator_axis.set_ylabel("Prey strategy")
        if row_index == 0:
            predator_axis.set_title("Predator payoff matrix")
        _annotate_heatmap(
            predator_axis,
            predator_matrix,
            predator_min + 0.5 * (predator_max - predator_min),
        )

    figure.suptitle(
        "Population-integral payoff matrices across spatial initial conditions\n"
        f"$w_1=w_2={weight:g}$, $t_{{sunset}}={t_sunset:g}$",
        fontsize=18,
    )
    figure.colorbar(prey_image, ax=axes[:, 1], label="prey payoff", shrink=0.98)
    figure.colorbar(
        predator_image,
        ax=axes[:, 2],
        label="predator payoff",
        shrink=0.98,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, bbox_inches="tight", dpi=200)
    plt.close(figure)


def main():
    args = parse_args()
    shared_base_config = None
    if args.run_config is None:
        activity_regimes = resolve_activity_regimes(None)
    else:
        shared_base_config, activity_regimes = load_shared_run_configuration(
            args.run_config
        )
    selected_cases = [INITIAL_CONDITION_CASES[name] for name in args.conditions]

    ensure_directory(args.cache_dir)
    results = []

    for case_index, case in enumerate(selected_cases, start=1):
        print(
            f"[{case_index}/{len(selected_cases)}] Computing {case.label}...",
            flush=True,
        )
        config = build_case_config(
            case,
            args,
            activity_regimes,
            shared_base_config=shared_base_config,
        )
        run_result = run_payoff_experiment(
            output_dir=args.cache_dir / case.slug,
            config=config,
            max_workers=args.max_workers,
            activity_regimes=activity_regimes,
            save_population_heatmaps=False,
        )
        x, initial_profiles = sample_initial_condition_profiles(
            config,
            args.profile_points,
        )
        results.append(
            {
                "case": case,
                "config": config,
                "x": x,
                "initial_profiles": initial_profiles,
                "prey_matrix": run_result["prey_matrix"],
                "predator_matrix": run_result["predator_matrix"],
            }
        )

    render_comparison_figure(
        results,
        activity_regimes,
        args.output_path,
        args.weight,
        (
            args.t_sunset
            if shared_base_config is None
            else shared_base_config["t_sunset"]
        ),
    )
    print(f"Saved comparison figure to {args.output_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())