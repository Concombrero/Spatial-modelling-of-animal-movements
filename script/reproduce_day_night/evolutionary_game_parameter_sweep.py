import argparse
import math
import sys
import time
from itertools import product
from pathlib import Path


CURRENT_DIRECTORY = Path(__file__).resolve().parent
if str(CURRENT_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIRECTORY))

from evolutionary_game import (
    DEFAULT_DT,
    DEFAULT_NUMBER_OF_CYCLES,
    DEFAULT_NUMBER_OF_POINTS,
    DEFAULT_PREDATOR_TOTAL_MASS,
    DEFAULT_PREY_TOTAL_MASS,
    DEFAULT_ROUNDS,
    DEFAULT_SELECTION_EVENTS,
    DEFAULT_SELECTION_PERCENTAGE,
    build_config,
    run_evolutionary_game,
)
from payoff_matrix import (
    DEFAULT_ATTRACTION,
    DEFAULT_DIFFUSION,
    DEFAULT_INITIAL_CENTERS,
    DEFAULT_INITIAL_WIDTH,
    DEFAULT_REACTION_RATES,
    DEFAULT_SIGHT_RADIUS,
    DEFAULT_SMELL_RADIUS,
)


OUTPUT_DIRECTORY = (
    Path(__file__).resolve().parent / "output/evolutionary_game_parameter_sweep"
)
DEFAULT_SHARE_PLOT_NAME = "strategy_share.png"
PARAMETER_SPECS = (
    ("w1", "prey sight weight", (0.5,)),
    ("w2", "predator sight weight", (0.5,)),
    ("r_smell_1", "prey smell radius", (DEFAULT_SMELL_RADIUS,)),
    ("r_sight_1", "prey sight radius", (DEFAULT_SIGHT_RADIUS,)),
    ("r_smell_2", "predator smell radius", (DEFAULT_SMELL_RADIUS,)),
    ("r_sight_2", "predator sight radius", (DEFAULT_SIGHT_RADIUS,)),
)


def add_parameter_axis_arguments(parser, parameter_name, label, default_values):
    option_prefix = parameter_name.replace("_", "-")
    default_text = " ".join(f"{float(value):g}" for value in default_values)
    parser.add_argument(
        f"--{option_prefix}-values",
        nargs="+",
        type=float,
        default=list(default_values),
        help=(
            f"Explicit values for {label}. Default: {default_text}."
        ),
    )
    parser.add_argument(
        f"--{option_prefix}-range",
        nargs=3,
        type=float,
        metavar=("START", "STOP", "STEP"),
        help=(
            f"Inclusive START STOP STEP range for {label}. "
            f"Overrides --{option_prefix}-values when provided."
        ),
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run the evolutionary predator-prey game on a Cartesian product of "
            "w1, w2, R_smell_1, R_sight_1, R_smell_2, and R_sight_2, saving only "
            "one strategy-share plot per parameter set."
        )
    )

    for parameter_name, label, default_values in PARAMETER_SPECS:
        add_parameter_axis_arguments(parser, parameter_name, label, default_values)

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
            "How many selection transfers to apply per round and species. "
            f"Default: {DEFAULT_SELECTION_EVENTS}."
        ),
    )
    parser.add_argument(
        "--selection-percentage",
        type=float,
        default=DEFAULT_SELECTION_PERCENTAGE,
        help=(
            "Population-share percentage transferred per selection event. "
            f"Default: {DEFAULT_SELECTION_PERCENTAGE:g}."
        ),
    )
    parser.add_argument(
        "--prey-total-mass",
        type=float,
        default=DEFAULT_PREY_TOTAL_MASS,
        help=(
            "Total initial prey mass shared across prey circadian subgroups. "
            f"Default: {DEFAULT_PREY_TOTAL_MASS:g}."
        ),
    )
    parser.add_argument(
        "--predator-total-mass",
        type=float,
        default=DEFAULT_PREDATOR_TOTAL_MASS,
        help=(
            "Total initial predator mass shared across predator circadian subgroups. "
            f"Default: {DEFAULT_PREDATOR_TOTAL_MASS:g}."
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
            "Diffusion coefficients for prey and predator. Default: "
            f"{DEFAULT_DIFFUSION[0]:g} {DEFAULT_DIFFUSION[1]:g}."
        ),
    )
    parser.add_argument(
        "--initial-centers",
        nargs=2,
        type=float,
        metavar=("X1", "X2"),
        default=list(DEFAULT_INITIAL_CENTERS),
        help=(
            "Initial Gaussian centers for prey and predator. Default: "
            f"{DEFAULT_INITIAL_CENTERS[0]:g} {DEFAULT_INITIAL_CENTERS[1]:g}."
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
        help="Root directory where one subfolder per parameter set is saved.",
    )
    parser.add_argument(
        "--share-plot-name",
        type=str,
        default=DEFAULT_SHARE_PLOT_NAME,
        help=(
            "Filename used inside each parameter folder for the saved share plot. "
            f"Default: {DEFAULT_SHARE_PLOT_NAME}."
        ),
    )
    parser.add_argument(
        "--echo-round-progress",
        action="store_true",
        help=(
            "Forward the per-round prey/predator progress printed by the inner "
            "evolutionary-game run."
        ),
    )
    return parser.parse_args()


def format_value_label(value):
    return f"{float(value):g}"


def expand_range(start, stop, step, *, parameter_name):
    start = float(start)
    stop = float(stop)
    step = float(step)

    if not math.isfinite(start) or not math.isfinite(stop) or not math.isfinite(step):
        raise ValueError(f"{parameter_name} range values must be finite.")
    if step <= 0.0:
        raise ValueError(f"{parameter_name} range step must be positive.")
    if stop < start:
        raise ValueError(f"{parameter_name} range stop must be at least start.")

    tolerance = 1.0e-12 * max(1.0, abs(start), abs(stop), abs(step))
    values = []
    current = start
    while current <= (stop + tolerance):
        values.append(float(round(current, 12)))
        current += step
    return tuple(values)


def resolve_axis_values(args, parameter_name):
    explicit_values = getattr(args, f"{parameter_name}_values")
    range_values = getattr(args, f"{parameter_name}_range")
    if range_values is not None:
        return expand_range(*range_values, parameter_name=parameter_name)
    return tuple(float(value) for value in explicit_values)


def resolve_parameter_axes(args):
    return {
        parameter_name: resolve_axis_values(args, parameter_name)
        for parameter_name, _, _ in PARAMETER_SPECS
    }


def build_case_output_directory(output_dir, case_parameters):
    return Path(output_dir) / (
        f"w1_{format_value_label(case_parameters['w1'])}"
        f"_w2_{format_value_label(case_parameters['w2'])}"
        f"_rsmell1_{format_value_label(case_parameters['r_smell_1'])}"
        f"_rsight1_{format_value_label(case_parameters['r_sight_1'])}"
        f"_rsmell2_{format_value_label(case_parameters['r_smell_2'])}"
        f"_rsight2_{format_value_label(case_parameters['r_sight_2'])}"
    )


def build_case_config(args, case_parameters):
    return build_config(
        w1=case_parameters["w1"],
        w2=case_parameters["w2"],
        t_sunset=args.t_sunset,
        number_of_points=args.number_of_points,
        dt=args.dt,
        number_of_cycles=args.number_of_cycles,
        rounds=args.rounds,
        selection_events=args.selection_events,
        selection_percentage=args.selection_percentage,
        prey_total_mass=args.prey_total_mass,
        predator_total_mass=args.predator_total_mass,
        sight_radius=(
            case_parameters["r_sight_1"],
            case_parameters["r_sight_2"],
        ),
        smell_radius=(
            case_parameters["r_smell_1"],
            case_parameters["r_smell_2"],
        ),
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


def move_share_plot(source_path, target_path):
    source_path = Path(source_path)
    target_path = Path(target_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    if source_path != target_path:
        source_path.replace(target_path)
    return target_path


def run_parameter_sweep(args):
    axes = resolve_parameter_axes(args)
    parameter_names = tuple(parameter_name for parameter_name, _, _ in PARAMETER_SPECS)
    axis_sizes = [len(axes[parameter_name]) for parameter_name in parameter_names]
    total_cases = math.prod(axis_sizes)
    sweep_start_time = time.perf_counter()

    print(
        f"Running {total_cases} evolutionary-game cases into {Path(args.output_dir)}",
        flush=True,
    )

    for case_index, values in enumerate(
        product(*(axes[parameter_name] for parameter_name in parameter_names)),
        start=1,
    ):
        case_parameters = dict(zip(parameter_names, values))
        output_dir = build_case_output_directory(args.output_dir, case_parameters)
        config = build_case_config(args, case_parameters)
        case_start_time = time.perf_counter()

        print(
            f"Starting case [{case_index}/{total_cases}] "
            f"w1={case_parameters['w1']:g}, "
            f"w2={case_parameters['w2']:g}, "
            f"R_smell_1={case_parameters['r_smell_1']:g}, "
            f"R_sight_1={case_parameters['r_sight_1']:g}, "
            f"R_smell_2={case_parameters['r_smell_2']:g}, "
            f"R_sight_2={case_parameters['r_sight_2']:g}",
            flush=True,
        )

        result = run_evolutionary_game(
            config,
            output_dir,
            echo=args.echo_round_progress,
            save_outputs={
                "round_payoffs": False,
                "distribution_history": False,
                "selection_events": False,
                "run_config": False,
                "share_plot": True,
            },
        )

        saved_plot = move_share_plot(
            result["share_plot"],
            Path(output_dir) / args.share_plot_name,
        )
        case_elapsed = time.perf_counter() - case_start_time
        total_elapsed = time.perf_counter() - sweep_start_time
        print(
            f"[{case_index}/{total_cases}] "
            f"w1={case_parameters['w1']:g}, "
            f"w2={case_parameters['w2']:g}, "
            f"R_smell_1={case_parameters['r_smell_1']:g}, "
            f"R_sight_1={case_parameters['r_sight_1']:g}, "
            f"R_smell_2={case_parameters['r_smell_2']:g}, "
            f"R_sight_2={case_parameters['r_sight_2']:g} "
            f"-> {saved_plot} "
            f"(case {case_elapsed:.1f}s, total {total_elapsed:.1f}s)",
            flush=True,
        )

    return 0


def main():
    args = parse_args()
    return run_parameter_sweep(args)


if __name__ == "__main__":
    raise SystemExit(main())