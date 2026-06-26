import argparse
from pathlib import Path

from payoff_matrix import (
    ACTIVITY_REGIME_CODES,
    DEFAULT_ATTRACTION,
    DEFAULT_DIFFUSION,
    DEFAULT_INITIAL_CENTERS,
    DEFAULT_INITIAL_WIDTH,
    DEFAULT_REACTION_RATES,
    DEFAULT_SMELL_RADIUS,
    DEFAULT_SIGHT_RADIUS,
    DEFAULT_T_SUNSET,
    DT,
    MAX_WORKERS,
    NUMBER_OF_CYCLES,
    NUMBER_OF_POINTS,
    OBSERVATION_WINDOW,
    build_output_paths,
    build_config,
    is_run_config_compatible,
    run_payoff_experiment,
)


DEFAULT_WEIGHT_VALUES = (0, 0.25, 0.5, 0.75, 1.0)
OUTPUT_DIRECTORY = Path(__file__).resolve().parent / "output/Pay-off/weight_sweep"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run the predator-prey payoff-matrix script for every pair "
            "(w1, w2) drawn from a weight list and save each run in its own folder."
        )
    )
    parser.add_argument(
        "--weight-values",
        nargs="+",
        type=float,
        default=list(DEFAULT_WEIGHT_VALUES),
        help=(
            "Sight-weight values used for both w1 and w2. "
            "The script evaluates every ordered pair in this list unless "
            "--w1-values or --w2-values overrides one axis."
        ),
    )
    parser.add_argument(
        "--w1-values",
        nargs="+",
        type=float,
        help=(
            "Sight-weight values used only for w1. If omitted, the script uses "
            "--weight-values for the w1 axis."
        ),
    )
    parser.add_argument(
        "--w2-values",
        nargs="+",
        type=float,
        help=(
            "Sight-weight values used only for w2. If omitted, the script uses "
            "--weight-values for the w2 axis."
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
        default=DEFAULT_SIGHT_RADIUS,
        help=f"Sight radius used by both populations. Default: {DEFAULT_SIGHT_RADIUS:g}.",
    )
    parser.add_argument(
        "--smell-radius",
        type=float,
        default=DEFAULT_SMELL_RADIUS,
        help=f"Smell radius used by both populations. Default: {DEFAULT_SMELL_RADIUS:g}.",
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
        help="Root directory where one subfolder per (w1, w2) pair is saved.",
    )
    parser.add_argument(
        "--heatmap-prey",
        choices=ACTIVITY_REGIME_CODES,
        help=(
            "Prey activity code filter for the saved population heatmaps. "
            "Use together with --heatmap-predator to save only one activity pair "
            "inside each weight folder."
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
        help="Number of parallel worker processes used within each payoff-matrix run.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip a weight-pair folder when its payoff_matrix.csv already exists.",
    )
    return parser.parse_args()


def format_weight_label(weight):
    return f"{float(weight):g}"


def build_run_output_directory(output_dir, prey_weight, predator_weight):
    return Path(output_dir) / (
        f"w1_{format_weight_label(prey_weight)}_w2_{format_weight_label(predator_weight)}"
    )


def resolve_weight_axes(args):
    shared_values = tuple(float(weight) for weight in args.weight_values)
    w1_values = (
        tuple(float(weight) for weight in args.w1_values)
        if args.w1_values is not None
        else shared_values
    )
    w2_values = (
        tuple(float(weight) for weight in args.w2_values)
        if args.w2_values is not None
        else shared_values
    )
    return w1_values, w2_values


def is_completed_run_directory(run_output_dir, config):
    run_output_dir = Path(run_output_dir)
    output_paths = build_output_paths(run_output_dir)
    if not output_paths["csv_output_path"].exists():
        return False

    return is_run_config_compatible(config, output_paths["run_config_output_path"])


def validate_args(args):
    if len(args.weight_values) < 1:
        raise ValueError("weight_values must contain at least one value.")

    weight_lists = [args.weight_values]
    if args.w1_values is not None:
        if len(args.w1_values) < 1:
            raise ValueError("w1_values must contain at least one value.")
        weight_lists.append(args.w1_values)
    if args.w2_values is not None:
        if len(args.w2_values) < 1:
            raise ValueError("w2_values must contain at least one value.")
        weight_lists.append(args.w2_values)

    for weight_list in weight_lists:
        if any(weight < 0.0 or weight > 1.0 for weight in weight_list):
            raise ValueError("Each weight value must lie in [0, 1].")

    if (args.heatmap_prey is None) != (args.heatmap_predator is None):
        raise ValueError(
            "--heatmap-prey and --heatmap-predator must be provided together."
        )


def main():
    args = parse_args()
    validate_args(args)

    w1_values, w2_values = resolve_weight_axes(args)
    total_runs = len(w1_values) * len(w2_values)
    scheduled_runs = 0
    skipped_runs = 0

    for prey_weight in w1_values:
        for predator_weight in w2_values:
            scheduled_runs += 1
            run_output_dir = build_run_output_directory(
                args.output_dir,
                prey_weight,
                predator_weight,
            )
            config = build_config(
                t_sunset=args.t_sunset,
                weights=(prey_weight, predator_weight),
                sight_radius=args.sight_radius,
                smell_radius=args.smell_radius,
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

            if args.skip_existing and is_completed_run_directory(run_output_dir, config):
                skipped_runs += 1
                print(
                    f"\n[{scheduled_runs}/{total_runs}] Skipping completed run for "
                    f"w1={prey_weight:g}, w2={predator_weight:g}",
                    flush=True,
                )
                continue

            print(
                f"\n[{scheduled_runs}/{total_runs}] Running payoff matrix for "
                f"w1={prey_weight:g}, w2={predator_weight:g}",
                flush=True,
            )
            run_payoff_experiment(
                output_dir=run_output_dir,
                config=config,
                max_workers=args.max_workers,
                heatmap_prey_code=args.heatmap_prey,
                heatmap_predator_code=args.heatmap_predator,
            )

    print(
        f"\nProcessed {total_runs} requested run(s) under {Path(args.output_dir)}"
    )
    if skipped_runs > 0:
        print(f"Skipped {skipped_runs} run(s) that were already complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())