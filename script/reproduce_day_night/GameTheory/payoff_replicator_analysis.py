import argparse
from pathlib import Path
import sys
import warnings

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
from scipy.integrate import solve_ivp

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from script.reproduce_day_night.GameTheory.payoff_csv_utils import (
    load_payoff_game_data,
)
from script.reproduce_day_night.paths import game_theory_payoff_output_path
from script.reproduce_day_night.shared_config import (
    ACTIVITY_COLORS,
    TWO_POPULATION_SIMULATION_CONFIG,
    apply_plot_typography,
    display_activity_code,
)


apply_plot_typography()


DEFAULT_PAYOFF_MATRIX_PATH = game_theory_payoff_output_path("payoff_matrix.csv")
DEFAULT_OUTPUT_DIRECTORY = game_theory_payoff_output_path("replicator_analysis")
TWO_POPULATION_ANALYSIS_CONFIG = TWO_POPULATION_SIMULATION_CONFIG["analysis"]
DEFAULT_TIME_SPAN = TWO_POPULATION_ANALYSIS_CONFIG["replicator_time_span"]
DEFAULT_TIME_STEPS = TWO_POPULATION_ANALYSIS_CONFIG["replicator_time_steps"]
DEFAULT_COMBINED_FIGURE_FILENAME = "strategy_frequencies.png"
LEGACY_PREY_FIGURE_FILENAME = "prey_strategy_frequencies.png"
LEGACY_PREDATOR_FIGURE_FILENAME = "predator_strategy_frequencies.png"


class MissingDependencyError(RuntimeError):
    pass


def require_nashpy():
    try:
        import nashpy as nash
    except ModuleNotFoundError as error:
        raise MissingDependencyError(
            "nashpy is required for Nash-equilibrium and replicator analyses. "
            f"Install it in the active interpreter ({sys.executable}) with "
            "`python -m pip install nashpy`, or run the script with the "
            "project environment."
        ) from error

    return nash


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Load a payoff matrix file or run directory, compute Nash equilibria, "
            "and plot asymmetric replicator dynamics for the corresponding "
            "two-population game."
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
        "--time-span",
        type=float,
        default=DEFAULT_TIME_SPAN,
        help=f"Final simulation time for replicator dynamics. Default: {DEFAULT_TIME_SPAN:g}.",
    )
    parser.add_argument(
        "--time-steps",
        type=int,
        default=DEFAULT_TIME_STEPS,
        help=(
            "Number of output time points used for the replicator dynamics plot. "
            f"Default: {DEFAULT_TIME_STEPS}."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIRECTORY,
        help=(
            "Directory where the plots are saved. Default: "
            f"{DEFAULT_OUTPUT_DIRECTORY}."
        ),
    )
    parser.add_argument(
        "--plot-style",
        choices=("line", "stacked"),
        default="line",
        help="Plot prey and predator frequencies as line plots or stacked area charts.",
    )
    return parser.parse_args()


def resolve_payoff_matrices(prey_payoff_matrix, predator_payoff_matrix=None):
    if predator_payoff_matrix is None:
        predator_matrix = np.asarray(prey_payoff_matrix, dtype=float)
        prey_matrix = -predator_matrix
    else:
        prey_matrix = np.asarray(prey_payoff_matrix, dtype=float)
        predator_matrix = np.asarray(predator_payoff_matrix, dtype=float)

    if prey_matrix.shape != predator_matrix.shape:
        raise ValueError("Prey and predator payoff matrices must have the same shape.")

    if prey_matrix.ndim != 2:
        raise ValueError("Payoff matrices must be two-dimensional.")

    return prey_matrix, predator_matrix


def _append_equilibria(
    equilibria,
    seen_pairs,
    prey_payoff_matrix,
    predator_payoff_matrix,
    algorithm_name,
    raw_pairs,
):
    appended = 0
    for prey_strategy, predator_strategy in raw_pairs:
        prey_strategy = normalize_distribution(prey_strategy)
        predator_strategy = normalize_distribution(predator_strategy)
        equilibrium_key = (
            tuple(np.round(prey_strategy, 12)),
            tuple(np.round(predator_strategy, 12)),
        )
        if equilibrium_key in seen_pairs:
            continue

        seen_pairs.add(equilibrium_key)
        prey_payoff = float(prey_strategy @ prey_payoff_matrix @ predator_strategy)
        predator_payoff = float(
            prey_strategy @ predator_payoff_matrix @ predator_strategy
        )
        equilibria.append(
            {
                "algorithm": algorithm_name,
                "prey_strategy": prey_strategy,
                "predator_strategy": predator_strategy,
                "predator_payoff": predator_payoff,
                "prey_payoff": prey_payoff,
            }
        )
        appended += 1
    return appended


def compute_nash_equilibria(prey_payoff_matrix, predator_payoff_matrix=None):
    """Find Nash equilibria with Nashpy for either a zero-sum or general-sum game."""
    nash = require_nashpy()

    prey_matrix, predator_matrix = resolve_payoff_matrices(
        prey_payoff_matrix,
        predator_payoff_matrix,
    )
    game = nash.Game(prey_matrix, predator_matrix)
    equilibria = []
    seen_pairs = set()

    algorithms = (
        ("support_enumeration", lambda: game.support_enumeration(non_degenerate=False)),
        ("vertex_enumeration", game.vertex_enumeration),
        ("lemke_howson_enumeration", game.lemke_howson_enumeration),
    )

    for algorithm_name, equilibrium_factory in algorithms:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            raw_equilibria = list(equilibrium_factory())

        appended = _append_equilibria(
            equilibria,
            seen_pairs,
            prey_matrix,
            predator_matrix,
            algorithm_name,
            raw_equilibria,
        )
        if appended > 0:
            break

    return equilibria


def normalize_distribution(values, *, atol=1.0e-12):
    distribution = np.asarray(values, dtype=float)
    distribution = np.where(np.abs(distribution) <= atol, 0.0, distribution)
    distribution = np.clip(distribution, 0.0, None)
    total = float(np.sum(distribution))
    if total <= atol:
        raise ValueError("Distribution must have positive mass.")
    return distribution / total


def resolve_strategy_colors(strategy_labels):
    fallback_colors = plt.get_cmap("tab10")(
        np.linspace(0.0, 0.9, len(strategy_labels))
    )
    return [
        ACTIVITY_COLORS.get(str(label), fallback_colors[index])
        for index, label in enumerate(strategy_labels)
    ]


def asymmetric_replicator_rhs(
    _time,
    state,
    prey_payoff_matrix,
    predator_payoff_matrix,
):
    """Replicator dynamics for a two-population game."""
    prey_strategy_count, predator_strategy_count = prey_payoff_matrix.shape
    prey_distribution = normalize_distribution(state[:prey_strategy_count])
    predator_distribution = normalize_distribution(state[prey_strategy_count:])

    prey_payoffs = prey_payoff_matrix @ predator_distribution
    predator_payoffs = predator_payoff_matrix.T @ prey_distribution

    mean_prey_payoff = float(prey_distribution @ prey_payoffs)
    mean_predator_payoff = float(predator_distribution @ predator_payoffs)

    prey_derivative = prey_distribution * (prey_payoffs - mean_prey_payoff)
    predator_derivative = predator_distribution * (
        predator_payoffs - mean_predator_payoff
    )
    return np.concatenate([prey_derivative, predator_derivative])


def simulate_replicator_dynamics(
    payoff_matrix,
    time_span,
    time_steps,
    predator_payoff_matrix=None,
):
    """Simulate prey and predator strategy shares from a uniform initial condition."""
    prey_matrix, predator_matrix = resolve_payoff_matrices(
        payoff_matrix,
        predator_payoff_matrix,
    )
    prey_strategy_count, predator_strategy_count = prey_matrix.shape
    initial_prey_distribution = np.full(
        prey_strategy_count,
        1.0 / prey_strategy_count,
        dtype=float,
    )
    initial_predator_distribution = np.full(
        predator_strategy_count,
        1.0 / predator_strategy_count,
        dtype=float,
    )
    initial_state = np.concatenate(
        [initial_prey_distribution, initial_predator_distribution]
    )
    evaluation_times = np.linspace(0.0, time_span, time_steps)

    solution = solve_ivp(
        asymmetric_replicator_rhs,
        t_span=(0.0, time_span),
        y0=initial_state,
        t_eval=evaluation_times,
        args=(prey_matrix, predator_matrix),
        rtol=1.0e-8,
        atol=1.0e-10,
    )
    if not solution.success:
        raise RuntimeError(f"Replicator dynamics solver failed: {solution.message}")

    # Renormalize each snapshot to remove small numerical drift from the ODE solver.
    prey_history = np.apply_along_axis(
        normalize_distribution,
        0,
        solution.y[:prey_strategy_count, :],
    )
    predator_history = np.apply_along_axis(
        normalize_distribution,
        0,
        solution.y[prey_strategy_count:, :],
    )
    return solution.t, prey_history, predator_history


def format_mixed_strategy(strategy_labels, strategy):
    nonzero_terms = []
    for label, probability in zip(strategy_labels, strategy):
        if probability > 1.0e-10:
            nonzero_terms.append(
                f"{display_activity_code(label)}={probability:.6f}"
            )
    return ", ".join(nonzero_terms) if nonzero_terms else "None"


def print_analysis(payoff_game_data, equilibria):
    np.set_printoptions(precision=4, suppress=True)
    if payoff_game_data.is_zero_sum:
        print("Predator payoff matrix A:")
        print(payoff_game_data.predator_values)
    else:
        print("Prey payoff matrix A:")
        print(payoff_game_data.prey_values)
        print()
        print("Predator payoff matrix B:")
        print(payoff_game_data.predator_values)
    print()
    print(f"{payoff_game_data.row_player_label} strategies:")
    for index, label in enumerate(payoff_game_data.row_strategies, start=1):
        print(f"  {index}. {display_activity_code(label)}")
    print()
    print(f"{payoff_game_data.column_player_label} strategies:")
    for index, label in enumerate(payoff_game_data.column_strategies, start=1):
        print(f"  {index}. {display_activity_code(label)}")
    print()

    if not equilibria:
        print("No Nash equilibria were returned by Nashpy support enumeration.")
        return

    print(f"Found {len(equilibria)} Nash equilibrium result(s):")
    for index, equilibrium in enumerate(equilibria, start=1):
        print(f"  Equilibrium {index}:")
        print(f"    Algorithm: {equilibrium['algorithm']}")
        print(
            f"    {payoff_game_data.row_player_label} mixed strategy: "
            f"{format_mixed_strategy(payoff_game_data.row_strategies, equilibrium['prey_strategy'])}"
        )
        print(
            f"    {payoff_game_data.column_player_label} mixed strategy: "
            f"{format_mixed_strategy(payoff_game_data.column_strategies, equilibrium['predator_strategy'])}"
        )
        print(
            "    Expected payoffs: "
            f"prey={equilibrium['prey_payoff']:.6f}, "
            f"predator={equilibrium['predator_payoff']:.6f}"
        )


def plot_strategy_frequencies(
    axis,
    time_grid,
    history,
    strategy_labels,
    title,
    colors,
    plot_style,
    equilibrium=None,
):
    """Plot one population's evolving strategy frequencies on a given axis."""
    if plot_style == "stacked":
        axis.stackplot(
            time_grid,
            *history,
            labels=strategy_labels,
            colors=colors,
            alpha=0.9,
        )
    else:
        for index, label in enumerate(strategy_labels):
            axis.plot(
                time_grid,
                history[index],
                label=label,
                color=colors[index],
                linewidth=2.2,
            )

    if equilibrium is not None:
        for index, equilibrium_share in enumerate(equilibrium):
            if equilibrium_share <= 1.0e-10:
                continue
            axis.axhline(
                equilibrium_share,
                color=colors[index],
                linestyle="--",
                linewidth=1.8,
                alpha=0.55,
                zorder=4,
            )

    axis.set_xlim(time_grid[0], time_grid[-1])
    axis.set_ylim(0.0, 1.0)
    axis.set_xlabel("Time")
    axis.set_ylabel("Strategy share")
    axis.set_title(title)
    axis.grid(True, alpha=0.25)


def build_shared_legend_handles(prey_labels, prey_colors, predator_labels, predator_colors, plot_style):
    handles_by_label = {}

    def add_handles(labels, colors):
        for label, color in zip(labels, colors):
            if label in handles_by_label:
                continue
            if plot_style == "stacked":
                handles_by_label[label] = Patch(
                    facecolor=color,
                    edgecolor="none",
                    alpha=0.9,
                    label=label,
                )
            else:
                handles_by_label[label] = Line2D(
                    [0],
                    [0],
                    color=color,
                    linewidth=2.2,
                    label=label,
                )

    add_handles(prey_labels, prey_colors)
    add_handles(predator_labels, predator_colors)
    return list(handles_by_label.values())


def save_combined_strategy_frequency_figure(
    time_grid,
    prey_history,
    prey_labels,
    prey_colors,
    prey_equilibrium,
    predator_history,
    predator_labels,
    predator_colors,
    predator_equilibrium,
    output_path,
    plot_style,
    row_player_label,
    column_player_label,
):
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(14, 6),
        sharey=True,
        constrained_layout=True,
    )

    plot_strategy_frequencies(
        axes[0],
        time_grid,
        prey_history,
        prey_labels,
        f"{row_player_label} Replicator Dynamics",
        prey_colors,
        plot_style,
        prey_equilibrium,
    )
    plot_strategy_frequencies(
        axes[1],
        time_grid,
        predator_history,
        predator_labels,
        f"{column_player_label} Replicator Dynamics",
        predator_colors,
        plot_style,
        predator_equilibrium,
    )

    axes[1].set_ylabel("")
    legend_handles = build_shared_legend_handles(
        prey_labels,
        prey_colors,
        predator_labels,
        predator_colors,
        plot_style,
    )
    figure.legend(
        handles=legend_handles,
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        frameon=False,
        title="Strategies\n(dashed = Nash equilibrium)",
    )
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def main():
    try:
        args = parse_args()
        if args.time_steps < 2:
            raise ValueError("--time-steps must be at least 2.")
        if args.time_span <= 0.0:
            raise ValueError("--time-span must be positive.")

        require_nashpy()

        payoff_game_data = load_payoff_game_data(args.payoff_matrix)
        equilibria = compute_nash_equilibria(
            payoff_game_data.prey_values,
            payoff_game_data.predator_values,
        )
        print_analysis(payoff_game_data, equilibria)

        time_grid, prey_history, predator_history = simulate_replicator_dynamics(
            payoff_game_data.prey_values,
            args.time_span,
            args.time_steps,
            payoff_game_data.predator_values,
        )

        output_directory = Path(args.output_dir).expanduser().resolve()
        output_directory.mkdir(parents=True, exist_ok=True)
        combined_plot_path = output_directory / DEFAULT_COMBINED_FIGURE_FILENAME
        prey_colors = resolve_strategy_colors(payoff_game_data.row_strategies)
        predator_colors = resolve_strategy_colors(payoff_game_data.column_strategies)
        prey_display_labels = [
            display_activity_code(label) for label in payoff_game_data.row_strategies
        ]
        predator_display_labels = [
            display_activity_code(label)
            for label in payoff_game_data.column_strategies
        ]
        prey_equilibrium = equilibria[0]["prey_strategy"] if equilibria else None
        predator_equilibrium = equilibria[0]["predator_strategy"] if equilibria else None

        for legacy_path in (
            output_directory / LEGACY_PREY_FIGURE_FILENAME,
            output_directory / LEGACY_PREDATOR_FIGURE_FILENAME,
        ):
            if legacy_path.is_file():
                legacy_path.unlink()

        save_combined_strategy_frequency_figure(
            time_grid,
            prey_history,
            prey_display_labels,
            prey_colors,
            prey_equilibrium,
            predator_history,
            predator_display_labels,
            predator_colors,
            predator_equilibrium,
            combined_plot_path,
            args.plot_style,
            payoff_game_data.row_player_label,
            payoff_game_data.column_player_label,
        )

        print()
        print(f"Saved combined strategy plot to {combined_plot_path}")
    except MissingDependencyError as error:
        raise SystemExit(str(error)) from error


if __name__ == "__main__":
    main()