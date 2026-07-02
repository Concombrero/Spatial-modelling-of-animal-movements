import argparse
from pathlib import Path
import warnings

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

from payoff_csv_utils import load_payoff_matrix_csv


DEFAULT_PAYOFF_MATRIX_PATH = Path(__file__).resolve().parent / "output/Pay-off/payoff_matrix.csv"
DEFAULT_OUTPUT_DIRECTORY = Path(__file__).resolve().parent / "output/Pay-off/replicator_analysis"
DEFAULT_TIME_SPAN = 40.0
DEFAULT_TIME_STEPS = 800


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Load a predator-prey payoff matrix CSV, compute Nash equilibria, "
            "and plot asymmetric replicator dynamics for the corresponding "
            "zero-sum game."
        )
    )
    parser.add_argument(
        "--payoff-matrix",
        type=Path,
        default=DEFAULT_PAYOFF_MATRIX_PATH,
        help=(
            "Path to the payoff_matrix.csv file. Default: "
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


def _append_equilibria(equilibria, seen_pairs, payoff_matrix, algorithm_name, raw_pairs):
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
        predator_payoff = float(prey_strategy @ payoff_matrix @ predator_strategy)
        equilibria.append(
            {
                "algorithm": algorithm_name,
                "prey_strategy": prey_strategy,
                "predator_strategy": predator_strategy,
                "predator_payoff": predator_payoff,
                "prey_payoff": -predator_payoff,
            }
        )
        appended += 1
    return appended


def compute_nash_equilibria(payoff_matrix):
    """Find Nash equilibria with Nashpy, falling back if support enumeration misses one."""
    import nashpy as nash

    game = nash.Game(-payoff_matrix, payoff_matrix)
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
            payoff_matrix,
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


def asymmetric_replicator_rhs(_time, state, payoff_matrix):
    """Replicator dynamics for a two-population zero-sum game."""
    prey_strategy_count, predator_strategy_count = payoff_matrix.shape
    prey_distribution = normalize_distribution(state[:prey_strategy_count])
    predator_distribution = normalize_distribution(state[prey_strategy_count:])

    prey_payoffs = (-payoff_matrix) @ predator_distribution
    predator_payoffs = payoff_matrix.T @ prey_distribution

    mean_prey_payoff = float(prey_distribution @ prey_payoffs)
    mean_predator_payoff = float(predator_distribution @ predator_payoffs)

    prey_derivative = prey_distribution * (prey_payoffs - mean_prey_payoff)
    predator_derivative = predator_distribution * (
        predator_payoffs - mean_predator_payoff
    )
    return np.concatenate([prey_derivative, predator_derivative])


def simulate_replicator_dynamics(payoff_matrix, time_span, time_steps):
    """Simulate prey and predator strategy shares from a uniform initial condition."""
    prey_strategy_count, predator_strategy_count = payoff_matrix.shape
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
        args=(payoff_matrix,),
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
            nonzero_terms.append(f"{label}={probability:.6f}")
    return ", ".join(nonzero_terms) if nonzero_terms else "None"


def print_analysis(payoff_matrix_data, equilibria):
    np.set_printoptions(precision=4, suppress=True)
    payoff_matrix = payoff_matrix_data.values
    print("Predator payoff matrix A:")
    print(payoff_matrix)
    print()
    print(f"{payoff_matrix_data.row_player_label} strategies:")
    for index, label in enumerate(payoff_matrix_data.row_strategies, start=1):
        print(f"  {index}. {label}")
    print()
    print(f"{payoff_matrix_data.column_player_label} strategies:")
    for index, label in enumerate(payoff_matrix_data.column_strategies, start=1):
        print(f"  {index}. {label}")
    print()

    if not equilibria:
        print("No Nash equilibria were returned by Nashpy support enumeration.")
        return

    print(f"Found {len(equilibria)} Nash equilibrium result(s):")
    for index, equilibrium in enumerate(equilibria, start=1):
        print(f"  Equilibrium {index}:")
        print(f"    Algorithm: {equilibrium['algorithm']}")
        print(
            f"    {payoff_matrix_data.row_player_label} mixed strategy: "
            f"{format_mixed_strategy(payoff_matrix_data.row_strategies, equilibrium['prey_strategy'])}"
        )
        print(
            f"    {payoff_matrix_data.column_player_label} mixed strategy: "
            f"{format_mixed_strategy(payoff_matrix_data.column_strategies, equilibrium['predator_strategy'])}"
        )
        print(
            "    Expected payoffs: "
            f"prey={equilibrium['prey_payoff']:.6f}, "
            f"predator={equilibrium['predator_payoff']:.6f}"
        )


def plot_strategy_frequencies(
    time_grid,
    history,
    strategy_labels,
    title,
    output_path,
    colors,
    plot_style,
):
    """Plot one population's evolving strategy frequencies with consistent colors."""
    figure, axis = plt.subplots(figsize=(11, 6))
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

    axis.set_xlim(time_grid[0], time_grid[-1])
    axis.set_ylim(0.0, 1.0)
    axis.set_xlabel("Time")
    axis.set_ylabel("Strategy frequency")
    axis.set_title(title)
    axis.grid(True, alpha=0.25)
    axis.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    figure.tight_layout(rect=(0.0, 0.0, 0.82, 1.0))
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def main():
    args = parse_args()
    if args.time_steps < 2:
        raise ValueError("--time-steps must be at least 2.")
    if args.time_span <= 0.0:
        raise ValueError("--time-span must be positive.")

    payoff_matrix_data = load_payoff_matrix_csv(args.payoff_matrix)
    payoff_matrix = payoff_matrix_data.values
    equilibria = compute_nash_equilibria(payoff_matrix)
    print_analysis(payoff_matrix_data, equilibria)

    time_grid, prey_history, predator_history = simulate_replicator_dynamics(
        payoff_matrix,
        args.time_span,
        args.time_steps,
    )

    output_directory = Path(args.output_dir).expanduser().resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    prey_plot_path = output_directory / "prey_strategy_frequencies.png"
    predator_plot_path = output_directory / "predator_strategy_frequencies.png"
    prey_colors = plt.get_cmap("tab10")(
        np.linspace(0.0, 0.9, len(payoff_matrix_data.row_strategies))
    )
    predator_colors = plt.get_cmap("tab10")(
        np.linspace(0.0, 0.9, len(payoff_matrix_data.column_strategies))
    )

    plot_strategy_frequencies(
        time_grid,
        prey_history,
        strategy_labels=payoff_matrix_data.row_strategies,
        title=f"{payoff_matrix_data.row_player_label} Replicator Dynamics",
        output_path=prey_plot_path,
        colors=prey_colors,
        plot_style=args.plot_style,
    )
    plot_strategy_frequencies(
        time_grid,
        predator_history,
        strategy_labels=payoff_matrix_data.column_strategies,
        title=f"{payoff_matrix_data.column_player_label} Replicator Dynamics",
        output_path=predator_plot_path,
        colors=predator_colors,
        plot_style=args.plot_style,
    )

    print()
    print(f"Saved prey strategy plot to {prey_plot_path}")
    print(f"Saved predator strategy plot to {predator_plot_path}")


if __name__ == "__main__":
    main()