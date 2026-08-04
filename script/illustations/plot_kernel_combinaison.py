import argparse
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from script.simulations.shared_config import apply_plot_typography


apply_plot_typography()


def kernel1(x, R):
    """Gaussian kernel with standard deviation R."""
    return np.exp(-0.5 * (x / R) ** 2) / (R * np.sqrt(2 * np.pi))


def kernel2(x, R):
    """Exponential kernel with scale R."""
    return 0.5 * np.exp(-np.abs(x) / R) / R


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compare Gaussian, exponential, and mixed perception kernels for "
            "two chosen radius parameters."
        )
    )
    parser.add_argument(
        "gaussian_radius",
        nargs="?",
        type=float,
        default=0.2,
        help="Standard deviation of the Gaussian kernel. Default: 0.2.",
    )
    parser.add_argument(
        "exponential_radius",
        nargs="?",
        type=float,
        default=0.1,
        help="Scale of the exponential kernel. Default: 0.1.",
    )
    parser.add_argument(
        "output_path",
        nargs="?",
        type=Path,
        help="Optional saved figure path.",
    )
    return parser.parse_args()


def build_grid(gaussian_radius, exponential_radius):
    support_radius = max(float(gaussian_radius), float(exponential_radius))
    point_count = max(2, int(5000 * support_radius))
    return np.linspace(-5 * support_radius, 5 * support_radius, point_count)


def main():
    args = parse_args()
    gaussian_radius = float(args.gaussian_radius)
    exponential_radius = float(args.exponential_radius)

    x = build_grid(gaussian_radius, exponential_radius)
    k1 = kernel1(x, gaussian_radius)
    k2 = kernel2(x, exponential_radius)
    fig, axes = plt.subplots(1, 3, figsize=(24, 6), sharex=True, sharey=True)

    axes[0].plot(x, k1, label="Gaussian kernel", color="blue")
    axes[0].set_title("Gaussian kernel (smell-based perception)")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("Kernel value")
    axes[0].grid()

    axes[1].plot(x, k2, label="Exponential kernel", color="green")
    axes[1].set_title("Exponential kernel (sight-based perception)")
    axes[1].set_xlabel("x")
    axes[1].grid()
    for w in [0.25, 0.5, 0.75]:
        k_comb = w * k1 + (1 - w) * k2
        axes[2].plot(
            x,
            k_comb,
            label="Combined kernel ($w$={:.2f})".format(w),
            linestyle="--",
        )
    axes[2].set_title("Kernel combination")
    axes[2].set_xlabel("x")
    axes[2].legend()
    axes[2].grid()

    fig.tight_layout()

    if args.output_path is not None:
        output_path = args.output_path.expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300)
        print(f"Figure saved to {output_path}")
    else:
        plt.show()

    plt.close(fig)
    return 0



if __name__ == "__main__":
    raise SystemExit(main())