import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from script.reproduce_day_night.shared_config import apply_plot_typography


apply_plot_typography()




def kernel1(x, R):
    """Gaussian kernel with standard deviation R."""
    return np.exp(-0.5 * (x / R) ** 2) / (R * np.sqrt(2 * np.pi))

def kernel2(x, R):
    """Exponential kernel with scale R."""
    return 0.5 * np.exp(-np.abs(x) / R) / R


def main(argc, argv):
    
    R1 = float(argv[1]) if argc > 1 else 0.2
    R2 = float(argv[2]) if argc > 2 else 0.1
    x = np.linspace(-5*max(R1, R2), 5*max(R1, R2), int(5000*max(R1, R2)))
    k1 = kernel1(x, R1)
    k2 = kernel2(x, R2)
    fig, axes = plt.subplots(1, 3, figsize=(24, 6), sharex=True, sharey=True)

    axes[0].plot(x, k1, label='Gaussian Kernel', color='blue')
    axes[0].set_title('Gaussian Kernel (Smell-based Perception)')
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('Kernel Value')
    axes[0].grid()

    axes[1].plot(x, k2, label='Exponential Kernel', color='green')
    axes[1].set_title('Exponential Kernel (Sight-based Perception)')
    axes[1].set_xlabel('x')
    axes[1].grid()
    for w in [0.25, 0.5, 0.75]:
        k_comb = w * k1 + (1 - w) * k2
        axes[2].plot(x, k_comb, label='Combined Kernel ($w$={:.2f})'.format(w), linestyle='--')
    axes[2].set_title('Kernel Combination')
    axes[2].set_xlabel('x')
    axes[2].legend()
    axes[2].grid()

    #fig.suptitle("Combination of Gaussian and Exponential Kernels with R1={} and R2={}".format(R1, R2), fontsize=16)
    fig.tight_layout()
    
    plt.show()
        
    if argc == 4:
        output_path = Path(argv[3])
        fig.savefig(f"{output_path}", dpi=300)
        print(f"Figure saved to {output_path}")

    
if __name__ == "__main__":
	raise SystemExit(main(len(sys.argv), sys.argv))