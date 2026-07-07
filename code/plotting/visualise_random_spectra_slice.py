"""Randomly plot selected slices from a 2D NMR spectra .npy file."""

import numpy as np
import matplotlib.pyplot as plt


# Edit these settings, then run the script from your IDE.
FILE_PATH = "data/plasma/aligned_nmr_spectra_128K_Plasma_WS625to680Zero.npy"
N_SAMPLES = 4
SLICE_START = 72_000
SLICE_STOP = 74_000
RANDOM_SEED = 41
SAVE_PATH = None  # Example: "random_spectra_slice.png"


def main():
    spectra = np.load(FILE_PATH, mmap_mode="r")
    if spectra.ndim != 2:
        raise ValueError(f"Expected a 2D array, got shape {spectra.shape}")

    start = max(0, SLICE_START)
    stop = min(SLICE_STOP, spectra.shape[1])
    if start >= stop:
        raise ValueError(f"Invalid slice [{SLICE_START}:{SLICE_STOP}]")

    rng = np.random.default_rng(RANDOM_SEED)
    count = min(N_SAMPLES, spectra.shape[0])
    indices = rng.choice(spectra.shape[0], size=count, replace=False)

    fig, axes = plt.subplots(count, 1, figsize=(14, 2.5 * count), sharex=True)
    axes = np.atleast_1d(axes)
    x = np.arange(start, stop)

    for ax, index in zip(axes, indices):
        ax.plot(x, spectra[index, start:stop], linewidth=0.8)
        ax.set_title(f"Sample row {index}")
        ax.set_ylabel("Intensity")
        ax.grid(alpha=0.2)

    axes[-1].set_xlabel("Spectrum point index")
    fig.suptitle(f"{FILE_PATH}: points [{start}:{stop}]", fontsize=12)
    fig.tight_layout()

    if SAVE_PATH:
        fig.savefig(SAVE_PATH, dpi=200, bbox_inches="tight")
        print(f"Saved plot to {SAVE_PATH}")

    print(f"Plotted sample rows: {indices.tolist()}")
    plt.show()


if __name__ == "__main__":
    main()
