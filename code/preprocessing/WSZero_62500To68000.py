"""
Fixed-window water suppression for aligned 1D NMR spectra.

This script loads an aligned spectra matrix from a NumPy file, sets the
predefined water-peak index range (62500 to 68000) to zero in each spectrum
when that region is not already near zero, and saves the updated spectra to a
new output file.

Input:
- data/aligned/aligned_nmr_spectra_128K_WSZero.npy

Output:
- data/aligned/aligned_nmr_spectra_128K_WS625to680Zero.npy
"""

import numpy as np
import matplotlib.pyplot as plt

# =========================
# User-configurable inputs
# =========================
INPUT_FILE = "data/plasma/plasma_unique.npy"
OUTPUT_FILE = "data/plasma/plasma_unique_WS625to680Zero.npy"
LOWER_BOUND = 62500
UPPER_BOUND = 68000
THRESHOLD = 1e-3
NUM_COMPARISON_PLOTS = 4
RANDOM_SEED = 42

def zero_water_peak(spectra, lower_bound=62500, upper_bound=68000, threshold=1e-3):
    """
    Zero out the region around the water peak in 1D NMR spectra.
    
    Parameters:
    - spectra: numpy array of shape (num_spectra, num_points), where each row is a spectrum.
    - lower_bound: The lower frequency bound for the water peak region.
    - upper_bound: The upper frequency bound for the water peak region.
    - threshold: A value below which the spectrum is considered zeroed out.
    
    Returns:
    - Updated spectra with the water peak region zeroed out.
    """
    num_spectra, num_points = spectra.shape
    
    for i in range(num_spectra):
        spectrum = spectra[i]
        
        # Identify the region around the water peak to zero
        start_idx = int(lower_bound)
        end_idx = int(upper_bound)
        
        # Check if the region is already near zero by comparing with the threshold
        region = spectrum[start_idx:end_idx]
        
        if np.any(np.abs(region) > threshold):  # If region is not zeroed
            spectrum[start_idx:end_idx] = 0  # Zero the region
            
    return spectra

def plot_comparisons(original_spectra, suppressed_spectra, num_plots=4, random_seed=42):
    """
    Plot overlaid comparisons of original vs suppressed spectra for random samples.
    """
    num_spectra = original_spectra.shape[0]
    if num_spectra == 0:
        return

    num_plots = min(num_plots, num_spectra)
    rng = np.random.default_rng(random_seed)
    sample_indices = rng.choice(num_spectra, size=num_plots, replace=False)

    cols = 2
    rows = int(np.ceil(num_plots / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(14, 4 * rows), squeeze=False)
    x = np.arange(original_spectra.shape[1])

    for i, ax in enumerate(axes.flat):
        if i >= num_plots:
            ax.axis("off")
            continue

        idx = int(sample_indices[i])
        ax.plot(x, original_spectra[idx], label="Original", alpha=0.8, lw=1.0)
        ax.plot(x, suppressed_spectra[idx], label="Suppressed", alpha=0.8, lw=1.0)
        ax.set_title(f"Spectrum {idx}")
        ax.set_xlabel("Point Index")
        ax.set_ylabel("Intensity")
        ax.grid(alpha=0.25)
        ax.legend()

    plt.tight_layout()
    plt.show()

# Load the spectra from the .npy file
spectra = np.load(INPUT_FILE)
original_spectra = spectra.copy()

# Zero out the water peak region in all spectra
updated_spectra = zero_water_peak(
    spectra,
    lower_bound=LOWER_BOUND,
    upper_bound=UPPER_BOUND,
    threshold=THRESHOLD
)

# Save the updated spectra to a new file
np.save(OUTPUT_FILE, updated_spectra)

print("Water peak region has been zeroed out in all spectra.")
print(f"Input file: {INPUT_FILE}")
print(f"Output file: {OUTPUT_FILE}")

# Show 3-4 overlaid comparisons of original vs suppressed spectra
plot_comparisons(
    original_spectra,
    updated_spectra,
    num_plots=NUM_COMPARISON_PLOTS,
    random_seed=RANDOM_SEED
)
