"""
Align and preprocess NMR spectra to a common length for downstream analysis.

This script:
1. Loads spectra from `data/source/nmr_spectra.npy`.
2. Removes duplicate spectra rows.
3. Aligns all spectra to the longest spectrum length using a selected method
   (`interpolate`, `resample`, `duplicate`, or `average`; default is `interpolate`).
4. Visualizes original vs aligned spectra and saves a comparison figure
   (`spectrum_alignment_results.png`) with summary statistics.
5. Saves the aligned output to `aligned_spectra.npy`.

Main outputs:
- `aligned_spectra.npy`: 2D array of shape (n_spectra, target_length)
- `spectrum_alignment_results.png`: alignment quality visualization
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from scipy.signal import resample

def align_spectra_to_longest(spectra_list, method='interpolate'):
    """
    Align all spectra to the length of the longest spectrum
    
    Args:
        spectra_list: numpy array of shape (n_spectra, n_points) or list of 1D arrays
        method: 'interpolate', 'resample', 'duplicate', or 'average'
    
    Returns:
        aligned_spectra: numpy array of shape (n_spectra, max_length)
        alignment_info: dict with alignment information
    """
    # Handle input format
    if isinstance(spectra_list, np.ndarray) and len(spectra_list.shape) == 2:
        # All spectra have same length already
        n_spectra, current_length = spectra_list.shape
        spectra_lengths = [current_length] * n_spectra
        target_length = current_length
        print(f"Input spectra already have uniform length: {current_length}")
        return spectra_list, {
            'original_lengths': spectra_lengths,
            'target_length': target_length,
            'method': method,
            'spectra_count': n_spectra
        }
    elif isinstance(spectra_list, list):
        # Variable length spectra
        spectra_lengths = [len(spectrum) for spectrum in spectra_list]
        n_spectra = len(spectra_list)
    else:
        raise ValueError("Input must be a numpy array or list of arrays")
    
    # Find the longest spectrum
    target_length = max(spectra_lengths)
    print(f"Aligning {n_spectra} spectra to length {target_length}")
    print(f"Original lengths range: {min(spectra_lengths)} to {max(spectra_lengths)}")
    
    # Initialize output array
    aligned_spectra = np.zeros((n_spectra, target_length))
    
    for i, spectrum in enumerate(spectra_list):
        current_length = len(spectrum)
        
        if current_length == target_length:
            aligned_spectra[i] = spectrum
            continue
            
        if method == 'interpolate':
            # Use linear interpolation for smooth resizing
            x_old = np.linspace(0, 1, current_length)
            x_new = np.linspace(0, 1, target_length)
            f = interpolate.interp1d(x_old, spectrum, kind='linear', 
                                   bounds_error=False, fill_value=0)
            aligned_spectra[i] = f(x_new)
            
        elif method == 'resample':
            # Use scipy resample (FFT-based)
            aligned_spectra[i] = resample(spectrum, target_length)
            
        elif method == 'duplicate':
            # Simple duplication/averaging method
            if current_length < target_length:
                # Duplicate points
                ratio = target_length / current_length
                indices = np.floor(np.arange(target_length) / ratio).astype(int)
                indices = np.clip(indices, 0, current_length - 1)
                aligned_spectra[i] = spectrum[indices]
            else:
                # This shouldn't happen since we're aligning to the longest
                # But handle it just in case
                ratio = current_length / target_length
                for j in range(target_length):
                    start_idx = int(j * ratio)
                    end_idx = int((j + 1) * ratio)
                    end_idx = min(end_idx, current_length)
                    if start_idx < end_idx:
                        aligned_spectra[i, j] = np.mean(spectrum[start_idx:end_idx])
        
        elif method == 'average':
            # Block averaging method (mainly for upsampling in this case)
            if current_length <= target_length:
                # Interpolate up
                x_old = np.arange(current_length)
                x_new = np.linspace(0, current_length - 1, target_length)
                f = interpolate.interp1d(x_old, spectrum, kind='linear', 
                                       bounds_error=False, fill_value=0)
                aligned_spectra[i] = f(x_new)
            else:
                # This shouldn't happen, but handle it
                block_size = current_length / target_length
                for j in range(target_length):
                    start = int(j * block_size)
                    end = int((j + 1) * block_size)
                    end = min(end, current_length)
                    if start < end:
                        aligned_spectra[i, j] = np.mean(spectrum[start:end])
    
    # Create alignment info
    alignment_info = {
        'original_lengths': spectra_lengths,
        'target_length': target_length,
        'method': method,
        'spectra_count': n_spectra,
        'length_stats': {
            'min': min(spectra_lengths),
            'max': max(spectra_lengths),
            'mean': np.mean(spectra_lengths),
            'std': np.std(spectra_lengths)
        }
    }
    
    return aligned_spectra, alignment_info

def visualize_alignment_results(original_spectra, aligned_spectra, alignment_info, 
                               n_examples=6, figsize=(18, 12)):
    """
    Visualize the results of spectrum alignment
    
    Args:
        original_spectra: list of original spectra (variable lengths)
        aligned_spectra: aligned spectra array
        alignment_info: alignment information dict
        n_examples: number of spectra to visualize
        figsize: figure size
    """
    n_examples = min(n_examples, len(original_spectra))
    
    fig, axes = plt.subplots(n_examples, 2, figsize=figsize)
    if n_examples == 1:
        axes = axes.reshape(1, -1)
    
    # Choose examples with different original lengths if possible
    original_lengths = alignment_info['original_lengths']
    unique_lengths = np.unique(original_lengths)
    
    if len(unique_lengths) >= n_examples:
        # Pick spectra with different lengths
        indices = []
        for length in unique_lengths[:n_examples]:
            idx = original_lengths.index(length)
            indices.append(idx)
    else:
        # Random selection
        indices = np.random.choice(len(original_spectra), n_examples, replace=False)
    
    for plot_idx, spec_idx in enumerate(indices):
        original_length = len(original_spectra[spec_idx])
        
        # Plot 1: Original spectrum
        axes[plot_idx, 0].plot(original_spectra[spec_idx], 'b-', alpha=0.7, linewidth=1)
        axes[plot_idx, 0].set_title(f'Original Spectrum {spec_idx+1}\n'
                                   f'Length: {original_length} points')
        axes[plot_idx, 0].set_xlabel('Point Index')
        axes[plot_idx, 0].set_ylabel('Intensity')
        axes[plot_idx, 0].grid(True, alpha=0.3)
        
        # Plot 2: Aligned spectrum
        axes[plot_idx, 1].plot(aligned_spectra[spec_idx], 'm-', alpha=0.7, linewidth=1)
        scaling_factor = alignment_info['target_length'] / original_length
        axes[plot_idx, 1].set_title(f'Aligned Spectrum {spec_idx+1}\n'
                                   f'Length: {alignment_info["target_length"]} points '
                                   f'(×{scaling_factor:.3f})')
        axes[plot_idx, 1].set_xlabel('Point Index')
        axes[plot_idx, 1].set_ylabel('Intensity')
        axes[plot_idx, 1].grid(True, alpha=0.3)
        
        # Add statistics
        stats_text = f'Mean: {np.mean(aligned_spectra[spec_idx]):.4f}\n' \
                    f'Std: {np.std(aligned_spectra[spec_idx]):.4f}\n' \
                    f'Range: [{np.min(aligned_spectra[spec_idx]):.3f}, {np.max(aligned_spectra[spec_idx]):.3f}]'
        axes[plot_idx, 1].text(0.02, 0.98, stats_text, transform=axes[plot_idx, 1].transAxes,
                              verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    plt.suptitle(f'Spectrum Alignment Results\n'
                 f'Method: {alignment_info["method"]}, Target Length: {alignment_info["target_length"]}', 
                 fontsize=16)
    plt.tight_layout()
    plt.savefig('spectrum_alignment_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print summary statistics
    print("\n" + "="*80)
    print("ALIGNMENT SUMMARY")
    print("="*80)
    
    stats = alignment_info['length_stats']
    print(f"Total spectra aligned: {alignment_info['spectra_count']}")
    print(f"Target length: {alignment_info['target_length']} points")
    print(f"Alignment method: {alignment_info['method']}")
    print(f"\nOriginal lengths:")
    print(f"  Min: {stats['min']} points")
    print(f"  Max: {stats['max']} points") 
    print(f"  Mean: {stats['mean']:.1f} points")
    print(f"  Std: {stats['std']:.1f} points")
    
    # Show scaling factors
    scaling_factors = [alignment_info['target_length'] / length for length in original_lengths]
    print(f"\nScaling factors:")
    print(f"  Min: {min(scaling_factors):.3f}×")
    print(f"  Max: {max(scaling_factors):.3f}×")
    print(f"  Mean: {np.mean(scaling_factors):.3f}×")
    
    return indices

spectra = np.load('data/tbi_tirupati/aligned_128K_TBI_Tirupati_WS625to680Zero.npy')
print(f"Loaded spectra shape: {spectra.shape}")
print("Original shape:", spectra.shape)        
# Remove duplicate rows
spectra = np.unique(spectra, axis=0)
print("After removing duplicates:", spectra.shape)
# Align spectra to longest length
aligned_spectra, alignment_info = align_spectra_to_longest(spectra, method='interpolate')

# Visualize results
visualize_alignment_results(spectra, aligned_spectra, alignment_info, n_examples=8)

# Save the aligned spectra
np.save('aligned_spectra.npy', aligned_spectra)
print(f"\nSaved aligned spectra to 'aligned_spectra.npy' with shape: {aligned_spectra.shape}")
