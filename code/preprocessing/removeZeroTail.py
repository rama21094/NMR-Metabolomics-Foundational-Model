# -*- coding: utf-8 -*-
"""
Created on Wed Sep 10 18:13:48 2025

@author: shank
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from scipy.signal import resample

def find_spectrum_end(spectrum, threshold=1e-6, min_consecutive=50):
    """
    Find the end of meaningful signal in a single spectrum
    
    Args:
        spectrum: 1D numpy array
        threshold: threshold below which values are considered zero
        min_consecutive: minimum number of consecutive zeros to consider as tail
    
    Returns:
        end_idx: index where meaningful signal ends
    """
    abs_spectrum = np.abs(spectrum)
    
    # Find regions below threshold
    below_threshold = abs_spectrum <= threshold
    
    # Find the last point that's above threshold
    above_threshold_indices = np.where(~below_threshold)[0]
    
    if len(above_threshold_indices) == 0:
        return len(spectrum) // 2  # If all zeros, return middle point
    
    last_signal_idx = above_threshold_indices[-1]
    
    # Look for consecutive zeros after the last signal
    start_search = min(last_signal_idx + 1, len(spectrum) - min_consecutive)
    
    for i in range(start_search, len(spectrum) - min_consecutive + 1):
        if np.all(below_threshold[i:i + min_consecutive]):
            return i
    
    # If no consecutive zeros found, return the last signal index with some padding
    padding = min(100, len(spectrum) // 20)
    return min(last_signal_idx + padding, len(spectrum))

def align_spectra_length(spectra_list, target_length, method='interpolate'):
    """
    Align all spectra to the same length
    
    Args:
        spectra_list: list of 1D numpy arrays (different lengths)
        target_length: desired final length
        method: 'interpolate', 'resample', 'duplicate', or 'average'
    
    Returns:
        aligned_spectra: numpy array of shape (n_spectra, target_length)
    """
    aligned_spectra = np.zeros((len(spectra_list), target_length))
    
    for i, spectrum in enumerate(spectra_list):
        current_length = len(spectrum)
        
        if current_length == target_length:
            aligned_spectra[i] = spectrum
        elif method == 'interpolate':
            # Use interpolation for smooth resizing
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
                # Average points
                ratio = current_length / target_length
                for j in range(target_length):
                    start_idx = int(j * ratio)
                    end_idx = int((j + 1) * ratio)
                    end_idx = min(end_idx, current_length)
                    if start_idx < end_idx:
                        aligned_spectra[i, j] = np.mean(spectrum[start_idx:end_idx])
        
        elif method == 'average':
            # Block averaging method
            if current_length <= target_length:
                # Interpolate up
                x_old = np.arange(current_length)
                x_new = np.linspace(0, current_length - 1, target_length)
                f = interpolate.interp1d(x_old, spectrum, kind='linear', 
                                       bounds_error=False, fill_value=0)
                aligned_spectra[i] = f(x_new)
            else:
                # Average down
                block_size = current_length / target_length
                for j in range(target_length):
                    start = int(j * block_size)
                    end = int((j + 1) * block_size)
                    end = min(end, current_length)
                    if start < end:
                        aligned_spectra[i, j] = np.mean(spectrum[start:end])
    
    return aligned_spectra

def advanced_zero_tail_removal(spectra, target_length=65536, threshold=1e-6, 
                             alignment_method='interpolate', min_consecutive=50,
                             preserve_ratio=True):
    """
    Advanced zero tail removal with alignment
    
    Args:
        spectra: numpy array of shape (n_samples, n_points)
        target_length: desired final length for all spectra
        threshold: threshold for considering values as zero
        alignment_method: method for length alignment
        min_consecutive: minimum consecutive zeros to consider as tail
        preserve_ratio: if True, preserve aspect ratio when resizing
    
    Returns:
        processed_spectra: numpy array of shape (n_samples, target_length)
        processing_info: dict with processing information for each spectrum
    """
    n_samples, original_length = spectra.shape
    trimmed_spectra = []
    processing_info = []
    
    print(f"Processing {n_samples} spectra from {original_length} points to {target_length} points")
    
    # Step 1: Remove zero tails from each spectrum individually
    for i in range(n_samples):
        spectrum = spectra[i]
        
        # Find where meaningful signal ends
        end_idx = find_spectrum_end(spectrum, threshold, min_consecutive)
        
        # Trim the spectrum
        trimmed_spectrum = spectrum[:end_idx]
        trimmed_spectra.append(trimmed_spectrum)
        
        # Store processing info
        info = {
            'original_length': original_length,
            'trimmed_length': end_idx,
            'removed_points': original_length - end_idx,
            'compression_ratio': end_idx / target_length if target_length > 0 else 1.0
        }
        processing_info.append(info)
        
        if i < 5:  # Print info for first 5 spectra
            print(f"Spectrum {i+1}: {original_length} → {end_idx} points "
                  f"(removed {original_length - end_idx}, ratio: {info['compression_ratio']:.3f})")
    
    # Step 2: Align all spectra to target length
    print(f"\nAligning spectra using method: {alignment_method}")
    aligned_spectra = align_spectra_length(trimmed_spectra, target_length, alignment_method)
    
    # Update processing info with final statistics
    for i, info in enumerate(processing_info):
        final_spectrum = aligned_spectra[i]
        info.update({
            'final_length': target_length,
            'final_mean': np.mean(final_spectrum),
            'final_std': np.std(final_spectrum),
            'final_range': (np.min(final_spectrum), np.max(final_spectrum)),
            'alignment_method': alignment_method
        })
    
    return aligned_spectra, processing_info

def visualize_processing_results(original_spectra, processed_spectra, processing_info, 
                               n_examples=8, figsize=(20, 16)):
    """
    Visualize the results of zero tail removal and alignment
    
    Args:
        original_spectra: original spectra array
        processed_spectra: processed spectra array
        processing_info: list of processing information dicts
        n_examples: number of spectra to visualize
        figsize: figure size
    """
    n_examples = min(n_examples, len(original_spectra), len(processed_spectra))
    
    fig, axes = plt.subplots(n_examples, 3, figsize=figsize)
    if n_examples == 1:
        axes = axes.reshape(1, -1)
    
    # Choose random indices for visualization
    indices = np.random.choice(len(original_spectra), n_examples, replace=False)
    
    for plot_idx, spec_idx in enumerate(indices):
        info = processing_info[spec_idx]
        
        # Plot 1: Original spectrum
        axes[plot_idx, 0].plot(original_spectra[spec_idx], 'b-', alpha=0.7, linewidth=1)
        axes[plot_idx, 0].axvline(x=info['trimmed_length'], color='r', linestyle='--', 
                                 label=f'Trim at {info["trimmed_length"]}')
        axes[plot_idx, 0].set_title(f'Original Spectrum {spec_idx+1}\n'
                                   f'Length: {info["original_length"]} points')
        axes[plot_idx, 0].set_xlabel('Point Index')
        axes[plot_idx, 0].set_ylabel('Intensity')
        axes[plot_idx, 0].legend()
        axes[plot_idx, 0].grid(True, alpha=0.3)
        
        # Plot 2: Trimmed spectrum (before alignment)
        trimmed_length = info['trimmed_length']
        trimmed_spectrum = original_spectra[spec_idx][:trimmed_length]
        axes[plot_idx, 1].plot(trimmed_spectrum, 'g-', alpha=0.7, linewidth=1)
        axes[plot_idx, 1].set_title(f'Trimmed Spectrum {spec_idx+1}\n'
                                   f'Length: {trimmed_length} points '
                                   f'(removed {info["removed_points"]})')
        axes[plot_idx, 1].set_xlabel('Point Index')
        axes[plot_idx, 1].set_ylabel('Intensity')
        axes[plot_idx, 1].grid(True, alpha=0.3)
        
        # Plot 3: Final processed spectrum
        axes[plot_idx, 2].plot(processed_spectra[spec_idx], 'm-', alpha=0.7, linewidth=1)
        axes[plot_idx, 2].set_title(f'Final Aligned Spectrum {spec_idx+1}\n'
                                   f'Length: {info["final_length"]} points '
                                   f'(ratio: {info["compression_ratio"]:.3f})')
        axes[plot_idx, 2].set_xlabel('Point Index')
        axes[plot_idx, 2].set_ylabel('Intensity')
        axes[plot_idx, 2].grid(True, alpha=0.3)
        
        # Add statistics text
        stats_text = f'Mean: {info["final_mean"]:.4f}\n' \
                    f'Std: {info["final_std"]:.4f}\n' \
                    f'Range: [{info["final_range"][0]:.3f}, {info["final_range"][1]:.3f}]'
        axes[plot_idx, 2].text(0.02, 0.98, stats_text, transform=axes[plot_idx, 2].transAxes,
                              verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle(f'Zero Tail Removal and Alignment Results\n'
                 f'Method: {processing_info[0]["alignment_method"]}, '
                 f'Target Length: {processing_info[0]["final_length"]}', fontsize=16)
    plt.tight_layout()
    plt.savefig('zero_tail_processing_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print summary statistics
    print("\n" + "="*80)
    print("PROCESSING SUMMARY")
    print("="*80)
    
    original_lengths = [info['original_length'] for info in processing_info]
    trimmed_lengths = [info['trimmed_length'] for info in processing_info]
    removed_points = [info['removed_points'] for info in processing_info]
    compression_ratios = [info['compression_ratio'] for info in processing_info]
    
    print(f"Total spectra processed: {len(processing_info)}")
    print(f"Original length (all): {original_lengths[0]} points")
    print(f"Target length: {processing_info[0]['final_length']} points")
    print(f"Alignment method: {processing_info[0]['alignment_method']}")
    print(f"\nTrimmed lengths - Min: {min(trimmed_lengths)}, Max: {max(trimmed_lengths)}, "
          f"Mean: {np.mean(trimmed_lengths):.1f}")
    print(f"Points removed - Min: {min(removed_points)}, Max: {max(removed_points)}, "
          f"Mean: {np.mean(removed_points):.1f}")
    print(f"Compression ratios - Min: {min(compression_ratios):.3f}, Max: {max(compression_ratios):.3f}, "
          f"Mean: {np.mean(compression_ratios):.3f}")
    
    return indices

# Example usage function
def test_advanced_processing():
    """Test function with dummy data"""
    print("Testing advanced zero tail removal with dummy data...")
    
    # Create dummy spectra with different zero tail lengths
    n_samples = 20
    original_length = 131072
    spectra = np.zeros((n_samples, original_length))
    
    # Add realistic patterns with different active regions
    for i in range(n_samples):
        # Random active region length
        active_length = np.random.randint(30000, 80000)
        
        # Add some peaks in the active region
        for _ in range(np.random.randint(10, 25)):
            center = np.random.randint(1000, active_length-1000)
            width = np.random.randint(50, 200)
            height = np.random.uniform(0.5, 3.0)
            x = np.arange(active_length)
            peak = height * np.exp(-((x - center) / width) ** 2)
            spectra[i, :active_length] += peak
        
        # Add some noise to the active region
        noise_level = np.random.uniform(0.01, 0.1)
        spectra[i, :active_length] += np.random.normal(0, noise_level, active_length)
        
        # Add very small residual noise to some "zero" regions
        if np.random.random() < 0.3:  # 30% chance
            residual_end = min(active_length + np.random.randint(5000, 20000), original_length)
            residual_noise = np.random.normal(0, 1e-8, residual_end - active_length)
            spectra[i, active_length:residual_end] = residual_noise
    
    print(f"Created {n_samples} dummy spectra with length {original_length}")
    
    # Process the spectra
    target_length = 65536
    processed_spectra, processing_info = advanced_zero_tail_removal(
        spectra, 
        target_length=target_length,
        threshold=1e-6,
        alignment_method='interpolate',
        min_consecutive=100
    )
    
    # Visualize results
    visualize_processing_results(spectra, processed_spectra, processing_info, n_examples=8)
    
    return spectra, processed_spectra, processing_info

if __name__ == "__main__":
    # Test with dummy data
    test_advanced_processing()
    spectra = np.load('data/source/nmr_spectra.npy')
    print(f"Loaded spectra shape: {spectra.shape}")
    print("Original shape:", spectra.shape)        
    # Remove duplicate rows
    spectra = np.unique(spectra, axis=0)
    print("After removing duplicates:", spectra.shape)
    start_idx = 32200
    end_idx = 33224        
    # Create a boolean mask or use slicing to keep all points except those in [32200, 33224)
    # Method 1: Using np.delete
    spectra = np.delete(spectra, slice(start_idx, end_idx), axis=1)
    print(f"Original spectra shape: {spectra.shape}")    
    # Process the spectra
    target_length = 65536
    processed_spectra, processing_info = advanced_zero_tail_removal(
        spectra, 
        target_length=target_length,
        threshold=1e-6,
        alignment_method='interpolate',
        min_consecutive=100
    )
    
    # Visualize results
    visualize_processing_results(spectra, processed_spectra, processing_info, n_examples=8)
    # # Remove zero tails
    # spectra, trimmed_length = remove_zero_tails(spectra)
    # print(f"Trimmed spectra shape: {spectra.shape}")
