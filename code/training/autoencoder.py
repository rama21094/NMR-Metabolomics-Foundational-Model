# -*- coding: utf-8 -*-
"""
Created on Fri Sep  5 00:23:44 2025

@author: shank
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# 1. Load and Preprocess Data
# Load your CPMG NMR spectra from .npy file
data = np.load('data/source/nmr_spectra.npy')  # Replace with your .npy file path
n_samples, n_points = data.shape
print(f"Data shape: {data.shape}")

# Normalize spectra (e.g., min-max normalization to [0, 1])
data_min = np.min(data, axis=1, keepdims=True)
data_max = np.max(data, axis=1, keepdims=True)
data_normalized = (data - data_min) / (data_max - data_min + 1e-10)  # Avoid division by zero

# 2. Create Masked Spectra
def create_masked_spectra(spectra, mask_ratio=0.2, mask_length=100):
    """
    Create masked spectra by randomly masking segments of the spectrum.
    
    Args:
        spectra: NumPy array of shape (n_samples, n_points)
        mask_ratio: Fraction of spectrum to mask (e.g., 0.2 for 20%)
        mask_length: Length of each masked segment in points
        
    Returns:
        masked_spectra: Spectra with masked regions (same shape as input)
        masks: Binary mask indicating masked regions (1 for masked, 0 for unmasked)
    """
    masked_spectra = spectra.copy()
    masks = np.zeros_like(spectra)
    
    for i in range(spectra.shape[0]):
        n_masks = int(mask_ratio * n_points / mask_length)  # Number of segments to mask
        for _ in range(n_masks):
            start_idx = np.random.randint(0, n_points - mask_length)
            masked_spectra[i, start_idx:start_idx + mask_length] = 0
            masks[i, start_idx:start_idx + mask_length] = 1
    
    return masked_spectra, masks

# Generate masked spectra
mask_ratio = 0.2  # Mask 20% of the spectrum
mask_length = 100  # Length of each masked segment (adjust based on your data)
masked_data, masks = create_masked_spectra(data_normalized, mask_ratio, mask_length)

# 3. Define Convolutional Autoencoder Model
def build_autoencoder(input_shape):
    """
    Build a 1D convolutional autoencoder for spectral inpainting.
    
    Args:
        input_shape: Tuple of (n_points,) for input spectrum
        
    Returns:
        model: Keras autoencoder model
    """
    inputs = layers.Input(shape=input_shape)
    
    # Encoder
    x = layers.Conv1D(64, kernel_size=5, padding='same', activation='relu')(inputs)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Conv1D(32, kernel_size=5, padding='same', activation='relu')(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    encoded = layers.Conv1D(16, kernel_size=5, padding='same', activation='relu')(x)
    
    # Decoder
    x = layers.Conv1D(16, kernel_size=5, padding='same', activation='relu')(encoded)
    x = layers.UpSampling1D(size=2)(x)
    x = layers.Conv1D(32, kernel_size=5, padding='same', activation='relu')(x)
    x = layers.UpSampling1D(size=2)(x)
    decoded = layers.Conv1D(1, kernel_size=5, padding='same', activation='sigmoid')(x)
    
    # Build model
    autoencoder = models.Model(inputs, decoded)
    autoencoder.compile(optimizer='adam', loss='mse')
    return autoencoder

# Reshape data for Conv1D (add channel dimension)
input_shape = (n_points, 1)
data_normalized = data_normalized.reshape(-1, n_points, 1)
masked_data = masked_data.reshape(-1, n_points, 1)

# Build and summarize model
autoencoder = build_autoencoder(input_shape)
autoencoder.summary()

# 4. Train the Model
# Train the autoencoder to reconstruct original spectra from masked spectra
history = autoencoder.fit(
    masked_data, data_normalized,
    epochs=50,
    batch_size=32,
    validation_split=0.2,
    shuffle=True
)

# 5. Evaluate the Model
# Predict on a few test spectra
n_examples = 5
test_spectra = masked_data[:n_examples]
reconstructed_spectra = autoencoder.predict(test_spectra)

# Denormalize reconstructed spectra for visualization
reconstructed_spectra = reconstructed_spectra * (data_max[:n_examples] - data_min[:n_examples]) + data_min[:n_examples]
test_spectra = test_spectra * (data_max[:n_examples] - data_min[:n_examples]) + data_min[:n_examples]
original_spectra = data[:n_examples]

# 6. Visualize Results
plt.figure(figsize=(15, 5 * n_examples))
for i in range(n_examples):
    plt.subplot(n_examples, 1, i + 1)
    plt.plot(original_spectra[i], label='Original Spectrum', alpha=0.8)
    plt.plot(test_spectra[i].squeeze(), label='Masked Spectrum', alpha=0.6)
    plt.plot(reconstructed_spectra[i].squeeze(), label='Reconstructed Spectrum', alpha=0.6)
    plt.legend()
    plt.title(f'Spectrum {i+1}')
plt.tight_layout()
plt.show()

# Plot training history
plt.figure(figsize=(10, 4))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Mean Squared Error')
plt.legend()
plt.title('Training and Validation Loss')
plt.show()
