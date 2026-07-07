# -*- coding: utf-8 -*-
"""
Created on Mon Sep 22 03:19:31 2025

@author: shank
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Sep  5 10:36:52 2025

@author: shank
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import random
from tqdm import tqdm
import math
import datetime

class NMRSpectrumDataset(Dataset):
    """Dataset for NMR spectra with masking for self-supervised learning"""
    
    def __init__(self, spectra, mask_ratio=0.25, patch_size=16):
        """
        Args:
            spectra: numpy array of shape (n_samples, n_points)
            mask_ratio: fraction of patches to mask
            patch_size: size of each patch for masking
        """
        self.spectra = torch.FloatTensor(spectra)
        self.mask_ratio = mask_ratio
        self.patch_size = patch_size
        self.n_patches = spectra.shape[1] // patch_size
        
        # Normalize spectra
        self.spectra = self.normalize_spectra(self.spectra)
        
    def normalize_spectra(self, spectra):
        """Normalize each spectrum individually"""
        # Z-score normalization per spectrum
        mean = spectra.mean(dim=1, keepdim=True)
        std = spectra.std(dim=1, keepdim=True) + 1e-8
        return (spectra - mean) / std
    
    def __len__(self):
        return len(self.spectra)
    
    def __getitem__(self, idx):
        spectrum = self.spectra[idx]
        
        # Create patches
        patches = spectrum.unfold(0, self.patch_size, self.patch_size)
        
        # Create mask
        mask = torch.zeros(self.n_patches, dtype=torch.bool)
        n_masked = int(self.n_patches * self.mask_ratio)
        masked_indices = torch.randperm(self.n_patches)[:n_masked]
        mask[masked_indices] = True
        
        # Apply mask to spectrum
        masked_spectrum = spectrum.clone()
        for i, is_masked in enumerate(mask):
            if is_masked:
                start_idx = i * self.patch_size
                end_idx = start_idx + self.patch_size
                masked_spectrum[start_idx:end_idx] = 0  # Zero out masked regions
        
        return {
            'original': spectrum,
            'masked': masked_spectrum,
            'mask': mask,
            'patches': patches
        }

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for transformer"""
    
    def __init__(self, d_model, max_len=10000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class NMRTransformerEncoder(nn.Module):
    """Transformer-based encoder for NMR spectra"""
    
    def __init__(self, spectrum_length, patch_size=16, d_model=256, nhead=8, 
                 num_layers=6, dim_feedforward=1024, dropout=0.1):
        super().__init__()
        
        self.patch_size = patch_size
        self.d_model = d_model
        self.n_patches = spectrum_length // patch_size
        
        # Patch embedding
        self.patch_embedding = nn.Linear(patch_size, d_model)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, self.n_patches)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Reconstruction head
        self.reconstruction_head = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, patch_size)
        )
        
        # Mask token (learnable parameter for masked patches)
        self.mask_token = nn.Parameter(torch.randn(1, 1, d_model))
        
    def forward(self, x, mask=None):
        batch_size, seq_len = x.shape
        
        # Create patches
        patches = x.unfold(1, self.patch_size, self.patch_size)  # (batch, n_patches, patch_size)
        
        # Embed patches
        embeddings = self.patch_embedding(patches)  # (batch, n_patches, d_model)
        
        # Add mask tokens for masked patches
        if mask is not None:
            mask_tokens = self.mask_token.expand(batch_size, -1, -1)
            for i in range(batch_size):
                embeddings[i, mask[i]] = mask_tokens[i]
        
        # Add positional encoding
        embeddings = embeddings.transpose(0, 1)  # (n_patches, batch, d_model)
        embeddings = self.pos_encoding(embeddings)
        embeddings = embeddings.transpose(0, 1)  # (batch, n_patches, d_model)
        
        # Apply transformer
        encoded = self.transformer(embeddings)
        
        # Reconstruct patches
        reconstructed_patches = self.reconstruction_head(encoded)
        
        # Reshape back to spectrum format
        reconstructed = reconstructed_patches.reshape(batch_size, -1)
        
        return reconstructed, encoded

class NMRMaskedAutoencoder(nn.Module):
    """Complete Masked Autoencoder for NMR spectra"""
    
    def __init__(self, spectrum_length, patch_size=16, **kwargs):
        super().__init__()
        self.encoder = NMRTransformerEncoder(spectrum_length, patch_size, **kwargs)
        self.patch_size = patch_size
        
    def forward(self, x, mask=None):
        return self.encoder(x, mask)

def compute_loss(model, batch, device):
    """Compute reconstruction loss for a batch"""
    original = batch['original'].to(device)
    masked = batch['masked'].to(device)
    mask = batch['mask'].to(device)
    
    # Forward pass
    reconstructed, _ = model(masked, mask)
    
    # Calculate loss only on masked regions
    loss = 0
    batch_size = original.size(0)
    
    for i in range(batch_size):
        masked_indices = mask[i]
        if masked_indices.any():
            # Get masked patches from original and reconstructed
            for patch_idx in torch.where(masked_indices)[0]:
                start = patch_idx * model.patch_size
                end = start + model.patch_size
                loss += F.mse_loss(
                    reconstructed[i, start:end], 
                    original[i, start:end]
                )
    
    loss = loss / batch_size
    return loss

def validate_model(model, val_dataloader, device):
    """Validate the model and return average validation loss"""
    model.eval()
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch in val_dataloader:
            loss = compute_loss(model, batch, device)
            total_loss += loss.item()
            num_batches += 1
    
    avg_val_loss = total_loss / num_batches
    return avg_val_loss

def train_ssl_model(model, train_dataloader, val_dataloader=None, num_epochs=100, 
                   lr=1e-4, device='cuda', scheduler_type='cosine', patience=5):
    """
    Training loop for SSL model with validation and flexible scheduler
    
    Args:
        model: The model to train
        train_dataloader: Training data loader
        val_dataloader: Validation data loader (optional)
        num_epochs: Number of training epochs
        lr: Learning rate
        device: Device to use
        scheduler_type: 'cosine' or 'plateau'
        patience: Patience for ReduceLROnPlateau scheduler
    """
    
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    
    # Choose scheduler
    if scheduler_type.lower() == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs)
    elif scheduler_type.lower() == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=patience
        )
    else:
        raise ValueError("scheduler_type must be 'cosine' or 'plateau'")
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    print(f"Using {scheduler_type} scheduler")
    if val_dataloader is not None:
        print(f"Training with validation set ({len(val_dataloader.dataset)} samples)")
    else:
        print("Training without validation set")
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        epoch_train_loss = 0
        pbar = tqdm(train_dataloader, desc=f'Epoch {epoch+1}/{num_epochs}')
        
        for batch in pbar:
            optimizer.zero_grad()
            
            loss = compute_loss(model, batch, device)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_train_loss += loss.item()
            pbar.set_postfix({'Train Loss': f'{loss.item():.6f}'})
        
        avg_train_loss = epoch_train_loss / len(train_dataloader)
        train_losses.append(avg_train_loss)
        
        current_time = datetime.datetime.now() 
        timestamp = current_time.strftime("%Y-%m-%d_%H-%M-%S")
        # Validation phase
        val_loss_str = ""
        if val_dataloader is not None:
            avg_val_loss = validate_model(model, val_dataloader, device)
            val_losses.append(avg_val_loss)
            val_loss_str = f", Val Loss = {avg_val_loss:.6f}"
            
            # Save best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': avg_train_loss,
                    'val_loss': avg_val_loss,
                }, 'best_model_{timestamp}.pth')
        
        # Update scheduler
        if scheduler_type.lower() == 'cosine':
            scheduler.step()
            lr_str = f", LR = {scheduler.get_last_lr()[0]:.6f}"
        elif scheduler_type.lower() == 'plateau':
            if val_dataloader is not None:
                scheduler.step(avg_val_loss)
            else:
                scheduler.step(avg_train_loss)
            lr_str = f", LR = {optimizer.param_groups[0]['lr']:.6f}"
        
        print(f'Epoch {epoch+1}: Train Loss = {avg_train_loss:.6f}{val_loss_str}{lr_str}')
        
        # Save checkpoint every 20 epochs
        if (epoch + 1) % 20 == 0:
            checkpoint_data = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'train_losses': train_losses,
            }
            if val_dataloader is not None:
                checkpoint_data['val_loss'] = avg_val_loss
                checkpoint_data['val_losses'] = val_losses
            
            torch.save(checkpoint_data, f'nmr_ssl_checkpoint_epoch_{epoch+1}.pth')
    
    return train_losses, val_losses

def plot_training_curves(train_losses, val_losses=None, save_path='training_curves.png'):
    """Plot training and validation loss curves"""
    plt.figure(figsize=(12, 5))
    
    # Loss plot
    plt.subplot(1, 2 if val_losses else 1, 1)
    plt.plot(train_losses, label='Training Loss', color='blue')
    if val_losses:
        plt.plot(val_losses, label='Validation Loss', color='red')
    plt.title('Training Progress')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.yscale('log')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Learning rate plot (if we have validation data, show overfitting detection)
    if val_losses:
        plt.subplot(1, 2, 2)
        # Calculate smoothed losses for trend detection
        smooth_train = np.convolve(train_losses, np.ones(5)/5, mode='valid')
        smooth_val = np.convolve(val_losses, np.ones(5)/5, mode='valid')
        
        plt.plot(range(len(smooth_train)), smooth_train, label='Smoothed Train', alpha=0.8)
        plt.plot(range(len(smooth_val)), smooth_val, label='Smoothed Val', alpha=0.8)
        plt.title('Smoothed Loss Trends')
        plt.xlabel('Epoch (smoothed)')
        plt.ylabel('MSE Loss')
        plt.yscale('log')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def visualize_reconstruction(model, dataset, device='cuda', n_examples=3):
    """Visualize reconstruction results"""
    model.eval()
    
    fig, axes = plt.subplots(n_examples, 3, figsize=(15, 4*n_examples))
    if n_examples == 1:
        axes = axes.reshape(1, -1)
    
    with torch.no_grad():
        for i in range(n_examples):
            sample = dataset[i]
            original = sample['original'].unsqueeze(0).to(device)
            masked = sample['masked'].unsqueeze(0).to(device)
            mask = sample['mask'].unsqueeze(0).to(device)
            
            reconstructed, _ = model(masked, mask)
            
            # Convert back to numpy
            original_np = original.cpu().numpy().flatten()
            masked_np = masked.cpu().numpy().flatten()
            reconstructed_np = reconstructed.cpu().numpy().flatten()
            
            # Plot
            axes[i, 0].plot(original_np)
            axes[i, 0].set_title(f'Original Spectrum {i+1}')
            axes[i, 0].set_xlabel('Frequency Point')
            axes[i, 0].set_ylabel('Intensity')
            
            axes[i, 1].plot(masked_np)
            axes[i, 1].set_title(f'Masked Spectrum {i+1}')
            axes[i, 1].set_xlabel('Frequency Point')
            axes[i, 1].set_ylabel('Intensity')
            
            axes[i, 2].plot(original_np, label='Original', alpha=0.7)
            axes[i, 2].plot(reconstructed_np, label='Reconstructed', alpha=0.7)
            axes[i, 2].set_title(f'Reconstruction Comparison {i+1}')
            axes[i, 2].set_xlabel('Frequency Point')
            axes[i, 2].set_ylabel('Intensity')
            axes[i, 2].legend()
    
    plt.tight_layout()
    plt.savefig('reconstruction_examples.png', dpi=300, bbox_inches='tight')
    plt.show()

def remove_zero_tails(spectra, threshold=1e-6):
    """
    Remove zero tails from NMR spectra
    
    Args:
        spectra: numpy array of shape (n_samples, n_points)
        threshold: threshold below which values are considered zero
    
    Returns:
        trimmed_spectra: numpy array with zero tails removed
        trim_length: length after trimming
    """
    # Find the last non-zero point across all spectra
    abs_spectra = np.abs(spectra)
    max_nonzero_idx = 0
    
    for i in range(len(spectra)):
        # Find last significant point in this spectrum
        nonzero_indices = np.where(abs_spectra[i] > threshold)[0]
        if len(nonzero_indices) > 0:
            last_nonzero = nonzero_indices[-1]
            max_nonzero_idx = max(max_nonzero_idx, last_nonzero)
    
    # Add some padding to avoid cutting important data
    padding = min(100, spectra.shape[1] // 20)  # 5% padding or 100 points, whichever is smaller
    trim_length = min(max_nonzero_idx + padding, spectra.shape[1])
    
    print(f"Trimming spectra from {spectra.shape[1]} to {trim_length} points")
    print(f"Removed {spectra.shape[1] - trim_length} trailing zeros")
    
    return spectra[:, :trim_length], trim_length

def main():
    # Set device
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load your NMR spectra
    print("Loading NMR spectra...")
    try:
        spectra = np.load('data/aligned/aligned_nmr_spectra_128K_WSZero.npy')
        print(f"Loaded spectra shape: {spectra.shape}")
        print("Original shape:", spectra.shape)        
        # Remove duplicate rows
        spectra = np.unique(spectra, axis=0)
        print("After removing duplicates:", spectra.shape)
        
        print(f"Original spectra shape: {spectra.shape}")
        
        # # Remove zero tails
        # spectra, trimmed_length = remove_zero_tails(spectra)
        # print(f"Trimmed spectra shape: {spectra.shape}")
        
    except FileNotFoundError:
        print("data/source/water_suppressed_data/source/nmr_spectra.npy not found. Creating dummy data for demonstration...")
        # Create dummy data for demonstration
        n_samples, n_points = 1000, 2048
        spectra = np.random.randn(n_samples, n_points)
        # Add some realistic NMR-like patterns
        for i in range(n_samples):
            # Add some peaks
            for _ in range(np.random.randint(5, 15)):
                center = np.random.randint(100, n_points-100)
                width = np.random.randint(5, 20)
                height = np.random.uniform(0.5, 2.0)
                x = np.arange(n_points)
                spectra[i] += height * np.exp(-((x - center) / width) ** 2)
        print(f"Created dummy spectra shape: {spectra.shape}")
    
    # Split data into train and validation sets (80/20 split)
    train_spectra, val_spectra = train_test_split(
        spectra, test_size=0.2, random_state=42, shuffle=True
    )
    
    print(f"Training set shape: {train_spectra.shape}")
    print(f"Validation set shape: {val_spectra.shape}")
    
    # Parameters
    spectrum_length = spectra.shape[1]
    patch_size = 256  # Adjust based on your spectrum resolution
    mask_ratio = 0.2
    batch_size = 128
    
    # Ensure spectrum length is divisible by patch size
    if spectrum_length % patch_size != 0:
        new_length = (spectrum_length // patch_size) * patch_size
        train_spectra = train_spectra[:, :new_length]
        val_spectra = val_spectra[:, :new_length]
        spectrum_length = new_length
        print(f"Adjusted spectrum length to {spectrum_length} to be divisible by patch size")
    
    # Create datasets and dataloaders
    train_dataset = NMRSpectrumDataset(train_spectra, mask_ratio=mask_ratio, patch_size=patch_size)
    val_dataset = NMRSpectrumDataset(val_spectra, mask_ratio=mask_ratio, patch_size=patch_size)
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    print(f"Training dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    print(f"Number of patches per spectrum: {train_dataset.n_patches}")
    print(f"Patches to mask per spectrum: {int(train_dataset.n_patches * mask_ratio)}")
    
    # Create model
    model = NMRMaskedAutoencoder(
        spectrum_length=spectrum_length,
        patch_size=patch_size,
        d_model=256,
        nhead=8,
        num_layers=6,
        dim_feedforward=1024,
        dropout=0.1
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Training parameters
    num_epochs = 300
    learning_rate = 1e-3
    scheduler_type = 'plateau'  # Change to 'plateau' if desired
    
    # Train model
    print(f"Starting training with {scheduler_type} scheduler...")
    train_losses, val_losses = train_ssl_model(
        model=model, 
        train_dataloader=train_dataloader, 
        val_dataloader=val_dataloader,
        num_epochs=num_epochs, 
        lr=learning_rate, 
        device=device,
        scheduler_type=scheduler_type,
        patience=10  # Only used for plateau scheduler
    )
    
    # Plot training curves
    plot_training_curves(train_losses, val_losses, 'training_validation_curves.png')
    
    # Visualize reconstructions
    print("Generating reconstruction examples...")
    visualize_reconstruction(model, val_dataset, device=device, n_examples=3)
    
    # Get the current date and time
    current_time = datetime.datetime.now()    
    # Format the time into a string (e.g., '2025-09-22_14-30-00')
    timestamp = current_time.strftime("%Y-%m-%d_%H-%M-%S")
    
    # Save final model
    final_model_path = f'nmr_{scheduler_type}LR_{num_epochs}epoch_ps{patch_size}_bs{batch_size}_mr{mask_ratio}_{timestamp}.pth'
    torch.save({
        'model_state_dict': model.state_dict(),
        'spectrum_length': spectrum_length,
        'patch_size': patch_size,
        'model_config': {
            'd_model': 256,
            'nhead': 8,
            'num_layers': 6,
            'dim_feedforward': 1024,
            'dropout': 0.1
        },
        'train_losses': train_losses,
        'val_losses': val_losses,
        'scheduler_type': scheduler_type
    }, final_model_path)
    
    print(f"Training completed! Final model saved as '{final_model_path}'")
    print(f"Best model saved as 'best_model.pth'")
    
    return model, train_dataset, val_dataset

if __name__ == "__main__":
    model, train_dataset, val_dataset = main()
