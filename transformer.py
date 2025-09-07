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

def train_ssl_model(model, dataloader, num_epochs=100, lr=1e-4, device='cuda'):
    """Training loop for SSL model"""
    
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs)
    
    model.train()
    losses = []
    
    for epoch in range(num_epochs):
        epoch_loss = 0
        pbar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{num_epochs}')
        
        for batch in pbar:
            optimizer.zero_grad()
            
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
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            pbar.set_postfix({'Loss': f'{loss.item():.6f}'})
        
        scheduler.step()
        avg_loss = epoch_loss / len(dataloader)
        losses.append(avg_loss)
        
        print(f'Epoch {epoch+1}: Average Loss = {avg_loss:.6f}, LR = {scheduler.get_last_lr()[0]:.6f}')
        
        # Save checkpoint every 20 epochs
        if (epoch + 1) % 20 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, f'nmr_ssl_checkpoint_epoch_{epoch+1}.pth')
    
    return losses

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

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load your NMR spectra
    print("Loading NMR spectra...")
    try:
        spectra = np.load('nmr_spectra.npy')
        print(f"Loaded spectra shape: {spectra.shape}")
    except FileNotFoundError:
        print("nmr_spectra.npy not found. Creating dummy data for demonstration...")
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
    
    # Parameters
    spectrum_length = spectra.shape[1]
    patch_size = 32  # Adjust based on your spectrum resolution
    mask_ratio = 0.25
    batch_size = 32
    
    # Ensure spectrum length is divisible by patch size
    if spectrum_length % patch_size != 0:
        new_length = (spectrum_length // patch_size) * patch_size
        spectra = spectra[:, :new_length]
        spectrum_length = new_length
        print(f"Adjusted spectrum length to {spectrum_length} to be divisible by patch size")
    
    # Create dataset and dataloader
    dataset = NMRSpectrumDataset(spectra, mask_ratio=mask_ratio, patch_size=patch_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    
    print(f"Dataset size: {len(dataset)}")
    print(f"Number of patches per spectrum: {dataset.n_patches}")
    print(f"Patches to mask per spectrum: {int(dataset.n_patches * mask_ratio)}")
    
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
    
    # Train model
    print("Starting training...")
    losses = train_ssl_model(model, dataloader, num_epochs=50, lr=1e-4, device=device)
    
    # Plot training loss
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.yscale('log')
    plt.grid(True)
    plt.savefig('training_loss.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Visualize reconstructions
    print("Generating reconstruction examples...")
    visualize_reconstruction(model, dataset, device=device, n_examples=3)
    
    # Save final model
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
        }
    }, 'nmr_ssl_final_model.pth')
    
    print("Training completed! Model saved as 'nmr_ssl_final_model.pth'")
    
    return model, dataset

if __name__ == "__main__":
    model, dataset = main()
