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

class NMRAugmentation:
    """Realistic NMR-specific augmentations"""
    
    @staticmethod
    def add_noise(spectrum, noise_level=0.02):
        """Add Gaussian noise"""
        noise = torch.randn_like(spectrum) * noise_level
        return spectrum + noise
    
    @staticmethod
    def scale_intensity(spectrum, scale_range=(0.8, 1.2)):
        """Random intensity scaling"""
        scale = random.uniform(*scale_range)
        return spectrum * scale
    
    @staticmethod
    def shift_baseline(spectrum, shift_range=(-0.05, 0.05)):
        """Add baseline shift"""
        shift = random.uniform(*shift_range)
        return spectrum + shift
    
    @staticmethod
    def shift_spectrum(spectrum, max_shift=100):
        """Circular shift (mimics chemical shift calibration errors)"""
        shift = random.randint(-max_shift, max_shift)
        return torch.roll(spectrum, shift, dims=0)
    
    @staticmethod
    def scale_peaks(spectrum, scale_range=(0.9, 1.1), threshold=0.1):
        """Randomly scale individual peaks"""
        mask = torch.abs(spectrum) > threshold
        peak_scale = random.uniform(*scale_range)
        augmented = spectrum.clone()
        augmented[mask] *= peak_scale
        return augmented
    
    @staticmethod
    def add_phase_distortion(spectrum, phase_amount=0.05):
        """Simulate phase correction errors"""
        # Simple linear phase distortion
        n_points = len(spectrum)
        phase = torch.linspace(0, phase_amount, n_points)
        phase_factor = torch.cos(phase * np.pi)
        return spectrum * phase_factor
    
    @staticmethod
    def apply_augmentations(spectrum, num_augmentations=2):
        """Apply random combination of augmentations"""
        augmentations = [
            NMRAugmentation.add_noise,
            NMRAugmentation.scale_intensity,
            NMRAugmentation.shift_baseline,
            NMRAugmentation.shift_spectrum,
            NMRAugmentation.scale_peaks,
            NMRAugmentation.add_phase_distortion
        ]
        
        augmented = spectrum.clone()
        selected_augs = random.sample(augmentations, min(num_augmentations, len(augmentations)))
        
        for aug_fn in selected_augs:
            augmented = aug_fn(augmented)
        
        return augmented

class ContrastiveNMRDataset(Dataset):
    """Dataset for contrastive learning with augmentation"""
    
    def __init__(self, spectra, augment=True):
        self.spectra = torch.FloatTensor(spectra)
        self.augment = augment
        
        # Normalize
        self.spectra = self.normalize_spectra(self.spectra)
        print(f"Contrastive dataset: {len(spectra)} spectra, augmentation={'ON' if augment else 'OFF'}")
    
    def normalize_spectra(self, spectra):
        """Max normalization"""
        normalized_spectra = torch.zeros_like(spectra)
        for i in range(len(spectra)):
            spectrum = spectra[i]
            max_val = torch.max(torch.abs(spectrum))
            if max_val > 1e-8:
                normalized_spectra[i] = spectrum / max_val
            else:
                normalized_spectra[i] = spectrum
        return normalized_spectra
    
    def __len__(self):
        return len(self.spectra)
    
    def __getitem__(self, idx):
        spectrum = self.spectra[idx]
        
        if self.augment:
            # Create two different augmented views
            view1 = NMRAugmentation.apply_augmentations(spectrum, num_augmentations=2)
            view2 = NMRAugmentation.apply_augmentations(spectrum, num_augmentations=2)
        else:
            view1 = spectrum
            view2 = spectrum
        
        return {
            'view1': view1,
            'view2': view2,
            'original': spectrum,
            'idx': idx
        }

class NMREncoder(nn.Module):
    """1D CNN Encoder for NMR spectra"""
    
    def __init__(self, input_length=131072, embedding_dim=128):
        super().__init__()
        
        # Progressive downsampling with 1D convolutions
        self.conv_blocks = nn.ModuleList([
            # Block 1: 131072 -> 65536
            nn.Sequential(
                nn.Conv1d(1, 32, kernel_size=15, stride=2, padding=7),
                nn.BatchNorm1d(32),
                nn.ReLU(),
                nn.Dropout(0.1)
            ),
            # Block 2: 65536 -> 32768
            nn.Sequential(
                nn.Conv1d(32, 64, kernel_size=15, stride=2, padding=7),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Dropout(0.1)
            ),
            # Block 3: 32768 -> 16384
            nn.Sequential(
                nn.Conv1d(64, 128, kernel_size=15, stride=2, padding=7),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Dropout(0.1)
            ),
            # Block 4: 16384 -> 8192
            nn.Sequential(
                nn.Conv1d(128, 256, kernel_size=15, stride=2, padding=7),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Dropout(0.1)
            ),
            # Block 5: 8192 -> 4096
            nn.Sequential(
                nn.Conv1d(256, 256, kernel_size=15, stride=2, padding=7),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Dropout(0.1)
            ),
        ])
        
        # Global pooling and projection
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        self.projection_head = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, embedding_dim)
        )
    
    def forward(self, x):
        # Add channel dimension
        x = x.unsqueeze(1)  # (batch, 1, length)
        
        # Apply conv blocks
        for conv_block in self.conv_blocks:
            x = conv_block(x)
        
        # Global pooling
        x = self.global_pool(x)  # (batch, 256, 1)
        x = x.squeeze(-1)  # (batch, 256)
        
        # Projection
        embedding = self.projection_head(x)
        
        # L2 normalize for contrastive learning
        embedding = F.normalize(embedding, dim=1)
        
        return embedding

class ContrastiveLoss(nn.Module):
    """NT-Xent (Normalized Temperature-scaled Cross Entropy) Loss"""
    
    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, embeddings1, embeddings2):
        """
        embeddings1, embeddings2: (batch_size, embedding_dim)
        """
        batch_size = embeddings1.size(0)
        
        # Concatenate embeddings
        embeddings = torch.cat([embeddings1, embeddings2], dim=0)  # (2*batch, dim)
        
        # Compute similarity matrix
        similarity_matrix = torch.matmul(embeddings, embeddings.T)  # (2*batch, 2*batch)
        similarity_matrix = similarity_matrix / self.temperature
        
        # Create labels: positive pairs are (i, i+batch_size)
        labels = torch.arange(batch_size, device=embeddings.device)
        labels = torch.cat([labels + batch_size, labels], dim=0)
        
        # Mask out self-similarity
        mask = torch.eye(2 * batch_size, device=embeddings.device, dtype=torch.bool)
        similarity_matrix = similarity_matrix.masked_fill(mask, -1e9)
        
        # Compute loss
        loss = F.cross_entropy(similarity_matrix, labels)
        
        return loss

def train_contrastive(model, train_loader, val_loader, num_epochs=200, lr=1e-3, 
                     device='cuda:1', timestamp='', patience=25):
    """Train with contrastive learning"""
    
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    criterion = ContrastiveLoss(temperature=0.5)
    
    # Cosine annealing
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs)
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    
    print(f"Training contrastive model for {num_epochs} epochs")
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        epoch_loss = 0
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        
        for batch in pbar:
            view1 = batch['view1'].to(device)
            view2 = batch['view2'].to(device)
            
            optimizer.zero_grad()
            
            # Get embeddings
            emb1 = model(view1)
            emb2 = model(view2)
            
            # Contrastive loss
            loss = criterion(emb1, emb2)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            pbar.set_postfix({'Loss': f'{loss.item():.4f}', 'LR': f'{scheduler.get_last_lr()[0]:.6f}'})
        
        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                view1 = batch['view1'].to(device)
                view2 = batch['view2'].to(device)
                
                emb1 = model(view1)
                emb2 = model(view2)
                loss = criterion(emb1, emb2)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_without_improvement = 0
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
            }, f'contrastive_best_{timestamp}.pth')
            print(f"✓ Best model saved (val_loss: {avg_val_loss:.4f})")
        else:
            epochs_without_improvement += 1
        
        scheduler.step()
        
        print(f'Epoch {epoch+1}: Train={avg_train_loss:.4f}, Val={avg_val_loss:.4f}, '
              f'Patience={epochs_without_improvement}/{patience}')
        
        if epochs_without_improvement >= patience:
            print("Early stopping")
            break
    
    return model, train_losses, val_losses

def generate_augmented_dataset(spectra, augmentation_factor=5):
    """Generate augmented versions of the dataset"""
    original_spectra = torch.FloatTensor(spectra)
    augmented_list = [original_spectra]
    
    print(f"Generating {augmentation_factor}x augmented data...")
    
    for i in tqdm(range(augmentation_factor - 1), desc="Augmenting"):
        augmented_batch = []
        for spectrum in original_spectra:
            aug_spectrum = NMRAugmentation.apply_augmentations(spectrum, num_augmentations=3)
            augmented_batch.append(aug_spectrum)
        augmented_list.append(torch.stack(augmented_batch))
    
    # Concatenate all
    all_spectra = torch.cat(augmented_list, dim=0)
    
    print(f"Original: {len(original_spectra)}, Augmented: {len(all_spectra)} "
          f"({augmentation_factor}x increase)")
    
    return all_spectra.numpy()

def main():
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    print("Loading NMR spectra...")
    spectra = np.load('data/aligned/aligned_nmr_spectra_128K_WSNoise.npy')
    print(f"Original data shape: {spectra.shape}")
    
    # Clean data
    nan_mask = np.isnan(spectra).any(axis=1)
    inf_mask = np.isinf(spectra).any(axis=1)
    bad_mask = nan_mask | inf_mask
    if bad_mask.any():
        spectra = spectra[~bad_mask]
    
    std_values = np.std(spectra, axis=1)
    zero_std_mask = std_values < 1e-10
    if zero_std_mask.any():
        spectra = spectra[~zero_std_mask]
    
    spectra = np.unique(spectra, axis=0)
    print(f"Cleaned data shape: {spectra.shape}")
    
    # AUGMENT DATA
    augmented_spectra = generate_augmented_dataset(spectra, augmentation_factor=5)
    
    # Split augmented data
    train_spectra, val_spectra = train_test_split(
        augmented_spectra, test_size=0.2, random_state=42, shuffle=True
    )
    
    print(f"Training set: {train_spectra.shape}")
    print(f"Validation set: {val_spectra.shape}")
    
    # Create contrastive datasets
    train_dataset = ContrastiveNMRDataset(train_spectra, augment=True)
    val_dataset = ContrastiveNMRDataset(val_spectra, augment=True)
    
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, 
                            num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, 
                           num_workers=4, pin_memory=True)
    
    # Create CNN encoder
    model = NMREncoder(input_length=spectra.shape[1], embedding_dim=128)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print("\n" + "="*60)
    print("CONTRASTIVE LEARNING with DATA AUGMENTATION")
    print(f"  - Original: {len(spectra)} → Augmented: {len(augmented_spectra)}")
    print(f"  - 1D CNN encoder")
    print(f"  - Contrastive SSL (SimCLR-style)")
    print("="*60 + "\n")
    
    model, train_losses, val_losses = train_contrastive(
        model, train_loader, val_loader, 
        num_epochs=200, lr=1e-3, device=device, 
        timestamp=timestamp, patience=25
    )
    
    # Save final encoder
    torch.save(model.state_dict(), f'nmr_encoder_{timestamp}.pth')
    print(f"Encoder saved to: nmr_encoder_{timestamp}.pth")
    
    return model, train_dataset, val_dataset

if __name__ == "__main__":
    model, train_dataset, val_dataset = main()