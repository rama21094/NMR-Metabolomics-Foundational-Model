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
import os
import datetime
import os
import sys

# Global dataset split fractions (modifiable)
# These should sum to 1.0 (train + val + test). Modify as needed.
TRAIN_SPLIT = 0.80
VAL_SPLIT = 0.20
TEST_SPLIT = 0

class NMRSpectrumDataset(Dataset):
    """Dataset for NMR spectra with masking for self-supervised learning"""

    def __init__(self, spectra, mask_ratio_min=0.20, mask_ratio_max=0.60, patch_size=256, mask_strategy='sparse_random', mask_fill='zero',
                 augment=False, per_point_std=None, noise_scale=1.0, normalize_input=True,
                 baseline_window_start=62500, baseline_window_end=68000,
                 correct_post_mask_baseline=True, baseline_tol=1e-8):
        """
        Args:
            spectra: numpy array of shape (n_samples, n_points)
            mask_ratio_min: lower bound of the per-sample masking fraction, drawn
                uniformly at random on every __getitem__ call
            mask_ratio_max: upper bound of the per-sample masking fraction
            patch_size: size of each patch for masking (SMALLER for finer granularity)
            mask_strategy: 'sparse_random', 'scattered_peaks', or 'random'
            mask_fill: 'zero', 'mean', 'noise', or a float value (for masked patch fill)
            augment: whether to apply data augmentation
            per_point_std: tensor of per-point std for augmentation
            noise_scale: scale factor for augmentation noise
            normalize_input: if True, apply per-spectrum min-max normalization
            baseline_window_start: start index (inclusive) of known zero baseline window
            baseline_window_end: end index (exclusive) of known zero baseline window
            correct_post_mask_baseline: if True, re-center unmasked points so known zero window stays near zero
            baseline_tol: tolerance for applying baseline correction
        """
        self.normalize_input = bool(normalize_input)
        if self.normalize_input:
            spectra = self._normalize_numpy(spectra)
        self.spectra = torch.FloatTensor(spectra)
        if not 0.0 < mask_ratio_min <= mask_ratio_max < 1.0:
            raise ValueError(
                f"mask_ratio_min/max must satisfy 0 < min <= max < 1, got "
                f"min={mask_ratio_min}, max={mask_ratio_max}"
            )
        self.mask_ratio_min = float(mask_ratio_min)
        self.mask_ratio_max = float(mask_ratio_max)
        self.patch_size = patch_size
        self.n_patches = spectra.shape[1] // patch_size
        self.mask_strategy = mask_strategy
        self.mask_fill = mask_fill
        self.augment = augment
        self.per_point_std = per_point_std.to('cpu') if per_point_std is not None else None
        self.noise_scale = noise_scale
        self.baseline_window_start = int(baseline_window_start)
        self.baseline_window_end = int(baseline_window_end)
        self.correct_post_mask_baseline = bool(correct_post_mask_baseline)
        self.baseline_tol = float(baseline_tol)

        if self.normalize_input:
            print(f"Normalized data range: [{self.spectra.min():.3f}, {self.spectra.max():.3f}]")
        else:
            print(f"Input data range (no normalization): [{self.spectra.min():.3f}, {self.spectra.max():.3f}]")
        print(f"Mask strategy: {mask_strategy} with per-sample ratio in [{self.mask_ratio_min:.2f}, {self.mask_ratio_max:.2f}]")
        print(f"Mask fill strategy: {mask_fill}")
        if self.correct_post_mask_baseline:
            print(
                f"Post-mask baseline correction: enabled "
                f"[{self.baseline_window_start}:{self.baseline_window_end}]"
            )
        else:
            print("Post-mask baseline correction: disabled")
        if self.augment:
            print(f"Data augmentation enabled with noise scale {self.noise_scale}")
        else:
            print("Data augmentation disabled")
    
    def _normalize_numpy(self, spectra):
        """Normalize each spectrum to [0, 1] range"""
        normalized = np.zeros_like(spectra)
        for i in range(len(spectra)):
            spectrum = spectra[i]
            min_val = spectrum.min()
            max_val = spectrum.max()
            if max_val - min_val > 1e-8:
                normalized[i] = (spectrum - min_val) / (max_val - min_val)
            else:
                normalized[i] = spectrum
        return normalized
    
    def log_normalize_spectra(self, spectra):
        """Log normalization - handles large dynamic ranges well"""
        normalized_spectra = torch.zeros_like(spectra)
        
        for i in range(len(spectra)):
            spectrum = spectra[i]
            # Shift to make all values positive (add 1 to avoid log(0))
            min_val = torch.min(spectrum)
            shifted = spectrum - min_val + 1.0
            normalized_spectra[i] = torch.log1p(shifted)  # log1p = log(1 + x)
    
        return normalized_spectra

    def create_mask(self, n_patches):
        """Create different types of masks - optimized for learning peak relationships"""
        mask = torch.zeros(n_patches, dtype=torch.bool)
        ratio = random.uniform(self.mask_ratio_min, self.mask_ratio_max)
        n_masked = max(1, int(n_patches * ratio))

        if self.mask_strategy == 'sparse_random':
            # Random sparse masking - ensures context remains
            masked_indices = torch.randperm(n_patches)[:n_masked].tolist()

        elif self.mask_strategy == 'scattered_peaks':
            # Mask small scattered regions (1-4 patches at a time)
            masked_set = set()
            # try to create groups until we have enough masked patches
            attempts = 0
            while len(masked_set) < n_masked and attempts < n_patches * 2:
                group_size = random.randint(1, min(4, n_masked - len(masked_set)))
                start = random.randint(0, max(0, n_patches - group_size))
                for j in range(group_size):
                    masked_set.add(start + j)
                attempts += 1

            # If still short, fill with random unique indices
            if len(masked_set) < n_masked:
                remaining = n_masked - len(masked_set)
                perm = torch.randperm(n_patches).tolist()
                for idx in perm:
                    if idx not in masked_set:
                        masked_set.add(idx)
                        remaining -= 1
                        if remaining == 0:
                            break

            masked_indices = list(sorted(masked_set))

        elif self.mask_strategy == 'single_peak':
            # Mask only 1-3 individual patches randomly
            actual_masked = min(3, n_masked)
            masked_indices = torch.randperm(n_patches)[:actual_masked].tolist()

        else:
            # Default sparse random
            masked_indices = torch.randperm(n_patches)[:n_masked].tolist()

        # Ensure exact number of masked indices (unique)
        masked_indices = list(dict.fromkeys(masked_indices))
        if len(masked_indices) < n_masked:
            # add more unique indices
            perm = torch.randperm(n_patches).tolist()
            for idx in perm:
                if idx not in masked_indices:
                    masked_indices.append(idx)
                if len(masked_indices) >= n_masked:
                    break

        masked_indices = masked_indices[:n_masked]
        mask[torch.tensor(masked_indices, dtype=torch.long)] = True
        return mask
    
    def __len__(self):
        return len(self.spectra)
    
    def __getitem__(self, idx):
        spectrum = self.spectra[idx]
        
        # Apply data augmentation if enabled
        if self.augment and self.per_point_std is not None:
            per_point_std_device = self.per_point_std.to(spectrum.device)
            noise = torch.randn_like(spectrum) * per_point_std_device * self.noise_scale
            spectrum = spectrum + noise
        
        mask = self.create_mask(self.n_patches)
        masked_spectrum = spectrum.clone()
        point_mask = torch.zeros_like(masked_spectrum, dtype=torch.bool)

        for i, is_masked in enumerate(mask):
            if is_masked:
                start = i * self.patch_size
                end = start + self.patch_size
                point_mask[start:end] = True
                if self.mask_fill == 'zero':
                    masked_spectrum[start:end] = 0.0
                elif self.mask_fill == 'mean':
                    patch_mean = spectrum[start:end].mean()
                    masked_spectrum[start:end] = patch_mean
                elif self.mask_fill == 'noise':
                    patch_std = spectrum[start:end].std()
                    if patch_std < 1e-8:
                        patch_std = 1.0
                    masked_spectrum[start:end] = torch.randn(self.patch_size) * patch_std
                else:
                    # Try to interpret as float
                    try:
                        fill_value = float(self.mask_fill)
                    except Exception:
                        fill_value = 0.0
                    masked_spectrum[start:end] = fill_value

        # Re-center unmasked values using the known zero baseline window.
        if self.correct_post_mask_baseline:
            n_points = masked_spectrum.shape[0]
            s = max(0, min(self.baseline_window_start, n_points))
            e = max(0, min(self.baseline_window_end, n_points))
            if s < e:
                window_unmasked = ~point_mask[s:e]
                if torch.any(window_unmasked):
                    window_vals = masked_spectrum[s:e][window_unmasked]
                    baseline_offset = torch.median(window_vals)
                    if torch.abs(baseline_offset) > self.baseline_tol:
                        masked_spectrum[~point_mask] = masked_spectrum[~point_mask] - baseline_offset

        return {
            'original': spectrum,
            'masked': masked_spectrum,
            'mask': mask,
            'patches': spectrum.unfold(0, self.patch_size, self.patch_size)
        }

    # def __getitem__(self, idx):
    #     spectrum = self.spectra[idx]
        
    #     # Create patches
    #     patches = spectrum.unfold(0, self.patch_size, self.patch_size)
        
    #     # Create mask
    #     mask = self.create_mask(self.n_patches)
        
    #     # Apply mask to spectrum - use zeros for simplicity
    #     masked_spectrum = spectrum.clone()
    #     for i, is_masked in enumerate(mask):
    #         if is_masked:
    #             start_idx = i * self.patch_size
    #             end_idx = start_idx + self.patch_size
    #             masked_spectrum[start_idx:end_idx] = 0
        
    #     return {
    #         'original': spectrum,
    #         'masked': masked_spectrum,
    #         'mask': mask,
    #         'patches': patches
    #     }

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
    """Simplified Transformer encoder for NMR spectra"""
    
    def __init__(self, spectrum_length, patch_size=512, d_model=256, nhead=8, 
                 num_layers=4, dim_feedforward=512, dropout=0.15):
        super().__init__()
        
        self.patch_size = patch_size
        self.d_model = d_model
        self.n_patches = spectrum_length // patch_size
        
        # Patch embedding with layer norm
        self.patch_embedding = nn.Sequential(
            nn.Linear(patch_size, d_model),
            nn.LayerNorm(d_model)
        )
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, self.n_patches)
        
        # Simplified transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True
        )
        # Pre-norm encoder layers do not support PyTorch's nested-tensor fast
        # path. Disable it explicitly instead of emitting a warning per model.
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers,
            enable_nested_tensor=False,
        )
        
        # # Improved reconstruction head with regularization
        # self.reconstruction_head = nn.Sequential(
        #     nn.Linear(d_model, dim_feedforward // 2),
        #     nn.GELU(),
        #     nn.Dropout(dropout),
        #     nn.LayerNorm(dim_feedforward // 2),
        #     nn.Linear(dim_feedforward // 2, patch_size),
        #     # nn.Tanh()  # Bound outputs to [-1, 1]
        # )

        # Better reconstruction head with skip connection
        self.reconstruction_head = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, patch_size)
            # NO Tanh!
        )
        
        # Add direct skip from embedding to output
        self.skip_projection = nn.Linear(patch_size, patch_size)

        # Learnable mask token
        self.mask_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        
        # Initialize weights
        self.apply(self._init_weights)

    # Better weight initialization in _init_weights:
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            # Use smaller init for better gradient flow
            nn.init.xavier_uniform_(module.weight, gain=0.5)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)
        
    # def forward(self, x, mask=None):
    #     batch_size, seq_len = x.shape
        
    #     # Create patches
    #     patches = x.unfold(1, self.patch_size, self.patch_size)
        
    #     # Embed patches
    #     embeddings = self.patch_embedding(patches)
        
    #     # Add mask tokens for masked patches
    #     if mask is not None:
    #         mask_tokens = self.mask_token.expand(batch_size, -1, -1)
    #         for i in range(batch_size):
    #             embeddings[i, mask[i]] = mask_tokens[i]
        
    #     # Add positional encoding
    #     embeddings = embeddings.transpose(0, 1)
    #     embeddings = self.pos_encoding(embeddings)
    #     embeddings = embeddings.transpose(0, 1)
        
    #     # Apply transformer
    #     encoded = self.transformer(embeddings)
        
    #     # Reconstruct patches
    #     reconstructed_patches = self.reconstruction_head(encoded)
        
    #     # Reshape back to spectrum format
    #     reconstructed = reconstructed_patches.reshape(batch_size, -1)
        
    #     return reconstructed, encoded
    
    def forward(self, x, mask=None):
        batch_size, seq_len = x.shape
        
        # Create patches
        patches = x.unfold(1, self.patch_size, self.patch_size)  # [B, n_patches, patch_size]
        original_patches = patches.clone()  # Save for skip connection
        
        # Embed patches
        embeddings = self.patch_embedding(patches)
        
        # Add mask tokens for masked patches
        if mask is not None:
            mask_tokens = self.mask_token.expand(batch_size, -1, -1)
            for i in range(batch_size):
                embeddings[i, mask[i]] = mask_tokens[i]
        
        # Positional encoding + transformer
        embeddings = embeddings.transpose(0, 1)
        embeddings = self.pos_encoding(embeddings)
        embeddings = embeddings.transpose(0, 1)
        encoded = self.transformer(embeddings)
        
        # Reconstruct patches
        reconstructed_patches = self.reconstruction_head(encoded)  # [B, n_patches, patch_size]
        
        # SKIP CONNECTION: Add back original unmasked patches (scaled)
        skip_contribution = self.skip_projection(original_patches)
        
        if mask is not None:
            # Only add skip for unmasked patches
            for i in range(batch_size):
                reconstructed_patches[i, ~mask[i]] += 0.3 * skip_contribution[i, ~mask[i]]
        else:
            reconstructed_patches += 0.3 * skip_contribution
        
        # Reshape back to spectrum format
        reconstructed = reconstructed_patches.reshape(batch_size, -1)
        
        return reconstructed, encoded

class NMRMaskedAutoencoder(nn.Module):
    """Complete Masked Autoencoder for NMR spectra"""
    
    def __init__(self, spectrum_length, patch_size=512, **kwargs):
        super().__init__()
        self.encoder = NMRTransformerEncoder(spectrum_length, patch_size, **kwargs)
        self.patch_size = patch_size
        
    def forward(self, x, mask=None):
        return self.encoder(x, mask)

# def compute_loss(model, batch, device):
#     """Simple MSE loss on masked regions"""
#     # Move tensors to device with non-blocking (DataLoader should use pin_memory=True)
#     original = batch['original'].to(device, non_blocking=True)
#     masked = batch['masked'].to(device, non_blocking=True)
#     mask = batch['mask'].to(device, non_blocking=True)

#     # Forward pass: reconstructed is (B, L)
#     reconstructed, _ = model(masked, mask)

#     batch_size = original.size(0)
#     patch_size = model.patch_size
#     n_patches = original.shape[1] // patch_size

#     # Reshape to (B, n_patches, patch_size)
#     orig_patches = original.unfold(1, patch_size, patch_size).contiguous()
#     # unfold returns shape (B, n_patches, patch_size)
#     rec_patches = reconstructed.view(batch_size, n_patches, patch_size)

#     # Ensure mask has shape (B, n_patches)
#     if mask.dim() == 1:
#         mask = mask.unsqueeze(0).expand(batch_size, -1)

#     mask_bool = mask.bool()

#     # Compute squared error per element and sum over patch dimension
#     se_per_patch = ((rec_patches - orig_patches) ** 2).sum(dim=2)  # (B, n_patches)

#     # Apply mask and sum
#     masked_se = se_per_patch * mask_bool.to(se_per_patch.dtype)
#     total_masked_elements = int(mask_bool.sum().item()) * patch_size
#     total_loss = masked_se.sum()

#     if total_masked_elements > 0:
#         loss = total_loss / total_masked_elements
#     else:
#         loss = torch.tensor(0.0, device=device, requires_grad=True)

#     return loss

def compute_loss(model, batch, device, reconstruction_weight=0.3):
    """
    Dual loss: primary on masked regions, auxiliary on full reconstruction
    """
    original = batch['original'].to(device, non_blocking=True)
    masked = batch['masked'].to(device, non_blocking=True)
    mask = batch['mask'].to(device, non_blocking=True)

    reconstructed, _ = model(masked, mask)

    batch_size = original.size(0)
    patch_size = model.patch_size
    n_patches = original.shape[1] // patch_size

    orig_patches = original.unfold(1, patch_size, patch_size).contiguous()
    rec_patches = reconstructed.view(batch_size, n_patches, patch_size)

    if mask.dim() == 1:
        mask = mask.unsqueeze(0).expand(batch_size, -1)
    mask_bool = mask.bool()

    # PRIMARY LOSS: Masked regions (higher weight)
    se_per_patch = ((rec_patches - orig_patches) ** 2).mean(dim=2)  # Mean over patch
    masked_loss = (se_per_patch * mask_bool.float()).sum() / (mask_bool.sum() + 1e-8)

    # AUXILIARY LOSS: Unmasked regions (forces peak preservation)
    unmasked_loss = (se_per_patch * (~mask_bool).float()).sum() / ((~mask_bool).sum() + 1e-8)

    # Combined loss
    total_loss = masked_loss + reconstruction_weight * unmasked_loss

    return total_loss, masked_loss, unmasked_loss

def validate_model(model, val_dataloader, device):
    """Validate the model and return average validation loss"""
    model.eval()
    total_loss = 0
    total_masked_loss = 0
    total_unmasked_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch in val_dataloader:
            loss, masked_loss, unmasked_loss = compute_loss(model, batch, device)
            total_loss += loss.item()
            total_masked_loss += masked_loss.item()
            total_unmasked_loss += unmasked_loss.item()
            num_batches += 1
    
    if num_batches > 0:
        avg_val_loss = total_loss / num_batches
        avg_masked_loss = total_masked_loss / num_batches
        avg_unmasked_loss = total_unmasked_loss / num_batches
    else:
        avg_val_loss = float('inf')
        avg_masked_loss = float('inf')
        avg_unmasked_loss = float('inf')

    return avg_val_loss, avg_masked_loss, avg_unmasked_loss

def describe_architecture(model):
    """Read the architecture back off the built model, for the checkpoint.

    Why this exists: nn.MultiheadAttention stores in_proj_weight as
    (3*d_model, d_model) regardless of nhead, so a checkpoint loaded with the
    WRONG nhead still passes load_state_dict(strict=True) silently -- while
    splitting d_model into a different number of heads than it was trained
    with, i.e. quietly reinterpreting the trained weights. Because nhead was
    not recorded, every masking eval script had to guess it, and they defaulted
    to 8 while training used 4. Recording the real values makes that class of
    silent mismatch impossible. Derived from the model rather than from CONFIG
    so it reflects what was actually built.
    """
    enc = model.encoder
    layer0 = enc.transformer.layers[0]
    return {
        'patch_size': int(enc.patch_embedding[0].weight.shape[1]),
        'd_model': int(enc.patch_embedding[0].weight.shape[0]),
        'nhead': int(layer0.self_attn.num_heads),
        'num_layers': int(len(enc.transformer.layers)),
        'dim_feedforward': int(layer0.linear1.out_features),
        'dropout': float(layer0.dropout.p),
    }


def train_ssl_model(model, train_dataloader, timestamp, val_dataloader=None, num_epochs=10,
                   lr=1e-4, device='cuda', warmup_epochs=10, min_lr=1e-6,
                   patience=30, model_name_prefix="nmr_ssl_model",
                   augment_enabled=False, augment_every=10):
    """Training loop with early stopping"""
    
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01, eps=1e-8)
    # Use automatic mixed precision for faster throughput and lower memory
    scaler = torch.cuda.amp.GradScaler()
    
    # Warmup + Cosine Annealing scheduler
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return epoch / warmup_epochs
        else:
            cos_epoch = epoch - warmup_epochs
            cos_epochs = num_epochs - warmup_epochs
            return min_lr/lr + (1 - min_lr/lr) * 0.5 * (1 + math.cos(math.pi * cos_epoch / cos_epochs))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    val_masked_loss = float('inf')
    val_unmasked_loss = float('inf')
    epochs_without_improvement = 0
    
    # Enable cuDNN autotuner for potential speedups on fixed-size inputs
    try:
        torch.backends.cudnn.benchmark = True
    except Exception:
        pass

    print(f"Training with warmup ({warmup_epochs} epochs) + cosine annealing")
    print(f"Initial LR: {lr}, Min LR: {min_lr}")
    print(f"Early stopping patience: {patience} epochs")
    
    for epoch in range(num_epochs):
        # Enable/disable data augmentation for this epoch
        if augment_enabled:
            train_dataloader.dataset.augment = (epoch % augment_every == 0)
        else:
            train_dataloader.dataset.augment = False
        
        # Training phase
        model.train()
        epoch_train_loss = 0
        # tqdm's live progress bar overwrites a single line via carriage returns,
        # which only works on a real terminal. When stdout is redirected/piped
        # (e.g. `| tee log.txt`), it falls back to printing a new line per
        # refresh -- one line per batch. Disable it in that case; the epoch
        # summary printed after this loop already reports the same info.
        pbar = tqdm(train_dataloader, desc=f'Epoch {epoch+1}/{num_epochs}', disable=not sys.stdout.isatty())
        
        for batch_idx, batch in enumerate(pbar):
            optimizer.zero_grad()

            # In train_ssl_model, replace compute_loss call:
            with torch.amp.autocast('cuda', enabled=(device != 'cpu')):
                loss, masked_loss, unmasked_loss = compute_loss(model, batch, device)            

            # # Mixed precision forward/backward
            # with torch.cuda.amp.autocast(enabled=(device != 'cpu')):
            #     loss = compute_loss(model, batch, device)

            if torch.isnan(loss):
                print(f"NaN loss detected at epoch {epoch+1}, batch {batch_idx}")
                continue

            scaler.scale(loss).backward()

            # Unscale before clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            scaler.step(optimizer)
            scaler.update()

            epoch_train_loss += loss.item()
            # Update progress bar:
            pbar.set_postfix({
                'Total': f'{loss.item():.4f}',
                'Masked': f'{masked_loss.item():.4f}',
                'Unmasked': f'{unmasked_loss.item():.4f}',
                'LR': f'{scheduler.get_last_lr()[0]:.6f}',         
                'Best Val': f'{best_val_loss:.4f}',
                'Val Masked Loss': f'{val_masked_loss:.4f}',
                'Val Unmasked Loss': f'{val_unmasked_loss:.4f}',
                'Patience': f'{epochs_without_improvement}/{patience}'
            })
        
        avg_train_loss = epoch_train_loss / len(train_dataloader)
        train_losses.append(avg_train_loss)
        
        # Validation phase
        val_loss_str = ""
        improved = False
        if val_dataloader is not None:
            avg_val_loss, val_masked_loss, val_unmasked_loss = validate_model(model, val_dataloader, device)
            val_losses.append(avg_val_loss)
            val_loss_str = f", Val Loss = {avg_val_loss:.4f}"
            
            if avg_val_loss < best_val_loss - 1e-6:
                best_val_loss = avg_val_loss
                epochs_without_improvement = 0
                improved = True
                
                # Save best model
                batch_size = train_dataloader.batch_size
                mask_ratio_min = train_dataloader.dataset.mask_ratio_min
                mask_ratio_max = train_dataloader.dataset.mask_ratio_max
                patch_size = train_dataloader.dataset.patch_size

                best_model_name = f"{model_name_prefix}_bs{batch_size}_mr{mask_ratio_min:.2f}-{mask_ratio_max:.2f}_ps{patch_size}_best.pth"

                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'train_loss': avg_train_loss,
                    'val_loss': avg_val_loss,
                    'best_val_loss': best_val_loss,
                    'hyperparameters': {
                        'batch_size': batch_size,
                        'mask_ratio_min': mask_ratio_min,
                        'mask_ratio_max': mask_ratio_max,
                        'patch_size': patch_size,
                        'learning_rate': lr,
                        'num_epochs': num_epochs,
                        'warmup_epochs': warmup_epochs,
                        'min_lr': min_lr,
                        # Full architecture, read off the model -- see
                        # describe_architecture() for why this is essential.
                        **describe_architecture(model),
                    }
                }, best_model_name)
                print(f"✓ New best model saved: {best_model_name}")
            else:
                epochs_without_improvement += 1
        else:
            if avg_train_loss < best_val_loss - 1e-6:
                best_val_loss = avg_train_loss
                epochs_without_improvement = 0
                improved = True
            else:
                epochs_without_improvement += 1
        
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        
        improvement_str = " ✓ IMPROVED" if improved else ""
        print(f'Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}{val_loss_str}, '
              f'LR = {current_lr:.6f}, No improvement: {epochs_without_improvement}/{patience}{improvement_str}')
        
        # Early stopping
        if epochs_without_improvement >= patience:
            print(f"Early stopping triggered after {patience} epochs without improvement")
            print(f"Best validation loss: {best_val_loss:.4f}")
            break
        
    print(f"\n{'='*60}")
    print(f"Training completed after {epoch+1} epochs")
    print(f"Best validation loss achieved: {best_val_loss:.4f}")
    print(f"{'='*60}\n")
    
    return train_losses, val_losses

def plot_training_curves(train_losses, val_losses=None, save_path='training_curves.png'):
    """Plot and save training curves"""
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, label='Training Loss', color='blue', linewidth=2)
    if val_losses:
        plt.plot(epochs, val_losses, label='Validation Loss', color='red', linewidth=2)
    plt.title('Training Progress', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('MSE Loss', fontsize=12)
    plt.yscale('log')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    if val_losses and len(train_losses) > 10:
        plt.subplot(1, 3, 2)
        window_size = max(5, len(train_losses) // 20)
        
        def smooth(data, window):
            if len(data) < window:
                return data
            smoothed = []
            for i in range(len(data)):
                start_idx = max(0, i - window // 2)
                end_idx = min(len(data), i + window // 2 + 1)
                smoothed.append(np.mean(data[start_idx:end_idx]))
            return smoothed
        
        smooth_train = smooth(train_losses, window_size)
        smooth_val = smooth(val_losses, window_size)
        
        plt.plot(epochs, smooth_train, label='Smoothed Train', color='blue', alpha=0.8, linewidth=2)
        plt.plot(epochs, smooth_val, label='Smoothed Val', color='red', alpha=0.8, linewidth=2)
        plt.title(f'Smoothed Trends (window={window_size})', fontsize=14)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('MSE Loss', fontsize=12)
        plt.yscale('log')
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
    
    if val_losses:
        plt.subplot(1, 3, 3)
        val_train_diff = np.array(val_losses) - np.array(train_losses[:len(val_losses)])
        plt.plot(epochs[:len(val_train_diff)], val_train_diff, color='purple', linewidth=2)
        plt.title('Val - Train Loss', fontsize=14)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss Difference', fontsize=12)
        plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        plt.grid(True, alpha=0.3)
        
        final_diff = val_train_diff[-1] if len(val_train_diff) > 0 else 0
        if final_diff > 0.1:
            plt.text(0.05, 0.95, 'Possible\nOverfitting', transform=plt.gca().transAxes, 
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7),
                    verticalalignment='top', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Training curves saved to: {save_path}")
    plt.show()
    
    print(f"\n=== Training Statistics ===")
    print(f"Total epochs: {len(train_losses)}")
    print(f"Final training loss: {train_losses[-1]:.6f}")
    if val_losses:
        print(f"Final validation loss: {val_losses[-1]:.6f}")
        print(f"Best validation loss: {min(val_losses):.6f}")
        best_epoch = val_losses.index(min(val_losses)) + 1
        print(f"Best validation achieved at epoch: {best_epoch}")

def visualize_reconstruction(model, dataset, device='cuda', n_examples=5, save_path='reconstruction_examples.png'):
    """Visualize reconstruction results"""
    model.eval()
    
    total_samples = len(dataset)
    random_indices = random.sample(range(total_samples), min(n_examples, total_samples))
    
    fig, axes = plt.subplots(n_examples, 3, figsize=(18, 4*n_examples))
    if n_examples == 1:
        axes = axes.reshape(1, -1)
    
    print(f"\nVisualizing reconstruction for random samples: {random_indices}")
    
    with torch.no_grad():
        for plot_idx, sample_idx in enumerate(random_indices):
            sample = dataset[sample_idx]
            original = sample['original'].unsqueeze(0).to(device)
            masked = sample['masked'].unsqueeze(0).to(device)
            mask = sample['mask'].unsqueeze(0).to(device)
            
            reconstructed, _ = model(masked, mask)
            
            original_np = original.cpu().numpy().flatten()
            masked_np = masked.cpu().numpy().flatten()
            reconstructed_np = reconstructed.cpu().numpy().flatten()
            
            mse_error = np.mean((original_np - reconstructed_np) ** 2)
            correlation = np.corrcoef(original_np, reconstructed_np)[0, 1]
            
            axes[plot_idx, 0].plot(original_np, color='blue', linewidth=1)
            axes[plot_idx, 0].set_title(f'Original Spectrum (Sample {sample_idx})', fontsize=12)
            axes[plot_idx, 0].set_xlabel('Frequency Point')
            axes[plot_idx, 0].set_ylabel('Intensity')
            axes[plot_idx, 0].grid(True, alpha=0.3)
            
            axes[plot_idx, 1].plot(masked_np, color='orange', linewidth=1)
            axes[plot_idx, 1].set_title(f'Masked Spectrum (Sample {sample_idx})', fontsize=12)
            axes[plot_idx, 1].set_xlabel('Frequency Point')
            axes[plot_idx, 1].set_ylabel('Intensity')
            axes[plot_idx, 1].grid(True, alpha=0.3)
            
            axes[plot_idx, 2].plot(original_np, label='Original', alpha=0.8, color='blue', linewidth=1)
            axes[plot_idx, 2].plot(reconstructed_np, label='Reconstructed', alpha=0.8, color='red', linewidth=1)
            axes[plot_idx, 2].set_title(f'Reconstruction (Sample {sample_idx})\nMSE: {mse_error:.4f}, Corr: {correlation:.3f}', fontsize=12)
            axes[plot_idx, 2].set_xlabel('Frequency Point')
            axes[plot_idx, 2].set_ylabel('Intensity')
            axes[plot_idx, 2].legend()
            axes[plot_idx, 2].grid(True, alpha=0.3)
            
            print(f"Sample {sample_idx}: MSE = {mse_error:.6f}, Correlation = {correlation:.4f}")
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Reconstruction examples saved to: {save_path}")
    plt.show()
    
    return random_indices

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Train NMR Masked Autoencoder')
    parser.add_argument('--augment', action='store_true', help='Enable data augmentation')
    parser.add_argument('--augment-every', type=int, default=10, help='Augment data every N epochs')
    parser.add_argument('--noise-scale', type=float, default=1.0, help='Scale for augmentation noise')
    parser.add_argument(
        '--data-path',
        nargs='+',
        default=['data/combined/combine_unique_MetaboLights_Workbench_Water_EDTA_Suppressed_rowMinMax_v3.npy'],
        help='Path(s) to NMR spectra .npy file(s). Multiple files can be passed separated by space.'
    )
    parser.add_argument('--mask-ratio-min', type=float, default=0.20, help='Lower bound of per-sample random masking ratio')
    parser.add_argument('--mask-ratio-max', type=float, default=0.60, help='Upper bound of per-sample random masking ratio')
    parser.add_argument('--device', default='auto', help="'auto', or an explicit device string like 'cuda:0' / 'cpu'")
    # Architecture / schedule knobs. These were previously hardcoded in CONFIG;
    # exposing them is what makes the patch-size experiment runnable (see
    # docs/SSL_vs_classical_analysis.md experiment #4). patch_size sets the
    # model's spectral resolution: 131072/patch_size tokens, so 1024 -> 128
    # tokens and 128 -> 1024 tokens. Attention cost grows quadratically in the
    # token count, so reducing patch_size is expensive -- check the printed
    # per-epoch time before committing to a long run.
    parser.add_argument('--patch-size', type=int, default=1024,
                        help='Points per patch. 131072/patch_size = number of tokens.')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--num-epochs', type=int, default=2000)
    parser.add_argument('--learning-rate', type=float, default=1e-4)
    parser.add_argument('--patience', type=int, default=200, help='Early-stopping patience in epochs')
    parser.add_argument('--d-model', type=int, default=128)
    parser.add_argument('--nhead', type=int, default=4)
    parser.add_argument('--num-layers', type=int, default=3)
    parser.add_argument('--dim-feedforward', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.2)
    args = parser.parse_args()
    if not 0.0 < args.mask_ratio_min <= args.mask_ratio_max < 1.0:
        raise ValueError("--mask-ratio-min/--mask-ratio-max must satisfy 0 < min <= max < 1")
    if args.d_model % args.nhead:
        raise ValueError(f"--d-model ({args.d_model}) must be divisible by --nhead ({args.nhead})")

    # CONFIG: All configurable parameters
    CONFIG = {
        'data_path': args.data_path,
        'num_epochs': args.num_epochs,
        'batch_size': args.batch_size,
        'mask_ratio_min': args.mask_ratio_min,
        'mask_ratio_max': args.mask_ratio_max,
        'patch_size': args.patch_size,
        'learning_rate': args.learning_rate,
        'warmup_epochs': 20,
        'patience': args.patience,
        'd_model': args.d_model,
        'nhead': args.nhead,
        'num_layers': args.num_layers,
        'dim_feedforward': args.dim_feedforward,
        'dropout': args.dropout,
        'train_split': 0.80,
        'val_split': 0.20,
        'test_split': 0.0
    }
    
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")
    
    print("Loading NMR spectra...")
    try:
        data_paths = CONFIG['data_path']
        if isinstance(data_paths, str):
            data_paths = [data_paths]
        spectra_list = []
        for data_path in data_paths:
            arr = np.load(data_path)
            if arr.ndim != 2:
                raise ValueError(f"Each .npy file must be 2D, got {data_path} with shape {arr.shape}")
            if spectra_list and arr.shape[1] != spectra_list[0].shape[1]:
                raise ValueError(
                    f"All .npy files must have the same number of frequency points. "
                    f"{data_paths[0]} has {spectra_list[0].shape[1]} points, but {data_path} has {arr.shape[1]}"
                )
            print(f"Loading {data_path} with shape {arr.shape}")
            spectra_list.append(arr)
        if len(spectra_list) == 1:
            spectra = spectra_list[0]
        else:
            spectra = np.concatenate(spectra_list, axis=0)
        print(f"Loaded combined spectra shape: {spectra.shape}")
        
        print("Validating data...")
        nan_mask = np.isnan(spectra).any(axis=1)
        inf_mask = np.isinf(spectra).any(axis=1)
        bad_mask = nan_mask | inf_mask
        
        if bad_mask.any():
            print(f"Removing {bad_mask.sum()} spectra with NaN/Inf values")
            spectra = spectra[~bad_mask]
        
        std_values = np.std(spectra, axis=1)
        zero_std_mask = std_values < 1e-10
        if zero_std_mask.any():
            print(f"Removing {zero_std_mask.sum()} constant spectra")
            spectra = spectra[~zero_std_mask]
        
        original_size = len(spectra)
        spectra = np.unique(spectra, axis=0)
        print(f"Removed {original_size - len(spectra)} duplicate spectra")
        
        print(f"Final clean spectra shape: {spectra.shape}")
        print(f"Data range: [{np.min(spectra):.3f}, {np.max(spectra):.3f}]")
        
        # Compute per-point statistics for augmentation
        if args.augment:
            per_point_std = np.std(spectra, axis=0)
            print(f"Computed per-point std for augmentation, shape: {per_point_std.shape}")
        else:
            per_point_std = None
        
    except FileNotFoundError:
        print("File not found. Please check the file path.")
        return None, None, None
    
    if len(spectra) < 100:
        print(f"Warning: Only {len(spectra)} spectra available.")
    
    # Ensure split fractions sum to 1.0; if not, normalize and warn
    total_frac = CONFIG['train_split'] + CONFIG['val_split'] + CONFIG['test_split']
    if abs(total_frac - 1.0) > 1e-6:
        print(f"Warning: TRAIN/VAL/TEST fractions sum to {total_frac:.4f}, normalizing to 1.0")
        TRAIN = CONFIG['train_split'] / total_frac
        VAL = CONFIG['val_split'] / total_frac
        TEST = CONFIG['test_split'] / total_frac
    else:
        TRAIN, VAL, TEST = CONFIG['train_split'], CONFIG['val_split'], CONFIG['test_split']

    # First split out the test set
    if TEST > 0:
        train_val_spectra, test_spectra = train_test_split(
            spectra, test_size=TEST, random_state=42, shuffle=True
        )
    else:
        train_val_spectra = spectra
        test_spectra = np.empty((0, spectra.shape[1]))

    # Then split train/val from the remaining
    if VAL > 0:
        val_fraction_of_trainval = VAL / (TRAIN + VAL)
        train_spectra, val_spectra = train_test_split(
            train_val_spectra, test_size=val_fraction_of_trainval, random_state=42, shuffle=True
        )
    else:
        train_spectra = train_val_spectra
        val_spectra = np.empty((0, spectra.shape[1]))

    print(f"Training set: {train_spectra.shape}")
    print(f"Validation set: {val_spectra.shape}")
    print(f"Test set: {test_spectra.shape}")
    
    # UPDATED PARAMETERS - Optimized for small dataset and peak learning
    spectrum_length = spectra.shape[1]
    patch_size = CONFIG['patch_size']      # Even smaller patches for finer granularity
    mask_ratio_min = CONFIG['mask_ratio_min']
    mask_ratio_max = CONFIG['mask_ratio_max']
    batch_size = CONFIG['batch_size']       # Larger batches for more stable gradients with small dataset

    if spectrum_length % patch_size != 0:
        new_length = (spectrum_length // patch_size) * patch_size
        train_spectra = train_spectra[:, :new_length]
        val_spectra = val_spectra[:, :new_length]
        spectrum_length = new_length
        print(f"Adjusted spectrum length to {spectrum_length}")

    train_dataset = NMRSpectrumDataset(
        train_spectra,
        mask_ratio_min=mask_ratio_min,
        mask_ratio_max=mask_ratio_max,
        patch_size=patch_size,
        mask_strategy='sparse_random',  # Changed - keeps context
        augment=False,  # Will be controlled during training
        per_point_std=torch.from_numpy(per_point_std) if per_point_std is not None else None,
        noise_scale=args.noise_scale
    )
    val_dataset = NMRSpectrumDataset(
        val_spectra,
        mask_ratio_min=mask_ratio_min,
        mask_ratio_max=mask_ratio_max,
        patch_size=patch_size,
        mask_strategy='sparse_random'
    )
    # Test dataset (if available)
    has_test = False
    if isinstance(test_spectra, np.ndarray):
        has_test = test_spectra.size > 0
    else:
        try:
            has_test = len(test_spectra) > 0
        except Exception:
            has_test = False

    if has_test:
        test_dataset = NMRSpectrumDataset(
            test_spectra,
            mask_ratio_min=mask_ratio_min,
            mask_ratio_max=mask_ratio_max,
            patch_size=patch_size,
            mask_strategy='sparse_random',
            augment=False,
            per_point_std=None,
            noise_scale=args.noise_scale
        )
    else:
        test_dataset = None
    # Tune number of workers dynamically (avoid oversubscribing CPU). Use persistent workers and prefetch to keep GPU fed.
    cpu_count = os.cpu_count() or 4
    suggested_workers = max(2, min(16, cpu_count // 2))

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=suggested_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=suggested_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4
    )
    if test_dataset is not None:
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=suggested_workers,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=4
        )
    else:
        test_dataloader = None
    
    print(f"Patches per spectrum: {train_dataset.n_patches}")
    print(
        f"Patches to mask: {int(train_dataset.n_patches * mask_ratio_min)}"
        f"-{int(train_dataset.n_patches * mask_ratio_max)} (random per sample)"
    )
    
    # OPTIMIZED MODEL for small dataset
    model = NMRMaskedAutoencoder(
        spectrum_length=spectrum_length,
        patch_size=patch_size,
        d_model=CONFIG['d_model'],          # Further reduced for small dataset
        nhead=CONFIG['nhead'],              # Reduced attention heads
        num_layers=CONFIG['num_layers'],         # Fewer layers to prevent overfitting
        dim_feedforward=CONFIG['dim_feedforward'],  # Smaller feedforward
        dropout=CONFIG['dropout']           # More dropout for regularization
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    num_epochs = CONFIG['num_epochs']
    learning_rate = CONFIG['learning_rate']
    warmup_epochs = CONFIG['warmup_epochs']
    patience = CONFIG['patience']  # Reduced patience for early stopping
    
    current_time = datetime.datetime.now()
    timestamp = current_time.strftime("%Y%m%d_%H%M%S")
    data_paths = CONFIG['data_path']
    if isinstance(data_paths, str):
        data_paths = [data_paths]
    dataset_name = os.path.splitext(os.path.basename(data_paths[0]))[0]
    if len(data_paths) > 1:
        dataset_name = f"{dataset_name}_merged{len(data_paths)}"
    model_name_prefix = f"./models/masked_ssl/{dataset_name}_{timestamp}"
    
    print("\n" + "="*60)
    print("Starting training with OPTIMIZED configuration for small dataset:")
    print(f"  - Dataset size: {len(train_spectra)} train, {len(val_spectra)} val")
    print(f"  - Max normalization (no clipping)")
    print(f"  - Sparse random masking ({mask_ratio_min:.0%}-{mask_ratio_max:.0%} per sample, randomized) - keeps context")
    print(f"  - Small patches ({patch_size}) for fine detail")
    print(f"  - Small model (3 layers, d=128) to prevent overfitting")
    print(f"  - Larger batches ({batch_size}) for stable gradients")
    print(f"  - Standard MSE loss")
    print("="*60 + "\n")
    
    train_losses, val_losses = train_ssl_model(
        model=model, 
        train_dataloader=train_dataloader, 
        val_dataloader=val_dataloader,
        num_epochs=num_epochs, 
        lr=learning_rate, 
        device=device,
        warmup_epochs=warmup_epochs,
        min_lr=1e-6,
        patience=patience,
        model_name_prefix=model_name_prefix,
        timestamp=timestamp,
        augment_enabled=args.augment,
        augment_every=args.augment_every
    )
    
    print("\nGenerating training curves...")
    plot_training_curves(train_losses, val_losses, f'{model_name_prefix}_training_curves.png')
    
    print("\nGenerating reconstruction examples...")
    random_indices = visualize_reconstruction(
        model, val_dataset, device=device, n_examples=5, 
        save_path=f'{model_name_prefix}_reconstructions.png'
    )
    # Evaluate on test set if available
    test_results = None
    if test_dataloader is not None:
        print("\nEvaluating on test dataset...")
        test_loss, test_masked_loss, test_unmasked_loss = validate_model(model, test_dataloader, device)
        test_results = {
            'test_loss': float(test_loss),
            'test_masked_loss': float(test_masked_loss),
            'test_unmasked_loss': float(test_unmasked_loss),
            'test_size': len(test_dataset)
        }
        print(f"Test Loss = {test_loss:.6f}, Masked = {test_masked_loss:.6f}, Unmasked = {test_unmasked_loss:.6f}")

        # Save test results to JSON
        import json
        test_out = f"{model_name_prefix}_test_results.json"
        with open(test_out, 'w') as jf:
            json.dump(test_results, jf, indent=2)
        print(f"Saved test results to {test_out}")

        # Save reconstruction examples for test set
        if test_dataset is not None:
            visualize_reconstruction(
                model, test_dataset, device=device, n_examples=8,
                save_path=f'{model_name_prefix}_test_reconstructions.png'
            )

    return model, train_dataset, val_dataset, test_dataset, test_results

if __name__ == "__main__":
    model, train_dataset, val_dataset, test_dataset, test_results = main()

    # # === Quick unit test for NMRSpectrumDataset __getitem__ ===
    # print("\n--- Unit test: NMRSpectrumDataset __getitem__ output ---")
    # # Use a small dummy spectrum for demonstration
    # dummy = np.linspace(-1, 1, 32*8).reshape(8, 32)  # 8 samples, 32 points each
    # ds = NMRSpectrumDataset(dummy, mask_ratio=0.25, patch_size=8, mask_strategy='scattered_peaks', mask_fill='mean')
    # sample = ds[0]
    # print("original:", sample['original'][:16].numpy())
    # print("masked:", sample['masked'][:16].numpy())
    # print("mask:", sample['mask'].numpy())
    # print("patches shape:", sample['patches'].shape)
    # # Show which patches were masked and their fill value
    # for i, is_masked in enumerate(sample['mask']):
    #     if is_masked:
    #         start = i * ds.patch_size
    #         end = start + ds.patch_size
    #         print(f"Patch {i} masked, fill: {sample['masked'][start:end].mean().item():.4f}")
