#!/usr/bin/env python3
"""
Gradient Analysis Script for NMR Masked Autoencoder

This script performs two main analyses:
1. Gradient checks with/without skip connections to detect vanishing gradients
2. Testing multiple intermediate skip connection configurations

Usage:
    python gradient_analysis.py --data-path <spectra.npy> --device cuda:0

Outputs:
- Gradient magnitude logs for each configuration
- Comparison summary
- Best performing configuration recommendation
"""

import torch
import torch.nn as nn
import numpy as np
import argparse
import os
from pathlib import Path
import json
from tqdm import tqdm
import datetime

# Import our model components
from trainer_revised import NMRSpectrumDataset, compute_loss
from torch.utils.data import DataLoader


class NMRTransformerEncoderWithSkips(nn.Module):
    """Modified NMRTransformerEncoder with configurable skip connections"""

    def __init__(self, spectrum_length, patch_size=512, d_model=256,
                 nhead=8, num_layers=4, dim_feedforward=512, dropout=0.15,
                 skip_mode='input_to_final'):
        """
        skip_mode options:
        - 'none': No skip connections
        - 'input_to_final': Original skip (input patches to final output)
        - 'layers_to_final': Each encoder layer output summed to final
        - 'residual_layers': Residual connections between layers (standard transformer)
        """
        super().__init__()

        self.skip_mode = skip_mode
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

        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True
        )
        self.transformer_layers = nn.ModuleList([
            encoder_layer for _ in range(num_layers)
        ])

        # Reconstruction head
        self.reconstruction_head = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, patch_size)
        )

        # Skip projection (for input_to_final mode)
        if skip_mode == 'input_to_final':
            self.skip_projection = nn.Linear(patch_size, patch_size)

        # Learnable mask token
        self.mask_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight, gain=0.5)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)

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

        # Positional encoding
        embeddings = embeddings.transpose(0, 1)
        embeddings = self.pos_encoding(embeddings)
        embeddings = embeddings.transpose(0, 1)

        # Apply transformer layers with optional intermediate skips
        layer_outputs = []
        current_embeddings = embeddings

        for layer in self.transformer_layers:
            current_embeddings = layer(current_embeddings)
            layer_outputs.append(current_embeddings)

        # Combine layer outputs based on skip_mode
        if self.skip_mode == 'none':
            encoded = current_embeddings  # Just the final layer
        elif self.skip_mode == 'input_to_final':
            encoded = current_embeddings  # Will add skip later
        elif self.skip_mode == 'layers_to_final':
            # Sum all layer outputs
            encoded = torch.stack(layer_outputs, dim=0).sum(dim=0)
        elif self.skip_mode == 'residual_layers':
            # This is actually the default transformer behavior
            encoded = current_embeddings
        else:
            encoded = current_embeddings

        # Reconstruct patches
        reconstructed_patches = self.reconstruction_head(encoded)

        # Apply skip connection if enabled
        if self.skip_mode == 'input_to_final':
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


class NMRMaskedAutoencoderWithSkips(nn.Module):
    """Complete Masked Autoencoder with configurable skips"""

    def __init__(self, spectrum_length, patch_size=512, skip_mode='input_to_final', **kwargs):
        super().__init__()
        self.encoder = NMRTransformerEncoderWithSkips(
            spectrum_length, patch_size, skip_mode=skip_mode, **kwargs
        )
        self.patch_size = patch_size

    def forward(self, x, mask=None):
        return self.encoder(x, mask)


def collect_gradients(model, loss):
    """Collect gradient norms for all parameters"""
    loss.backward(retain_graph=True)

    grad_norms = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.data.norm(2).item()
            grad_norms[name] = grad_norm
        else:
            grad_norms[name] = 0.0

    # Clear gradients for next run
    model.zero_grad()
    return grad_norms


def analyze_gradients_for_config(config_name, model, dataloader, device, n_batches=10):
    """Run gradient analysis for a single configuration"""
    print(f"\n{'='*50}")
    print(f"Analyzing gradients for: {config_name}")
    print(f"{'='*50}")

    model.to(device)
    model.train()

    all_grad_norms = []
    losses = []

    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= n_batches:
            break

        # Forward pass
        with torch.cuda.amp.autocast(enabled=(device != 'cpu')):
            loss, _, _ = compute_loss(model, batch, device)

        losses.append(loss.item())

        # Collect gradients
        grad_norms = collect_gradients(model, loss)
        all_grad_norms.append(grad_norms)

    # Aggregate gradient norms across batches
    avg_grad_norms = {}
    for name in all_grad_norms[0].keys():
        avg_grad_norms[name] = np.mean([gn[name] for gn in all_grad_norms])

    # Group by layer type
    layer_groups = {
        'embedding': [],
        'positional': [],
        'transformer': [],
        'reconstruction': [],
        'skip': [],
        'mask_token': []
    }

    for name, norm in avg_grad_norms.items():
        if 'patch_embedding' in name:
            layer_groups['embedding'].append(norm)
        elif 'pos_encoding' in name:
            layer_groups['positional'].append(norm)
        elif 'transformer_layers' in name:
            layer_groups['transformer'].append(norm)
        elif 'reconstruction_head' in name:
            layer_groups['reconstruction'].append(norm)
        elif 'skip_projection' in name:
            layer_groups['skip'].append(norm)
        elif 'mask_token' in name:
            layer_groups['mask_token'].append(norm)

    # Compute statistics
    group_stats = {}
    for group_name, norms in layer_groups.items():
        if norms:
            group_stats[group_name] = {
                'mean': np.mean(norms),
                'std': np.std(norms),
                'min': np.min(norms),
                'max': np.max(norms),
                'count': len(norms)
            }
        else:
            group_stats[group_name] = {'mean': 0, 'std': 0, 'min': 0, 'max': 0, 'count': 0}

    avg_loss = np.mean(losses)

    results = {
        'config': config_name,
        'avg_loss': avg_loss,
        'group_stats': group_stats,
        'detailed_norms': avg_grad_norms
    }

    # Print summary
    print(f"Average Loss: {avg_loss:.6f}")
    print("Gradient Norms by Layer Group:")
    for group, stats in group_stats.items():
        if stats['count'] > 0:
            print(f"  {group}: mean={stats['mean']:.2e}, std={stats['std']:.2e}, "
                  f"min={stats['min']:.2e}, max={stats['max']:.2e} (n={stats['count']})")

    return results


def run_full_training_comparison(configs, train_dataloader, val_dataloader, device, base_model_kwargs):
    """Run full training for each configuration and compare final performance"""
    print(f"\n{'='*60}")
    print("RUNNING FULL TRAINING COMPARISON")
    print(f"{'='*60}")

    results = {}

    for config_name, skip_mode in configs.items():
        print(f"\n--- Training with {config_name} ---")

        # Create model
        model = NMRMaskedAutoencoderWithSkips(
            skip_mode=skip_mode,
            **base_model_kwargs
        )

        # Quick training (reduced epochs for comparison)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
        scaler = torch.cuda.amp.GradScaler()

        model.to(device)
        model.train()

        train_losses = []
        val_losses = []

        # Train for 50 epochs (reduced for speed)
        n_epochs = 50

        for epoch in range(n_epochs):
            epoch_loss = 0
            for batch in train_dataloader:
                optimizer.zero_grad()

                with torch.cuda.amp.autocast(enabled=(device != 'cpu')):
                    loss, _, _ = compute_loss(model, batch, device)

                scaler.scale(loss).backward()
                scaler.unscale(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()

                epoch_loss += loss.item()

            avg_train_loss = epoch_loss / len(train_dataloader)
            train_losses.append(avg_train_loss)

            # Validation
            if val_dataloader is not None:
                model.eval()
                val_loss, _, _ = validate_model(model, val_dataloader, device)
                val_losses.append(val_loss)
                model.train()
            else:
                val_loss = avg_train_loss
                val_losses.append(val_loss)

            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}: Train={avg_train_loss:.6f}, Val={val_loss:.6f}")

        final_train_loss = train_losses[-1]
        final_val_loss = val_losses[-1]

        results[config_name] = {
            'final_train_loss': final_train_loss,
            'final_val_loss': final_val_loss,
            'train_losses': train_losses,
            'val_losses': val_losses
        }

        print(f"Final: Train={final_train_loss:.6f}, Val={final_val_loss:.6f}")

    return results


def validate_model(model, dataloader, device):
    """Quick validation function"""
    model.eval()
    total_loss = 0
    total_masked = 0
    total_unmasked = 0
    n_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            loss, masked_loss, unmasked_loss = compute_loss(model, batch, device)
            total_loss += loss.item()
            total_masked += masked_loss.item()
            total_unmasked += unmasked_loss.item()
            n_batches += 1

    return total_loss / n_batches, total_masked / n_batches, total_unmasked / n_batches


def main():
    parser = argparse.ArgumentParser(description='Gradient analysis for NMR MAE')
    parser.add_argument('--data-path', default='data/aligned/aligned_nmr_spectra_128K_WS625to680Zero.npy',
                       help='Path to NMR spectra data')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to run on')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size for analysis')
    parser.add_argument('--n-batches-grad', type=int, default=10,
                       help='Number of batches for gradient analysis')
    parser.add_argument('--output-dir', default='results/gradient_analysis',
                       help='Directory to save results')
    parser.add_argument('--run-full-training', action='store_true',
                       help='Also run full training comparison (slower)')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load data
    print("Loading data...")
    spectra = np.load(args.data_path)
    print(f"Loaded spectra shape: {spectra.shape}")

    # Create small dataset for analysis
    dataset = NMRSpectrumDataset(
        spectra[:100],  # Use subset for speed
        mask_ratio=0.15,
        patch_size=1024,
        mask_strategy='sparse_random'
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

    # Model configuration
    base_model_kwargs = {
        'spectrum_length': spectra.shape[1],
        'patch_size': 1024,
        'd_model': 128,
        'nhead': 4,
        'num_layers': 3,
        'dim_feedforward': 256,
        'dropout': 0.2
    }

    # Configurations to test
    configs = {
        'no_skip': 'none',
        'input_to_final_skip': 'input_to_final',
        'layers_to_final_skip': 'layers_to_final',
        'residual_layers': 'residual_layers'
    }

    # Run gradient analysis
    gradient_results = {}
    for config_name, skip_mode in configs.items():
        model = NMRMaskedAutoencoderWithSkips(skip_mode=skip_mode, **base_model_kwargs)
        result = analyze_gradients_for_config(
            config_name, model, dataloader, args.device, args.n_batches_grad
        )
        gradient_results[config_name] = result

    # Compare gradient magnitudes
    print(f"\n{'='*60}")
    print("GRADIENT MAGNITUDE COMPARISON")
    print(f"{'='*60}")

    comparison = {}
    for config_name, result in gradient_results.items():
        group_stats = result['group_stats']
        transformer_mean = group_stats['transformer']['mean']
        reconstruction_mean = group_stats['reconstruction']['mean']
        skip_mean = group_stats['skip']['mean'] if group_stats['skip']['count'] > 0 else 0

        comparison[config_name] = {
            'avg_loss': result['avg_loss'],
            'transformer_grad_mean': transformer_mean,
            'reconstruction_grad_mean': reconstruction_mean,
            'skip_grad_mean': skip_mean
        }

        print(f"{config_name}:")
        print(f"  Loss: {result['avg_loss']:.6f}")
        print(f"  Transformer grads: {transformer_mean:.2e}")
        print(f"  Reconstruction grads: {reconstruction_mean:.2e}")
        if skip_mean > 0:
            print(f"  Skip grads: {skip_mean:.2e}")
        print()

    # Determine best configuration based on gradient flow
    best_config = min(comparison.keys(),
                     key=lambda x: comparison[x]['avg_loss'])

    print(f"Best configuration (lowest loss): {best_config}")
    print(f"Loss: {comparison[best_config]['avg_loss']:.6f}")

    # Check for vanishing gradients
    no_skip_transformer_grads = comparison['no_skip']['transformer_grad_mean']
    skip_transformer_grads = comparison['input_to_final_skip']['transformer_grad_mean']

    if no_skip_transformer_grads < 1e-6:
        print("WARNING: No-skip configuration shows signs of vanishing gradients!")
        print(".2e")
    else:
        print("No clear vanishing gradient issue detected in no-skip configuration.")

    if skip_transformer_grads > no_skip_transformer_grads:
        print("Skip connections appear to improve gradient flow to transformer layers.")
    else:
        print("Skip connections do not significantly improve gradient flow.")

    # Run full training comparison if requested
    if args.run_full_training:
        print("\nRunning full training comparison...")
        training_results = run_full_training_comparison(
            configs, dataloader, None, args.device, base_model_kwargs
        )

        # Update best config based on training
        best_train_config = min(training_results.keys(),
                               key=lambda x: training_results[x]['final_val_loss'])

        print(f"\nBest configuration from full training: {best_train_config}")
        print(f"Final validation loss: {training_results[best_train_config]['final_val_loss']:.6f}")

        comparison.update(training_results)

    # Save results
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(args.output_dir, f'gradient_analysis_{timestamp}.json')

    with open(results_file, 'w') as f:
        json.dump({
            'gradient_results': gradient_results,
            'comparison': comparison,
            'best_config': best_config,
            'args': vars(args)
        }, f, indent=2)

    print(f"\nResults saved to: {results_file}")

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Gradient analysis completed for {len(configs)} configurations")
    print(f"Best performing configuration: {best_config}")
    print(f"Results saved to: {results_file}")


if __name__ == "__main__":
    import math  # For positional encoding
    main()#</content>
#<parameter name="filePath">/home/nmrbox/0012/shasharma/Desktop/NMR_Metabolomics/gradient_analysis.py