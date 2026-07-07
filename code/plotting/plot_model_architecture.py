#!/usr/bin/env python3
"""
Utility to generate textual and visual descriptions of the NMR MAE model.

Generates:
 - a textual summary saved to `<output_prefix>_summary.txt` and printed to stdout
 - a visual block diagram saved to `<output_prefix>_architecture.png`

Usage:
    python plot_model_architecture.py --model-path <checkpoint.pth> --output-prefix mymodel

Optional:
    --spectrum-length N   : set spectrum length for a forward pass to inspect shapes

The script will try to infer `patch_size`, `d_model`, `dim_feedforward`, and `num_layers`
from the checkpoint's `model_state_dict`. If some values cannot be inferred, reasonable
defaults are used.
"""

import argparse
import torch
import matplotlib.pyplot as plt
import numpy as np
import textwrap
import os
from trainer_revised import NMRMaskedAutoencoder


def infer_arch_from_state(state_dict):
    """Try to infer patch_size, d_model, dim_feedforward and num_layers from state_dict."""
    info = {
        'patch_size': None,
        'd_model': None,
        'dim_feedforward': None,
        'num_layers': None,
    }

    keys = list(state_dict.keys())

    # Infer patch_size and d_model from patch_embedding weight
    for k in keys:
        if 'patch_embedding' in k and k.endswith('weight'):
            w = state_dict[k]
            try:
                info['d_model'] = int(w.shape[0])
                info['patch_size'] = int(w.shape[1])
            except Exception:
                pass
            break

    # Infer dim_feedforward from reconstruction head first linear
    for k in keys:
        if 'reconstruction_head' in k and k.endswith('weight'):
            # typical naming: encoder.reconstruction_head.0.weight -> [dim_feedforward, d_model]
            w = state_dict[k]
            if info['d_model'] is None:
                try:
                    # fallback: assume second dim is d_model
                    info['d_model'] = int(w.shape[1])
                except Exception:
                    pass
            try:
                if w.shape[1] == info['d_model']:
                    info['dim_feedforward'] = int(w.shape[0])
                    break
                else:
                    # maybe this is a different layer; continue searching
                    pass
            except Exception:
                pass

    # Infer num_layers by checking transformer layer keys
    layer_idxs = set()
    for k in keys:
        if 'transformer.layers.' in k:
            try:
                part = k.split('transformer.layers.')[1]
                idx = int(part.split('.')[0])
                layer_idxs.add(idx)
            except Exception:
                pass
    if layer_idxs:
        info['num_layers'] = int(max(layer_idxs) + 1)

    return info


def build_model_from_state(state_dict, spectrum_length=None, fallback=None):
    # Fallback defaults
    if fallback is None:
        fallback = {'patch_size': 1024, 'd_model': 128, 'num_layers': 3, 'dim_feedforward': 256, 'dropout': 0.2}

    inferred = infer_arch_from_state(state_dict)
    patch_size = inferred['patch_size'] or fallback['patch_size']
    d_model = inferred['d_model'] or fallback['d_model']
    num_layers = inferred['num_layers'] or fallback['num_layers']
    dim_feedforward = inferred['dim_feedforward'] or fallback['dim_feedforward']
    dropout = fallback.get('dropout', 0.2)

    # Try to infer nhead (attention heads) by searching for in_proj_weight shapes
    nhead = fallback.get('nhead', 4)
    # state_dict may be either the nested ckpt or raw state dict
    sd = state_dict if isinstance(state_dict, dict) else {}
    # If nested, try to extract model_state_dict
    if 'model_state_dict' in sd:
        sd = sd['model_state_dict']

    try:
        for k, v in sd.items():
            if k.endswith('in_proj_weight'):
                # in_proj_weight shape is (3*d_model, d_model)
                shape0 = v.shape[0]
                possible_d_model = v.shape[1]
                if possible_d_model and d_model and possible_d_model == d_model:
                    # cannot directly get nhead; assume common choices
                    # try to find head_dim by looking for in_proj_bias split or using fallback
                    # We'll leave nhead as fallback unless explicit key found
                    pass
            # Some implementations store 'self_attn.num_heads' or similar
            if 'num_heads' in k:
                try:
                    nhead = int(np.array(v).item())
                except Exception:
                    pass
    except Exception:
        pass

    if spectrum_length is None:
        # choose spectrum length as patch_size * 16 (arbitrary) so model positional encodings exist
        spectrum_length = patch_size * 16

    model = NMRMaskedAutoencoder(
        spectrum_length=spectrum_length,
        patch_size=patch_size,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout
    )

    # try load state dict permissively
    try:
        model.load_state_dict(state_dict, strict=False)
    except Exception as e:
        # try nested key
        if 'model_state_dict' in state_dict:
            try:
                model.load_state_dict(state_dict['model_state_dict'], strict=False)
            except Exception:
                pass
    return model, {'patch_size': patch_size, 'd_model': d_model, 'num_layers': num_layers, 'dim_feedforward': dim_feedforward, 'spectrum_length': spectrum_length, 'nhead': nhead}


def textual_summary(model, out_path=None):
    lines = []
    lines.append('Model textual summary')
    lines.append('-' * 60)
    total_params = 0
    for name, param in model.named_parameters():
        pcount = param.numel()
        total_params += pcount
        lines.append(f"{name:60s}  {pcount:12,d}")
    lines.append('-' * 60)
    lines.append(f"Total parameters: {total_params:,}")
    text = '\n'.join(lines)
    print(text)
    if out_path:
        with open(out_path, 'w') as f:
            f.write(text)
    return text


def draw_architecture(meta, out_png, out_pdf=None):
    """Draw a simple block-diagram of the model using matplotlib.
    meta: dict with keys patch_size,d_model,num_layers,dim_feedforward,spectrum_length
    """
    patch_size = meta['patch_size']
    d_model = meta['d_model']
    num_layers = meta['num_layers']
    dim_ff = meta['dim_feedforward']
    spec_len = meta['spectrum_length']
    n_patches = spec_len // patch_size
    nhead = meta.get('nhead', 4)
    head_dim = int(d_model // nhead) if nhead and d_model else None

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.axis('off')

    # Blocks positions
    x = 0.05
    y = 0.5
    h = 0.18
    w_block = 0.18
    gap = 0.02

    def rect(x, y, w, h, label, fontsize=10, facecolor='#cfe2f3'):
        r = plt.Rectangle((x, y-h/2), w, h, facecolor=facecolor, edgecolor='k')
        ax.add_patch(r)
        ax.text(x + w/2, y, label, ha='center', va='center', fontsize=fontsize, wrap=True)
        return r

    # Input
    rect(x, y, w_block, h, f'Input\nSpectrum (L={spec_len})', fontsize=10)
    x += w_block + gap

    # Patch embedding
    rect(x, y, w_block, h, f'Patch Embedding\nLinear({patch_size}→{d_model})', fontsize=10)
    x += w_block + gap

    # Positional Encoding
    rect(x, y, w_block, h, f'Positional Encoding\n(n_patches={n_patches})', fontsize=10)
    x += w_block + gap


    # Transformer stack (detailed)
    transformer_label = f'Transformer Encoder\n{num_layers} layers\nMulti-Head Self-Attention\n(nhead={nhead}, head_dim={head_dim})\nFeedForward dim={dim_ff}'
    rect(x, y, w_block, h, transformer_label, fontsize=8)
    x += w_block + gap

    # Reconstruction head
    rect(x, y, w_block, h, f'Reconstruction Head\nLinear({d_model}→{dim_ff}→{patch_size})', fontsize=10)
    x += w_block + gap

    # Output
    rect(x, y, w_block, h, f'Output\nReconstructed Spectrum', fontsize=10)

    # Add arrows between major blocks
    ax.annotate('', xy=(0.12, y), xytext=(0.0, y), arrowprops=dict(arrowstyle='->'))
    ax.annotate('', xy=(0.3, y), xytext=(0.12, y), arrowprops=dict(arrowstyle='->'))
    ax.annotate('', xy=(0.48, y), xytext=(0.3, y), arrowprops=dict(arrowstyle='->'))
    ax.annotate('', xy=(0.66, y), xytext=(0.48, y), arrowprops=dict(arrowstyle='->'))
    ax.annotate('', xy=(0.84, y), xytext=(0.66, y), arrowprops=dict(arrowstyle='->'))

    # Mask token and skip connection annotations
    ax.text(0.25, 0.8, 'Mask token inserted into embeddings for masked patches', ha='center', fontsize=9)
    ax.text(0.6, 0.2, 'Skip connection: original patches added to reconstruction (scaled)', ha='center', fontsize=9)

    # Draw detailed transformer internal flow under the transformer block
    tx = 0.48 - w_block/2
    ty = y
    inner_w = w_block
    inner_h = h * 0.9
    # draw small blocks vertically inside transformer region
    sub_x = tx + 0.02
    sub_w = inner_w - 0.04
    sub_h = inner_h / 5
    sub_y_top = ty + inner_h/2 - sub_h/2
    # Attention
    att_rect = plt.Rectangle((sub_x, sub_y_top - sub_h/2), sub_w, sub_h, facecolor='#fde9d9', edgecolor='k')
    ax.add_patch(att_rect)
    ax.text(sub_x + sub_w/2, sub_y_top, f'Multi-Head Self-Attention\n(nhead={nhead}, head_dim={head_dim})', ha='center', va='center', fontsize=7)
    # Add&Norm
    ay = sub_y_top - sub_h - 0.01
    an_rect = plt.Rectangle((sub_x, ay - sub_h/2), sub_w, sub_h, facecolor='#e2efda', edgecolor='k')
    ax.add_patch(an_rect)
    ax.text(sub_x + sub_w/2, ay, 'Add & Norm (residual)', ha='center', va='center', fontsize=7)
    # Feed-forward
    fy = ay - sub_h - 0.01
    ff_rect = plt.Rectangle((sub_x, fy - sub_h/2), sub_w, sub_h, facecolor='#fde2f3', edgecolor='k')
    ax.add_patch(ff_rect)
    ax.text(sub_x + sub_w/2, fy, f'FeedForward\n({d_model}→{dim_ff}→{d_model})', ha='center', va='center', fontsize=7)
    # Add&Norm 2
    ay2 = fy - sub_h - 0.01
    an2_rect = plt.Rectangle((sub_x, ay2 - sub_h/2), sub_w, sub_h, facecolor='#e2efda', edgecolor='k')
    ax.add_patch(an2_rect)
    ax.text(sub_x + sub_w/2, ay2, 'Add & Norm (residual)', ha='center', va='center', fontsize=7)

    # Small arrows for residuals
    ax.annotate('', xy=(sub_x - 0.01, sub_y_top), xytext=(sub_x + sub_w + 0.01, sub_y_top), arrowprops=dict(arrowstyle='->', linestyle='--'))
    ax.annotate('', xy=(sub_x - 0.01, fy), xytext=(sub_x + sub_w + 0.01, fy), arrowprops=dict(arrowstyle='->', linestyle='--'))

    # Title
    ax.set_title('NMR Masked Autoencoder — Architecture Overview', fontsize=14)

    plt.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches='tight')
    if out_pdf:
        fig.savefig(out_pdf, bbox_inches='tight')
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description='Plot model architecture and textual summary for NMR MAE')
    p.add_argument('--model-path', required=False, default=None, help='Path to checkpoint (.pth). If omitted, first .pth in models/SSL_models/ will be used')
    p.add_argument('--output-prefix', default='nmr_mae', help='Output prefix for files')
    p.add_argument('--spectrum-length', type=int, default=131072, help='(optional) spectrum length to run a dummy forward pass')
    p.add_argument('--save-pdf', action='store_true', help='Also save architecture as PDF')
    args = p.parse_args()

    model_path = args.model_path
    if model_path is None:
        # try to auto-discover a checkpoint in models/SSL_models/
        candidates = []
        try:
            for fname in os.listdir('models/SSL_models'):
                if fname.endswith('.pth'):
                    candidates.append(os.path.join('models/SSL_models', fname))
        except Exception:
            candidates = []
        if not candidates:
            raise FileNotFoundError('No checkpoint provided and none found in models/SSL_models/')
        model_path = candidates[0]
        print(f'Auto-selected checkpoint: {model_path}')

    ckpt = torch.load(model_path, map_location='cpu')
    state = ckpt.get('model_state_dict', ckpt)

    model, meta = build_model_from_state(state, spectrum_length=args.spectrum_length)

    # textual summary
    out_txt = f"{args.output_prefix}_summary.txt"
    textual_summary(model, out_txt)

    # simple shape inspection via forward pass if spectrum length provided
    if args.spectrum_length is not None:
        model.eval()
        with torch.no_grad():
            dummy = torch.randn(1, args.spectrum_length)
            # build a dummy mask of shape (n_patches)
            n_patches = args.spectrum_length // meta['patch_size']
            mask = torch.zeros(n_patches, dtype=torch.bool).unsqueeze(0)
            out = model(dummy, mask)
            # out is (reconstructed, encoded)
            try:
                reconstructed, encoded = out
                print('\nForward pass shapes:')
                print(' reconstructed:', tuple(reconstructed.shape))
                print(' encoded:', tuple(encoded.shape))
            except Exception:
                pass

    # visual
    out_png = f"{args.output_prefix}_architecture.png"
    out_pdf = f"{args.output_prefix}_architecture.pdf" if args.save_pdf else None
    draw_architecture(meta, out_png, out_pdf)
    print(f"Saved architecture visual to {out_png}")
    if out_pdf:
        print(f"Saved architecture visual to {out_pdf}")


if __name__ == '__main__':
    main()
