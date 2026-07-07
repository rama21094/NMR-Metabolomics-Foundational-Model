#!/usr/bin/env python3
"""
Compare summary JSON files (different mask ratios) and plot comparative charts
for Masked, Unmasked and Peaks metrics.

Usage:
    python plot_compare_summaries.py --input-dir results/testing/combined --out-dir results/testing/combined/compare_plots

The script will look for `summary*.json` files under `--input-dir` and attempt to
parse mask ratios from filenames (e.g. `summary_25mask500epoch.json` -> 25%).
If a ratio cannot be parsed, the file will be included and labelled by filename.
"""

import argparse
import json
import os
import re
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def find_summary_files(input_dir, explicit_files=None):
    if explicit_files:
        return [Path(f) for f in explicit_files]
    p = Path(input_dir)
    files = sorted(p.glob('summary*.json'))
    return files


def parse_mask_ratio_from_name(name):
    s = name
    # Look for patterns like 25mask, 35mask, 50mask, or _25_
    m = re.search(r'(\d{1,3})(?=\s*mask)', s, flags=re.IGNORECASE)
    if not m:
        m = re.search(r'_(\d{1,3})mask', s, flags=re.IGNORECASE)
    if not m:
        # fallback: any number followed by 'mask' or '%'
        m = re.search(r'(\d{1,3})(?=%)', s)
    if m:
        val = int(m.group(1))
        # interpret as percent
        return float(val) / 100.0, f'{val}%'

    # Special case: a plain 'summary.json' (user told this was 15%) -> treat as 15%
    if 'summary.json' in s or s.endswith('summary.json'):
        return 0.15, '15%'

    # Can't parse: return None and use filename as label
    return None, s


def load_json(fn: Path):
    with open(fn, 'r') as f:
        return json.load(f)


def collect_metrics(files):
    entries = []
    for f in files:
        data = load_json(f)
        ratio, label = parse_mask_ratio_from_name(f.name)
        entries.append({'path': f, 'ratio': ratio, 'label': label, 'data': data})
    # sort by ratio when available, otherwise by filename
    entries = sorted(entries, key=lambda e: (e['ratio'] is None, e['ratio'] if e['ratio'] is not None else e['label']))
    return entries


def extract_metric(entry, category, metric):
    # Return (mean, std) if available, else (np.nan, np.nan)
    try:
        cat = entry['data'][category]
        stat = cat[metric]
        return float(stat.get('mean', np.nan)), float(stat.get('std', np.nan))
    except Exception:
        return np.nan, np.nan


def plot_group(entries, category, metrics, out_path):
    labels = [e['label'] for e in entries]
    x = np.arange(len(entries))

    n_metrics = len(metrics)
    fig, axes = plt.subplots(n_metrics, 1, figsize=(6, 3*n_metrics), sharex=True)
    if n_metrics == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics):
        means = []
        stds = []
        for e in entries:
            m, s = extract_metric(e, category, metric)
            means.append(m)
            stds.append(s)

        means = np.array(means, dtype=float)
        stds = np.array(stds, dtype=float)

        # Bar plot with error bars
        ax.bar(x, means, yerr=stds, capsize=6, alpha=0.8)
        ax.set_ylabel(metric)
        if metric.lower().find('mse') >= 0 or metric.lower().find('rmse') >= 0:
            # use log scale for error magnitudes if values span orders
            ax.set_yscale('log')

        ax.grid(True, alpha=0.3)

    plt.xticks(x, labels, rotation=45, ha='right')
    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description='Compare summary JSONs across mask ratios')
    p.add_argument('--input-dir', default='results/testing/combined', help='Directory containing summary JSON files')
    p.add_argument('--files', nargs='*', help='Explicit files to compare (overrides --input-dir)')
    p.add_argument('--out-dir', default='results/testing/combined/compare_plots', help='Output directory for comparison plots')
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    files = find_summary_files(args.input_dir, args.files)
    if not files:
        raise FileNotFoundError(f'No summary JSON files found in {args.input_dir}')

    entries = collect_metrics(files)

    # --- Masked metrics ---
    masked_metrics = ['mse_masked', 'mae_masked', 'pearson_masked', 'peak_recovery_masked']
    out_masked = os.path.join(args.out_dir, 'compare_masked.png')
    plot_group(entries, 'Masked', masked_metrics, out_masked)

    # --- Unmasked metrics ---
    unmasked_metrics = ['mse_unmasked', 'mae_unmasked', 'pearson_unmasked']
    out_unmasked = os.path.join(args.out_dir, 'compare_unmasked.png')
    plot_group(entries, 'Unmasked', unmasked_metrics, out_unmasked)

    # --- Peaks metrics ---
    peaks_metrics = ['n_peaks_original', 'n_peaks_reconstructed', 'peak_position_error', 'peak_intensity_error', 'peak_f1']
    out_peaks = os.path.join(args.out_dir, 'compare_peaks.png')
    plot_group(entries, 'Peaks', peaks_metrics, out_peaks)

    print('Saved comparison plots to', args.out_dir)


if __name__ == '__main__':
    main()
