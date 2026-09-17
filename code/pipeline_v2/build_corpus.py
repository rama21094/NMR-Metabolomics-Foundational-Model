#!/usr/bin/env python3
"""Build an NMR corpus on ONE common ppm axis, with absolute referencing preserved.

Replaces read1D_align.py and create_workbench_cpmg_serum_plasma_npy.py, which
share two defects that took a week to diagnose and cannot be repaired after the
fact (docs/PI_outline.md 5v, 6a).

WHAT WAS WRONG
--------------
1. Referencing was computed correctly and then thrown away. Both old scripts built
   a correct per-spectrum axis from `procs` and then re-referenced every spectrum
   so that its RIGHTMOST DETECTED PEAK became 0 ppm:

       shift = reference_ppm - rightmost_peak_ppm
       aligned_ppm_axis = ppm_axis + shift

   That peak is not TSP and is not the same compound between spectra, so absolute
   chemical shift was destroyed for every spectrum in the corpus.

2. Each spectrum was resampled onto ITS OWN ppm range:

       new_x = np.linspace(start, end, target_points)   # start,end = this spectrum

   so a 12 ppm and a 20 ppm acquisition both ended up with 131,072 points but
   spanning different ppm. A peak occupied 33 points in one and 55 in the other.
   1,130 of 9,670 corpus rows (11.7%) are affected.

   Both scripts DID compute a common axis and never used it. This is the one-line
   difference that matters.

WHY IT CANNOT BE PATCHED AFTERWARDS
-----------------------------------
Seven separate attempts to recover the geometry by fitting the stored spectra
against each other all failed (6a). The correlation-versus-scale surface has a
broad plateau with a spurious optimum at compressive scales, so a wrong answer
scores about as well as the right one. Reprocessing from raw is the only sound fix,
and it is cheap: every parameter needed is in the Bruker files.

WHAT THIS SCRIPT DOES DIFFERENTLY
---------------------------------
* Interpolates every spectrum onto ONE axis, fixed by --ppm-high/--ppm-low, so a
  given index means the same chemical shift in every row. Points outside a
  spectrum's own acquired range are zero-filled and the coverage is recorded.
* Keeps absolute ppm from `procs:OFFSET`. No peak-based re-referencing at all.
  If you want a reference-standard correction, do it afterwards as a separate,
  auditable step -- not silently inside the reader.
* Defines every region in ppm, never in point indices. The old water step zeroed
  fixed indices 62500-68000, which is only the water window if every spectrum
  happens to share an axis -- exactly the assumption that was false.
* Writes a provenance CSV with one row per spectrum carrying the parameters the
  geometry depends on, so the corpus can always be audited without re-reading raw.

Usage
-----
    # look before you leap: scan parameters, write no spectra
    python build_corpus.py /path/to/PlasmaNMRData --out-prefix plasma --dry-run

    # build
    python build_corpus.py /path/to/PlasmaNMRData --out-prefix plasma

    # several trees onto the SAME axis, then concatenate safely
    python build_corpus.py /path/to/SerumNMRData   --out-prefix serum
    python build_corpus.py /path/to/Workbench      --out-prefix workbench

Only numpy and scipy are required; nmrglue is used if present but not needed.
"""
from __future__ import annotations

import argparse
import csv
import collections
import os
import re
import sys
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------
# The common axis. Every dataset we hold covers -0.5 to 10 ppm (checked against
# the acquisition parameters of all 9,665 mapped corpus rows plus all four
# evaluation cohorts), so nothing is zero-padded inside this window and no real
# signal is discarded outside it.
#
# 131,072 points over 10.5 ppm is 8.01e-5 ppm/point. The finest source in the
# corpus is 9.15e-5 (a 12 ppm sweep at SI=131072), so this loses no resolution
# anywhere. --points 65536 halves memory at 1.60e-4 ppm/point, which still gives
# 12-30 points across a typical 1-3 Hz line; that is a reasonable choice, but it
# is a lossy one baked into an expensive step, so it is not the default.
# --------------------------------------------------------------------------
PPM_HIGH_DEFAULT = 10.0
PPM_LOW_DEFAULT = -0.5
POINTS_DEFAULT = 131_072

# Solvent window, in ppm. The old pipeline used point indices 62500-68000, which
# corresponded to 4.435-5.276 ppm only under the axis it assumed.
WATER_PPM = (4.50, 5.10)


# ============================ Bruker reading ==============================

def _parse_jcamp(path: Path) -> dict:
    """Minimal JCAMP-DX reader for acqus/procs. No dependencies."""
    out: dict[str, str] = {}
    try:
        text = path.read_text(errors="replace")
    except OSError:
        return out
    for m in re.finditer(r"^##\$?([^=]+)=\s*(.*)$", text, re.MULTILINE):
        out[m.group(1).strip()] = m.group(2).strip()
    return out


def _num(d: dict, key: str, default=None):
    v = d.get(key)
    if v is None:
        return default
    v = v.strip().strip("<>")
    try:
        return float(v)
    except ValueError:
        return default


def read_processed_spectrum(pdata_dir: Path):
    """Return (intensity, ppm_axis, params) for a Bruker pdata directory.

    The ppm axis is the REAL one, from procs:OFFSET/SW_p/SF/SI. It is returned
    unmodified -- no alignment, no referencing, no shifting.
    """
    procs_path = pdata_dir / "procs"
    one_r = pdata_dir / "1r"
    if not procs_path.exists() or not one_r.exists():
        return None

    procs = _parse_jcamp(procs_path)
    sf = _num(procs, "SF")
    sw_p = _num(procs, "SW_p")
    offset = _num(procs, "OFFSET")
    si = _num(procs, "SI")
    if not (sf and sw_p and si) or offset is None:
        return None
    si = int(si)

    bytordp = int(_num(procs, "BYTORDP", 0) or 0)
    dtypp = int(_num(procs, "DTYPP", 0) or 0)
    endian = ">" if bytordp == 1 else "<"
    dtype = np.dtype(f"{endian}f8") if dtypp == 2 else np.dtype(f"{endian}i4")

    data = np.fromfile(one_r, dtype=dtype).astype(np.float64, copy=False)
    if data.size == 0:
        return None
    if data.size != si:
        # Trust SI: it is what the axis is defined against.
        data = data[:si] if data.size > si else np.pad(data, (0, si - data.size))

    nc_proc = _num(procs, "NC_proc", 0.0) or 0.0
    data = data * (2.0 ** nc_proc)

    sw_ppm = sw_p / sf
    ppm = np.linspace(offset, offset - sw_ppm, si)   # descending, as acquired

    acqus = _parse_jcamp(pdata_dir.parent.parent / "acqus")
    params = {
        "proc_SF": sf, "proc_SW_p": sw_p, "proc_OFFSET": offset, "proc_SI": si,
        "sw_ppm": sw_ppm,
        "ppm_high": float(ppm[0]), "ppm_low": float(ppm[-1]),
        "acqu_SFO1": _num(acqus, "SFO1", ""), "acqu_BF1": _num(acqus, "BF1", ""),
        "acqu_SW": _num(acqus, "SW", ""), "acqu_TD": _num(acqus, "TD", ""),
        "acqu_NS": _num(acqus, "NS", ""), "acqu_RG": _num(acqus, "RG", ""),
        "acqu_PULPROG": (acqus.get("PULPROG", "") or "").strip("<>"),
        "acqu_NUC1": (acqus.get("NUC1", "") or "").strip("<>"),
        "acqu_SOLVENT": (acqus.get("SOLVENT", "") or "").strip("<>"),
        "acqu_DATE": _num(acqus, "DATE", ""),
        "acqu_TE": _num(acqus, "TE", ""),
        "field_mhz": round(_num(acqus, "BF1", 0.0) or 0.0),
    }
    return data, ppm, params


def find_pdata_dirs(root: Path, procno: str | None):
    """One processed dataset per EXPERIMENT, deterministic order.

    An experiment is the directory holding `acqus`; under it sits `pdata/<n>`,
    often several. Taking them all inflates the set -- the old serum path list had
    3,577 entries for 1,962 experiments, and deduplication then had to clean up
    after it.

    But pinning `--procno 1` is wrong too: in MTBLS540 eleven experiments under
    Sample132/1321 are processed into `pdata/2` with no `pdata/1` at all, so that
    rule would drop them silently. The default 'auto' therefore takes ONE per
    experiment, preferring procno 1 when it exists and otherwise the numerically
    smallest present, and reports every experiment where it had to choose
    something else -- the dry run over PlasmaNMRData reports exactly 12.
    """
    by_exp: dict[Path, list[Path]] = {}
    for dirpath, _dirnames, filenames in os.walk(root):
        if "1r" in filenames and "procs" in filenames:
            p = Path(dirpath)
            exp = p.parent.parent              # .../<exp>/pdata/<n> -> <exp>
            by_exp.setdefault(exp, []).append(p)

    def key(p):
        try:
            return (0, int(p.name))
        except ValueError:
            return (1, 0)

    out, nonstandard = [], []
    for exp in sorted(by_exp):
        cands = sorted(by_exp[exp], key=key)
        if procno and procno != "auto":
            pick = [c for c in cands if c.name == procno]
            if not pick:
                continue
            out.append(pick[0])
            continue
        chosen = next((c for c in cands if c.name == "1"), cands[0])
        if chosen.name != "1":
            nonstandard.append(chosen)
        out.append(chosen)
    if nonstandard:
        print(f"  note: {len(nonstandard):,} experiments have no pdata/1; used "
              f"the lowest present instead, e.g.")
        for c in nonstandard[:4]:
            print(f"    procno {c.name:<8} {c}")
    return out


# ============================ Processing ==================================

def to_common_axis(y, ppm_src, ppm_target):
    """Interpolate onto the common axis. THE FIX.

    np.interp needs ascending x, and NMR axes descend, so both are reversed.
    Points outside the source range become 0 rather than being extrapolated:
    extrapolating invents signal where the spectrometer acquired none.
    """
    xs = ppm_src[::-1]
    ys = y[::-1]
    out = np.interp(ppm_target[::-1], xs, ys, left=0.0, right=0.0)[::-1]
    return out


def suppress_window_ppm(y, ppm_axis, lo_ppm, hi_ppm, mode="zero", rng=None):
    """Zero or noise-fill a window defined in PPM. Never in point indices."""
    m = (ppm_axis >= lo_ppm) & (ppm_axis <= hi_ppm)
    if not m.any():
        return y
    out = y.copy()
    if mode == "zero":
        out[m] = 0.0
        return out
    base = y[(ppm_axis >= 9.0) | (ppm_axis <= -0.2)]
    if base.size < 64:
        base = y[np.argsort(np.abs(y))[: max(64, y.size // 10)]]
    med = float(np.median(base))
    mad = float(np.median(np.abs(base - med)))
    sigma = 1.4826 * mad if mad > 0 else float(np.std(base)) or 1e-12
    rng = rng or np.random.default_rng(0)
    out[m] = rng.normal(med, sigma, size=int(m.sum()))
    return out


def fingerprint(M, lo, hi, nbins=600):
    """Mean-pool a contiguous band into `nbins`, then mean-centre and unit-norm.

    An earlier version sampled 600 points on an even comb across the band. At
    0.013 ppm spacing that comb steps over most linewidths, so it compared an
    aliased handful of points rather than the spectrum, and two genuinely
    different samples from one study could clear r>0.999 on it. Pooling uses
    every point in the band instead, so a high correlation means the spectra
    really do agree.
    """
    w = (hi - lo) // nbins
    if w < 1:
        w, nbins = 1, hi - lo
    B = M[:, lo:lo + w * nbins].astype(np.float32, copy=True)
    F = B.reshape(B.shape[0], nbins, w).mean(axis=2)
    F -= F.mean(axis=1, keepdims=True)
    F /= np.linalg.norm(F, axis=1, keepdims=True) + 1e-12
    return F


def dedupe(M, lo, hi, threshold):
    """Drop later near-duplicates, keeping the first occurrence, order preserved.

    The old implementation correlated full 131,072-point rows pairwise, which is
    O(n^2) over the whole array. This uses a fingerprint over a clean band, which
    is the same decision far more cheaply.
    """
    F = fingerprint(M, lo, hi)
    keep, kept_F, drops = [], [], []
    for i in range(F.shape[0]):
        if kept_F:
            sims = np.asarray(kept_F) @ F[i]
            j = int(np.argmax(sims))
            if sims[j] > threshold:
                drops.append((i, keep[j], float(sims[j])))
                continue
        keep.append(i)
        kept_F.append(F[i])
    return np.asarray(keep, dtype=np.int64), drops


# ============================== Main ======================================

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", help="directory tree to scan for Bruker pdata")
    ap.add_argument("--out-prefix", required=True)
    ap.add_argument("--out-dir", default=".")
    ap.add_argument("--ppm-high", type=float, default=PPM_HIGH_DEFAULT)
    ap.add_argument("--ppm-low", type=float, default=PPM_LOW_DEFAULT)
    ap.add_argument("--points", type=int, default=POINTS_DEFAULT)
    ap.add_argument("--procno", default="auto",
                    help="'auto' (default) takes ONE processed dataset per "
                         "experiment, preferring pdata/1 and otherwise the lowest "
                         "procno present. A fixed number restricts to exactly that "
                         "one, which is usually WRONG: MTBLS10958 is processed in "
                         "pdata/700, so --procno 1 would drop that study entirely. "
                         "The old pipeline took every pdata/<n>, giving 3,577 "
                         "entries for 1,962 serum experiments.")
    ap.add_argument("--pulprog", default="cpmg",
                    help="case-insensitive regex the acqus PULPROG must match. "
                         "Default 'cpmg'. THIS MATTERS: the corpus was built from "
                         "CPMG spectra only, because CPMG's T2 filter suppresses "
                         "the broad protein and lipid envelope. The raw trees still "
                         "contain other experiments -- PlasmaNMRData holds 14 "
                         "noesygppr1d and 11 jresgpprqf among 7,116 -- and a "
                         "J-resolved experiment is not a 1D spectrum at all. The "
                         "old pipeline excluded these upstream by copying only "
                         "CPMG folders, so a reader that walks the raw tree must "
                         "re-apply the filter or it silently changes the corpus. "
                         "Use '.' to accept everything.")
    ap.add_argument("--nuc1", default="1H",
                    help="required acqus NUC1; empty string to disable")
    ap.add_argument("--water", default="zero", choices=["zero", "noise", "keep"])
    ap.add_argument("--water-ppm", type=float, nargs=2, default=WATER_PPM)
    ap.add_argument("--normalise", default="maxabs", choices=["maxabs", "none"])
    ap.add_argument("--dedupe", type=float, default=0.0,
                    help="correlation threshold, e.g. 0.999. 0 disables.")
    ap.add_argument("--min-coverage", type=float, default=0.99,
                    help="reject a spectrum whose acquired range covers less than "
                         "this fraction of the common axis. Such a spectrum would "
                         "be mostly zero-padding.")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--dry-run", action="store_true",
                    help="scan parameters and report, write nothing")
    args = ap.parse_args()

    root = Path(args.root).expanduser().resolve()
    outdir = Path(args.out_dir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    ppm_target = np.linspace(args.ppm_high, args.ppm_low, args.points)
    dppm = (args.ppm_high - args.ppm_low) / (args.points - 1)
    print(f"common axis: {args.ppm_high:+.3f} to {args.ppm_low:+.3f} ppm, "
          f"{args.points:,} points, {dppm:.4e} ppm/point")

    dirs = find_pdata_dirs(root, args.procno)
    print(f"found {len(dirs):,} Bruker pdata directories under {root}")
    if not dirs:
        sys.exit("nothing to do")

    # ---- pass 1: parameters only -----------------------------------------
    pul_re = re.compile(args.pulprog, re.I) if args.pulprog else None
    rows, widths = [], {}
    rejected = collections.Counter()
    for i, d in enumerate(dirs):
        r = read_processed_spectrum(d)
        if r is None:
            rows.append((d, None))
            rejected["unreadable"] += 1
            continue
        _, ppm, p = r
        pp = str(p.get("acqu_PULPROG", ""))
        if pul_re and not pul_re.search(pp):
            rows.append((d, None))
            rejected[f"PULPROG={pp or '(blank)'}"] += 1
            continue
        if args.nuc1 and str(p.get("acqu_NUC1", "") or "").strip() not in ("", args.nuc1):
            rows.append((d, None))
            rejected[f"NUC1={p.get('acqu_NUC1')}"] += 1
            continue
        rows.append((d, p))
        widths[round(p["sw_ppm"], 3)] = widths.get(round(p["sw_ppm"], 3), 0) + 1
        if (i + 1) % 500 == 0:
            print(f"\r  scanned {i + 1}/{len(dirs)}", end="", flush=True)
    print(f"\r  scanned {len(dirs)}/{len(dirs)}")
    ok = [r for r in rows if r[1] is not None]
    print(f"  accepted: {len(ok):,}   excluded: {len(rows) - len(ok):,}")
    if rejected:
        print("  excluded by reason:")
        for k, v in rejected.most_common():
            print(f"    {k:<40} {v:,}")
    print("\n  spectral widths present (ppm -> n):")
    for w in sorted(widths):
        print(f"    {w:<10} {widths[w]:,}")
    print("\n  This spread is exactly what the old pipeline mishandled. Here every")
    print("  one of these is interpolated onto the single axis above.")

    short = [(d, p) for d, p in ok
             if not (p["ppm_high"] >= args.ppm_high and p["ppm_low"] <= args.ppm_low)]
    if short:
        print(f"\n  !! {len(short):,} spectra do not fully cover "
              f"{args.ppm_high}..{args.ppm_low} ppm and will be zero-padded there.")
        for d, p in short[:5]:
            print(f"     {p['ppm_high']:+.2f}..{p['ppm_low']:+.2f}  {d}")
    if args.dry_run:
        print("\ndry run: nothing written")
        return

    # ---- pass 2: build ----------------------------------------------------
    keep_dirs = [d for d, p in ok]
    keep_par = [p for d, p in ok]
    n = len(keep_dirs)
    spec = np.lib.format.open_memmap(
        outdir / f"{args.out_prefix}_spectra.npy", mode="w+",
        dtype=np.float32, shape=(n, args.points))
    cover = np.zeros(n)
    rng = np.random.default_rng(0)

    for i, (d, p) in enumerate(zip(keep_dirs, keep_par)):
        r = read_processed_spectrum(d)
        if r is None:
            continue
        y, ppm_src, _ = r
        if args.water != "keep":
            y = suppress_window_ppm(y, ppm_src, args.water_ppm[0],
                                    args.water_ppm[1], args.water, rng)
        z = to_common_axis(y, ppm_src, ppm_target)
        cover[i] = float(np.mean((ppm_target <= p["ppm_high"]) &
                                 (ppm_target >= p["ppm_low"])))
        if args.normalise == "maxabs":
            m = float(np.max(np.abs(z)))
            if m > 0:
                z = z / m
        spec[i] = z.astype(np.float32)
        if (i + 1) % 200 == 0:
            print(f"\r  built {i + 1}/{n}", end="", flush=True)
    print(f"\r  built {n}/{n}")
    spec.flush()

    idx = np.arange(n)
    bad = cover < args.min_coverage
    if bad.any():
        print(f"  dropping {int(bad.sum()):,} spectra below "
              f"{args.min_coverage:.0%} axis coverage")
        idx = idx[~bad]

    if args.dedupe > 0:
        lo, hi = int(0.15 * args.points), int(0.85 * args.points)
        sub = np.array(spec[idx])
        k, drops = dedupe(sub, lo, hi, args.dedupe)
        print(f"  dedupe at r>{args.dedupe}: keeping {len(k):,} of {len(idx):,}")
        if drops:
            dpath = outdir / f"{args.out_prefix}_dropped.csv"
            with open(dpath, "w", newline="") as fh:
                w = csv.writer(fh)
                w.writerow(["dropped_pdata_path", "duplicate_of_pdata_path", "r"])
                for i, j, r in drops:
                    w.writerow([str(keep_dirs[idx[i]]), str(keep_dirs[idx[j]]),
                                f"{r:.6f}"])
            print(f"  wrote {dpath}  ({len(drops):,} rows) -- check these before"
                  f" trusting the dedupe")
        idx = idx[k]

    final = outdir / f"{args.out_prefix}_spectra.npy"
    if len(idx) != n:
        tmp = outdir / f"{args.out_prefix}_spectra_final.npy"
        out = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.float32,
                                        shape=(len(idx), args.points))
        for j in range(0, len(idx), args.batch):
            sl = idx[j:j + args.batch]
            out[j:j + len(sl)] = spec[sl]
        out.flush(); del out, spec
        os.replace(tmp, final)
    else:
        del spec

    np.save(outdir / f"{args.out_prefix}_ppm_axis.npy", ppm_target)

    prov = outdir / f"{args.out_prefix}_provenance.csv"
    cols = ["row", "pdata_path", "axis_coverage", "sw_ppm", "ppm_high", "ppm_low",
            "proc_OFFSET", "proc_SF", "proc_SW_p", "proc_SI", "field_mhz",
            "acqu_SFO1", "acqu_BF1", "acqu_SW", "acqu_TD", "acqu_NS", "acqu_RG",
            "acqu_PULPROG", "acqu_NUC1", "acqu_SOLVENT", "acqu_DATE", "acqu_TE"]
    with open(prov, "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(cols)
        for newrow, i in enumerate(idx):
            p = keep_par[i]
            w.writerow([newrow, str(keep_dirs[i]), f"{cover[i]:.4f}"] +
                       [p.get(c, "") for c in cols[3:]])

    print(f"\nwrote {final}  ({len(idx):,} x {args.points:,})")
    print(f"wrote {outdir / (args.out_prefix + '_ppm_axis.npy')}")
    print(f"wrote {prov}")
    print("\nEvery row is on the same ppm axis and keeps its acquired absolute")
    print("referencing. Verify with code/pipeline_v2/verify_corpus.py before use.")


if __name__ == "__main__":
    main()
