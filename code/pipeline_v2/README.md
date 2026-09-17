# Corrected preprocessing pipeline

Replaces `read1D_align.py` and `create_workbench_cpmg_serum_plasma_npy.py`.

## What was wrong (docs/PI_outline.md 5v)

Both old scripts:

1. Built a correct per-spectrum ppm axis from `procs`, then **discarded it** by
   re-referencing every spectrum so its *rightmost detected peak* became 0 ppm.
   That peak is not TSP and is not the same compound between spectra, so absolute
   chemical shift was destroyed corpus-wide.
2. Resampled each spectrum to 131,072 points **across its own ppm range**, so a
   12 ppm and a 20 ppm acquisition both had 131,072 points spanning different ppm.
   1,130 of 9,670 corpus rows (11.7%) are affected.

Both scripts computed a common axis and never used it. That is the one-line fix.

Seven attempts to repair this after the fact all failed (6a). Reprocessing is the
only sound route.

## Running it

```bash
python build_corpus.py /path/to/PlasmaNMRData --out-prefix plasma --dry-run
python build_corpus.py /path/to/PlasmaNMRData --out-prefix plasma
python build_corpus.py /path/to/SerumNMRData  --out-prefix serum
python build_corpus.py /path/to/MetabolomicsWorkBench_SingleFolder --out-prefix workbench
```

### Use exactly these three roots

Confirmed against the directory tree; the pdata counts match the parameter CSVs
exactly, which is what identifies them as the trees the `.npy` files came from:

| root | experiments | pdata |
|---|---|---|
| `PlasmaNMRData` | 7,051 | 7,116 |
| `SerumNMRData` | 1,961 | 1,961 |
| `MetabolomicsWorkBench_SingleFolder` | 82 studies | 950 |

**Do not use** `MetabolightsCPMGSingleFolder` (16,384 dirs — a superset that still
contains what was later discarded), `MetabolightsData` (raw download, all pulse
programmes), `SerumNMRData_discarded` or
`MetabolomicsWorkBench_SingleFolder_NonSelected` (both are rejects). The
serum/plasma split and the discards are hand curation that cannot be re-derived
from parameters.

Start with `--dry-run`: it reads only parameters and prints the spectral widths
present. That table is the thing the old pipeline got wrong, so look at it.

**Use the same `--ppm-high/--ppm-low/--points` for every tree.** They default to
+10.0 to -0.5 ppm over 131,072 points. Then the three arrays share an axis and
`np.concatenate` is meaningful — which was never true before.

**`--pulprog` matters as much as the axis fix.** The corpus is CPMG-only by
design: the T2 filter suppresses the broad protein and lipid envelope. The
original pipeline enforced that upstream, by copying only CPMG folders
(`CPMGtoOneFolder.py`) before any spectrum was read. A reader that walks the raw
tree must re-apply the filter or it silently changes what the corpus is.

How much this matters, measured: of 520 TBI `pdata` directories only **177** are
CPMG -- 225 are `noesypr1d`, 117 `zgpr`, 1 `zgesgp`. `PlasmaNMRData` holds 14
`noesygppr1d` and 11 `jresgpprqf` among 7,116, and a J-resolved experiment is not
a 1D spectrum at all. The default regex is `cpmg`; pass `--pulprog .` to accept
everything, and read the exclusion table the dry run prints.

**`--procno` defaults to `auto`, and leave it there.** An experiment often holds
several `pdata/<n>`; the old pipeline read them all, which is why its serum path
list had 3,577 entries for 1,962 experiments and why deduplication had so much to
remove. But pinning `--procno 1` is also wrong: MTBLS10958 is processed in
`pdata/700`, so that rule would silently drop the whole study — 370 corpus rows.
`auto` takes one per experiment, preferring `pdata/1` and otherwise the lowest
present, and prints every experiment where it had to choose something else.

Then, always:

```bash
python verify_corpus.py plasma_spectra.npy plasma_ppm_axis.npy plasma_provenance.csv
```

## Reading the verification

It groups rows by the spectral width they were *acquired* at and scans the scale
that best maps each group's median onto the largest group's. **If the rebuild
worked, every group peaks at scale 1.00 with a clear margin over its runner-up.**

That margin is the whole test. From the old corpus:

```
correct    best scale 1.00, peak 0.844, runner-up 0.507   <- sharp, dominant
incorrect  best scale 1.90, peak 0.527, runner-up 0.482   <- broad plateau
```

A high correlation alone proves nothing; seven false positives came from reading
one. The script prints `PASS` only when every group peaks at 1.00 with a margin
above 0.15.

Smoke test on real TBI Bruker data: with the CPMG filter off, the lone SW 13.016
spectrum sitting among 519 at SW 16.02 peaks at **scale 1.00, 0.872 against
runner-up 0.541** — a different acquisition width landing correctly on the common
axis. With the filter on, 177 of those 520 experiments remain.

## Landmark output

`verify_corpus.py` also reports where lactate, alanine, creatine and glucose
actually fall. Because absolute ppm is now preserved, these should be near their
literature positions. A **consistent** offset across all four means the dataset
needs a reference correction — real information the old pipeline destroyed. On the
TBI smoke test every landmark sat +0.03 to +0.05 ppm high, so TBI does need one.

Apply any such correction as a separate, logged step. Do not put peak-based
referencing back inside the reader; that is what caused this.

## What changed, beyond the two fixes

- **Regions are specified in ppm, never point indices.** The old water step zeroed
  indices 62500-68000, which is the water window only if every spectrum shares an
  axis — the assumption that was false. Now `--water-ppm 4.40 5.30`.
- **No extrapolation.** Points outside a spectrum's acquired range are zero-filled,
  not extrapolated, and the coverage fraction is recorded per row.
  `--min-coverage` drops spectra that would be mostly padding.
- **Provenance CSV per row** with the parameters the geometry depends on, so the
  corpus can be audited later without re-reading raw data.
- **Deduplication uses a fingerprint** over a clean band rather than correlating
  full 131,072-point rows pairwise. Same decision, far cheaper. Off by default;
  enable with `--dedupe 0.999`.
- `NC_proc` scaling is applied, which the old readers ignored. It affects absolute
  intensity, not geometry, but it makes rows comparable before normalisation.

## Axis choice

131,072 points over 10.5 ppm is 8.01e-5 ppm/point. The finest source in the corpus
is 9.15e-5, so nothing is downsampled. `--points 65536` halves memory at 1.60e-4
ppm/point, still 12-30 points across a typical 1-3 Hz line — a reasonable choice,
but lossy and baked into an expensive step, so not the default. Note it changes the
model's input length.

Every dataset we hold covers -0.5 to 10 ppm, so nothing is padded inside the
window and no real signal falls outside it.
