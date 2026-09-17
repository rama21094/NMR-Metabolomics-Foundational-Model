# Task: recover the row -> experiment mapping for the serum NMR array

## Goal

Produce one CSV:

    npy_row, experiment_relpath, acqu_SW, proc_OFFSET, field_mhz

one line per row of `aligned_nmr_spectra_128K_WSNoise.npy` (2,148 rows x 131,072,
float64). Nothing else is needed — no spectra, no plots.

## Why

The pre-training corpus contains 2,145 rows that came from this array. Each source
study was acquired with a different spectral width, and the preprocessing resampled
every spectrum to 131,072 points **across its own ppm range**, so ppm-per-point
varies by study and those spectra are stretched relative to the rest of the corpus.
To fix them we need each row's acquisition spectral width, which means knowing
which Bruker experiment each row came from. Two of the studies here (MTBLS10958,
MTBLS11188) were acquired at 11.9878 ppm against the corpus's 20.024 — a 1.67x
stretch — so this is not a marginal correction.

## What you have

- `aligned_nmr_spectra_128K_WSNoise.npy` — the array, 2,148 rows.
- `spectra_paths.txt` — 3,577 lines, the **pre-deduplication** serum path list, in
  the order the pipeline read them. Lines look like
  `SerumNMRData\MTBLS10958_MO_002_siero\20232604_MO_002_siero\2\pdata\1`
- `bruker_params_SerumNMRData.csv` — already has `acqu_SW`, `proc_OFFSET`,
  `field_mhz_rounded` per experiment, keyed on a `relpath` column such as
  `MTBLS10958_MO_002_siero/20232604_MO_002_siero/2`. **Do not re-extract these.**
- `read1D_align.py` — the pipeline that produced the array.
- The raw `SerumNMRData` tree.

## The key fact that makes this tractable

`read1D_align.py` deduplicates with `remove_duplicate_spectra()`, which walks the
spectra **in order**, keeps the first occurrence, and drops any later spectrum
correlating > 0.99 with one already kept. Order is preserved. So the 2,148 array
rows are an **ordered subsequence** of the 3,577 paths — row 0 is the first
surviving path, row 1 the next, and so on. Use that constraint; it makes the match
far more reliable than nearest-neighbour matching alone.

Note the paths file has ~3,577 entries for ~1,962 experiments because the same
experiment appears under several `pdata/<n>` processing numbers. That is most of
what deduplication removed.

## Suggested method

1. For each of the 3,577 paths in order, read the processed spectrum (`1r`) with
   `nmrglue`, and build its ppm axis exactly as `read1D_align.py` line 161 does:
   `np.linspace(offset, offset - sw/sfo1, si)` from `procs` OFFSET, SW_p, SF, SI.
2. Reproduce the pipeline's transform on it: align to the rightmost peak
   (`find_rightmost_peak` / `align_spectrum_to_reference`), normalise by
   `max(abs(spectrum))`, then `resample_to_length(..., 131072, method="linear")`.
   Water handling does not matter if you avoid the water region in step 3.
3. Fingerprint both your reproduced spectra and the array rows over a clean band —
   use array indices 85000:93500, subsampled to ~600 points, mean-centred and
   normalised to unit L2 norm. Avoid the water region and the EDTA region.
4. Walk the 3,577 reproduced spectra in order with a pointer into the 2,148 array
   rows. Assign a path to the current array row when their fingerprint correlation
   exceeds 0.999, then advance the pointer. Skip paths that do not match — those
   are the deduplicated ones.
5. Join each matched path to `bruker_params_SerumNMRData.csv` on the experiment
   relpath (strip the `SerumNMRData/` prefix and the trailing `/pdata/<n>`), and
   write the CSV.

## Acceptance checks — please report these

- How many of the 2,148 rows were assigned a path, and the distribution of the
  matching correlations (a clean match should be > 0.999).
- The count of rows per `acqu_SW` value. Expect a large group near 20.02-20.03 and
  a distinct group at 11.9878.
- Confirm the assigned paths are strictly increasing in `spectra_paths.txt` line
  number. If they are not, the ordered-subsequence assumption broke and the result
  should not be trusted.
- Any row left unassigned — report it rather than guessing.

If step 2 proves hard to reproduce exactly, say so rather than forcing a match: a
partial mapping with honest correlations is more useful than a complete one built
on a loose threshold.
