import numpy as np

# Load the .npy file
arr = np.load("data/source/nmr_spectra.npy")

# Convert each row to a tuple (hashable) and find unique rows
_, unique_idx, counts = np.unique(arr, axis=0, return_index=True, return_counts=True)

# Duplicate rows are where counts > 1
duplicate_rows = np.where(counts > 1)[0]

if len(duplicate_rows) == 0:
    print("No duplicate rows found.")
else:
    print("Duplicate rows found at indices:")
    for dup in duplicate_rows:
        # all row indices that match this duplicate
        dup_indices = np.where((arr == arr[unique_idx[dup]]).all(axis=1))[0]
        print(f"Row {dup} duplicates at indices: {dup_indices}")

