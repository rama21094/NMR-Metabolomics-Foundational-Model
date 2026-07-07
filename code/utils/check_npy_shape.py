import numpy as np
import os
from pathlib import Path

root = Path('.')

for path in sorted(root.rglob('*.npy')):
    try:
        arr = np.load(path, mmap_mode='r')
        print(f"{path}: shape={arr.shape}, dtype={arr.dtype}")
    except Exception as exc:
        print(f"Could not read array shape for {path}: {exc}")
