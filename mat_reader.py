"""Show the variables of a .mat file in the terminal, and optionally as a plot. Nothing is saved.

Lists every variable with its shape, then prints the values (large arrays shortened with ...).
MATLAB 7.3 files (e.g. the sleep videos) are HDF5 and can be huge, so for those only the shapes are listed,
and --plot shows the first frame of a 3D array.

Usage:
    python mat_reader.py results/wpow_cal60s/Data/IdentifiedDatasets/11_19_2026_21_50_29/Fmodels.mat
    python mat_reader.py <file>.mat --var Fmodels --full     # every value
    python mat_reader.py <file>.mat --plot                   # 2D: image, 1D: line
"""
import argparse
import sys

import h5py
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio

p = argparse.ArgumentParser()
p.add_argument("file", help=".mat file")
p.add_argument("--var", help="only this variable (default: all)")
p.add_argument("--full", action="store_true", help="print every value instead of a shortened view")
p.add_argument("--plot", action="store_true", help="also show every variable as a plot")
args = p.parse_args()

try:
    data = {k: v for k, v in sio.loadmat(args.file).items() if not k.startswith("__")}
except NotImplementedError:  # MATLAB 7.3: HDF5, read lazily
    data = dict(h5py.File(args.file, "r").items())
if args.var:
    data = {args.var: data[args.var]}

if args.full:
    np.set_printoptions(threshold=sys.maxsize, linewidth=200)
for name, v in data.items():
    print(f"{name}: shape {v.shape}, type {v.dtype}")
    if isinstance(v, np.ndarray):
        print(v, "\n")

if args.plot:
    for name, v in data.items():
        # 3D: first frame only. MATLAB 7.3 stores the dimensions reversed (frames first), MATLAB 5 doesn't.
        if v.ndim > 2:
            a = v[0].T if isinstance(v, h5py.Dataset) else v[..., 0]
        else:
            a = v[()]
        a = np.asarray(a, dtype=float)
        a = a.squeeze()
        plt.figure(figsize=(10, 5))
        if a.ndim == 2 and min(a.shape) > 1:
            plt.imshow(a, aspect="auto", cmap="RdBu_r", vmin=-np.abs(a).max(), vmax=np.abs(a).max(),
                       interpolation="nearest")
            plt.colorbar()
            plt.xlabel("column")
            plt.ylabel("row")
        else:
            plt.plot(a.ravel())
            plt.xlabel("index")
        plt.title(f"{name} {v.shape}" + (" (first frame)" if v.ndim > 2 else ""))
    plt.show()
