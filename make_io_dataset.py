"""Convert an ECoG .mat file into HySI-SAT text files (one comma-separated row per channel).

The ECoG channels become the output file. HySI-SAT always reads an input file,
so a dummy all-zero input is written as well; use it with nu = 0.

Usage:
    python make_io_dataset.py Dataset/ECoG_A1_S1.mat --channels 0,1 --start 0 --length 2000
"""
import argparse
from pathlib import Path

import numpy as np
import scipy.io as sio

p = argparse.ArgumentParser()
p.add_argument("file")
p.add_argument("--channels", help="comma-separated channel indices (default: all)")
p.add_argument("--start", type=int, default=0)
p.add_argument("--length", type=int, help="number of samples to use (default: all)")
args = p.parse_args()

x = sio.loadmat(args.file)["ECOG"]
if args.channels:
    x = x[[int(c) for c in args.channels.split(",")]]
end = args.start + args.length if args.length else x.shape[1]
x = x[:, args.start:end]

name = f"Data/{Path(args.file).stem}"
np.savetxt(f"{name}_output.txt", x, fmt="%f", delimiter=",")
np.savetxt(f"{name}_input_dummy.txt", np.zeros((1, x.shape[1])), fmt="%f", delimiter=",")
print(f"Wrote {name}_output.txt ({x.shape[0]} channels x {x.shape[1]} samples) and {name}_input_dummy.txt")
