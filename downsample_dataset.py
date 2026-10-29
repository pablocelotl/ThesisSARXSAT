"""Downsample a HySI-SAT dataset (one comma-separated row per signal) to a lower sampling rate.

Every row is low-pass filtered before it is downsampled (scipy.signal.resample_poly), so that
faster oscillations do not fold back (alias) into the slow signal. A matching all-zero dummy
input is written too, as with make_io_dataset.py.

Usage:
    python downsample_dataset.py Data/hpc_g3_output.txt --fs 1000 --target 250 --zscore
writes Data/hpc_g3_250Hz_output.txt and Data/hpc_g3_250Hz_input_dummy.txt.
"""
import argparse
from fractions import Fraction
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly

from make_io_dataset import write_io_dataset

p = argparse.ArgumentParser()
p.add_argument("output", help="dataset to downsample, e.g. Data/hpc_g3_output.txt")
p.add_argument("--fs", type=float, default=1000.0, help="sampling rate of the dataset in Hz (default: 1000)")
p.add_argument("--target", type=float, required=True, help="new sampling rate in Hz")
p.add_argument("--zscore", action="store_true",
               help="normalize every signal to zero mean and unit std after downsampling")
p.add_argument("--name", help="output name in Data/ (default: <dataset name>_<target>Hz)")
args = p.parse_args()

if args.target >= args.fs:
    raise SystemExit("--target must be lower than --fs")
ratio = Fraction(args.target / args.fs).limit_denominator(10000)

x = np.loadtxt(args.output, delimiter=",", ndmin=2)
y = resample_poly(x, ratio.numerator, ratio.denominator, axis=1)
if args.zscore:
    y = (y - y.mean(axis=1, keepdims=True)) / y.std(axis=1, ddof=1, keepdims=True)

print(f"{x.shape[0]} signals: {x.shape[1]} samples at {args.fs:g} Hz -> {y.shape[1]} samples at "
      f"{args.target:g} Hz ({y.shape[1] / args.target:.1f} s)")
target = f"{args.target:g}".replace(".", "p")
name = f"Data/{args.name or Path(args.output).stem.removesuffix('_output') + f'_{target}Hz'}"
write_io_dataset(y, name)
