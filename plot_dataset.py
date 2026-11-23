"""Plot a HySI-SAT dataset (one comma-separated row per signal) against time, one panel per signal.

Usage:
    python plot_dataset.py Data/hpc_g3_1Hz_output.txt --fs 1
    python plot_dataset.py Data/hpc_g3_250Hz_output.txt --fs 250 --start 100 --length 60
--start and --length are in seconds. The figure is saved next to the dataset as <name>.png;
add --show to also open it.
"""
import argparse
from pathlib import Path

import matplotlib
import numpy as np

p = argparse.ArgumentParser()
p.add_argument("output", help="dataset, e.g. Data/hpc_g3_1Hz_output.txt")
p.add_argument("--fs", type=float, default=1000.0, help="sampling rate in Hz (default: 1000)")
p.add_argument("--start", type=float, default=0.0, help="first second to plot (default: 0)")
p.add_argument("--length", type=float, help="number of seconds to plot (default: all)")
p.add_argument("--labels", help="comma-separated signal names (default: row 0, row 1, ...)")
p.add_argument("--show", action="store_true", help="open the figure as well as saving it")
args = p.parse_args()
if not args.show:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

x = np.loadtxt(args.output, delimiter=",", ndmin=2)
i0 = int(args.start * args.fs)
i1 = x.shape[1] if args.length is None else min(x.shape[1], i0 + int(args.length * args.fs))
t = np.arange(i0, i1) / args.fs
labels = args.labels.split(",") if args.labels else [f"row {i}" for i in range(x.shape[0])]

fig, axes = plt.subplots(x.shape[0], 1, sharex=True, squeeze=False,
                         figsize=(12, 1.6 * x.shape[0] + 1))
for ax, row, label in zip(axes[:, 0], x, labels):
    ax.plot(t, row[i0:i1], lw=0.7, color="C0")
    ax.set_ylabel(label)
    ax.grid(alpha=0.3)
axes[-1, 0].set_xlabel("time (s)")
axes[-1, 0].set_xlim(t[0], t[-1])
fig.suptitle(f"{Path(args.output).name}: {x.shape[0]} signals, {args.fs:g} Hz, {t[0]:.0f}-{t[-1]:.0f} s")
fig.tight_layout()

png = Path(args.output).with_suffix(".png")
fig.savefig(png, dpi=120)
print(f"saved {png}")
if args.show:
    plt.show()
