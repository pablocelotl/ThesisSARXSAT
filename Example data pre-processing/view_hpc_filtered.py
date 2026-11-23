"""Scrollable viewer for hpc_filtered.mat (notch-filtered hippocampus LFP, 1 kHz).

Top: all channels stacked, in a time window you can scroll through.
Bottom: overview of one channel over the whole recording; the shaded band is
the window shown on top.

With --groups N the channels are colored by their group of correlated channels (lfp_groups.py),
and the N group means (the signals used in the combined dataset) are added in black at the bottom.

Controls: drag the slider, or use the left/right arrow keys (shift = 10 windows).

Usage:
    python view_hpc_filtered.py hpc_filtered.mat [--window 10] [--overview-channel 11] [--groups 3]
"""
import argparse

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
from matplotlib.widgets import Slider

p = argparse.ArgumentParser()
p.add_argument("file", nargs="?", default="hpc_filtered.mat")
p.add_argument("--window", type=float, default=10, help="window length in seconds")
p.add_argument("--overview-channel", type=int, default=11, help="1-based, as in MATLAB")
p.add_argument("--groups", type=int, help="color the channels by this many groups and show the group means")
args = p.parse_args()

fs = 1000
HPC_LFP_filtered = np.stack(sio.loadmat(args.file, squeeze_me=True)["HPC_LFP_filtered"])
numChannels, t = HPC_LFP_filtered.shape
duration = t / fs
win = int(args.window * fs)
spacing = 4 * np.median(HPC_LFP_filtered.std(axis=1))  # vertical gap between channels

# Rows to draw: the channels, followed by the group means when --groups is given
rows = HPC_LFP_filtered
labels = [f"ch {i}" for i in range(1, numChannels + 1)]
colors = [None] * numChannels
if args.groups:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # lfp_groups.py is in the repo root
    from lfp_groups import describe_groups, group_lfp_channels
    groups, grouped = group_lfp_channels(list(HPC_LFP_filtered), args.groups)
    describe_groups(list(HPC_LFP_filtered), groups)
    for g, group in enumerate(groups):
        for c in group:
            colors[c] = f"C{g}"
            labels[c] += f" (g{g})"
    rows = np.vstack((HPC_LFP_filtered, grouped))
    labels += [f"mean g{g}" for g in range(len(groups))]
    colors += ["k"] * len(groups)
numRows = rows.shape[0]

fig, (ax, ax_all) = plt.subplots(2, 1, gridspec_kw={"height_ratios": [4, 1]}, figsize=(12, 8))
fig.subplots_adjust(bottom=0.1, hspace=0.3)

# Stacked channels, channel 1 on top
lines = [ax.plot([], [], lw=0.6, color=c)[0] for c in colors]
ax.set_yticks([-i * spacing for i in range(1, numRows + 1)])
ax.set_yticklabels(labels)
ax.set_ylim(-(numRows + 1) * spacing, 0)
ax.set_xlabel("Time (s)")

# Whole-recording overview, decimated for speed
step = 50
ch = args.overview_channel - 1
ax_all.plot(np.arange(0, t, step) / fs, HPC_LFP_filtered[ch, ::step], lw=0.4, color="k")
ax_all.set_xlim(0, duration)
ax_all.set_xlabel("Time (s)")
ax_all.set_ylabel(f"ch {args.overview_channel}")
band = None

slider = Slider(fig.add_axes([0.15, 0.02, 0.7, 0.02]), "Start (s)", 0,
                max(duration - args.window, 0), valinit=0)


def update(start):
    i0 = int(start * fs)
    i1 = min(i0 + win, t)
    time = np.arange(i0, i1) / fs
    for i, line in enumerate(lines, start=1):
        line.set_data(time, rows[i - 1, i0:i1] - i * spacing)
    ax.set_xlim(i0 / fs, i1 / fs)
    ax.set_title(f"Filtered hippocampus LFP, {i0 / fs:.1f}-{i1 / fs:.1f} s of {duration:.1f} s")
    global band
    if band is not None:
        band.remove()
    band = ax_all.axvspan(i0 / fs, i1 / fs, color="tab:orange", alpha=0.3)
    fig.canvas.draw_idle()


def on_key(event):
    step = {"right": 1, "left": -1, "shift+right": 10, "shift+left": -10}.get(event.key)
    if step:
        slider.set_val(float(np.clip(slider.val + step * args.window, slider.valmin, slider.valmax)))


slider.on_changed(update)
fig.canvas.mpl_connect("key_press_event", on_key)
update(0)
plt.show()
