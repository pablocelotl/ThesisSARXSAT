"""Interactive frame viewer for sleep_video_half1.mat / sleep_video_half2.mat.

The .mat files are MATLAB v7.3 (HDF5), so they are read with h5py. Frames are
read one at a time from disk, so the multi-GB video never has to fit in RAM.

Controls: drag the slider, or use the left/right arrow keys (shift = 50 frames).

Usage:
    python view_sleep_video.py sleep_video_half1.mat [--frame 999]
"""
import argparse

import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider

p = argparse.ArgumentParser()
p.add_argument("file")
p.add_argument("--frame", type=int, default=999, help="initial frame (0-based)")
args = p.parse_args()

frameFreq = 50

f = h5py.File(args.file, "r")
video = f[next(iter(f.keys()))]  # MATLAB H x W x N is stored transposed: N x W x H
numFrames = video.shape[0]


def frame(k):
    return video[k].T  # back to MATLAB orientation (H x W)


# Fixed colour limits from a sample of frames, so brightness is comparable between frames
sample = np.stack([frame(k) for k in np.linspace(0, numFrames - 1, 50, dtype=int)])
vmin, vmax = np.nanpercentile(sample, [1, 99])

fig, ax = plt.subplots()
fig.subplots_adjust(bottom=0.15)
im = ax.imshow(frame(args.frame), vmin=vmin, vmax=vmax, cmap="viridis")
fig.colorbar(im, ax=ax)
title = ax.set_title("")

slider = Slider(fig.add_axes([0.15, 0.04, 0.7, 0.03]), "Frame", 0, numFrames - 1,
                valinit=args.frame, valstep=1)


def update(k):
    k = int(k)
    im.set_data(frame(k))
    title.set_text(f"Frame {k + 1}/{numFrames}  (t = {k / frameFreq:.2f} s)")
    fig.canvas.draw_idle()


def on_key(event):
    step = {"right": 1, "left": -1, "shift+right": 50, "shift+left": -50}.get(event.key)
    if step:
        slider.set_val(int(np.clip(slider.val + step, 0, numFrames - 1)))


slider.on_changed(update)
fig.canvas.mpl_connect("key_press_event", on_key)
update(args.frame)
plt.show()
