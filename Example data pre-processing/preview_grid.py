"""Preview how much a frame of the sleep video is blurred by grid-box averaging.

Helper for choosing --grid in make_video_dataset.py; not part of the pipeline.
Left: the original frame. Right: the same frame with every grid box replaced by
its average over brain pixels, exactly as make_video_dataset.py computes it
(background and near-constant pixels left out, boxes below --min-coverage dropped
and shown blank). The red outline on the original is the brain mask.

The "Box size" slider sets the box side in pixels; the grid it gives (ROWSxCOLS,
to pass as --grid) and the number of kept boxes (= number of outputs) are shown
in the title. The "Min std" slider sets the activity threshold of the mask (pass
it as --min-std). Controls: drag the sliders, or use the left/right arrow keys to
step through frames (shift = 50 frames) and up/down to change the box size.

Usage:
    python preview_grid.py sleep_video_half1.mat [--frame 999] [--box 24] [--min-std 0.01]
"""
import argparse

import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider

p = argparse.ArgumentParser()
p.add_argument("file")
p.add_argument("--frame", type=int, default=999, help="initial frame (0-based)")
p.add_argument("--box", type=int, default=24, help="initial box side in pixels")
p.add_argument("--min-coverage", type=float, default=0.5,
               help="drop boxes whose fraction of brain pixels is below this (default: 0.5)")
p.add_argument("--min-std", type=float, default=0.01,
               help="initial threshold: pixels whose std over time is below this are not brain")
args = p.parse_args()

frameFreq = 50

f = h5py.File(args.file, "r")
video = f[next(iter(f.keys()))]  # MATLAB H x W x N is stored transposed: N x W x H
numFrames = video.shape[0]
H, W = video.shape[2], video.shape[1]


def frame(k):
    return video[k].T  # back to MATLAB orientation (H x W)


# Per-pixel activity and colour limits from a sample of frames, as in make_video_dataset.py
sample = np.stack([frame(k) for k in np.linspace(0, numFrames - 1, 50, dtype=int)])
nonzero = np.any(sample != 0, axis=0)
pixel_std = sample.std(axis=0)
mask = nonzero & (pixel_std >= args.min_std)
vmin, vmax = np.nanpercentile(sample[:, nonzero], [1, 99])


def blurred(x, s):
    """Box-averaged frame for box side s, the grid it uses and the number of kept boxes."""
    gr, gc = max(1, round(H / s)), max(1, round(W / s))
    row_edges = np.linspace(0, H, gr + 1).astype(int)
    col_edges = np.linspace(0, W, gc + 1).astype(int)

    def box_sums(a):
        return np.add.reduceat(np.add.reduceat(a, row_edges[:-1], axis=0), col_edges[:-1], axis=1)

    counts = box_sums(mask.astype(float))
    keep = counts / np.outer(np.diff(row_edges), np.diff(col_edges)) >= args.min_coverage
    means = np.where(keep, box_sums(x * mask) / np.where(counts > 0, counts, 1), np.nan)
    # Expand each box mean back over its pixels; keep the brain outline for comparison
    out = np.repeat(np.repeat(means, np.diff(row_edges), axis=0), np.diff(col_edges), axis=1)
    out[~mask] = np.nan
    return out, gr, gc, keep.sum()


def masked(x):
    return np.where(mask, x, np.nan)


fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11, 7))
fig.subplots_adjust(bottom=0.22)
cmap = plt.get_cmap("viridis").copy()
cmap.set_bad("black")
im0 = ax0.imshow(np.where(nonzero, frame(args.frame), np.nan), vmin=vmin, vmax=vmax, cmap=cmap)
im1 = ax1.imshow(masked(frame(args.frame)), vmin=vmin, vmax=vmax, cmap=cmap)
outline = []
fig.colorbar(im1, ax=[ax0, ax1], shrink=0.8)
title = fig.suptitle("")

sFrame = Slider(fig.add_axes([0.15, 0.13, 0.6, 0.03]), "Frame", 0, numFrames - 1,
                valinit=args.frame, valstep=1)
sBox = Slider(fig.add_axes([0.15, 0.08, 0.6, 0.03]), "Box size [px]", 1, min(H, W),
              valinit=args.box, valstep=1)
sStd = Slider(fig.add_axes([0.15, 0.03, 0.6, 0.03]), "Min std", 0, 0.1,
              valinit=args.min_std, valfmt="%.3f")


def update_mask(_=None):
    global mask
    mask = nonzero & (pixel_std >= sStd.val)
    for c in outline:
        for line in getattr(c, "collections", [c]):  # matplotlib < 3.8 keeps one per level
            line.remove()
    outline[:] = [ax0.contour(mask, levels=[0.5], colors="red", linewidths=1)]
    ax0.set_title(f"Original ({H} x {W} px)\nbrain = {mask.sum() / nonzero.sum():.0%} "
                  f"of non-zero px")
    update()


def update(_=None):
    k, s = int(sFrame.val), int(sBox.val)
    x = frame(k)
    out, gr, gc, n = blurred(x, s)
    im0.set_data(np.where(nonzero, x, np.nan))
    im1.set_data(out)
    ax1.set_title(f"--grid {gr}x{gc}\n(~{H / gr:.1f} x {W / gc:.1f} px per box)")
    title.set_text(f"Frame {k + 1}/{numFrames}  (t = {k / frameFreq:.2f} s)   "
                   f"{n} of {gr * gc} boxes kept = {n} outputs")
    fig.canvas.draw_idle()


def on_key(event):
    step = {"right": 1, "left": -1, "shift+right": 50, "shift+left": -50}.get(event.key)
    if step:
        sFrame.set_val(int(np.clip(sFrame.val + step, 0, numFrames - 1)))
    step = {"up": 1, "down": -1}.get(event.key)
    if step:
        sBox.set_val(int(np.clip(sBox.val + step, 1, min(H, W))))


sFrame.on_changed(update)
sBox.on_changed(update)
sStd.on_changed(update_mask)
fig.canvas.mpl_connect("key_press_event", on_key)
update_mask()
plt.show()
