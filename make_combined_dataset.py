"""Combine the cortex video boxes (50 Hz) and the hippocampal LFP channels (1 kHz) into one HySI-SAT dataset.

The box signals (one row per box, from make_video_dataset.py) are linearly interpolated to 1 kHz:
frame k is placed at LFP sample offset + 20*k, and the 19 samples between two frames lie on the
straight line between them. Only the time span covered by both recordings is kept. The output file
has the box rows first (in the order of the _boxes.txt file), then the hippocampal channels.
A matching all-zero dummy input is written too, as with make_io_dataset.py; use it with nu = 0.

Usage:
    python make_combined_dataset.py Data/sleep_video_grid8x5_output.txt \
        "Example data pre-processing/hpc_filtered.mat" --zscore --length 2000
"""
import argparse
from pathlib import Path

import numpy as np
import scipy.io as sio

from make_io_dataset import write_io_dataset

p = argparse.ArgumentParser()
p.add_argument("boxes", help="box signals at the video rate, e.g. Data/sleep_video_grid8x5_output.txt")
p.add_argument("hpc", help="hippocampal LFP .mat file, e.g. hpc_filtered.mat")
p.add_argument("--hpc-key", default="HPC_LFP_filtered", help="variable in the .mat file (default: HPC_LFP_filtered)")
p.add_argument("--video-fs", type=float, default=50.0, help="video frame rate in Hz (default: 50)")
p.add_argument("--fs", type=float, default=1000.0, help="LFP sampling rate in Hz (default: 1000)")
p.add_argument("--offset", type=int, default=0,
               help="LFP sample at which video frame 0 was taken (default: 0, both start together)")
p.add_argument("--start", type=int, default=0, help="first sample of the combined 1 kHz dataset to keep")
p.add_argument("--length", type=int, help="number of 1 kHz samples to keep (default: all)")
p.add_argument("--zscore", action="store_true",
               help="normalize every signal to zero mean and unit std (the LFP is ~1000x larger than the boxes)")
p.add_argument("--name", help="output name in Data/ (default: <boxes name>_hpc_1kHz)")
args = p.parse_args()

ratio = args.fs / args.video_fs
if ratio != int(ratio):
    raise SystemExit(f"fs / video-fs = {ratio} is not a whole number of LFP samples per frame")
ratio = int(ratio)

boxes = np.loadtxt(args.boxes, delimiter=",", ndmin=2)
hpc = np.stack([np.asarray(ch, dtype=float).ravel()
                for ch in sio.loadmat(args.hpc, squeeze_me=True)[args.hpc_key]])

# LFP sample index of every video frame, and the span covered by both recordings
frame_t = args.offset + ratio * np.arange(boxes.shape[1])
t0 = max(frame_t[0], 0)
t1 = min(frame_t[-1], hpc.shape[1] - 1) + 1
t0 += args.start
if args.length:
    t1 = min(t1, t0 + args.length)
if t1 <= t0:
    raise SystemExit("The video and LFP do not overlap in the requested range; check --offset/--start")
t = np.arange(t0, t1)

boxes_1k = np.stack([np.interp(t, frame_t, row) for row in boxes])
x = np.vstack((boxes_1k, hpc[:, t0:t1]))
if args.zscore:
    x = (x - x.mean(axis=1, keepdims=True)) / x.std(axis=1, ddof=1, keepdims=True)

print(f"Video: {boxes.shape[0]} boxes x {boxes.shape[1]} frames ({boxes.shape[1] / args.video_fs:.2f} s); "
      f"LFP: {hpc.shape[0]} channels x {hpc.shape[1]} samples ({hpc.shape[1] / args.fs:.2f} s)")
print(f"Keeping LFP samples {t0}..{t1 - 1} ({(t1 - t0) / args.fs:.2f} s); "
      f"rows 0..{boxes.shape[0] - 1} are boxes, rows {boxes.shape[0]}..{x.shape[0] - 1} are LFP channels")

name = f"Data/{args.name or Path(args.boxes).stem.removesuffix('_output') + '_hpc_1kHz'}"
write_io_dataset(x, name)
