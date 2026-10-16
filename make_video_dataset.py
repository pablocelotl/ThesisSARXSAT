"""Turn the sleep voltage-imaging video into HySI-SAT text files by averaging over a grid.

Each frame (384 x 241, MATLAB orientation) is split into a ROWS x COLS grid and every
grid box is averaged, giving one output signal per box: rows = boxes, columns = frames.
Background pixels (exactly 0 in every frame, outside the brain) are left out of the
averages, and so are pixels that barely change over time (the near-zero halo around the
cortex left by the pre-processing; see --min-std). Boxes with too little brain in them
are dropped. A matching all-zero
dummy input is written too, as with make_io_dataset.py; use it with nu = 0.

The .mat files are MATLAB v7.3 (HDF5), so they are read with h5py, a block of frames
at a time. Several files are concatenated in time (e.g. half1 then half2).

Usage:
    python make_video_dataset.py "Example data pre-processing/sleep_video_half1.mat" \
        "Example data pre-processing/sleep_video_half2.mat" --grid 4x4 --start 0 --length 2000
"""
import argparse

import h5py
import numpy as np

from make_io_dataset import write_io_dataset

p = argparse.ArgumentParser()
p.add_argument("files", nargs="+", help="sleep_video_half*.mat files, in time order")
p.add_argument("--grid", default="4x4", help="ROWSxCOLS boxes per frame (default: 4x4)")
p.add_argument("--start", type=int, default=0, help="first frame (0-based, across all files)")
p.add_argument("--length", type=int, help="number of frames to use (default: all)")
p.add_argument("--min-coverage", type=float, default=0.5,
               help="drop boxes whose fraction of brain pixels is below this (default: 0.5)")
p.add_argument("--min-std", type=float, default=0.01,
               help="pixels whose std over time is below this are not brain (default: 0.01)")
p.add_argument("--name", help="output name in Data/ (default: sleep_video_grid<ROWS>x<COLS>)")
args = p.parse_args()

gr, gc = (int(n) for n in args.grid.lower().split("x"))
videos = [next(iter(h5py.File(f, "r").values())) for f in args.files]  # each stored N x W x H
numFrames = sum(v.shape[0] for v in videos)
H, W = videos[0].shape[2], videos[0].shape[1]
start = args.start
end = min(start + args.length, numFrames) if args.length else numFrames


def frames(i0, i1):
    """Frames i0..i1-1 across all files, as (n, H, W) in MATLAB orientation."""
    out, offset = [], 0
    for v in videos:
        a, b = max(i0 - offset, 0), min(i1 - offset, v.shape[0])
        if a < b:
            out.append(v[a:b].transpose(0, 2, 1))
        offset += v.shape[0]
    return np.concatenate(out)


# Box edges (boxes differ by at most one pixel when the frame does not divide evenly)
row_edges = np.linspace(0, H, gr + 1).astype(int)
col_edges = np.linspace(0, W, gc + 1).astype(int)


def box_sums(x):
    """Sum of x (n, H, W) over every grid box -> (n, gr, gc)."""
    return np.add.reduceat(np.add.reduceat(x, row_edges[:-1], axis=1), col_edges[:-1], axis=2)


# Brain mask: pixels that are non-zero in a sample of frames and change enough over it
sample = np.stack([frames(k, k + 1)[0] for k in np.linspace(0, numFrames - 1, 50, dtype=int)])
mask = np.any(sample != 0, axis=0) & (sample.std(axis=0) >= args.min_std)
counts = box_sums(mask[None].astype(float))[0]
areas = np.outer(np.diff(row_edges), np.diff(col_edges))
keep = counts / areas >= args.min_coverage
if not keep.any():
    raise SystemExit("No grid box has enough brain pixels; lower --min-coverage")

# Average each kept box over its brain pixels, a block of frames at a time
x = np.empty((keep.sum(), end - start))
block = 500
for i0 in range(start, end, block):
    i1 = min(i0 + block, end)
    sums = box_sums(frames(i0, i1) * mask)
    x[:, i0 - start:i1 - start] = (sums / np.where(counts > 0, counts, 1))[:, keep].T
    print(f"\rframes {i1 - start}/{end - start}", end="", flush=True)
print()

name = f"Data/{args.name or f'sleep_video_grid{gr}x{gc}'}"
write_io_dataset(x, name)

# Which grid box each output row is, so results can be mapped back onto the frame
with open(f"{name}_boxes.txt", "w") as f:
    f.write(f"# frames {start}..{end - 1} of {numFrames} (50 Hz), frame {H}x{W}, grid {gr}x{gc}, "
            f"min-std {args.min_std}, min-coverage {args.min_coverage}\n")
    f.write("# output_row, grid_row, grid_col, pixel_rows, pixel_cols, brain_fraction\n")
    for k, (r, c) in enumerate(zip(*np.nonzero(keep))):
        f.write(f"{k}, {r}, {c}, {row_edges[r]}-{row_edges[r + 1] - 1}, "
                f"{col_edges[c]}-{col_edges[c + 1] - 1}, {counts[r, c] / areas[r, c]:.2f}\n")
print(f"Kept {keep.sum()} of {gr * gc} boxes; box layout in {name}_boxes.txt")
