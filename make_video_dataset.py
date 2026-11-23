"""Turn the sleep voltage-imaging video into HySI-SAT text files by averaging over a grid.

Each frame (384 x 241, MATLAB orientation) is split into a ROWS x COLS grid and every
grid box is averaged, giving one output signal per box: rows = boxes, columns = frames.
Background pixels (exactly 0 in every frame, outside the brain) are left out of the
averages, and so are pixels that barely change over time (the near-zero halo around the
cortex left by the pre-processing; see --min-std). Boxes with too little brain in them
are dropped. With --areas, the cortical areas from brain_areas.py
replace the grid boxes (output Data/sleep_video_areas_*). A matching all-zero
dummy input is written too, as with make_io_dataset.py; use it with nu = 0.

The .mat files are MATLAB v7.3 (HDF5), so they are read with h5py, a block of frames
at a time. Several files are concatenated in time (e.g. half1 then half2).

Usage:
    python make_video_dataset.py "Example data pre-processing/sleep_video_half1.mat" \
        "Example data pre-processing/sleep_video_half2.mat" --grid 4x4 --start 0 --length 2000
    python make_video_dataset.py "Example data pre-processing/sleep_video_half1.mat" \
        "Example data pre-processing/sleep_video_half2.mat" --areas Data/brain_areas.txt
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
p.add_argument("--areas", help="average over the cortical areas in this label file (Data/brain_areas.txt, "
               "made with brain_areas.py) instead of the grid")
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


# Every pixel gets the number of its grid box, or of its cortical area with --areas (-1: none)
row_edges = np.linspace(0, H, gr + 1).astype(int)  # boxes differ by at most one pixel
col_edges = np.linspace(0, W, gc + 1).astype(int)
if args.areas:
    labels = np.loadtxt(args.areas, dtype=int)
    with open(args.areas) as f:
        area_names = [a.split(" = ")[1] for a in f.readline().lstrip("# ").strip().split(", ")]
else:
    labels = (np.searchsorted(row_edges, np.arange(H), side="right") - 1)[:, None] * gc \
        + (np.searchsorted(col_edges, np.arange(W), side="right") - 1)[None, :]
onehot = (labels.ravel()[:, None] == np.arange(labels.max() + 1)).astype(float)  # pixels x regions

# Brain mask: pixels that are non-zero in a sample of frames and change enough over it
sample = np.stack([frames(k, k + 1)[0] for k in np.linspace(0, numFrames - 1, 50, dtype=int)])
mask = np.any(sample != 0, axis=0) & (sample.std(axis=0) >= args.min_std)
counts = mask.ravel() @ onehot  # brain pixels per region
sizes = onehot.sum(axis=0)
keep = counts / np.maximum(sizes, 1) >= args.min_coverage
if not keep.any():
    raise SystemExit("No region has enough brain pixels; lower --min-coverage")

# Average each kept region over its brain pixels, a block of frames at a time
x = np.empty((keep.sum(), end - start))
block = 500
for i0 in range(start, end, block):
    i1 = min(i0 + block, end)
    sums = (frames(i0, i1) * mask).reshape(i1 - i0, -1) @ onehot[:, keep]
    x[:, i0 - start:i1 - start] = (sums / counts[keep]).T
    print(f"\rframes {i1 - start}/{end - start}", end="", flush=True)
print()

name = f"Data/{args.name or ('sleep_video_areas' if args.areas else f'sleep_video_grid{gr}x{gc}')}"
write_io_dataset(x, name)

# Which grid box or area each output row is, so results can be mapped back onto the frame
header = (f"# frames {start}..{end - 1} of {numFrames} (50 Hz), frame {H}x{W}, "
          f"{args.areas or f'grid {gr}x{gc}'}, min-std {args.min_std}, min-coverage {args.min_coverage}\n")
if args.areas:
    with open(f"{name}_areas.txt", "w") as f:
        f.write(header + "# output_row, area, brain_pixels, brain_fraction\n")
        for k, a in enumerate(np.flatnonzero(keep)):
            f.write(f"{k}, {area_names[a]}, {counts[a]:.0f}, {counts[a] / sizes[a]:.2f}\n")
    print(f"Kept {keep.sum()} of {len(keep)} areas; layout in {name}_areas.txt")
else:
    with open(f"{name}_boxes.txt", "w") as f:
        f.write(header + "# output_row, grid_row, grid_col, pixel_rows, pixel_cols, brain_fraction\n")
        for k, b in enumerate(np.flatnonzero(keep)):
            r, c = divmod(b, gc)
            f.write(f"{k}, {r}, {c}, {row_edges[r]}-{row_edges[r + 1] - 1}, "
                    f"{col_edges[c]}-{col_edges[c + 1] - 1}, {counts[b] / sizes[b]:.2f}\n")
    print(f"Kept {keep.sum()} of {gr * gc} boxes; box layout in {name}_boxes.txt")
