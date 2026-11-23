"""Turn a HySI-SAT dataset into band-power envelopes with a complex Morlet wavelet transform.

Follows the preprocessing of de Fonseca, Stella & Mazo, "Multivariate Sparse Modeling of Cross-Frequency
Couplings in Neurophysiological Data" (Sec. III-C), extended to the hippocampal high gamma and ripple bands:
    1. continuous wavelet transform with a complex Morlet wavelet of q = 6 cycles, at the dataset's own rate
       (1 kHz, so that ripples fit; the paper resampled to 200 Hz because its bands stopped at 45 Hz),
       at 1 Hz steps up to 45 Hz and 5 Hz steps from 50 to 250 Hz
    2. power |W|^2, averaged over the frequencies of each band (BANDS below)
    3. block-average to 20 Hz (50 samples at 1 kHz, the paper's 10 samples at 200 Hz)
    4. drop the edges, where the wavelet runs past the data (3 standard deviations of the slowest wavelet)
    5. log10, then remove a linear trend and z-score every (signal, band) feature
Rows taken from the 50 Hz video (the first --video-rows rows, as written by make_combined_dataset.py) only get
the bands below the video Nyquist frequency (25 Hz): delta, theta and alpha.

Writes Data/<name>_output.txt (one row per feature, signal by signal, bands in the order of BANDS),
Data/<name>_input_dummy.txt, Data/<name>_features.txt (which signal and band every row is),
Data/<name>_features.png (the 20 Hz features) and Data/<name>_features_full.png (a 10 s window at the full
rate, before the block averaging).

Usage:
    python frequency_analysis/wavelet_power.py Data/comb_g3_nrem_output.txt --video-rows 26 --name comb_g3_nrem_wpow
"""
import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pywt
from scipy import signal

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # make_io_dataset.py is in the repo root
from make_io_dataset import write_io_dataset

BANDS = {"delta": (1, 4), "theta": (4, 8), "alpha": (8, 13), "beta": (13, 30), "gamma": (30, 45),
         "hgamma": (60, 100), "ripple": (150, 250)}
FREQS = np.r_[np.arange(1, 46), np.arange(50, 251, 5)]  # Hz; the resolution f/q is >= 8 Hz above 50 Hz
FS_OUT = 20  # Hz, rate of the output features
Q = 6  # cycles of the Morlet wavelet
ZOOM = 10  # s, length of the window in the full-rate figure

p = argparse.ArgumentParser()
p.add_argument("output", help="dataset, one comma-separated row per signal, e.g. Data/comb_g3_nrem_output.txt")
p.add_argument("--fs", type=float, default=1000.0, help="sampling rate of the dataset in Hz (default: 1000)")
p.add_argument("--video-rows", type=int, default=0, help="number of leading rows that come from the 50 Hz video")
p.add_argument("--video-fs", type=float, default=50.0, help="video frame rate in Hz (default: 50)")
p.add_argument("--zoom", type=float, help="start in s of the full-rate figure window (default: the middle)")
p.add_argument("--name", help="output name in Data/ (default: <dataset name>_wpow)")
args = p.parse_args()

fs = args.fs
block = int(fs / FS_OUT)
x = np.loadtxt(args.output, delimiter=",", ndmin=2)

# Complex Morlet "cmorB-C": exp(-t^2/B) exp(2j pi C t). A Gaussian of standard deviation sigma = q / (2 pi f)
# at centre frequency f, as in the paper, is B = 2 (q / 2 pi)^2 with C = 1.
wavelet = f"cmor{2 * (Q / (2 * np.pi)) ** 2:.4f}-1.0"
scales = pywt.frequency2scale(wavelet, FREQS / fs)

features, names = [], []
for i, row in enumerate(x):
    coef, _ = pywt.cwt(row, scales, wavelet, sampling_period=1 / fs, method="fft")  # step 1
    power = np.abs(coef) ** 2  # step 2
    nyquist = (args.video_fs if i < args.video_rows else fs) / 2
    for band, (lo, hi) in BANDS.items():
        if hi <= nyquist:
            features.append(power[(FREQS >= lo) & (FREQS <= hi)].mean(axis=0).astype(np.float32))
            names.append((i, band))
full = np.array(features)

n = full.shape[1] // block * block
y = full[:, :n].reshape(len(full), -1, block).mean(axis=2)  # step 3
edge = int(np.ceil(3 * Q / (2 * np.pi * FREQS[0]) * FS_OUT))  # step 4, in output samples
y = y[:, edge:-edge]
full = full[:, edge * block:n - edge * block]  # the same time span at the full rate


def normalize(z):
    z = signal.detrend(np.log10(z), axis=1)  # step 5
    return (z - z.mean(axis=1, keepdims=True)) / z.std(axis=1, ddof=1, keepdims=True)


y_log, full_log = np.log10(y), np.log10(full)  # for comparing the two rates in the full-rate figure
y = normalize(y)
full = normalize(full)
print(f"{x.shape[0]} signals x {x.shape[1] / fs:.1f} s -> {len(y)} features x {y.shape[1]} samples at "
      f"{FS_OUT} Hz ({y.shape[1] / FS_OUT:.1f} s, {edge / FS_OUT:.2f} s dropped at each edge)")

name = f"Data/{args.name or Path(args.output).stem.removesuffix('_output') + '_wpow'}"
write_io_dataset(y, name)
with open(f"{name}_features.txt", "w") as f:
    f.write("# output_row, source_row, band\n")
    for k, (i, band) in enumerate(names):
        f.write(f"{k}, {i}, {band}\n")

# Figures. Top: every feature as one image row, grouped by signal. Bottom: the bands of the first LFP signal.
src = np.array([i for i, _ in names])
lfp = args.video_rows if args.video_rows < x.shape[0] else 0
lfp_rows = [k for k, (i, _) in enumerate(names) if i == lfp]


def label(i):
    return f"ctx box {i}" if i < args.video_rows else f"LFP {i - args.video_rows}"


def plot(z, t0, rate, title, fname, traces, coarse=None):
    t = t0 + np.arange(z.shape[1]) / rate
    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(13, 11), gridspec_kw={"height_ratios": [3, 2]}, sharex=True)
    im = ax0.imshow(z, aspect="auto", cmap="RdBu_r", vmin=-3, vmax=3, interpolation="nearest",
                    extent=(t[0], t[-1] + 1 / rate, len(z) - 0.5, -0.5))
    fig.colorbar(im, cax=ax0.inset_axes([1.01, 0, 0.012, 1]), label="z-scored log10 power")
    signals = np.unique(src)
    for i in signals[1:]:
        ax0.axhline(np.argmax(src == i) - 0.5, color="k", lw=0.4)
    ax0.set_yticks([np.flatnonzero(src == i).mean() for i in signals], [label(i) for i in signals], fontsize=7)
    ax0.set_ylabel("signal (one row per band, delta at the top)")
    ax0.set_title(title)
    ax0.tick_params(labelbottom=True)
    gap = 5 if coarse is None else 1.5  # z-scores span about +-3, log10 powers about +-0.7
    for j, k in enumerate(lfp_rows):
        ax1.plot(t, traces[k] + gap * j, lw=0.6, color=f"C{j}")
        if coarse is not None:
            tc = t0 + np.arange(coarse.shape[1]) / FS_OUT
            ax1.step(tc, coarse[k] + gap * j, where="post", lw=1, color="k")
    ax1.set_yticks([gap * j for j in range(len(lfp_rows))], [names[k][1] for k in lfp_rows])
    ax1.set_ylabel("z-scored log10 power, offset per band" if coarse is None
                   else "log10 power minus its window mean, offset per band")
    ax1.set_title(f"{label(lfp)}: band powers" + (" (colour: full rate, black: 20 Hz)" if coarse is not None else ""))
    ax1.set_xlabel("Time from the start of the dataset (s)")
    ax0.set_xlim(t[0], t[-1])
    fig.savefig(fname, dpi=120, bbox_inches="tight")


t0 = edge / FS_OUT
plot(y, t0, FS_OUT, f"All features of {Path(args.output).name} at {FS_OUT} Hz", f"{name}_features.png", y)
start = args.zoom if args.zoom is not None else t0 + y.shape[1] / FS_OUT / 2 - ZOOM / 2
i0 = int(round((start - t0) * FS_OUT))
i1 = i0 + ZOOM * FS_OUT
fine = full_log[:, i0 * block:i1 * block]
mean = fine.mean(axis=1, keepdims=True)
plot(full[:, i0 * block:i1 * block], t0 + i0 / FS_OUT, fs, f"Before block averaging: {ZOOM} s at {fs:g} Hz",
     f"{name}_features_full.png", fine - mean, coarse=y_log[:, i0:i1] - mean)
print(f"Wrote {name}_features.txt, {name}_features.png and {name}_features_full.png")
