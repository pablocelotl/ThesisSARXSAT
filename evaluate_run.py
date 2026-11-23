"""Evaluate one HySI-SAT run: which modes were found, what they look like, and how the signals couple in each.

Reads a run folder as produced by run_identify.py (downloaded from DelftBlue or local):
    <run>/DataOld.db                               settings of the run (ny, nu, dw, delta)
    <run>/Data/IdentifiedDatasets/<date_time>/     Fmodels, Sigmat, Yt, Rt (.mat)
and writes to <run>/evaluation/:
    summary.txt     modes, segments and fit
    overview.png    the dataset as an image (every feature over time), the mode sequence under it, and a few
                    outputs with their one-step prediction
    profiles.png    mean of every feature in every mode: which bands are high or low where, per mode
    A.png           the A matrix of every mode, y(k) = A y(k-1) + ..., and how much each entry differs between modes
    brain.png       coupling between signals per mode, drawn on the cortex map and a hippocampal probe (--layout)

Which models. The models HySI-SAT exports (Fmodels) come from its search: a linear program (Sparsification.pyx)
whose objective only makes the models of different intervals agree, with every error within delta. Nothing rewards
small errors, so the solver returns a vertex of the feasible set, with many errors at exactly +-delta. Here each
mode's model is re-estimated by least squares on the samples of that mode, keeping HySI-SAT's switching sequence;
summary.txt compares both. --raw-models plots HySI-SAT's own models instead.
A is the coefficient of the most recent lag, y(k-1); a row is the output predicted, a column the signal it is
predicted from. A coupling is "signal i helps predict signal j one sample later" (a directed, Granger-like link),
not a correlation.

For a wavelet dataset (frequency_analysis/wavelet_power.py), --features names every row by its signal and band,
and --layout places the cortex signals on the frame: a _boxes.txt (make_video_dataset.py --grid) or _areas.txt
(--areas, then Data/brain_areas.txt is read too). The signals after the cortex ones are the hippocampal groups.

Usage:
    python evaluate_run.py results/wpow_cal60s --fs 20 --features Data/comb3x2_g3_nrem_wpow_features.txt \
        --layout Data/sleep_video_grid3x2_boxes.txt
"""
import argparse
import glob
import os
import sqlite3

import matplotlib
import numpy as np
import scipy.io as sio

MODE_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
OTHER = "#b5b3ad"
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"

p = argparse.ArgumentParser()
p.add_argument("run", help="run folder, e.g. results/wpow_cal60s")
p.add_argument("--result", help="IdentifiedDatasets/<date_time> folder (default: the newest in the run)")
p.add_argument("--features", help="<name>_features.txt of a wavelet dataset: signal and band of every output row")
p.add_argument("--layout", help="_boxes.txt or _areas.txt of the cortex signals, for brain.png")
p.add_argument("--areas-map", default="Data/brain_areas.txt", help="area label image, for an _areas.txt layout")
p.add_argument("--outputs", help="comma-separated output rows to plot in overview.png (default: first 4)")
p.add_argument("--fs", type=float, help="sampling rate in Hz, to show time in seconds (default: samples)")
p.add_argument("--delta", type=float, help="noise bound (default: from DataOld.db)")
p.add_argument("--raw-models", action="store_true", help="use HySI-SAT's models instead of least-squares refits")
p.add_argument("--show", action="store_true", help="also open the plots in a window")
args = p.parse_args()
if not args.show:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, Rectangle  # noqa: E402

# --- Load the run -------------------------------------------------------------------------
result = args.result or max(glob.glob(os.path.join(args.run, "Data", "IdentifiedDatasets", "*", "")),
                            key=os.path.getmtime, default=None)
if not result:
    raise SystemExit(f"No Data/IdentifiedDatasets/<date_time> folder in {args.run}; did the run finish?")


def load(name):
    d = sio.loadmat(os.path.join(result, name + ".mat"))
    return d[[k for k in d if not k.startswith("__")][0]]


F = np.atleast_2d(load("Fmodels"))  # modes x (outputs * regressor length)
sigma = load("Sigmat").ravel().astype(int)  # mode per sample
Y = np.atleast_2d(load("Yt"))  # outputs x samples
R = np.atleast_2d(load("Rt"))  # samples x regressor: [ny lags of output 0, ..., of the last output, input lags]
outputs, T = Y.shape
modes = np.arange(len(F))

settings = {}
db = os.path.join(args.run, "DataOld.db")
if os.path.exists(db):
    row = sqlite3.connect(db).execute("SELECT nu, ny, inputs, outputs, dw, delta FROM Modelinfo "
                                      "ORDER BY rowid DESC LIMIT 1").fetchone()
    if row:
        settings = dict(zip(["nu", "ny", "inputs", "outputs", "dw", "delta"], row))
delta = args.delta if args.delta is not None else settings.get("delta")
ny = settings.get("ny", 1)

# --- Models: HySI-SAT's, and least-squares refits on each mode's samples ------------------
theta_raw = F.reshape(len(F), outputs, -1)
theta_ls = np.stack([np.linalg.lstsq(R[sigma == m], Y[:, sigma == m].T, rcond=None)[0].T for m in modes])
theta = theta_raw if args.raw_models else theta_ls
A = theta[:, :, ny - 1:outputs * ny:ny]  # modes x outputs x outputs, coefficients of y(k-1)


def residuals(th):
    return Y - np.einsum("tr,tor->ot", R, th[sigma])


E_raw, E_ls = residuals(theta_raw), residuals(theta_ls)
E = E_raw if args.raw_models else E_ls
Yhat = Y - E
one_model = np.linalg.lstsq(R, Y.T, rcond=None)[0].T
E_one = Y - one_model @ R.T

edges = np.flatnonzero(np.diff(sigma)) + 1
starts, ends = np.r_[0, edges], np.r_[edges, T]
seg_mode = sigma[starts]
mode_color = {m: MODE_COLORS[m] if m < len(MODE_COLORS) else OTHER for m in modes}
x = np.arange(T) / args.fs if args.fs else np.arange(T)
unit = "s" if args.fs else "samples"

# --- Names: signal and band of every output row -------------------------------------------
if args.features:
    feat = np.loadtxt(args.features, delimiter=",", dtype=str, comments="#")
    src = feat[:, 1].astype(int)
    band = [b.strip() for b in feat[:, 2]]
else:
    src, band = np.arange(outputs), [""] * outputs
bands = list(dict.fromkeys(band))
signals = np.unique(src)

layout = []  # one entry per cortex signal: (name, pixel mask)
if args.layout:
    rows = [line.strip().split(", ") for line in open(args.layout) if not line.startswith("#")]
    if args.layout.endswith("_areas.txt"):
        area_map = np.loadtxt(args.areas_map, dtype=int)
        with open(args.areas_map) as f:
            names = [a.split(" = ")[1] for a in f.readline().lstrip("# ").strip().split(", ")]
        layout = [(r[1], area_map == names.index(r[1])) for r in rows]
    else:
        H, W = (int(n) for n in open(args.layout).readline().split("frame ")[1].split(",")[0].split("x"))
        for r in rows:
            mask = np.zeros((H, W), bool)
            (r0, r1), (c0, c1) = (map(int, r[3].split("-")), map(int, r[4].split("-")))
            mask[r0:r1 + 1, c0:c1 + 1] = True
            layout.append((f"box ({r[1]},{r[2]})", mask))
n_ctx = len(layout)


def signal_name(i):
    if not args.features:
        return f"y{i}"
    return f"ctx {i}" if i < n_ctx else f"LFP {i - n_ctx}" if n_ctx else f"signal {i}"


row_names = [f"{signal_name(s)} {b}".strip() for s, b in zip(src, band)]

# --- Summary ------------------------------------------------------------------------------
rms = lambda e: np.sqrt(np.mean(e ** 2))  # noqa: E731
lines = [f"Run: {result}",
         "Settings: " + (", ".join(f"{k}={v}" for k, v in settings.items()) if settings else "(no DataOld.db)"),
         f"Data: {outputs} outputs x {T} samples ({x[-1] - x[0] + (x[1] - x[0]):g} {unit})",
         f"Modes: {len(F)}, segments: {len(starts)}, switches: {len(starts) - 1}",
         f"Plots use {'HySI-SAT' if args.raw_models else 'least-squares'} models",
         "",
         "Fit (rms error over all outputs; the data are z-scored, so 1 = no better than predicting 0):",
         f"  one model for all samples (least squares): {rms(E_one):.3f}",
         f"  HySI-SAT models:                            {rms(E_raw):.3f} (largest |error| {np.abs(E_raw).max():.3g}, "
         f"delta = {delta})",
         f"  least-squares models per mode:              {rms(E_ls):.3f} (largest |error| {np.abs(E_ls).max():.3g})",
         "",
         f"mode  samples  share  segments  segments ({unit})                          rms HySI  rms LS  "
         "corr(A HySI, A LS)"]
for m in modes:
    i = sigma == m
    segs = ", ".join(f"{x[s]:g}-{x[e - 1]:g}" for s, e in zip(starts[seg_mode == m], ends[seg_mode == m]))
    corr = np.corrcoef(theta_raw[m, :, :outputs * ny].ravel(), theta_ls[m, :, :outputs * ny].ravel())[0, 1]
    lines.append(f"{m:>4}  {i.sum():>7}  {i.mean():>5.1%}  {(seg_mode == m).sum():>8}  {segs:<40}  "
                 f"{rms(E_raw[:, i]):>8.3f}  {rms(E_ls[:, i]):>6.3f}  {corr:>18.2f}")
if args.features:
    lines += ["", "Least-squares fit per band (R^2, pooled over signals): one model -> per-mode models"]
    for b in bands:
        rows_b = [o for o in range(outputs) if band[o] == b]
        var = np.sum(Y[rows_b] ** 2)
        lines.append(f"  {b:<8} {1 - np.sum(E_one[rows_b] ** 2) / var:.2f} -> {1 - np.sum(E_ls[rows_b] ** 2) / var:.2f}")
if n_ctx:
    lines += ["", "Cortex signals:"] + [f"  ctx {i} = {name}" for i, (name, _) in enumerate(layout)]

out_dir = os.path.join(args.run, "evaluation")
os.makedirs(out_dir, exist_ok=True)
for old in ["residuals.png", "segments.png"]:  # plots of the previous version of this script
    if os.path.exists(os.path.join(out_dir, old)):
        os.remove(os.path.join(out_dir, old))
with open(os.path.join(out_dir, "summary.txt"), "w") as f:
    f.write("\n".join(lines) + "\n")
print("\n".join(lines))

# --- Plot style ---------------------------------------------------------------------------
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2,
    "text.color": INK, "font.size": 9, "lines.linewidth": 1.2, "axes.spines.top": False, "axes.spines.right": False,
})


def mode_legend(fig):
    handles = [Rectangle((0, 0), 1, 1, color=mode_color[m]) for m in modes]
    fig.legend(handles, [f"mode {m}" for m in modes], loc="upper right", ncol=len(modes), frameon=False)


def signal_ticks(ax, axis="y"):
    """Lines between signals and one tick per signal, for a plot with one row (or column) per output."""
    for i in signals[1:]:
        k = np.argmax(src == i) - 0.5
        (ax.axhline if axis == "y" else ax.axvline)(k, color=INK_2, lw=0.5)
    pos = [np.flatnonzero(src == i).mean() for i in signals]
    (ax.set_yticks if axis == "y" else ax.set_xticks)(pos, [signal_name(i) for i in signals], fontsize=7)


# --- overview.png: dataset image, mode sequence, a few outputs ------------------------------
show_rows = [int(o) for o in args.outputs.split(",")] if args.outputs else list(range(min(4, outputs)))
fig, axs = plt.subplots(2 + len(show_rows), 1, figsize=(12, 8 + 1.2 * len(show_rows)), sharex=True,
                        gridspec_kw={"height_ratios": [6, 0.5] + [1.2] * len(show_rows)})
im = axs[0].imshow(Y, aspect="auto", cmap="RdBu_r", vmin=-3, vmax=3, interpolation="nearest",
                   extent=(x[0], x[-1], outputs - 0.5, -0.5))
fig.colorbar(im, cax=axs[0].inset_axes([1.01, 0, 0.012, 1]), label="value (z-score)")
signal_ticks(axs[0])
axs[0].set_title("Dataset: every output row over time" + (" (one row per band, bands in the order "
                 + ", ".join(bands) + ")" if args.features else ""), loc="left")
for s, e, m in zip(starts, ends, seg_mode):
    axs[1].axvspan(x[s], x[e - 1], color=mode_color[m], lw=0)
    axs[1].text((x[s] + x[e - 1]) / 2, 0.5, str(m), ha="center", va="center", color="white", weight="bold")
axs[1].set_yticks([])
axs[1].set_ylabel("mode", rotation=0, ha="right", va="center")
for ax, o in zip(axs[2:], show_rows):
    for s, e, m in zip(starts, ends, seg_mode):
        ax.axvspan(x[s], x[e - 1], color=mode_color[m], alpha=0.14, lw=0)
    ax.plot(x, Y[o], color=INK, lw=0.9, label="measured")
    ax.plot(x, Yhat[o], color=INK_2, lw=0.8, ls="--", label="one-step prediction")
    ax.set_ylabel(row_names[o], rotation=0, ha="right", va="center")
axs[2].legend(loc="upper left", frameon=False, ncol=2, fontsize=8)
axs[-1].set_xlabel(f"time ({unit})")
axs[-1].set_xlim(x[0], x[-1])
mode_legend(fig)
fig.tight_layout(rect=(0, 0, 1, 0.98))
fig.savefig(os.path.join(out_dir, "overview.png"), dpi=130)

# --- profiles.png: mean of every feature per mode (signals x bands) ---------------------------
prof = np.full((len(modes), len(signals), len(bands)), np.nan)
for o in range(outputs):
    for m in modes:
        prof[m, np.searchsorted(signals, src[o]), bands.index(band[o])] = Y[o, sigma == m].mean()
lim = np.nanmax(np.abs(prof))
fig, axs = plt.subplots(1, len(modes), figsize=(1.5 + 2.4 * len(modes), 0.4 * len(signals) + 1.2),
                        sharey=True, squeeze=False)
for m, ax in zip(modes, axs[0]):
    im = ax.imshow(prof[m], cmap="RdBu_r", vmin=-lim, vmax=lim)
    for (i, j), v in np.ndenumerate(prof[m]):
        if not np.isnan(v):
            ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=6)
    ax.set_xticks(range(len(bands)), bands, rotation=90)
    ax.set_yticks(range(len(signals)), [signal_name(i) for i in signals])
    ax.set_title(f"mode {m} ({(sigma == m).mean():.0%})", color=mode_color[m], weight="bold")
fig.colorbar(im, ax=axs[0], shrink=0.8, label="mean in the mode (z-score)")
fig.suptitle("What every mode looks like: mean of every feature over the mode's samples", x=0.02, ha="left")
fig.savefig(os.path.join(out_dir, "profiles.png"), dpi=130, bbox_inches="tight")

# --- A.png: A matrix per mode, and its spread between modes ------------------------------------
panels = [(f"mode {m}", A[m], "RdBu_r") for m in modes] + [("std between modes", A.std(axis=0), "Greys")]
lim = np.percentile(np.abs(A), 99)
ncol = min(len(panels), 3)
nrow = int(np.ceil(len(panels) / ncol))
fig, axs = plt.subplots(nrow, ncol, figsize=(5.2 * ncol, 5 * nrow), squeeze=False)
for ax, (title, M, cmap) in zip(axs.flat, panels):
    im = ax.imshow(M, cmap=cmap, vmin=0 if cmap == "Greys" else -lim, vmax=lim, interpolation="nearest")
    signal_ticks(ax, "y")
    signal_ticks(ax, "x")
    ax.tick_params(axis="x", rotation=90)
    ax.set_title(title, loc="left")
    ax.set_xlabel("from: y(k-1)")
    ax.set_ylabel("to: y(k)")
    fig.colorbar(im, ax=ax, shrink=0.75)
for ax in axs.flat[len(panels):]:
    ax.axis("off")
fig.suptitle("A matrices" + (", bands in the order " + ", ".join(bands) + " inside every signal"
                             if args.features else ""), x=0.02, ha="left")
fig.tight_layout()
fig.savefig(os.path.join(out_dir, "A.png"), dpi=130)

# --- brain.png: coupling between signals per mode, on the cortex map ---------------------------
if layout:
    # Strength of the link i -> j: root mean square, over all band pairs of the two signals, of the t-value of the
    # A entry (coefficient / its standard error in the least-squares fit of the mode). t-values make modes of
    # different lengths comparable: a short mode has noisier coefficients, not stronger links. A link is drawn if
    # the band pairs together are significant: sum of t^2 above the 99.9% point of a chi-square distribution.
    from scipy.stats import chi2
    S = np.zeros((len(modes), len(signals), len(signals)))
    for m in modes:
        X, Ym = R[sigma == m], Y[:, sigma == m].T
        e = Ym - X @ theta_ls[m].T
        s2 = (e ** 2).sum(axis=0) / (len(X) - X.shape[1])  # error variance per output
        se = np.sqrt(np.outer(s2, np.diag(np.linalg.pinv(X.T @ X))))  # outputs x regressors
        t = (theta_ls[m] / np.where(se > 0, se, np.inf))[:, ny - 1:outputs * ny:ny]
        S[m] = [[np.sqrt(np.mean(t[np.ix_(src == j, src == i)] ** 2)) for i in signals] for j in signals]
    pairs = np.array([[(src == j).sum() * (src == i).sum() for i in signals] for j in signals])
    significant = S ** 2 * pairs > chi2.ppf(0.999, pairs)
    off = ~np.eye(len(signals), dtype=bool)
    # Most links pass that test (and the residuals are autocorrelated, which makes t-values optimistic), so only
    # the TOP_LINKS strongest significant links of every mode are drawn: its main pathways.
    TOP_LINKS = 10
    drawn = np.zeros_like(significant)
    for m in modes:
        strength = np.where(significant[m] & off, S[m], 0)
        drawn[m] = (strength >= np.sort(strength.ravel())[-TOP_LINKS]) & (strength > 0)
    H, W = layout[0][1].shape
    pos = [np.array(np.nonzero(mask)).mean(axis=1)[::-1] for _, mask in layout]  # (x, y) of every cortex signal
    n_hpc = len(signals) - n_ctx
    probe_x, probe_top, probe_bottom = W + 45, 0.2 * H, 0.85 * H
    step = (probe_bottom - probe_top) / max(n_hpc, 1)
    pos += [np.array([probe_x, probe_top + step * (k + 0.5)]) for k in range(n_hpc)]
    smax = S[:, off].max()
    regions = np.full((H, W), np.nan)
    for i, (_, mask) in enumerate(layout):
        regions[mask] = i

    fig, axs = plt.subplots(1, len(modes), figsize=(3.6 * len(modes) + 1, 5.6), squeeze=False)
    for m, ax in zip(modes, axs[0]):
        ax.imshow(np.where(np.isnan(regions), 0, 1), cmap="Greys", vmin=0, vmax=6, extent=(-0.5, W - 0.5, H - 0.5, -0.5))
        ax.contour(np.nan_to_num(regions, nan=-1), np.arange(-0.5, n_ctx), colors="white", linewidths=1.5)
        ax.add_patch(Rectangle((probe_x - 8, probe_top), 16, probe_bottom - probe_top, fc=GRID, ec=INK_2))
        for j in range(len(signals)):
            for i in range(len(signals)):
                if drawn[m, j, i]:
                    ax.add_patch(FancyArrowPatch(pos[i], pos[j], connectionstyle="arc3,rad=0.25", arrowstyle="-|>",
                                                 mutation_scale=10, lw=0.3 + 4 * S[m, j, i] / smax,
                                                 color=mode_color[m], alpha=0.8, shrinkA=9, shrinkB=9))
        ax.scatter(*np.array(pos).T, s=260, c="white", edgecolors=INK, zorder=3)
        for k, (px, py) in enumerate(pos):
            ax.text(px, py, str(k) if k < n_ctx else f"H{k - n_ctx}", ha="center", va="center", fontsize=7,
                    color=mode_color[m], weight="bold", zorder=4)
        ax.set_xlim(-10, probe_x + 30)
        ax.set_ylim(H + 10, -10)
        ax.axis("off")
        ax.set_title(f"mode {m} ({(sigma == m).mean():.0%}): {(significant[m] & off).sum()} significant links",
                     color=mode_color[m], weight="bold")
    names = [f"{i}: {name}" for i, (name, _) in enumerate(layout)] + [f"H{k}: LFP group {k}" for k in range(n_hpc)]
    fig.text(0.01, 0.01, "Cortex " + "; ".join(names[:n_ctx]) + ".   Hippocampus (probe): " + "; ".join(names[n_ctx:])
             + ".\nArrow i -> j: signal i helps predict signal j one sample later, significant at p < 0.001 over its "
             f"band pairs. Only the {TOP_LINKS} strongest per mode are drawn; width = RMS t-value of those A entries (thickest: {smax:.1f}).",
             fontsize=7, color=INK_2, va="bottom")
    fig.suptitle("Coupling between regions per mode (front of the brain at the top)", x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0.07, 1, 0.95))
    fig.savefig(os.path.join(out_dir, "brain.png"), dpi=130)

print(f"\nWrote summary.txt, overview.png, profiles.png, A.png{', brain.png' if layout else ''} to {out_dir}")
if args.show:
    plt.show()
