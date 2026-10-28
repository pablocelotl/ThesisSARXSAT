"""Evaluate one HySI-SAT run: summary statistics and plots of modes, fit and residuals.

Reads a run folder as produced by run_identify.py (downloaded from DelftBlue or local):
    <run>/DataOld.db                               settings of the run (ny, nu, dw, delta)
    <run>/Data/IdentifiedDatasets/<date_time>/     Fmodels, Sigmat, Yt, Rt (.mat)
and writes to <run>/evaluation/:
    summary.txt     modes, segments, fit within delta, per-output errors
    overview.png    switching sequence + selected outputs with the active mode shaded
    residuals.png   largest residual per sample and per output, against delta
    segments.png    length of every segment, per mode, against the dwell time

The prediction for sample t is y_hat[o, t] = Rt[t] @ theta[sigma[t]][o], the one-step-ahead
ARX prediction that HySI-SAT constrains to |y - y_hat| <= delta.

Usage:
    python evaluate_run.py results/sleep8x5_cal1000 [--outputs 0,1,2,3] [--fs 50] [--show]
"""
import argparse
import glob
import os
import sqlite3

import matplotlib
import numpy as np
import scipy.io as sio

# Categorical slots in fixed order (one per mode); text and surface tokens
MODE_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
OTHER = "#b5b3ad"
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"

p = argparse.ArgumentParser()
p.add_argument("run", help="run folder, e.g. results/sleep8x5_cal1000")
p.add_argument("--result", help="IdentifiedDatasets/<date_time> folder (default: the newest in the run)")
p.add_argument("--outputs", help="comma-separated output rows to plot (default: first 6)")
p.add_argument("--delta", type=float, help="noise bound (default: from DataOld.db)")
p.add_argument("--dw", type=int, help="minimum dwell time (default: from DataOld.db)")
p.add_argument("--fs", type=float, help="sampling rate in Hz, to show time in seconds (default: samples)")
p.add_argument("--show", action="store_true", help="also open the plots in a window")
args = p.parse_args()
if not args.show:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

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
R = np.atleast_2d(load("Rt"))  # samples x regressor length
outputs, T = Y.shape
rlen = R.shape[1]

# Settings of the run: the last row in the run's database
settings = {}
db = os.path.join(args.run, "DataOld.db")
if os.path.exists(db):
    row = sqlite3.connect(db).execute("SELECT nu, ny, inputs, outputs, dw, delta FROM Modelinfo "
                                      "ORDER BY rowid DESC LIMIT 1").fetchone()
    if row:
        settings = dict(zip(["nu", "ny", "inputs", "outputs", "dw", "delta"], row))
delta = args.delta if args.delta is not None else settings.get("delta")
dw = args.dw if args.dw is not None else settings.get("dw")
if delta is None:
    raise SystemExit("delta not found in DataOld.db; pass --delta")

# --- Predictions, residuals, segments -----------------------------------------------------
theta = F.reshape(len(F), outputs, rlen)
Yhat = np.einsum("tr,tor->ot", R, theta[sigma])
E = Y - Yhat
absE = np.abs(E)
tol = delta * 1.01  # HySI-SAT accepts errors up to delta + delta/100
within = absE <= tol

edges = np.flatnonzero(np.diff(sigma)) + 1
starts = np.r_[0, edges]
ends = np.r_[edges, T]
seg_mode = sigma[starts]
seg_len = ends - starts

modes = np.arange(len(F))
mode_color = {m: MODE_COLORS[m] if m < len(MODE_COLORS) else OTHER for m in modes}
x = np.arange(T) / args.fs if args.fs else np.arange(T)
xlabel = "time (s)" if args.fs else "sample"
show_rows = [int(o) for o in args.outputs.split(",")] if args.outputs else list(range(min(6, outputs)))

# --- Summary ------------------------------------------------------------------------------
lines = [f"Run: {result}",
         f"Settings: " + ", ".join(f"{k}={v}" for k, v in settings.items()) if settings else "Settings: (no DataOld.db)",
         f"Data: {outputs} outputs x {T} samples, regressor length {rlen}, {F.shape[1]} parameters per mode",
         f"Modes: {len(F)}, segments: {len(seg_len)}, switches: {len(seg_len) - 1}",
         f"Samples within delta={delta:g} (all outputs): {within.all(axis=0).mean():.1%};"
         f" output values within delta: {within.mean():.1%}",
         f"Largest residual: {absE.max():.4g} ({absE.max() / delta:.2f} x delta)",
         "",
         "mode  samples  share  segments  min_len  median_len  max_len  max|e|",
         ]
for m in modes:
    L = seg_len[seg_mode == m]
    n = int((sigma == m).sum())
    em = absE[:, sigma == m].max() if n else float("nan")
    lines.append(f"{m:>4}  {n:>7}  {n / T:>5.1%}  {len(L):>8}  {L.min() if len(L) else 0:>7}"
                 f"  {np.median(L) if len(L) else 0:>10.1f}  {L.max() if len(L) else 0:>7}  {em:>6.3g}")
lines += ["", "output  rms_error  max|e|  within_delta"]
for o in range(outputs):
    lines.append(f"{o:>6}  {np.sqrt(np.mean(E[o] ** 2)):>9.4g}  {absE[o].max():>6.3g}  {within[o].mean():>12.1%}")
if len(F) > len(MODE_COLORS):
    lines.append(f"\nNote: modes {len(MODE_COLORS)}.. are drawn in gray (only {len(MODE_COLORS)} distinct colors).")

out_dir = os.path.join(args.run, "evaluation")
os.makedirs(out_dir, exist_ok=True)
with open(os.path.join(out_dir, "summary.txt"), "w") as f:
    f.write("\n".join(lines) + "\n")
print("\n".join(lines))

# --- Plot style ---------------------------------------------------------------------------
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 9, "lines.linewidth": 1.2,
})


def shade_modes(ax, alpha=0.14):
    for s, e, m in zip(starts, ends, seg_mode):
        ax.axvspan(x[s], x[min(e, T - 1)], color=mode_color[m], alpha=alpha, lw=0)


def mode_legend(fig):
    handles = [plt.Rectangle((0, 0), 1, 1, color=mode_color[m]) for m in modes[:len(MODE_COLORS)]]
    labels = [f"mode {m}" for m in modes[:len(MODE_COLORS)]]
    if len(F) > len(MODE_COLORS):
        handles.append(plt.Rectangle((0, 0), 1, 1, color=OTHER))
        labels.append(f"modes {len(MODE_COLORS)}+")
    fig.legend(handles, labels, loc="upper right", ncol=min(len(labels), 8), frameon=False)


# --- overview.png: switching sequence + outputs ---------------------------------------------
n = len(show_rows)
fig, axs = plt.subplots(n + 1, 1, figsize=(11, 1.2 + 1.3 * n), sharex=True,
                        gridspec_kw={"height_ratios": [0.9] + [1] * n})
ax = axs[0]
for s, e, m in zip(starts, ends, seg_mode):
    ax.plot(x[s:e], sigma[s:e], color=mode_color[m], lw=2, solid_capstyle="butt")
ax.set_yticks(modes)
ax.set_ylabel("mode")
ax.set_title("Switching sequence", loc="left", color=INK)
for ax, o in zip(axs[1:], show_rows):
    shade_modes(ax)
    ax.plot(x, Y[o], color=INK, lw=1.0, label="measured")
    ax.plot(x, Yhat[o], color=INK_2, lw=0.9, ls="--", label="one-step prediction")
    ax.set_ylabel(f"output {o}")
axs[1].legend(loc="upper left", frameon=False, ncol=2, fontsize=8)
axs[-1].set_xlabel(xlabel)
mode_legend(fig)
fig.tight_layout(rect=(0, 0, 1, 0.97))
fig.savefig(os.path.join(out_dir, "overview.png"), dpi=150)

# --- residuals.png: largest residual per sample, and per output -------------------------------
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 6.5), gridspec_kw={"height_ratios": [1.2, 1]})
shade_modes(ax1)
ax1.plot(x, absE.max(axis=0), color=INK, lw=0.9)
ax1.axhline(delta, color=INK_2, lw=1, ls="--")
ax1.annotate(f"delta = {delta:g}", (1, delta), xycoords=("axes fraction", "data"), xytext=(-4, 4),
             textcoords="offset points", ha="right", color=INK_2)
ax1.set_ylim(0, max(absE.max(), delta) * 1.1)
ax1.set_xlabel(xlabel)
ax1.set_ylabel("max |error| over outputs")
ax1.set_title("Largest residual per sample (background: active mode)", loc="left", color=INK)

ax2.bar(np.arange(outputs), absE.max(axis=1), width=0.8, color="#2a78d6", label="max |error|")
ax2.scatter(np.arange(outputs), np.sqrt(np.mean(E ** 2, axis=1)), s=18, color=INK, zorder=3,
            label="rms error")
ax2.axhline(delta, color=INK_2, lw=1, ls="--")
ax2.set_xticks(np.arange(outputs))
ax2.set_xlabel("output")
ax2.set_ylabel("error")
ax2.grid(axis="x", visible=False)
ax2.set_ylim(0, max(absE.max(), delta) * 1.35)
ax2.legend(loc="upper left", frameon=False, ncol=2)
ax2.set_title("Error per output", loc="left", color=INK)
mode_legend(fig)
fig.tight_layout(rect=(0, 0, 1, 0.96))
fig.savefig(os.path.join(out_dir, "residuals.png"), dpi=150)

# --- segments.png: every segment's length, per mode ----------------------------------------
fig, ax = plt.subplots(figsize=(8, 1.2 + 0.5 * len(F)))
rng = np.random.default_rng(0)
for m in modes:
    L = seg_len[seg_mode == m]
    if not len(L):
        continue
    ax.plot([np.median(L)] * 2, [m - 0.3, m + 0.3], color=INK_2, lw=1.5, zorder=2)
    ax.scatter(L, m + rng.uniform(-0.15, 0.15, len(L)), s=48, color=mode_color[m],
               edgecolor=SURFACE, linewidth=1.5, zorder=3)
if dw:
    ax.axvline(dw, color=INK_2, lw=1, ls="--")
    ax.annotate(f"dwell time = {dw}", (dw, 1), xycoords=("data", "axes fraction"), xytext=(4, -10),
                textcoords="offset points", color=INK_2)
ax.set_yticks(modes)
ax.set_yticklabels([f"mode {m}" for m in modes])
ax.set_ylim(-0.6, len(F) - 0.4)
ax.invert_yaxis()
ax.set_xlabel("segment length (samples)")
ax.grid(axis="y", visible=False)
ax.set_title("Length of every segment (bar: median)", loc="left", color=INK)
fig.tight_layout()
fig.savefig(os.path.join(out_dir, "segments.png"), dpi=150)

print(f"\nWrote summary.txt, overview.png, residuals.png, segments.png to {out_dir}")
if args.show:
    plt.show()
