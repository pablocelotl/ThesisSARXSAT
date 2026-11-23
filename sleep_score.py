"""Score the sleep recording into NREM / REM / WAKE without EMG, from the hippocampal LFP (1 kHz).

Follows the logic of buzcode's SleepScoreMaster (Watson et al. 2016), which also works without EMG.
For every 1 s step a window of --window seconds around it gives three features:

    slow wave  log10 of the 0.5-4 Hz share of the 0.5-20 Hz power. High in NREM.
               From the LFP channel with the most delta, or from the mean cortex box signal (--boxes).
    theta      log10 of the theta (6-9 Hz) / delta (1-4 Hz) power. High in REM and active wake.
               From the LFP channel with the most theta.
    pseudo-EMG mean pairwise correlation of the 275-450 Hz LFP between all channels (Schomburg et al. 2014).
               Muscle activity reaches every channel at once, local neural activity does not. High when moving.
               Caveat: hpc_filtered.mat was decimated from 20 kHz with [::20], without an anti-aliasing
               filter, so this band also holds aliased common-mode noise; --no-emg leaves the feature out.

Each feature is split in two by a 2-component Gaussian mixture fitted on all steps. Theta and pseudo-EMG are
only used when their two components are well apart (Ashman's D >= --min-separation); the slow wave is always
used, with a warning when it is not bimodal. Rules:
    high pseudo-EMG              -> WAKE  (movement can also raise the slow-wave power)
    else high slow wave          -> NREM
    else high theta              -> REM
    else                         -> WAKE  (quiet wake)
Bouts shorter than --min-bout seconds are then merged into the preceding bout. The sharp-wave ripple rate
(150-250 Hz events) is computed only to check the result: it should be clearly highest in NREM.

Writes Data/<name>_states.csv (one row per second), Data/<name>_nrem_bouts.txt (start sample and length
in 1 kHz samples, for --start/--length of lfp_groups.py and make_combined_dataset.py with --offset 0)
and Data/<name>_states.png.

Usage:
    python sleep_score.py "Example data pre-processing/hpc_filtered.mat"
    python sleep_score.py "Example data pre-processing/hpc_filtered.mat" \
        --boxes Data/sleep_video_grid8x5_output.txt --name sleep_ctx
"""
import argparse

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
from scipy import signal
from sklearn.mixture import GaussianMixture

STATES = ["WAKE", "NREM", "REM"]
COLORS = {"WAKE": "tab:orange", "NREM": "tab:blue", "REM": "tab:red"}

p = argparse.ArgumentParser()
p.add_argument("hpc", help="hippocampal LFP .mat file, e.g. hpc_filtered.mat")
p.add_argument("--hpc-key", default="HPC_LFP_filtered", help="variable in the .mat file (default: HPC_LFP_filtered)")
p.add_argument("--fs", type=float, default=1000.0, help="LFP sampling rate in Hz (default: 1000)")
p.add_argument("--boxes", help="cortex box signals (make_video_dataset.py output); their mean gives the slow wave")
p.add_argument("--video-fs", type=float, default=50.0, help="video frame rate in Hz (default: 50)")
p.add_argument("--offset", type=int, default=0, help="LFP sample at which video frame 0 was taken (default: 0)")
p.add_argument("--sw-channel", type=int, help="LFP channel for the slow wave (0-based, default: most delta)")
p.add_argument("--theta-channel", type=int, help="LFP channel for theta (0-based, default: most theta)")
p.add_argument("--ripple-channel", type=int, help="LFP channel for ripples (0-based, default: most ripple power)")
p.add_argument("--window", type=float, default=4.0, help="feature window in seconds (default: 4)")
p.add_argument("--min-bout", type=float, default=10.0, help="shortest bout in seconds (default: 10)")
p.add_argument("--no-emg", action="store_true", help="do not use the pseudo-EMG")
p.add_argument("--min-separation", type=float, default=2.0,
               help="Ashman's D below which a feature counts as not bimodal (default: 2)")
p.add_argument("--min-prob", type=float, default=0.5,
               help="P(high slow wave) needed for NREM; e.g. 0.9 keeps only confident NREM (default: 0.5)")
p.add_argument("--name", default="sleep", help="output name in Data/ (default: sleep)")
p.add_argument("--show", action="store_true", help="also show the figure")
args = p.parse_args()

fs = args.fs
lfp = np.stack([np.asarray(ch, dtype=float).ravel()
                for ch in sio.loadmat(args.hpc, squeeze_me=True)[args.hpc_key]])
n_ch, n_samp = lfp.shape
n_sec = int(n_samp // fs)


def band_power(f, P, lo, hi):
    return P[..., (f >= lo) & (f < hi)].sum(axis=-1)


def windows(x, rate, t0=0.0):
    """Yield (second, window of x) for a --window long window centered on every second s + 0.5."""
    half = args.window / 2
    for s in range(n_sec):
        i0 = int(round((s + 0.5 - half - t0) * rate))
        i1 = int(round((s + 0.5 + half - t0) * rate))
        yield s, x[..., max(i0, 0):max(min(i1, x.shape[-1]), 0)]


def window_psd(x, rate):
    """PSD of every window of x (one row per second; NaN where the window falls outside x)."""
    out = None
    for s, w in windows(x, rate):
        if w.shape[-1] < rate * args.window / 2:
            continue
        f, P = signal.welch(w, rate, nperseg=int(rate))
        if out is None:
            out = np.full((n_sec, len(f)), np.nan)
        out[s] = P
    return f, out


# Default channels: the ones where each band takes the largest share of the power
f, P = signal.welch(lfp, fs, nperseg=int(4 * fs))
total = band_power(f, P, 0.5, 450)
sw_ch = args.sw_channel if args.sw_channel is not None else int(np.argmax(band_power(f, P, 0.5, 4) / total))
th_ch = args.theta_channel if args.theta_channel is not None else int(np.argmax(band_power(f, P, 6, 9) / total))
rp_ch = args.ripple_channel if args.ripple_channel is not None else int(np.argmax(band_power(f, P, 150, 250) / total))
print(f"LFP: {n_ch} channels x {n_samp} samples ({n_samp / fs:.1f} s); "
      f"theta from ch {th_ch}, ripples from ch {rp_ch}")

# Slow wave
if args.boxes:
    boxes = np.loadtxt(args.boxes, delimiter=",", ndmin=2)
    boxes = (boxes - boxes.mean(axis=1, keepdims=True)) / boxes.std(axis=1, keepdims=True)
    ctx = boxes.mean(axis=0)
    # shift the video so that its windows line up with the LFP seconds
    pad = int(round(args.offset / fs * args.video_fs))
    ctx = np.concatenate((np.full(max(pad, 0), np.nan), ctx[max(-pad, 0):]))
    f_sw, P_sw = window_psd(np.nan_to_num(ctx), args.video_fs)
    P_sw[[s for s, w in windows(ctx, args.video_fs) if np.isnan(w).any() or len(w) == 0]] = np.nan
    print(f"Slow wave from the mean of {boxes.shape[0]} cortex boxes ({boxes.shape[1] / args.video_fs:.1f} s)")
else:
    f_sw, P_sw = window_psd(lfp[sw_ch], fs)
    print(f"Slow wave from LFP ch {sw_ch}")
slow_wave = np.log10(band_power(f_sw, P_sw, 0.5, 4) / band_power(f_sw, P_sw, 0.5, 20))

# Theta
f_th, P_th = window_psd(lfp[th_ch], fs)
theta = np.log10(band_power(f_th, P_th, 6, 9) / band_power(f_th, P_th, 1, 4))

# Pseudo-EMG
hf = signal.sosfiltfilt(signal.butter(4, [275, 450], btype="bandpass", fs=fs, output="sos"), lfp, axis=1)
iu = np.triu_indices(n_ch, 1)
emg = np.array([np.corrcoef(w)[iu].mean() for _, w in windows(hf, fs)])

# Sharp-wave ripples (for checking only): 150-250 Hz envelope of the common-average referenced channel,
# robust z-score above 3 for at least 15 ms with a peak above 5
rp = signal.sosfiltfilt(signal.butter(4, [150, 250], btype="bandpass", fs=fs, output="sos"),
                        lfp[rp_ch] - lfp.mean(axis=0))
env = np.abs(signal.hilbert(rp))
z = (env - np.median(env)) / (1.4826 * np.median(np.abs(env - np.median(env))))
above = np.concatenate(([0], (z > 3).astype(int), [0]))
on, off = np.flatnonzero(np.diff(above) == 1), np.flatnonzero(np.diff(above) == -1)
peak = np.array([z[a:b].max() for a, b in zip(on, off)])
ripple_t = on[((off - on) >= 0.015 * fs) & (peak > 5)] / fs
ripple_rate = np.array([np.sum((ripple_t >= s + 0.5 - args.window / 2) & (ripple_t < s + 0.5 + args.window / 2))
                        for s in range(n_sec)]) / args.window


def split(x, name):
    """Fit 2 Gaussians to x; return P(upper component) for every step, the threshold and Ashman's D."""
    ok = np.isfinite(x)
    gmm = GaussianMixture(2, n_init=5, random_state=0).fit(x[ok, None])
    mu, sd = gmm.means_.ravel(), np.sqrt(gmm.covariances_.ravel())
    hi = int(np.argmax(mu))
    d = np.sqrt(2) * abs(mu[0] - mu[1]) / np.sqrt(sd[0] ** 2 + sd[1] ** 2)
    prob = np.full(len(x), np.nan)
    prob[ok] = gmm.predict_proba(x[ok, None])[:, hi]
    grid = np.linspace(mu.min(), mu.max(), 1000)
    thr = grid[np.argmin(np.abs(gmm.predict_proba(grid[:, None])[:, hi] - 0.5))]
    bimodal = d >= args.min_separation
    print(f"  {name:10s} threshold {thr:7.3f}   Ashman's D {d:4.2f}   high in {np.mean(prob[ok] > 0.5):5.1%} "
          f"of the steps{'' if bimodal else '   NOT bimodal'}")
    return prob, thr, bimodal, gmm


print("Gaussian-mixture split of each feature:")
p_sw, thr_sw, bimodal_sw, gmm_sw = split(slow_wave, "slow wave")
p_th, thr_th, use_th, gmm_th = split(theta, "theta")
p_emg, thr_emg, use_emg, gmm_emg = split(emg, "pseudo-EMG")
use_emg &= not args.no_emg
print(f"Using: slow wave{', theta' if use_th else ''}{', pseudo-EMG' if use_emg else ''}")
if not bimodal_sw:
    print("Warning: the slow wave is not bimodal, so the NREM boundary is uncertain; "
          "check the histograms in the figure.")

valid = np.isfinite(slow_wave)
moving = use_emg & (p_emg > 0.5)
nrem = ~moving & (p_sw > args.min_prob)
rem = ~moving & ~nrem & use_th & (p_th > 0.5)
label = np.where(nrem, 1, np.where(rem, 2, 0))
label[~valid] = -1

# Merge bouts shorter than --min-bout into the bout before them (or after, for the first one)
while True:
    edges = np.flatnonzero(np.diff(label)) + 1
    starts, ends = np.r_[0, edges], np.r_[edges, n_sec]
    short = [k for k in range(len(starts)) if ends[k] - starts[k] < args.min_bout and label[starts[k]] >= 0
             and len(starts) > 1]
    if not short:
        break
    k = min(short, key=lambda k: ends[k] - starts[k])
    label[starts[k]:ends[k]] = label[starts[k - 1]] if k > 0 else label[ends[k]]

names = np.array(STATES + ["NONE"])[label]
print("\nState      time (s)   share   ripples/s")
for st in STATES:
    m = names == st
    print(f"  {st:6s} {m.sum():9d}   {m.mean():6.1%}   {ripple_rate[m].mean() if m.any() else np.nan:8.2f}")

edges = np.flatnonzero(np.diff(label)) + 1
starts, ends = np.r_[0, edges], np.r_[edges, n_sec]
bouts = [(s, e) for s, e in zip(starts, ends) if label[s] == 1]
print(f"\nNREM bouts ({len(bouts)}):")
for s, e in bouts:
    print(f"  {s:5d}-{e:5d} s  ({e - s:4d} s)   --start {int(s * fs)} --length {int((e - s) * fs)}")

name = f"Data/{args.name}"
np.savetxt(f"{name}_states.csv",
           np.column_stack((np.arange(n_sec), label, slow_wave, theta, emg, ripple_rate, p_sw)),
           fmt=["%d", "%d", "%.5f", "%.5f", "%.5f", "%.3f", "%.4f"], delimiter=",",
           header="second,state(0=WAKE 1=NREM 2=REM -1=none),slow_wave,theta,pseudo_emg,ripples_per_s,p_slow_wave",
           comments="")
np.savetxt(f"{name}_nrem_bouts.txt", [(int(s * fs), int((e - s) * fs)) for s, e in bouts], fmt="%d",
           delimiter=",", header="start_sample,length_samples (1 kHz)", comments="")

# Figure: spectrogram, the features with their thresholds, the hypnogram, and the feature histograms
fig = plt.figure(figsize=(14, 10))
gs = fig.add_gridspec(6, 4, width_ratios=[6, 6, 6, 2], hspace=0.15, wspace=0.3)
t = np.arange(n_sec) + 0.5
ax0 = fig.add_subplot(gs[0, :3])
fsp, tsp, Sxx = signal.spectrogram(lfp[th_ch], fs, nperseg=int(2 * fs), noverlap=int(fs))
keep = fsp <= 30
ax0.pcolormesh(tsp, fsp[keep], np.log10(Sxx[keep] + 1e-12), shading="auto", cmap="magma")
ax0.set_ylabel(f"ch {th_ch}\nHz")
axes = [ax0]
for row, (x, thr, gmm, ylabel) in enumerate([(slow_wave, thr_sw, gmm_sw, "slow wave"),
                                             (theta, thr_th, gmm_th, "theta/delta"),
                                             (emg, thr_emg, gmm_emg, "pseudo-EMG")], start=1):
    ax = fig.add_subplot(gs[row, :3], sharex=ax0)
    ax.plot(t, x, lw=0.7, color="k")
    ax.axhline(thr, color="tab:green", ls="--", lw=1)
    ax.set_ylabel(ylabel)
    axes.append(ax)
    axh = fig.add_subplot(gs[row, 3], sharey=ax)
    xf = x[np.isfinite(x)]
    axh.hist(xf, bins=60, orientation="horizontal", density=True, color="0.6")
    grid = np.linspace(xf.min(), xf.max(), 300)
    for w, mu, var in zip(gmm.weights_, gmm.means_.ravel(), gmm.covariances_.ravel()):
        axh.plot(w * np.exp(-(grid - mu) ** 2 / (2 * var)) / np.sqrt(2 * np.pi * var), grid, lw=1)
    axh.axhline(thr, color="tab:green", ls="--", lw=1)
    axh.tick_params(labelleft=False)
ax = fig.add_subplot(gs[4, :3], sharex=ax0)
ax.plot(t, ripple_rate, lw=0.7, color="k")
ax.set_ylabel("ripples/s")
axes.append(ax)
ax = fig.add_subplot(gs[5, :3], sharex=ax0)
for st in STATES:
    ax.fill_between(t, 0, 1, where=names == st, step="mid", color=COLORS[st], label=st, lw=0)
ax.set_yticks([])
ax.set_xlabel("Time (s)")
ax.legend(loc="upper right", ncol=3, fontsize=8)
axes.append(ax)
for ax in axes[:-1]:
    ax.tick_params(labelbottom=False)
ax0.set_xlim(0, n_sec)
fig.suptitle(f"Sleep scoring: {', '.join(f'{st} {np.mean(names == st):.0%}' for st in STATES)}")
fig.savefig(f"{name}_states.png", dpi=120, bbox_inches="tight")
print(f"\nWrote {name}_states.csv, {name}_nrem_bouts.txt and {name}_states.png")
if args.show:
    plt.show()
