"""Python port of natural_sleep_preprocessing.m.

Loads the hippocampus LFP (block4_sleep_OE.mat) and the cortex voltage-imaging
video (.raw, big-endian float32), notch-filters the LFP at 50 Hz and its
harmonics, and plots the results.

Usage:
    python natural_sleep_preprocessing.py block4_sleep_OE.mat \
        block444999fr_50Hz_241x384_Bandpass_Cheby_high_0.5FHz_D_F0_.raw
"""
import argparse

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
from scipy import signal

p = argparse.ArgumentParser()
p.add_argument("sleep_file", help="block4_sleep_OE.mat")
p.add_argument("ctx_file", help="cortex voltage-imaging .raw file")
args = p.parse_args()

# %% Load hippocampus and cortex data
Data_OE = sio.loadmat(args.sleep_file, squeeze_me=True, struct_as_record=False)["Data_OE"]

# Big-endian float32, memory-mapped so the (multi-GB) video is not read into RAM at once
raw_video = np.memmap(args.ctx_file, dtype=">f4", mode="r")

# %% Create and plot hippocampus lfp dataset
# originally 20kHz ---> downsample to 1kHz
HPC_LFP = [np.asarray(ch, dtype=float).ravel()[::20] for ch in Data_OE.HPC_LFP]
t = HPC_LFP[0].shape[0]
fs = 1000

fd = [49, 99, 149, 199, 249, 299, 349]
fu = [51, 101, 151, 201, 251, 301, 351]

# designfilt('bandstopiir','FilterOrder',2,...,'DesignMethod','butter') is a
# 2nd-order Butterworth bandstop, i.e. butter(N=1) with btype='bandstop'
d = [signal.butter(1, [lo, hi], btype="bandstop", fs=fs, output="sos") for lo, hi in zip(fd, fu)]

# filter signal
HPC_LFP_filtered = []
for x in HPC_LFP:
    for sos in d:
        x = signal.sosfiltfilt(sos, x)
    HPC_LFP_filtered.append(x)


def zscore(x):
    # MATLAB zscore uses the sample standard deviation (ddof=1)
    return (x - x.mean()) / x.std(ddof=1)


def plot_spectrogram(ax, x):
    # MATLAB spectrogram(x, fs, 0.9*fs, 2^14, fs): Hamming window of length fs,
    # 90% overlap, 2^14-point FFT, PSD shown in dB
    f, tt, Sxx = signal.spectrogram(x, fs=fs, window="hamming", nperseg=fs,
                                    noverlap=int(0.9 * fs), nfft=2**14,
                                    detrend=False, scaling="density", mode="psd")
    im = ax.pcolormesh(tt, f, 10 * np.log10(Sxx + np.finfo(float).eps), shading="auto")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    plt.colorbar(im, ax=ax, label="Power/frequency (dB/Hz)")


fig, axs = plt.subplots(2, 1)
plot_spectrogram(axs[0], signal.detrend(zscore(HPC_LFP[10])))
axs[0].set_title("Unfiltered")
plot_spectrogram(axs[1], signal.detrend(zscore(HPC_LFP_filtered[10])))
axs[1].set_title("Filtered")

HPC_LFP_filtered_normalized = [signal.detrend(zscore(x)) for x in HPC_LFP_filtered]

plt.figure()
time = np.arange(t) / fs
for i, x in enumerate(HPC_LFP_filtered_normalized, start=1):
    plt.plot(time, x - i * 1000)
plt.xlabel("Time (s)")
plt.ylabel("LFP/channel")

# %% Create and plot cortex voltage imaging dataset
# Convert into frames
W = 241  # frame width
H = 384  # frame height
frameFreq = 50

numFrames = len(Data_OE.frames)
# MATLAB reshape is column-major, hence order="F"
sleep_video = raw_video[:W * H * numFrames].reshape((W, H, numFrames), order="F")
sleep_video_half1 = sleep_video[:, :, :20000].transpose(1, 0, 2)
sleep_video_half2 = sleep_video[:, :, 20000:].transpose(1, 0, 2)

plt.figure()
plt.imshow(sleep_video_half1[:, :, 999], aspect="auto")
plt.colorbar()

plt.show()
