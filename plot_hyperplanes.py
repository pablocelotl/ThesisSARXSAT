"""Visualize the PWARX partition (H.mat) exported by HySI-SAT for one identification run.

H has shape (modes, modes-1, n+1). Mode i is active where H[i] @ [x; 1] <= 0 for every row,
with x the regressor [y lags..., u lags...] as stored in Rt.mat (last column of Rt is the constant 1).
Each row of H[i] equals W[j] - W[i] for one other mode j, where W are the LinearSVC scores, so
the partition is equivalently "mode = argmax_i W[i] @ [x; 1]".

Usage: python plot_hyperplanes.py Data/IdentifiedDatasets/<run> [--axes 4 5]
"""
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import scipy.io
from matplotlib.colors import ListedColormap


def load(run):
    H = scipy.io.loadmat(os.path.join(run, "H.mat"))["H"]
    R = scipy.io.loadmat(os.path.join(run, "Rt.mat"))["Rt"]
    sigma = scipy.io.loadmat(os.path.join(run, "Sigmat.mat"))["sigmat"].ravel()
    U = scipy.io.loadmat(os.path.join(run, "Ut.mat"))["Ut"]
    Y = scipy.io.loadmat(os.path.join(run, "Yt.mat"))["Yt"]
    return H, R, sigma, U.shape[0], Y.shape[0]


def scores_from_H(H):
    """Recover SVM score vectors W (up to a common offset) from H: W[0] = 0, W[i] = H[0][i-1]."""
    W = np.vstack([np.zeros(H.shape[2]), H[0]])
    others = lambda j: [i for i in range(H.shape[0]) if i != j]
    err = max(abs(H[j][r] - (W[i] - W[j])).max() for j in range(H.shape[0]) for r, i in enumerate(others(j)))
    assert err < 1e-8, "H does not have the expected pairwise structure"
    return W


def regressor_labels(n, inputs, outputs):
    """Guess ny, nu from the regressor length; fall back to x1..xn if ambiguous."""
    fits = [(ny, nu) for ny in range(1, 11) for nu in range(1, 11) if ny * outputs + nu * inputs == n]
    if len(fits) != 1:
        return ["x{}".format(i + 1) for i in range(n)]
    ny, nu = fits[0]
    labels = ["y{}(k-{})".format(a + 1, ny - l) for a in range(outputs) for l in range(ny)]
    labels += ["u{}(k-{})".format(a + 1, nu - l) for a in range(inputs) for l in range(nu)]
    return labels


def mode_cmap(modes):
    colors = list(plt.get_cmap("tab20").colors) + list(plt.get_cmap("tab20b").colors)
    return ListedColormap(colors[:modes])


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run", help="directory of an identification run (contains H.mat, Rt.mat, ...)")
    p.add_argument("--axes", type=int, nargs=2, help="two regressor indices for the 2D slice (default: largest spread)")
    p.add_argument("--show", action="store_true", help="open the figures instead of only saving them")
    args = p.parse_args()

    H, R, sigma, inputs, outputs = load(args.run)
    modes, n = H.shape[0], H.shape[2] - 1
    W = scores_from_H(H)
    labels = regressor_labels(n, inputs, outputs)
    region = np.argmax(R @ W.T, axis=1)

    print("H.mat: array of shape {} = (modes, modes-1 boundaries per mode, {} regressors + constant)".format(H.shape, n))
    print("Regressor order:", labels + ["1"])
    print("Mode i active where H[i] @ [x; 1] <= 0 (all rows)")
    print("Samples inside the region of their identified mode: {:.1%}".format((region == sigma).mean()))
    empty = sorted(set(range(modes)) - set(region))
    print("Modes whose region contains no data point ({}): {}".format(len(empty), empty))

    cmap = mode_cmap(modes)
    ax_i, ax_j = args.axes if args.axes else np.argsort(R[:, :n].std(axis=0))[-2:][::-1]

    ## 2D slice: vary two regressors, hold the others at their median
    fig1, ax = plt.subplots(figsize=(9, 7))
    xs = np.linspace(R[:, ax_i].min(), R[:, ax_i].max(), 400)
    ys = np.linspace(R[:, ax_j].min(), R[:, ax_j].max(), 400)
    xx, yy = np.meshgrid(xs, ys)
    grid = np.tile(np.median(R, axis=0), (xx.size, 1))
    grid[:, ax_i], grid[:, ax_j] = xx.ravel(), yy.ravel()
    zz = np.argmax(grid @ W.T, axis=1).reshape(xx.shape)
    ax.pcolormesh(xx, yy, zz, cmap=cmap, vmin=-0.5, vmax=modes - 0.5, alpha=0.35, shading="auto")
    ax.contour(xx, yy, zz, levels=np.arange(modes) + 0.5, colors="k", linewidths=0.6)
    sc = ax.scatter(R[:, ax_i], R[:, ax_j], c=sigma, cmap=cmap, vmin=-0.5, vmax=modes - 0.5, s=12, edgecolors="k", linewidths=0.2)
    fig1.colorbar(sc, ax=ax, ticks=range(modes), label="mode")
    ax.set_xlabel(labels[ax_i])
    ax.set_ylabel(labels[ax_j])
    ax.set_title("Regions on a 2D slice (other regressors at median)\nbackground = region, dots = identified mode of each sample")

    ## Per mode: samples labelled vs samples that land in its region
    fig2, ax = plt.subplots(figsize=(11, 4))
    idx = np.arange(modes)
    ax.bar(idx - 0.2, np.bincount(sigma, minlength=modes), 0.4, label="samples assigned to mode (sigmat)")
    ax.bar(idx + 0.2, np.bincount(sigma[region == sigma], minlength=modes), 0.4, label="... of which inside the mode's region")
    ax.set_xticks(idx)
    ax.set_xlabel("mode")
    ax.set_ylabel("samples")
    ax.legend()
    ax.set_title("How well the hyperplanes reproduce the switching sequence ({:.0%} overall)".format((region == sigma).mean()))

    ## Over time: identified mode vs the mode the partition predicts
    fig3, ax = plt.subplots(figsize=(11, 4))
    k = np.arange(1, len(sigma) + 1)
    ax.scatter(k, sigma, s=6, label="identified mode (sigmat)")
    ax.scatter(k[region != sigma], region[region != sigma], s=6, marker="x", c="r", label="region mode, where it differs")
    ax.set_xlabel("[k]")
    ax.set_ylabel("mode")
    ax.legend()
    ax.grid()

    for fig, name in ((fig1, "H_slice.png"), (fig2, "H_per_mode.png"), (fig3, "H_over_time.png")):
        fig.tight_layout()
        fig.savefig(os.path.join(args.run, name), dpi=130)
        print("saved", os.path.join(args.run, name))
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
