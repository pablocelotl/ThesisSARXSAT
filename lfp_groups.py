"""Group highly correlated hippocampal LFP channels and average every group into one signal.

Channels are clustered by hierarchical clustering (average linkage) on the distance 1 - |Pearson correlation|, 
so the channels within a group move together (or exactly opposite to each other).

Usage on its own, to see the groups:
    python lfp_groups.py "Example data pre-processing/hpc_filtered.mat" --groups 3
and to also write the group means as a HySI-SAT dataset (Data/<name>_output.txt + _input_dummy.txt):
    python lfp_groups.py "Example data pre-processing/hpc_filtered.mat" --groups 3 \
        --write --zscore --start 100000 --length 10000
"""
import argparse

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

LFP_DEPTH_GROUP_COUNT = 3


def cluster_lfp_channels(
    channels: list[np.ndarray],
    n_groups: int = LFP_DEPTH_GROUP_COUNT,
) -> list[list[int]]:
    """Group similar LFP channels by absolute Pearson correlation."""
    signals = np.stack(channels, axis=0).astype(np.float64, copy=False)
    if not 1 < n_groups <= signals.shape[0]:
        raise ValueError("n_groups must be between 2 and the number of LFP channels")
    channel_std = signals.std(axis=1, keepdims=True)
    if np.any(channel_std <= np.finfo(np.float64).eps):
        raise ValueError("An LFP channel has zero variance and cannot be clustered")
    standardized = (signals - signals.mean(axis=1, keepdims=True)) / channel_std
    correlation = np.corrcoef(standardized)
    distance = 1.0 - np.abs(correlation)
    np.fill_diagonal(distance, 0.0)
    tree = linkage(squareform(distance, checks=False), method="average")
    labels = fcluster(tree, n_groups, criterion="maxclust")
    return [
        np.flatnonzero(labels == label).tolist()
        for label in sorted(
            np.unique(labels),
            key=lambda label: np.flatnonzero(labels == label).min(),
        )
    ]


def group_lfp_channels(channels, n_groups=LFP_DEPTH_GROUP_COUNT):
    """Return (groups, grouped): the channel indices of every group and the group means (groups x samples)."""
    groups = cluster_lfp_channels(channels, n_groups)
    grouped = np.stack(
        [np.mean(np.stack([channels[channel] for channel in group]), axis=0) for group in groups],
        axis=0,
    )
    return groups, grouped


def describe_groups(channels, groups):
    """Print the channels of every group, warning about channels that are anti-correlated with the
    first channel of their group (those partly cancel out in the group mean)."""
    correlation = np.corrcoef(np.stack(channels, axis=0))
    for g, group in enumerate(groups):
        r = [correlation[group[0], c] for c in group]
        print(f"group {g}: channels {group}  min |r| within group = {min(abs(x) for x in r):.2f}")
        negative = [c for c, x in zip(group, r) if x < 0]
        if negative:
            print(f"  warning: channels {negative} are anti-correlated with channel {group[0]} "
                  f"and partly cancel out in the group mean")


if __name__ == "__main__":
    import scipy.io as sio

    p = argparse.ArgumentParser()
    p.add_argument("hpc", help="hippocampal LFP .mat file, e.g. hpc_filtered.mat")
    p.add_argument("--hpc-key", default="HPC_LFP_filtered", help="variable in the .mat file")
    p.add_argument("--groups", type=int, default=LFP_DEPTH_GROUP_COUNT, help="number of groups")
    p.add_argument("--write", action="store_true", help="write the group means as a HySI-SAT dataset")
    p.add_argument("--start", type=int, default=0, help="first sample to keep (with --write)")
    p.add_argument("--length", type=int, help="number of samples to keep (with --write, default: all)")
    p.add_argument("--zscore", action="store_true",
                   help="normalize every group mean to zero mean and unit std (with --write)")
    p.add_argument("--name", help="output name in Data/ (default: hpc_g<groups>)")
    args = p.parse_args()

    hpc_lfp_filtered = [np.asarray(ch, dtype=float).ravel()
                        for ch in sio.loadmat(args.hpc, squeeze_me=True)[args.hpc_key]]
    groups, _ = group_lfp_channels(hpc_lfp_filtered, args.groups)
    describe_groups(hpc_lfp_filtered, groups)

    if args.write:
        from make_io_dataset import write_io_dataset

        # groups are found on the whole recording, before cutting out the requested range
        _, grouped = group_lfp_channels(hpc_lfp_filtered, args.groups)
        end = args.start + args.length if args.length else grouped.shape[1]
        x = grouped[:, args.start:end]
        if args.zscore:
            x = (x - x.mean(axis=1, keepdims=True)) / x.std(axis=1, ddof=1, keepdims=True)
        write_io_dataset(x, f"Data/{args.name or f'hpc_g{args.groups}'}")
