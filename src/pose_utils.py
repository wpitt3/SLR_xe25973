#!/usr/bin/env python3
import glob
import os
import pickle
import gzip
import random

import numpy as np
import torch
import matplotlib.pyplot as plt

GROUPS = {
    "body": list(range(0, 17)),
    "feet": list(range(17, 23)),
    "face": list(range(23, 91)),
    "lhand": list(range(91, 112)),
    "rhand": list(range(112, 133)),
}

GROUP_COLORS = {
    "body": "#1b9e77",
    "feet": "#66a61e",
    "face": "#7570b3",
    "lhand": "#e7298a",
    "rhand": "#d95f02",
    "CSL-News lhand": "#1b9e77",
    "CSL-News rhand": "#d95f02",
    "CSL-Daily rhand": "#7570b3",
    "CSL-Daily lhand": "#e7298a",
}

MSKA_KEYPOINTS = set([
    0, 1, 3, 5, 7, 9, 2, 4, 6, 8, 10,  # upper body
    *range(91, 112), *range(112, 133),  # hands
    23, 26, 29, 33, 36, 39, 41, 43, 46, 48, 53, 56, 59, 62, 65, 68, 71, 72, 73, 74, 75, 76, 77, 79, 80, 81,  # face
])

COCO_BODY_EDGES = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
]

HAND_EDGES_LOCAL = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
]


def _hand_edges(offset):
    return [(a + offset, b + offset) for a, b in HAND_EDGES_LOCAL]


def _mska_keypoints():
    groups = {g: [i for i in idx if i in MSKA_KEYPOINTS] for g, idx in GROUPS.items()}
    return {g: idx for g, idx in groups.items() if idx}


EDGES = {"body": COCO_BODY_EDGES, "lhand": _hand_edges(91), "rhand": _hand_edges(112)}

GROUPED_MSKA_KEYPOINTS = _mska_keypoints()

def load_dataset_file(filename):
    try:
        with open(filename, "rb") as f:
            return pickle.load(f)
    except pickle.UnpicklingError:
        with gzip.open(filename, "rb") as f:
            return pickle.load(f)


def load_csl_pose_pkl(path):
    person_i = 0
    with open(path, "rb") as f:
        d = pickle.load(f)

    keypoints = np.asarray(d["keypoints"])
    scores = np.asarray(d["scores"])
    w, h = d["w_h"]

    if keypoints.ndim == 4:
        keypoints = keypoints[:, person_i]
        scores = scores[:, person_i]
    else:
        raise ValueError(f"{path}: unexpected keypoints shape {keypoints.shape}")

    kp = np.concatenate([keypoints, scores[..., None]], axis=-1)
    return kp, int(w), int(h)


def load_csl_news_folder(folder, pattern="**/*.pkl", max_files=20000):
    filepaths = sorted(glob.glob(os.path.join(folder, pattern), recursive=True))
    rng = random.Random(0)
    filepaths = sorted(rng.sample(filepaths, min(max_files, len(filepaths))))
    data = {}
    skipped = []

    for i, fp in enumerate(filepaths):
        name = os.path.splitext(os.path.basename(fp))[0]
        try:
            kp, w, h = load_csl_pose_pkl(fp)
        except Exception as e:
            skipped.append((fp, str(e)))
            continue

        data[name] = {
            "keypoint": torch.from_numpy(kp).float(),
            "num_frames": kp.shape[0],
            "width": w,
            "height": h,
        }

    print(f"Loaded: {len(data)}, Skipped: {len(skipped)}")
    return data, skipped


def compute_group_confidence_histograms(data, groups, n_bins=50):
    counts = {g: np.zeros(n_bins, dtype=np.int64) for g in groups}
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    for sample in data.values():
        kp = sample["keypoint"]
        kp = kp.numpy() if torch.is_tensor(kp) else np.asarray(kp)
        conf = kp[..., 2]
        for g, idx in groups.items():
            counts[g] += np.histogram(conf[:, idx].ravel(), bins=edges)[0]

    return edges, counts


def prefix_dicts(d, prefix):
    return {f"{prefix}{k}": v for k, v in d.items()}


def plot_compare_confidence_distribution(data_a, data_b, prefix_a, prefix_b, mska_only=False, n_bins=50, title=None, xlabel=None, yLim=None):
    groups = GROUPED_MSKA_KEYPOINTS if mska_only else GROUPS

    edges, counts_a = compute_group_confidence_histograms(data_a, groups, n_bins=n_bins)
    _, counts_b = compute_group_confidence_histograms(data_b, groups, n_bins=n_bins)
    del(counts_a['face'])
    del(counts_b['face'])
    del (counts_a['body'])
    del (counts_b['body'])
    counts = {**prefix_dicts(counts_a, prefix_a), **prefix_dicts(counts_b, prefix_b)}

    return _plot_confidence(edges, counts, title=title, yLim=yLim)


def plot_confidence_distribution(data, mska_only=False, n_bins=50, title=None, yLim=None):
    groups = GROUPED_MSKA_KEYPOINTS if mska_only else GROUPS

    edges, counts = compute_group_confidence_histograms(data, groups, n_bins=n_bins)
    return _plot_confidence(edges, counts, title=title, yLim=yLim)


def _plot_confidence(edges, counts, title=None, yLim=None):
    centers = 0.5 * (edges[:-1] + edges[1:])

    fig, ax = plt.subplots(figsize=(9, 4.5))

    for g, c in counts.items():
        y = c.astype(float)
        y = y / y.sum()
        ax.plot(centers, y, label=g, color=GROUP_COLORS.get(g, None), lw=2)

    if yLim:
        ax.set_ylim(0, yLim)
    ax.set_xlabel("Pose confidence")
    ax.set_ylabel("Fraction")
    ax.set_title(title or f"Per body part confidence distribution")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig, ax
