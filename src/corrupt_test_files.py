#!/usr/bin/env python3

import pickle, os
import numpy as np
import torch
LSHO, RSHO, CENTER = 5, 6, [5, 6, 11, 12]


def shoulder_width(xy):
    return float(np.median(np.linalg.norm(xy[:, LSHO] - xy[:, RSHO], axis=1)) + 1e-6)


def jitter(kp, sev, rng):
    out = kp.copy()
    out[:, :, :2] += rng.normal(0, sev * shoulder_width(kp[:, :, :2]), out[:, :, :2].shape).astype(np.float32)
    return out


def rotate(kp, sev, rng):
    out = kp.copy()
    ctr = out[:, CENTER, :2].mean(1, keepdims=True)
    a = np.deg2rad(rng.normal(0, sev))
    R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]], np.float32)
    out[:, :, :2] = (out[:, :, :2] - ctr) @ R.T + ctr
    return out


def confidence_gate(kp, sev, rng):
    out = kp.copy()
    low = out[:, :, 2] < sev
    out[low, :2] = 0.0; out[low, 2] = 0.0
    return out


def frame_freeze(kp, sev, rng):
    out = kp.copy()
    for t in range(1, out.shape[0]):
        if rng.random() < sev:
            out[t] = out[t - 1]
    return out


CORRUPTIONS = {"jitter": jitter, "rotate": rotate, "confidence_gate": confidence_gate, "frame_freeze": frame_freeze}


def corrupt(test_input_location, output, corruptions, seed):
    ops = [(n, float(s)) for n, s in (c.split(":") for c in corruptions)]
    rng = np.random.default_rng(seed)
    data = pickle.load(open(test_input_location, "rb"))
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)

    for k, s in data.items():
        kp = np.asarray(s["keypoint"], dtype=np.float32)
        for n, sev in ops:
            kp = CORRUPTIONS[n](kp, sev, rng)
        s["keypoint"] = torch.from_numpy(np.ascontiguousarray(kp))
        s["num_frames"] = int(kp.shape[0])

    pickle.dump(data, open(output, "wb"))

