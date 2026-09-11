#!/usr/bin/env python3
import subprocess
import os
import yaml
from corrupt_test_files import corrupt

BASE_CONFIG = "configs/csl-daily_s2g.yaml"
TEST_PKL = "data/RTM/CSL-Daily.test"
CORRUPT_DIR = "data/corrupt"

MODELS = {
    "baseline": "checkpoints/base/best_checkpoint.pth",
    "arm2":   "checkpoints/arm2/best_checkpoint.pth",
    "arm3":   "checkpoints/arm3/best_checkpoint.pth",
    "jitter":   "checkpoints/jitter/best_checkpoint.pth",
}

CONFIGS = [
    {"name": "clean",            "corr": []},
    {"name": "jitter_0.01",      "corr": ["jitter:0.01"]},
    {"name": "jitter_0.02",      "corr": ["jitter:0.02"]},
    {"name": "jitter_0.03",      "corr": ["jitter:0.03"]},
    {"name": "jitter_0.04",      "corr": ["jitter:0.04"]},
    {"name": "jitter_0.05",      "corr": ["jitter:0.05"]},
    {"name": "rotate_5",         "corr": ["rotate:5"]},
    {"name": "rotate_10",        "corr": ["rotate:10"]},
    {"name": "rotate_15",        "corr": ["rotate:15"]},
    {"name": "rotate_25",        "corr": ["rotate:25"]},
    {"name": "rotate_35",        "corr": ["rotate:35"]},
    {"name": "rotate_45",        "corr": ["rotate:45"]},
    {"name": "confgate_0.2",     "corr": ["confidence_gate:0.2"]},
    {"name": "confgate_0.35",    "corr": ["confidence_gate:0.35"]},
    {"name": "confgate_0.5",     "corr": ["confidence_gate:0.5"]},
    {"name": "freeze_0.1",       "corr": ["frame_freeze:0.1"]},
    {"name": "freeze_0.15",      "corr": ["frame_freeze:0.15"]},
    {"name": "freeze_0.2",       "corr": ["frame_freeze:0.2"]},
    {"name": "freeze_0.3",       "corr": ["frame_freeze:0.3"]},
    {"name": "freeze_0.4",       "corr": ["frame_freeze:0.4"]},
    {"name": "freeze_0.5",       "corr": ["frame_freeze:0.5"]},
]


def make_corrupted_input(cfg):
    if not cfg["corr"]:
        return TEST_PKL
    out = f"{CORRUPT_DIR}/test.{cfg['name']}"
    corrupt(TEST_PKL, out, cfg["corr"], 0)
    return out


def create_temp_yaml(pkl, tag):
    cfg = yaml.safe_load(open(BASE_CONFIG))
    cfg["data"]["test_label_path"] = pkl
    tmp_yml = f"{CORRUPT_DIR}/_cfg_{tag}.yaml"
    yaml.safe_dump(cfg, open(tmp_yml, "w"))
    return tmp_yml


def eval_model(ckpt, pkl, tag):
    tmp_yml = create_temp_yaml(pkl, tag)
    log = f"robust_logs/{tag}.log"
    os.makedirs("robust_logs", exist_ok=True)
    with open(log, "w") as f:
        subprocess.run(["python", "train.py", "--config", tmp_yml, "--eval", "--eval_test_only", "--resume", ckpt], stdout=f, stderr=subprocess.STDOUT)


def main():
    os.makedirs(CORRUPT_DIR, exist_ok=True)

    for cfg in CONFIGS:
        pkl = make_corrupted_input(cfg)
        for m, ckpt in MODELS.items():
            eval_model(ckpt, pkl, f"{m}__{cfg['name']}")
            print(f"{m:10s} {cfg['name']:10s}", flush=True)


if __name__ == "__main__":
    main()
