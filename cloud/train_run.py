"""Run the action model on EPIC-100 *train* videos to build RL / tuning episodes.

Only train-split videos are touched here (zero overlap with the 187 scored
manifest videos). Their window predictions + public train annotations become
the environment the highlight-selection policy is trained in.

    modal run cloud/train_run.py --n 30
    modal volume get ustad-t4 train_preds/ data/train_preds/
"""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import modal

T4 = Path(__file__).resolve().parent.parent

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg", "git")
    .pip_install("torch==2.5.1", "torchvision==0.20.1", index_url="https://download.pytorch.org/whl/cpu")
    .pip_install("numpy<2.3", "timm==1.0.11", "einops==0.8.0", "open_clip_torch==2.29.0", "pillow", "requests", "pyyaml")
    .run_commands("git clone --depth 1 https://github.com/epic-kitchens/epic-kitchens-download-scripts /opt/epicdl")
    .add_local_dir(T4 / "code", "/app/code", copy=True, ignore=["**/__pycache__/**", "**/.git/**"])
    .add_local_dir(T4 / "kit", "/app/kit", copy=True)
    .add_local_file(T4 / "weights" / "lavila_tsfb_ek100_multihead.pth", "/app/weights/lavila_tsfb_ek100_multihead.pth", copy=True)
)
vol = modal.Volume.from_name("ustad-t4", create_if_missing=True)
app = modal.App("ustad-t4-train", image=image)


@app.function(cpu=16, memory=24576, timeout=3 * 3600, volumes={"/vol": vol}, retries=1)
def train_preds(video_id: str) -> str:
    import glob
    import json
    import os
    import shutil
    import subprocess
    import sys
    from dataclasses import asdict

    sys.path.insert(0, "/app/code")
    import torch
    torch.set_num_threads(16)
    out = f"/vol/train_preds/{video_id}.preds.json"
    if os.path.exists(out):
        return f"{video_id}: already done"
    work = f"/tmp/t_{video_id}"
    os.makedirs(work, exist_ok=True)
    subprocess.run([sys.executable, "epic_downloader.py", "--videos", "--specific-videos", video_id, "--output-path", work],
                   cwd="/opt/epicdl", check=True, capture_output=True, text=True)
    path = (glob.glob(f"{work}/**/{video_id}.MP4", recursive=True) + glob.glob(f"{work}/**/{video_id}.mp4", recursive=True))[0]
    from ustad_t4 import actions, importance, video
    dur = video.duration(path)
    motion = importance.motion_curve(path, dur)
    preds = actions.predict(path)
    os.makedirs("/vol/train_preds", exist_ok=True)
    with open(out, "w") as f:
        json.dump({"duration": dur, "motion": motion, "windows": [asdict(w) for w in preds]}, f)
    shutil.rmtree(work, ignore_errors=True)
    vol.commit()
    return f"{video_id}: ok {dur:.0f}s, {len(preds)} windows"


@app.local_entrypoint()
def main(n: int = 30):
    ann = T4 / "data" / "epic_ann" / "EPIC_100_train.csv"
    end = defaultdict(float)

    def secs(ts: str) -> float:
        h, m, s = ts.split(":")
        return int(h) * 3600 + int(m) * 60 + float(s)

    for r in csv.DictReader(ann.open()):
        end[r["video_id"]] = max(end[r["video_id"]], secs(r["stop_timestamp"]))
    # 2-10 minute videos, at most two per participant, spread across kitchens.
    per_p, picked = defaultdict(int), []
    for vid, d in sorted(end.items()):
        p = vid.split("_")[0]
        if 120 <= d <= 600 and per_p[p] < 2:
            picked.append(vid)
            per_p[p] += 1
    picked = picked[:n]
    print(f"{len(picked)} train videos: {picked}")
    for res in train_preds.map(picked, return_exceptions=True, order_outputs=False):
        print(res, flush=True)
