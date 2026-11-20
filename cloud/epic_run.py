"""Process the EPIC-KITCHENS-100 manifest videos on Modal, stream-and-delete.

Each container downloads one video from the official EPIC endpoint (via the
official epic-kitchens-download-scripts), runs *the same* ustad_t4 pipeline that
run.sh runs, saves <video_id>.json plus raw window predictions to a Modal
volume, and deletes the video. No video is ever stored, so disk stays small.

    modal run cloud/epic_run.py                 # all pending EPIC videos
    modal run cloud/epic_run.py --limit 2       # smoke test on two videos
    modal volume get ustad-t4 outputs ./outputs_cloud   # fetch results
"""
from __future__ import annotations

import csv
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
T4 = HERE.parent

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg", "git")
    .pip_install("torch==2.5.1", "torchvision==0.20.1", index_url="https://download.pytorch.org/whl/cpu")
    .pip_install("numpy<2.3", "timm==1.0.11", "einops==0.8.0", "open_clip_torch==2.29.0", "pillow", "requests", "pyyaml")
    .run_commands("git clone --depth 1 https://github.com/epic-kitchens/epic-kitchens-download-scripts /opt/epicdl")
    .add_local_dir(T4 / "code", "/app/code", copy=True, ignore=["**/__pycache__/**", "**/.git/**"])
    .add_local_dir(T4 / "kit", "/app/kit", copy=True)
    .add_local_dir(T4 / "weights", "/app/weights", copy=True)
)

vol = modal.Volume.from_name("ustad-t4", create_if_missing=True)
app = modal.App("ustad-t4-epic", image=image)


@app.function(cpu=16, memory=24576, timeout=3 * 3600, volumes={"/vol": vol}, retries=1)
def process(video_id: str) -> str:
    import glob
    import json
    import os
    import shutil
    import subprocess
    import sys
    import time

    os.environ["USTAD_SAVE_PREDS"] = "/vol/preds"
    os.environ.setdefault("TEAM_ID", "INCOGNITO")
    sys.path.insert(0, "/app/code")
    import torch
    torch.set_num_threads(16)

    out_path = f"/vol/outputs/{video_id}.json"
    if os.path.exists(out_path):
        return f"{video_id}: already done"

    t0 = time.time()
    work = f"/tmp/epic_{video_id}"
    os.makedirs(work, exist_ok=True)
    subprocess.run(
        [sys.executable, "epic_downloader.py", "--videos", "--specific-videos", video_id, "--output-path", work],
        cwd="/opt/epicdl", check=True, capture_output=True, text=True,
    )
    found = glob.glob(f"{work}/**/{video_id}.MP4", recursive=True) + glob.glob(f"{work}/**/{video_id}.mp4", recursive=True)
    if not found:
        raise RuntimeError(f"{video_id}: download produced no file")
    src = found[0]
    # Name the file by its manifest id so the pipeline echoes the right video_id.
    path = f"{work}/{video_id}.mp4"
    os.replace(src, path)
    t_dl = time.time() - t0

    from ustad_t4 import run, schema

    doc = run.summarize(path)
    errors, _ = schema.check(doc, vocab=run.VOCAB)
    if errors:
        raise RuntimeError(f"{video_id}: invalid output {errors}")
    os.makedirs("/vol/outputs", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(doc, f, indent=2)
    shutil.rmtree(work, ignore_errors=True)
    vol.commit()
    return f"{video_id}: ok, download {t_dl:.0f}s, total {time.time() - t0:.0f}s, {len(doc['actions'])} actions"


@app.local_entrypoint()
def main(limit: int = 0):
    rows = [r for r in csv.DictReader((T4 / "kit" / "manifest.csv").open()) if r["dataset"] == "epic_kitchens_100"]
    done = set()
    try:
        done = {e.path.split("/")[-1].removesuffix(".json") for e in vol.listdir("outputs")}
    except Exception:
        pass
    # Longest first, so the slowest videos start earliest.
    todo = sorted((r for r in rows if r["video_id"] not in done), key=lambda r: -float(r["duration_sec"]))
    if limit:
        todo = sorted(todo, key=lambda r: float(r["duration_sec"]))[:limit]
    print(f"{len(done)} done, {len(todo)} to process")
    for res in process.map([r["video_id"] for r in todo], return_exceptions=True, order_outputs=False):
        print(res, flush=True)
