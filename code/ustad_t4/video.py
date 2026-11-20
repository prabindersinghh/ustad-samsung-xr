"""Video access through ffmpeg/ffprobe: duration and low-resolution frame sampling.

Frames are decoded straight into numpy through a pipe, so nothing is written to
disk and a 10-minute video costs a few MB of memory at the sizes used here.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess

import numpy as np

FFMPEG = os.environ.get("FFMPEG") or shutil.which("ffmpeg") or "ffmpeg"
FFPROBE = os.environ.get("FFPROBE") or shutil.which("ffprobe") or "ffprobe"


def duration(path: str) -> float:
    out = subprocess.run(
        [FFPROBE, "-v", "error", "-show_entries", "format=duration", "-of", "json", path],
        capture_output=True, text=True, check=True,
    ).stdout
    return float(json.loads(out)["format"]["duration"])


def sample_frames(path: str, fps: float, width: int, height: int, gray: bool = False) -> np.ndarray:
    """Decode the whole video at `fps`, resized to width x height.

    Returns an array of shape (n, h, w) for gray or (n, h, w, 3) for RGB, uint8.
    Frame i is taken at roughly i / fps seconds.
    """
    pix, ch = ("gray", 1) if gray else ("rgb24", 3)
    cmd = [
        FFMPEG, "-v", "error", "-i", path,
        "-vf", f"fps={fps},scale={width}:{height}",
        "-f", "rawvideo", "-pix_fmt", pix, "pipe:1",
    ]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    frame = width * height * ch
    n = len(raw) // frame
    arr = np.frombuffer(raw[: n * frame], dtype=np.uint8)
    return arr.reshape((n, height, width)) if gray else arr.reshape((n, height, width, 3))
