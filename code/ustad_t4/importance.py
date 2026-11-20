"""Per-second importance from motion and visual change - the model-free baseline.

Egocentric highlights tend to sit where the scene changes and the hands move.
This curve needs no weights, so it is the fallback whenever a model is missing,
and it is blended into the learned scores later.
"""
from __future__ import annotations

import numpy as np

from . import video


def motion_curve(path: str, duration: float, fps: float = 2.0) -> list[float]:
    frames = video.sample_frames(path, fps, 64, 36, gray=True).astype(np.float32)
    if len(frames) < 2:
        return [1.0] * max(1, int(round(duration)))
    diff = np.abs(np.diff(frames, axis=0)).mean(axis=(1, 2))
    diff = np.concatenate([[diff[0]], diff])
    per_sec = []
    n_sec = max(1, int(np.ceil(duration)))
    step = int(round(fps))
    for s in range(n_sec):
        chunk = diff[s * step:(s + 1) * step]
        per_sec.append(float(chunk.mean()) if len(chunk) else 0.0)
    arr = np.array(per_sec)
    # Smooth over 3 s and normalise to [0, 1].
    k = np.ones(3) / 3
    arr = np.convolve(arr, k, mode="same")
    lo, hi = arr.min(), arr.max()
    arr = (arr - lo) / (hi - lo) if hi > lo else np.ones_like(arr)
    return arr.tolist()
