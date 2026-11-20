"""Video-level label (Label accuracy, 10%, scored on FPVSum).

Primary: zero-shot CLIP (open_clip ViT-B-32) over a few frames against one text
prompt per kit label. Fallback when CLIP weights are not installed: kitchen
evidence from the action model -> "cooking", else "other".

Weights live in weights/clip/ so the run works offline.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from . import video

CLIP_DIR = Path(os.environ.get("USTAD_CLIP_DIR", Path(__file__).resolve().parents[2] / "weights" / "clip"))
CLIP_ARCH = "ViT-B-32"
CLIP_FILE = CLIP_DIR / "open_clip_vit_b32_laion2b.bin"

PROMPTS = {
    "bike": "a first-person video riding a mountain bike",
    "bikepolo": "a first-person video playing bike polo",
    "boxing": "a first-person video of boxing in a ring",
    "cooking": "a first-person video of cooking in a kitchen",
    "horse": "a first-person video riding a horse",
    "jumping": "a first-person video of skydiving or bungee jumping",
    "longboarding": "a first-person video longboarding down a road",
    "motor": "a first-person video riding a motorcycle",
    "parkour": "a first-person video doing parkour on rooftops",
    "plane": "a first-person video flying a small airplane",
    "rockclimbing": "a first-person video rock climbing a cliff",
    "scuba": "a first-person video scuba diving underwater",
    "skate": "a first-person video skateboarding in a skatepark",
    "ski": "a first-person video skiing down a snowy slope",
    "surfing": "a first-person video surfing a wave",
    "other": "a first-person video of an everyday activity",
}

_clip = None


def _load_clip():
    global _clip
    if _clip is None:
        import open_clip
        import torch

        model, _, preprocess = open_clip.create_model_and_transforms(CLIP_ARCH, pretrained=str(CLIP_FILE))
        tok = open_clip.get_tokenizer(CLIP_ARCH)
        model.eval()
        with torch.inference_mode():
            names = list(PROMPTS)
            text = model.encode_text(tok([PROMPTS[n] for n in names]))
            text = text / text.norm(dim=-1, keepdim=True)
        _clip = (model, preprocess, names, text)
    return _clip


def _clip_label(path: str, duration: float, labels: list[str]) -> dict:
    import torch
    from PIL import Image

    model, preprocess, names, text = _load_clip()
    n = 8
    fps = max(n / max(duration, 1.0), 0.02)
    frames = video.sample_frames(path, fps, 320, 180)[:n]
    if len(frames) == 0:
        return {"name": "other", "score": 0.0}
    with torch.inference_mode():
        imgs = torch.stack([preprocess(Image.fromarray(f)) for f in frames])
        emb = model.encode_image(imgs)
        emb = emb / emb.norm(dim=-1, keepdim=True)
        probs = (100.0 * emb @ text.T).softmax(dim=-1).mean(0).numpy()
    allowed = [i for i, nm in enumerate(names) if nm in labels]
    best = max(allowed, key=lambda i: probs[i])
    return {"name": names[best], "score": round(float(probs[best]), 4)}


HIGHLIGHT_TEXT = "a first-person video frame of the most exciting, intense moment of {}"
DULL_TEXT = "a first-person video frame of an uneventful, boring moment of {}"


def excitement_curve(path: str, duration: float, label_name: str) -> list[float] | None:
    """Per-second 'is this a highlight?' score from CLIP, in [0, 1].

    Each second is scored by how much closer its frame sits to an "exciting
    moment of <activity>" prompt than to an "uneventful moment" prompt.
    Returns None when CLIP is unavailable.
    """
    if not CLIP_FILE.exists():
        return None
    import torch
    from PIL import Image

    model, preprocess, _, _ = _load_clip()
    import open_clip
    tok = open_clip.get_tokenizer(CLIP_ARCH)
    activity = PROMPTS.get(label_name, PROMPTS["other"]).replace("a first-person video ", "")
    frames = video.sample_frames(path, 1.0, 320, 180)
    if len(frames) == 0:
        return None
    scores = []
    with torch.inference_mode():
        txt = model.encode_text(tok([HIGHLIGHT_TEXT.format(activity), DULL_TEXT.format(activity)]))
        txt = txt / txt.norm(dim=-1, keepdim=True)
        for i in range(0, len(frames), 32):
            imgs = torch.stack([preprocess(Image.fromarray(f)) for f in frames[i:i + 32]])
            emb = model.encode_image(imgs)
            emb = emb / emb.norm(dim=-1, keepdim=True)
            p = (100.0 * emb @ txt.T).softmax(dim=-1)[:, 0]
            scores.extend(p.tolist())
    arr = np.array(scores, dtype=np.float32)
    arr = np.convolve(arr, np.ones(3) / 3, mode="same")
    lo, hi = float(arr.min()), float(arr.max())
    arr = (arr - lo) / (hi - lo) if hi > lo else np.ones_like(arr)
    n = max(1, int(np.ceil(duration)))
    if len(arr) < n:
        arr = np.concatenate([arr, np.full(n - len(arr), arr[-1])])
    return arr[:n].tolist()


def cooking_prob(path: str, duration: float) -> float:
    """Probability mass CLIP gives to 'cooking' - used to decide whether the
    kitchen action model is worth running on this video."""
    if not CLIP_FILE.exists():
        return 1.0
    import torch
    from PIL import Image

    model, preprocess, names, text = _load_clip()
    frames = video.sample_frames(path, max(8 / max(duration, 1.0), 0.02), 320, 180)[:8]
    if len(frames) == 0:
        return 1.0
    with torch.inference_mode():
        imgs = torch.stack([preprocess(Image.fromarray(f)) for f in frames])
        emb = model.encode_image(imgs)
        emb = emb / emb.norm(dim=-1, keepdim=True)
        probs = (100.0 * emb @ text.T).softmax(dim=-1).mean(0)
    return float(probs[names.index("cooking")])


def classify(path: str, duration: float, preds, vocab: dict) -> dict:
    labels = vocab["labels"]
    if CLIP_FILE.exists():
        try:
            return _clip_label(path, duration, labels)
        except Exception as exc:  # fall through to the evidence rule
            print(f"label: CLIP failed ({exc}); using action evidence", flush=True)
    if preds:
        conf = float(np.mean([max(p for _, p in w.nouns[:1]) for w in preds]))
        if conf >= 0.25:
            return {"name": "cooking", "score": round(min(1.0, conf + 0.3), 4)}
    return {"name": "other", "score": 0.3}
