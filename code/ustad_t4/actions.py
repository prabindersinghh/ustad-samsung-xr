"""Egocentric action recognition with LaViLa (TimeSformer-B, fine-tuned on EPIC-100).

Checkpoint: LaViLa "ft_ek100_cls" multi-head (verb / noun / action heads),
trained on the EPIC-KITCHENS-100 *train* split. Reported top-1 on EPIC-100 val:
verb 69.0, noun 58.4, action 46.9. Source: facebookresearch/LaViLa model zoo.

The verb and noun heads index EPIC's canonical class ids, which are exactly the
first 97 verbs / 300 nouns of the kit's supported_classes.json (the kit appends
"other" to each). We use only those two heads, never the action head, whose index
map is built from annotation files.

The vision tower is built directly (no OpenAI CLIP download), so inference works
offline with nothing but weights/.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import video

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "third_party" / "lavila_min"))

WEIGHTS = Path(os.environ.get("USTAD_WEIGHTS", HERE.parents[1] / "weights" / "lavila_tsfb_ek100_multihead.pth"))

FPS = 8            # frames per second fed to the model
CLIP = 16          # frames per clip -> 2.0 s windows
STRIDE = 8         # frames between window starts -> 1.0 s
SIZE = 224
TOPK = 5
MEAN = np.array([108.3272985, 116.7460125, 104.09373615], dtype=np.float32)
STD = np.array([68.5005327, 66.6321579, 70.32316305], dtype=np.float32)

_model = None


@dataclass
class WindowPred:
    start: float
    end: float
    verbs: list = field(default_factory=list)   # [(class_idx, prob)] top-k
    nouns: list = field(default_factory=list)


def available() -> bool:
    return WEIGHTS.exists()


def _device():
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def load():
    global _model
    if _model is not None:
        return _model
    import torch
    import torch.nn as nn
    from lavila.models.openai_model import QuickGELU
    from lavila.models.timesformer import SpaceTimeTransformer

    ckpt = torch.load(str(WEIGHTS), map_location="cpu", weights_only=True)
    state = {k.replace("module.", "", 1): v for k, v in ckpt["state_dict"].items()}

    visual = SpaceTimeTransformer(
        num_frames=CLIP, time_init="zeros", attention_style="frozen-in-time",
        ln_pre=True, act_layer=QuickGELU, is_tanh_gating=False, drop_path_rate=0.0,
    )
    visual.head = nn.Identity()
    visual.pre_logits = nn.Identity()

    heads = sorted({int(k.split(".")[1]) for k in state if k.startswith("fc_cls.")})
    sizes = [state[f"fc_cls.{i}.weight"].shape[0] for i in heads]

    class Classifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.visual = visual
            self.fc_cls = nn.ModuleList([nn.Linear(visual.num_features, n) for n in sizes])

        def forward(self, x):
            emb = self.visual(x)
            return [m(emb) for m in self.fc_cls]

    model = Classifier()
    missing, unexpected = model.load_state_dict(state, strict=False)
    real_missing = [k for k in missing if "temporal_embed" not in k]
    if real_missing:
        raise RuntimeError(f"checkpoint does not match the model: missing {real_missing[:5]}")
    model.eval().to(_device())
    _model = (model, sizes)
    return _model


def _frames(path: str) -> np.ndarray:
    # Resize short side to 224 then centre-crop, as LaViLa's validation transform does.
    cmd_vf = f"fps={FPS},scale='if(gt(iw,ih),-2,{SIZE})':'if(gt(iw,ih),{SIZE},-2)',crop={SIZE}:{SIZE}"
    import subprocess
    raw = subprocess.run(
        [video.FFMPEG, "-v", "error", "-i", path, "-vf", cmd_vf, "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"],
        capture_output=True, check=True,
    ).stdout
    n = len(raw) // (SIZE * SIZE * 3)
    return np.frombuffer(raw[: n * SIZE * SIZE * 3], dtype=np.uint8).reshape(n, SIZE, SIZE, 3)


def predict(path: str, batch: int = 8) -> list[WindowPred]:
    import torch

    model, sizes = load()
    dev = _device()
    frames = _frames(path)
    if len(frames) < CLIP:  # very short video: pad by repeating the last frame
        pad = np.repeat(frames[-1:], CLIP - len(frames), axis=0) if len(frames) else np.zeros((CLIP, SIZE, SIZE, 3), np.uint8)
        frames = np.concatenate([frames, pad]) if len(frames) else pad
    starts = list(range(0, len(frames) - CLIP + 1, STRIDE))
    out: list[WindowPred] = []
    with torch.inference_mode():
        for i in range(0, len(starts), batch):
            idx = starts[i:i + batch]
            clips = np.stack([frames[s:s + CLIP] for s in idx]).astype(np.float32)
            clips = (clips - MEAN) / STD                        # B T H W C
            x = torch.from_numpy(clips).permute(0, 4, 1, 2, 3).to(dev)  # B C T H W
            logits = model(x)
            pv = torch.softmax(logits[0], dim=-1).cpu().numpy()
            pn = torch.softmax(logits[1], dim=-1).cpu().numpy()
            for j, s in enumerate(idx):
                tv = np.argsort(-pv[j])[:TOPK]
                tn = np.argsort(-pn[j])[:TOPK]
                out.append(WindowPred(
                    start=round(s / FPS, 2), end=round((s + CLIP) / FPS, 2),
                    verbs=[(int(k), float(pv[j][k])) for k in tv],
                    nouns=[(int(k), float(pn[j][k])) for k in tn],
                ))
    return out
