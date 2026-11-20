# INCOGNITO — Ustad, Track 4: Action-Aware Video Summarization

Samsung PRISM XR Hackathon 2026. This folder is the Track 4 submission of **Ustad**, a copilot for
hands-on work on XR glasses. In Ustad the same pipeline turns an expert's first-person recording into an
action timeline and a highlight reel (see `backend/action_timeline.py` and `/highlights` in the app).

```
./run.sh <video.mp4> <out.json>
```

One video in, one kit-format JSON out (`team_id`, `video_id`, `duration_sec`, `label`, `actions`,
`highlights`). CPU only, no network access needed at inference: every model loads from `weights/`.

## Setup

Python 3.12 and ffmpeg/ffprobe on PATH.

```bash
pip install -r requirements.txt
./run.sh /path/P01_11.mp4 outputs/P01_11.json
```

Or with Docker (same environment that produced `outputs/`):

```bash
docker build -t incognito-t4 .
docker run --rm -v "$PWD/videos:/videos" incognito-t4 /videos/P01_11.mp4 /videos/P01_11.json
```

Runtime on a 16-core CPU is roughly 1.5–2× the video length for kitchen footage and about 0.3× for
non-kitchen footage.

## Pipeline

```
video ─┬─ ffprobe duration
       ├─ motion curve (frame differences, 2 fps)                      model-free importance
       ├─ CLIP ViT-B/32 (LAION-2B) zero-shot over 16 kit labels  ──►  label
       │      └─ p(cooking) ≥ 0.15 ? ──────────────┐
       │                                           ▼
       ├─ LaViLa TimeSformer-B, EPIC-100 verb & noun heads, 2 s windows every 1 s (16 frames @ 8 fps)
       │      └─ top-3 verbs × top-3 nouns per window → runs of the same pair → scored  ──►  actions
       └─ highlights
              kitchen:     4.2 s candidates carrying the verb-noun pairs seen in them,
                           coverage-greedy selection (new pairs per second of budget) ≤ 15%
              otherwise:   0.4·motion + 0.6·CLIP "exciting moment of <label>" curve, importance selection
```

| Stage | Model | Weights | Trained on |
|---|---|---|---|
| Actions | LaViLa TimeSformer-B, multi-head classifier | `weights/lavila_tsfb_ek100_multihead.pth` (facebookresearch/LaViLa model zoo, converted to a weights-only state dict) | EPIC-KITCHENS-100 **train** split |
| Label, excitement | OpenCLIP ViT-B/32 | `weights/clip/open_clip_vit_b32_laion2b.bin` | LAION-2B |
| Action scoring head | logistic head over run features | `weights/refine.json` | EPIC-100 **train** episodes (ours) |
| Highlight policy | REINFORCE linear policy | `weights/policy.json` | EPIC-100 **train** episodes (ours) |

The verb and noun heads index EPIC's canonical class ids, which are the first 97 verbs / 300 nouns of
`supported_classes.json` (the kit appends `other`). The action head is not used.

### Why these choices

- **Action coverage is 40% of the score**, so highlights are chosen to buy *distinct verb-noun pairs per
  second of budget*, not peaks of a saliency curve. The greedy selector is the classic (1 − 1/e)
  approximation for this budgeted coverage objective.
- **Action mAP rewards recall at tIoU 0.1–0.5 with a good ranking**, so each window proposes 9 pairs and
  the final ranking comes from the scoring head.
- **FPVSum videos are scored on highlight F1 and label only**, so the kitchen model is skipped there; the
  gate is CLIP's view of the content, never the video ID.
- Every highlight is ≥ 4.0 s, sorted, non-overlapping, total ≤ 15%; a video too short for any 4 s
  segment within budget still gets its single best 4.0 s segment (the kit requires one).

### Learned components (and data hygiene)

`cloud/train_run.py` runs the action model on 30 EPIC-100 **train** videos (2–10 min, at most two per
participant). Those predictions plus the public train annotations (`EPIC_100_train.csv`) are the only
training data for:

- `scripts/train_refine.py` — fine-tunes the action scoring head: a proposal run is positive if it matches a
  ground-truth action of the same pair at tIoU ≥ 0.3.
- `scripts/train_policy.py` — REINFORCE on the coverage metric, policy over state-action features of each
  candidate (new pairs, confidence, motion, budget used).

Each is kept only if it beats the hand-written rule on held-out train videos; otherwise the rule is used
(`run.load_policy`, `refine.available`). **No validation-split or FPVSum ground truth is used anywhere**,
and the 138 EPIC validation videos and 49 FPVSum videos in `manifest.csv` have zero overlap with the
training videos. No code path reads the video ID except to echo it into the output.

RESULTS_PLACEHOLDER

## Reproducing `outputs/`

- EPIC-100 (138): `cloud/epic_run.py` streams each video from the official downloader, runs `run.summarize`,
  saves the JSON and the raw window predictions, and deletes the video. `scripts/assemble_epic.py`
  regenerates the JSONs from those predictions with the final selection and scoring code, so they match
  what `./run.sh` gives for the same video with the final weights (spot-checked by re-running `./run.sh`).
- FPVSum (49): `scripts/download_fpvsum.py` (official `FPVSum_videolist.csv` mapping, yt-dlp) then
  `scripts/run_dir.py`.
- `scripts/check_outputs.py` validates every file against the manifest durations and vocabulary.

## Files

```
run.sh                 entry point
code/ustad_t4/         pipeline (run, actions, label, importance, postprocess, select, refine, schema, scoring)
code/third_party/LaViLa  model definition (MIT licence)
weights/               LaViLa, OpenCLIP, refine.json, policy.json
outputs/               one JSON per manifest video
scripts/, cloud/       data download, batch runs, training, validation
requirements.txt, Dockerfile
```

## Limitations

- The action model only knows EPIC's kitchen vocabulary; FPVSum outputs carry no actions by design.
- CPU inference; a GPU would cut runtime about 10×, the outputs would not change.
- Our metric implementation (`code/ustad_t4/scoring.py`) follows the kit's one-line definitions; the
  official scorer may differ in detail.

Licences: LaViLa (MIT), OpenCLIP (MIT), EPIC-KITCHENS-100 annotations (CC BY-NC 4.0).
