# Thapar_INCOGNITO — Ustad, Track 4: Action-Aware Video Summarization

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
./get_weights.sh                     # no-op if weights/ already has both checkpoints
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
              kitchen:     4.2 s sliding candidates + one candidate fitted to each confident run;
                           a pair counts only if ≥ 50% of its run is inside the candidate;
                           coverage-greedy selection (new pairs per second of budget) ≤ 15%
              otherwise:   0.4·motion + 0.6·CLIP "exciting moment of <label>" curve, importance selection
       label rule: if the kitchen gate opened and the action model found hand-object actions, label = cooking
                   (content-based, from the pixels and model outputs; never from the video ID)
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
  approximation for this budgeted coverage objective. The organisers clarified (22 Nov) that an action
  is captured when ≥ 50% of its duration lies inside the union of the highlights, labels not required,
  so a candidate only gets credit for pairs whose predicted run is at least half inside it, and every
  confident run also proposes a candidate fitted to its own extent.
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

Caveat: LaViLa was itself fine-tuned on the EPIC-100 train split, so its predictions on train videos are
more confident than on unseen video. Scores measured on train episodes therefore overstate what to expect
on the validation videos; we use them only to compare selection rules against each other.

## Results we can measure ourselves

The official scorer and the scored subset are hidden, so these are our own checks. Train-episode numbers come
from 27 EPIC-100 **train** videos with a fixed 80/20 split (seed 7) and our implementation of the kit metrics.

| Check | Result |
|---|---|
| Output files | 185 of 187 validated (138 of 138 EPIC-100, 47 of 49 FPVSum), 0 invalid; median 352 actions per EPIC video |
| Highlight budget | mean 13.9% of video length and ≤ 15% for every video of 27 s or more; every segment ≥ 4.0 s, sorted, non-overlapping. Three EPIC clips of 11–25 s get the single 4 s segment the kit requires, which exceeds 15% of their length |
| FPVSum label agreement with the dataset category in the video name (reporting only, never used by the code) | 41 of 47 (87%) |
| Action-scoring head, held-out train videos | action mAP 0.735 → 0.737 (kept) |
| RL highlight policy, held-out train videos (any-overlap rule, before the clarification) | coverage 0.465 vs greedy rule 0.479 (not used) |
| Coverage under the ≥ 50% capture rule, all 27 train videos | 0.302 → 0.369 after run-fitted candidates and threshold 0.25 (19 better, 3 worse, 5 equal) |
| Threshold choice | 0.25: LaViLa is more confident on its own train videos, and 0.25 on the val videos gives the same confident-run density (≈ 16/min) as 0.3 on train |
| Coverage / action mAP of the shipped pipeline, held-out train videos | 0.396 / 0.737 (overstated, see caveat above) |

Two FPVSum videos (`Scuba_KD74Ywuy9ng`, `Scuba_Um4kqB8bQBg`) are no longer available on YouTube. The organisers
confirmed on 22 Nov that they are excluded from scoring, so `outputs/` has 185 files and `validate.py` reports them
as excluded, not missing.

## Reproducing `outputs/`

- EPIC-100 (138): `cloud/epic_run.py` streams each video from the official downloader, runs `run.summarize`,
  saves the JSON and the raw window predictions, and deletes the video. `scripts/assemble_epic.py`
  regenerates the JSONs from those predictions with the final selection and scoring code, so they match
  what `./run.sh` gives for the same video with the final weights (spot-checked by re-running `./run.sh`).
- FPVSum (49): `scripts/download_fpvsum.py` (official `FPVSum_videolist.csv` mapping, yt-dlp) then
  `scripts/run_dir.py`.
- `python validate.py --dir outputs/` (or `--file <json> -v`) checks every rule in the kit README: the six keys,
  types and ranges, duration within 0.5 s of the manifest, labels/verbs/nouns in `supported_classes.json`, segment
  bounds, highlights sorted and non-overlapping with one ≥ 4 s, the 15% budget (warning), file name = `video_id`,
  one `team_id` across all files, and full manifest coverage. The kit's own `validate.py` was not shipped to teams,
  so this is ours. `scripts/stats.py` prints the numbers above.
- `scripts/make_reel.py <video> <json> <reel.mp4>` cuts the highlight reel from the original footage.
- Weights: the submitted `TEAM_ID/` folder contains both checkpoints in `weights/`. In a git clone they are not in
  the repository (GitHub's 100 MB file limit), so `./get_weights.sh` fetches them from the `weights-v1` release and
  verifies `weights/SHA256SUMS`; it skips files already present. The Docker build runs it, so the image always has
  the weights.

## Team ID

`team_id` comes from the `TEAM_ID` environment variable (default `Thapar_INCOGNITO`, set in the Dockerfile).
`python scripts/set_team_id.py <TEAM_ID>` rewrites it in every file of `outputs/` without touching anything else.

## Note on the output schema

The organisers' overview slide shows a simplified JSON (`start`/`end`, `importance`). We follow the student kit's
`TEAM_README.md` and its example files exactly (`segment: [start, end]`, six top-level keys), since that is the
format `validate.py` checks.

## Files

```
run.sh                 entry point
code/ustad_t4/         pipeline (run, actions, label, importance, postprocess, select, refine, schema, scoring)
code/third_party/lavila_min  LaViLa model definition, two files (MIT licence)
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
