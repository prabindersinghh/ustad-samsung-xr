# Submission: Samsung PRISM XR Hackathon 2026, Track 4

| | |
|---|---|
| **Track** | 04 · Action-Aware Video Summarization |
| **Team** | Incognito — `Thapar_INCOGNITO` (CollegeName_TeamName); this is the `team_id` in every output |
| **College** | Thapar Institute of Engineering & Technology |
| **Members** | Prabinder Singh, Anish Grover, Prabhasees Singh |
| **Code** | https://github.com/prabindersinghh/ustad-samsung-xr (the repository root is the Track 4 package) |
| **Presentation** | [`presentation/Thapar Institute of Engineering & Technology_INCOGNITO.pptx`](presentation/Thapar%20Institute%20of%20Engineering%20%26%20Technology_INCOGNITO.pptx) · [PDF](presentation/Thapar%20Institute%20of%20Engineering%20%26%20Technology_INCOGNITO.pdf) |
| **Demo video** | [`video/Thapar_INCOGNITO_demo_final.mp4`](video/Thapar_INCOGNITO_demo_final.mp4), 2:15 |
| **AI disclosure** | [`AI_DISCLOSURE.md`](AI_DISCLOSURE.md) |

## In one line

Ustad recognises every hand–object action in a first-person video as a timed verb–noun pair, then keeps the 15% of
the footage that captures the most distinct actions. One command per video, CPU only, no network calls.

## Submission package (repository root, uploaded as `Thapar_INCOGNITO/`)

| Required | Contents |
|---|---|
| `run.sh` | `./run.sh <video.mp4> <out.json>` |
| `code/` | `ustad_t4/` pipeline + vendored LaViLa model code (MIT) |
| `weights/` | `lavila_tsfb_ek100_multihead.pth`, `clip/open_clip_vit_b32_laion2b.bin`, `refine.json`, `SHA256SUMS`. The uploaded folder contains the two large checkpoints, so an offline re-run works; in git they come from the `weights-v1` release via `get_weights.sh` (GitHub's 100 MB file limit). |
| `requirements.txt` | Python 3.12, PyTorch 2.5.1 / torchvision 0.20.1 (CPU), timm 1.0.11, einops 0.8.0, open_clip_torch 2.29.0 |
| `outputs/` | 185 `<video_id>.json` files |
| `README.md` | setup, architecture, results, reproduction |
| Also included | `Dockerfile`, `get_weights.sh`, `validate.py` |

## Verification

- `python validate.py --dir outputs/` → 185/185 present (2 Scuba videos excluded by the organisers), 0 invalid,
  one `team_id`. Checks every rule in the kit README; the kit's own validate.py was not shipped to teams.
- `docker build` from `python:3.12-slim` succeeds on a fresh clone, downloads and checksum-verifies the weights, and
  `run.sh` inside the image produced a valid JSON (Python 3.12.16, torch 2.5.1+cpu, torchvision 0.20.1+cpu).
- The pinned requirements resolve on Python 3.12; the full pipeline was re-run end to end on Python 3.12.

## Pipeline

1. **Perceive.** LaViLa TimeSformer-B reads 16-frame, 2 s windows every 1 s (8 fps) and predicts verb (97) and noun (300) classes.
2. **Score.** Top-3 verbs × top-3 nouns per window merge into timed action runs, ranked by a logistic head trained on EPIC-100 *train* videos only.
3. **Select.** Candidates (4.2 s sliding windows + one fitted to each confident run) credit an action only when ≥ 50% of it is inside, the organisers' capture rule. Greedy budgeted max-coverage fills 15%, every segment ≥ 4 s.
4. **Label.** OpenCLIP zero-shot over the 16 kit labels; content-based rule: kitchen gate open and actions found → `cooking`.

## Results

| Check | Result |
|---|---|
| Outputs | 185 of 187 (138 EPIC-KITCHENS-100 + 47 FPVSum; 2 excluded by organisers), 0 invalid |
| Mean highlight share | 13.9% |
| Action coverage, ≥ 50% rule, 27 EPIC train videos | 0.302 → 0.369 after the rule-aware selector (held-out 5: 0.317 → 0.396) |
| Action mAP, held-out train videos | 0.737 (scoring head kept; 0.735 without it) |
| FPVSum label agreement with the video-name category (reporting only) | 41 of 47 |

## Kitchen data, factory product

Track 4 is scored only on the organisers' datasets: EPIC-KITCHENS-100 (kitchens) and FPVSum (sports). All Track 4
numbers come from those videos. The factory footage in the film and deck shows where the same engine is used — the
Ustad app, kept in [`ustad-app/`](ustad-app/) — and is not part of the Track 4 evaluation. There we keep the verbs and map nouns outside a kitchen to
"object"; accuracy on factory footage has not been measured.

## Data hygiene

No video IDs or ground truth in the code. Nothing tuned on the 187 scored videos. FPVSum annotations treated as
hidden test labels and never used. LaViLa was trained on EPIC-100 train, so train-video metrics are optimistic.

## Limitations

Kitchen noun vocabulary; CPU latency ≈ 1.5–2× real time on 16 cores for kitchen footage; FPVSum highlights use
CLIP excitement + motion rather than an action model; three EPIC clips under 27 s exceed 15% because the kit
requires one 4 s segment; not yet tested on XR glasses hardware.
