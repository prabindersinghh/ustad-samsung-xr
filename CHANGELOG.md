# Changelog

What each commit did, oldest first. Commit IDs link to GitHub. Add a line here with every change a reviewer would
care about (see [CONTRIBUTING.md](CONTRIBUTING.md)).

## 20 Nov 2026 — Track 4 engine and outputs

| Commit | What changed and why |
|---|---|
| [`3a32f8f`](../../commit/3a32f8f) | Added the Track 4 engine (`ustad_t4`): LaViLa TimeSformer-B action recognition, OpenCLIP labels, coverage-greedy highlights, kit-format JSON, `run.sh`. |
| [`4cd6141`](../../commit/4cd6141) | Trained the action scoring head on EPIC-100 train (held-out mAP 0.735 → 0.737, kept); tuned selection; added the highlight reel cutter. |
| [`20e42c2`](../../commit/20e42c2) | README for Samsung XR, `get_weights.sh` with SHA-256 checksums (weights in release `weights-v1`), Track 4 results section. |
| [`cf075ef`](../../commit/cf075ef) | `finalize.sh`: pull cloud predictions, regenerate EPIC outputs from saved predictions, validate, print stats. |
| [`897f644`](../../commit/897f644) | First validated outputs: 136 EPIC-100 + 47 FPVSum; README results. |
| [`bfd8564`](../../commit/bfd8564) | Restored the original README style, changing only the hackathon, models and numbers. |
| [`7bdf3bd`](../../commit/7bdf3bd) | Output for P29_05 (137 of 138 EPIC). |
| [`34c1cd6`](../../commit/34c1cd6) | Output for P22_03: all 138 EPIC-100 videos, 185 of 187 in total. |

## 22 Nov 2026 — organisers' clarifications, packaging, submission

| Commit | What changed and why |
|---|---|
| [`288ea15`](../../commit/288ea15) | Organisers clarified that an action counts as captured when ≥ 50% of it is inside the highlights. Scorer and selector rebuilt around that rule (run-fitted candidates, threshold 0.25 calibrated by confident-run density); EPIC train coverage 0.302 → 0.369; EPIC outputs regenerated. |
| [`8b770c6`](../../commit/8b770c6) | Ignore local server logs (keeps runtime files out of git). |
| [`c560cef`](../../commit/c560cef) | `validate.py --file/--dir` (kit rules, manifest coverage, one team_id); content-based `cooking` label rule (fixed P07_13); `get_weights.sh` skips verified files and runs in the Docker build. |
| [`4ee141c`](../../commit/4ee141c) | `team_id` set to `Thapar_INCOGNITO` (CollegeName_TeamName) in all 185 outputs, `run.py`, Dockerfile; validator works without `kit/`; Track 4 SUBMISSION.md. |
| [`d441651`](../../commit/d441651) | `get_weights.sh` tolerates Windows line endings in `SHA256SUMS` (a CRLF checkout broke the Docker build); file pinned to LF. |
| [`01985e4`](../../commit/01985e4) | Added the presentation, the 2:15 demo film, the AI disclosure and a submission map. |
| [`5fd9e7d`](../../commit/5fd9e7d) | Final presentation (PPTX + PDF) named CollegeName_TeamName. |
| [`bc9ab08`](../../commit/bc9ab08) | README and SUBMISSION.md brought to the final state: Track 4 at a glance, coverage under the 50% rule, kitchen-data vs factory-product note, run commands. |
| [`0a7ff53`](../../commit/0a7ff53) | Restructured: the repository root is now the Track 4 package (kit layout); new README hero on EPIC kitchen footage with real labels. |
| [`d6e156d`](../../commit/d6e156d) | CONTRIBUTING.md for new contributors; neutral AI-disclosure wording; removed a redundant agent pointer file. |
| `7820adc` | CHANGELOG.md. |
| (latest) | Latest presentation (PPTX + PDF) from the team. |

## Verified at submission

- `python validate.py --dir outputs/` → 185/185 present (2 Scuba videos excluded by the organisers), 0 invalid.
- `docker build` from `python:3.12-slim` on a fresh clone fetches and verifies the weights; `run.sh` inside the image
  produced a valid JSON (Python 3.12.16, torch 2.5.1+cpu, torchvision 0.20.1+cpu).
