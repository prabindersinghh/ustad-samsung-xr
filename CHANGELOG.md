# Changelog

What each commit did, oldest first. Commit IDs link to GitHub. Add a line here with every change a reviewer would
care about (see [CONTRIBUTING.md](CONTRIBUTING.md)).

## 20 Nov 2026 — Track 4 engine and outputs

| Commit | What changed and why |
|---|---|
| [`ab41aa1`](../../commit/ab41aa1) | Added the Track 4 engine (`ustad_t4`): LaViLa TimeSformer-B action recognition, OpenCLIP labels, coverage-greedy highlights, kit-format JSON, `run.sh`. |
| [`909f7c9`](../../commit/909f7c9) | Trained the action scoring head on EPIC-100 train (held-out mAP 0.735 → 0.737, kept); tuned selection; added the highlight reel cutter. |
| [`9afa8fc`](../../commit/9afa8fc) | README for Samsung XR, `get_weights.sh` with SHA-256 checksums (weights in release `weights-v1`), Track 4 results section. |
| [`97e13ff`](../../commit/97e13ff) | `finalize.sh`: pull cloud predictions, regenerate EPIC outputs from saved predictions, validate, print stats. |
| [`179fb80`](../../commit/179fb80) | First validated outputs: 136 EPIC-100 + 47 FPVSum; README results. |
| [`9b816da`](../../commit/9b816da) | Output for P29_05 (137 of 138 EPIC). |
| [`0377585`](../../commit/0377585) | Output for P22_03: all 138 EPIC-100 videos, 185 of 187 in total. |

## 22 Nov 2026 — organisers' clarifications, packaging, submission

| Commit | What changed and why |
|---|---|
| [`d265e10`](../../commit/d265e10) | Organisers clarified that an action counts as captured when ≥ 50% of it is inside the highlights. Scorer and selector rebuilt around that rule (run-fitted candidates, threshold 0.25 calibrated by confident-run density); EPIC train coverage 0.302 → 0.369; EPIC outputs regenerated. |
| [`4b2080c`](../../commit/4b2080c) | `validate.py --file/--dir` (kit rules, manifest coverage, one team_id); content-based `cooking` label rule (fixed P07_13); `get_weights.sh` skips verified files and runs in the Docker build. |
| [`d789fb4`](../../commit/d789fb4) | `team_id` set to `Thapar_INCOGNITO` (CollegeName_TeamName) in all 185 outputs, `run.py`, Dockerfile; validator works without `kit/`; Track 4 SUBMISSION.md. |
| [`ec38160`](../../commit/ec38160) | `get_weights.sh` tolerates Windows line endings in `SHA256SUMS` (a CRLF checkout broke the Docker build); file pinned to LF. |
| [`42cddd7`](../../commit/42cddd7) | Restructured: the repository root is now the Track 4 package (kit layout); new README hero on EPIC kitchen footage with real labels. |
| [`8207c3c`](../../commit/8207c3c) | CONTRIBUTING.md for new contributors; neutral AI-disclosure wording; removed a redundant agent pointer file. |
| `ed924cd` | CHANGELOG.md. |
| (latest) | Latest presentation (PPTX + PDF) from the team. |
| (latest) | Track 4 only: deck and film show only EPIC-KITCHENS footage. |

## Verified at submission

- `python validate.py --dir outputs/` → 185/185 present (2 Scuba videos excluded by the organisers), 0 invalid.
- `docker build` from `python:3.12-slim` on a fresh clone fetches and verifies the weights; `run.sh` inside the image
  produced a valid JSON (Python 3.12.16, torch 2.5.1+cpu, torchvision 0.20.1+cpu).
