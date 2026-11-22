# Contributing

Start here if you are new to this repository. It is the **Track 4 package** for the Samsung PRISM XR
Hackathon 2026 (action-aware summarization of first-person video): `run.sh`, `code/ustad_t4/`, `weights/`,
`outputs/`, `validate.py`, `Dockerfile`.

History of what each commit did: [CHANGELOG.md](CHANGELOG.md). Results and design: [README.md](README.md).
Submission summary: [SUBMISSION.md](SUBMISSION.md).

## Get running in 10 minutes (Track 4)

```bash
# Python 3.12 and ffmpeg/ffprobe on PATH
./get_weights.sh                     # fetches + SHA-256-verifies the two large checkpoints (skips if present)
pip install -r requirements.txt
./run.sh some_video.mp4 out/some_video.json
python validate.py --file out/some_video.json -v
python validate.py --dir outputs/    # must stay: 185/185 present, 0 invalid
```

Or `docker build -t thapar-incognito-t4 .` and `docker run --rm -v "$PWD/v:/v" thapar-incognito-t4 /v/in.mp4 /v/out.json`.

## Where things live (Track 4)

| Path | Purpose |
|---|---|
| `code/ustad_t4/run.py` | entry point: duration → motion → actions → highlights → label → JSON |
| `code/ustad_t4/actions.py` | LaViLa TimeSformer-B inference (2 s windows, 1 s stride) |
| `code/ustad_t4/postprocess.py` | action runs, scoring, highlight candidates (≥ 50% capture rule) |
| `code/ustad_t4/select.py` | coverage-greedy / importance highlight selection, 15% budget, ≥ 4 s segments |
| `code/ustad_t4/label.py` | OpenCLIP zero-shot label, kitchen gate, excitement curve |
| `code/ustad_t4/schema.py` | the kit's JSON rules (used by `run.py` and `validate.py`) |
| `code/ustad_t4/scoring.py` | our re-implementation of the four metrics (for tuning on EPIC train only) |
| `scripts/` | tuning, training (scoring head, RL policy), assembling outputs, stats, reels |
| `cloud/` | Modal batch jobs that produced `outputs/` |
| `kit/` | the organisers' student kit, unchanged |

## Rules that keep the submission valid

1. **No video IDs or ground truth in code.** Everything must be decided from pixels and model outputs.
2. **Never tune or train on the 187 scored videos**, and never use FPVSum annotations (treated as hidden labels).
   Tune on EPIC-100 *train* videos with `scripts/eval_train.py` / `scripts/tune_selection.py`.
3. **Outputs are generated, never hand-edited.** Change code, then regenerate (`scripts/assemble_epic.py` for EPIC
   from saved predictions, `./run.sh` otherwise) and run `python validate.py --dir outputs/`.
4. **Keep `team_id` consistent** (`Thapar_INCOGNITO`); use `scripts/set_team_id.py` if it ever changes.
5. **No secrets in git.** Weights over 100 MB go to a GitHub release, listed in `weights/SHA256SUMS`.
6. Learned parts stay only if they beat the simple rule on held-out videos; record the numbers in the README.

## Commits and docs

- One logical change per commit. Subject in the imperative, ≤ 72 characters, prefixed with the area when useful
  (`Track 4:`, `README:`, `docs:`). The body says *why* and quotes any number that changed.
- Add a line to [CHANGELOG.md](CHANGELOG.md) for every change a reviewer would care about.
- If you change a result, update the README results table in the same commit.
