# AI disclosure — Thapar_INCOGNITO, Track 4

We used AI tools while building this submission. This page lists where, so the work can be judged fairly.

## AI models inside the submitted system

These are part of the pipeline and run locally from `weights/`; no hosted model or API is called.

| Model | Source | Use |
|---|---|---|
| LaViLa TimeSformer-B, EPIC-100 multi-head (verb/noun) | facebookresearch/LaViLa model zoo, pre-trained on EPIC-KITCHENS-100 train | per-second action recognition |
| OpenCLIP ViT-B/32 (LAION-2B) | open_clip | video label, kitchen gate, excitement curve |
| Logistic action-scoring head | trained by us on 27 EPIC-100 **train** videos | ranks action runs |

The REINFORCE highlight policy we trained lost to the greedy rule on held-out videos and is not used.

## AI assistance during development

- **Coding assistant:** an AI coding assistant helped write and refactor code, scripts, tests, the validator, Docker and
  cloud batch setup, and helped debug. Every design choice, run and result was reviewed by the team, and all
  numbers in the README, deck and film come from our own runs (`outputs/`, `scripts/stats.py`,
  `scripts/eval_train.py`).
- **Writing:** the README, docs, deck text and film script were drafted with the AI assistant and edited by the team.
- **Film:** narration is synthetic speech (Microsoft Edge neural TTS, voice "Andrew"); the music was generated
  procedurally by our own script; visuals are HTML/GSAP compositions rendered with HyperFrames. Footage is real:
  EPIC-KITCHENS-100 validation video P04_28 and our own recordings from a partner plant.
- **3D:** the glasses images and turntable are renders of a 3D model (`glasses.glb`) made with three.js, with the
  manufacturer's logo decals hidden.

## What AI did not do

No AI tool produced or edited the 185 output JSONs by hand; they come only from `run.sh` / the same pipeline code.
No ground truth for the 187 scored videos and no FPVSum annotations were used anywhere.
