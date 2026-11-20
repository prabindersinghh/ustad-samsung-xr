#!/usr/bin/env bash
# Download the two large checkpoints into weights/ and verify them.
# (refine.json and policy.json are small and already in the repository.)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE="https://github.com/prabindersinghh/ustad-samsung-xr/releases/download/weights-v1"
mkdir -p "$HERE/weights/clip"
curl -fL -o "$HERE/weights/lavila_tsfb_ek100_multihead.pth" "$BASE/lavila_tsfb_ek100_multihead.pth"
curl -fL -o "$HERE/weights/clip/open_clip_vit_b32_laion2b.bin" "$BASE/open_clip_vit_b32_laion2b.bin"
(cd "$HERE/weights" && sha256sum -c SHA256SUMS)
