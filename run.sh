#!/usr/bin/env bash
# Track 4 run contract: ./run.sh <video.mp4> <out.json>
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PYTHON:-python}"
PYTHONPATH="$HERE/code${PYTHONPATH:+:$PYTHONPATH}" exec "$PY" -m ustad_t4.run "$1" "$2"
