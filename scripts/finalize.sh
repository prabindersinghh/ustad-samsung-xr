#!/usr/bin/env bash
# Pull the cloud EPIC results, regenerate every EPIC JSON with the final code, validate all outputs.
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-../.venv/Scripts/python}"
rm -rf outputs_cloud && mkdir -p outputs_cloud
"$PY" -m modal volume get --force ustad-t4 preds/ outputs_cloud/ >/dev/null
"$PY" -m modal volume get --force ustad-t4 outputs/ outputs_cloud/ >/dev/null
P=outputs_cloud/preds; [ -d outputs_cloud/preds/preds ] && P=outputs_cloud/preds/preds
O=outputs_cloud/outputs; [ -d outputs_cloud/outputs/outputs ] && O=outputs_cloud/outputs/outputs
echo "preds: $(ls $P | wc -l), cloud jsons: $(ls $O | wc -l)"
"$PY" scripts/assemble_epic.py "$P" "$O" outputs
"$PY" scripts/check_outputs.py || true
"$PY" scripts/stats.py
