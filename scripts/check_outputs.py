"""Check every JSON in outputs/ against the kit rules and the manifest.

    python scripts/check_outputs.py [outputs_dir]

Reports missing videos, invalid files and warnings. Exit code 1 if anything is
invalid or missing. Stand-in for the kit's validate.py until we receive it.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
from ustad_t4 import schema  # noqa: E402


def main() -> int:
    out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "outputs"
    manifest = {r["video_id"]: float(r["duration_sec"]) for r in csv.DictReader((ROOT / "kit" / "manifest.csv").open())}
    vocab = json.loads((ROOT / "kit" / "supported_classes.json").read_text())
    team_ids, bad, warned = set(), 0, 0
    for vid, dur in manifest.items():
        f = out_dir / f"{vid}.json"
        if not f.exists():
            continue
        doc = json.loads(f.read_text())
        team_ids.add(doc.get("team_id"))
        errors, warnings = schema.check(doc, dur, vocab)
        if doc.get("video_id") != vid:
            errors.append("video_id does not match filename")
        if errors:
            bad += 1
            print(f"INVALID {vid}: " + "; ".join(errors))
        if warnings:
            warned += 1
    present = sum((out_dir / f"{v}.json").exists() for v in manifest)
    print(f"\n{present}/{len(manifest)} present, {bad} invalid, {warned} with warnings, team_ids={team_ids}")
    if len(team_ids) > 1:
        print("INVALID: team_id differs across files")
        bad += 1
    return 1 if bad or present < len(manifest) else 0


if __name__ == "__main__":
    sys.exit(main())
