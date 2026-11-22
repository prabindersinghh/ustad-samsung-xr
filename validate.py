"""Track 4 output validator, standing in for the kit's validate.py (not shipped to teams).

    python validate.py --file outputs/P02_109.json -v
    python validate.py --dir outputs/

Checks every rule in the kit README, sections 3-4: the six keys, types and ranges,
duration within 0.5 s of manifest.csv, label/verb/noun in supported_classes.json,
action segments inside the video, highlights sorted, non-overlapping, at least one
>= 4.0 s (warns on shorter ones and on > 15% budget). With --dir it also checks that
every manifest video is present, that each file name matches its video_id, and that
team_id is identical in every file. The two Scuba FPVSum videos the organisers
excluded (22 Nov) are reported but not counted as missing.
Exit code 0 only when everything passes.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "code"))
from ustad_t4 import schema  # noqa: E402

EXCLUDED = {"Scuba_KD74Ywuy9ng", "Scuba_Um4kqB8bQBg"}  # organisers: excluded from scoring
MANIFEST = {r["video_id"]: float(r["duration_sec"]) for r in csv.DictReader((ROOT / "kit" / "manifest.csv").open())}
VOCAB = json.loads((ROOT / "kit" / "supported_classes.json").read_text())


def check_file(path: Path, verbose: bool) -> tuple[bool, str | None]:
    try:
        doc = json.loads(path.read_text())
    except Exception as e:  # noqa: BLE001
        print(f"INVALID {path.name}: not valid JSON ({e})")
        return False, None
    vid = path.stem
    errors, warnings = schema.check(doc, MANIFEST.get(vid), VOCAB)
    if doc.get("video_id") != vid:
        errors.append(f"video_id {doc.get('video_id')!r} does not match file name")
    if vid not in MANIFEST:
        errors.append("video_id not in manifest.csv")
    for e in errors:
        print(f"INVALID {path.name}: {e}")
    if verbose:
        for w in warnings:
            print(f"warning {path.name}: {w}")
        if not errors:
            print(f"OK {path.name}")
    return not errors, doc.get("team_id")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--file", type=Path)
    g.add_argument("--dir", type=Path)
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args()
    if a.file:
        ok, _ = check_file(a.file, True)
        return 0 if ok else 1
    files = sorted(a.dir.glob("*.json"))
    results = [check_file(f, a.verbose) for f in files]
    bad = sum(not ok for ok, _ in results)
    teams = {t for _, t in results if t is not None}
    have = {f.stem for f in files}
    missing = [v for v in MANIFEST if v not in have and v not in EXCLUDED]
    excluded = [v for v in MANIFEST if v not in have and v in EXCLUDED]
    for v in missing:
        print(f"MISSING {v}.json")
    if len(teams) > 1:
        print(f"INVALID: team_id differs across files: {sorted(teams)}")
        bad += 1
    expected = len(MANIFEST) - len(excluded)
    print(f"\n{len(files)} files, {expected - len(missing)}/{expected} manifest videos present "
          f"({len(excluded)} excluded by the organisers: {', '.join(excluded) or 'none'}), "
          f"{bad} invalid, team_id={sorted(teams)}")
    return 0 if not bad and not missing else 1


if __name__ == "__main__":
    sys.exit(main())
