"""Download the 49 FPVSum videos listed in the Track 4 manifest.

Manifest IDs look like `Bike_lhTFVa0h9Rk` or `BikePolo_5yf797Lh4Oc_1`: a category,
then an 11-character YouTube ID, then an optional suffix. YouTube IDs can contain
underscores, so the ID is taken as the 11 characters after the first underscore.

Each file is saved as <video_id>.mp4 (720p max) so the pipeline can find it by
manifest ID. Already-downloaded files are skipped, so the script can be re-run.
Its duration is checked against the manifest and mismatches are reported, since
the scorer only accepts duration_sec within 0.5 s of the manifest value.
"""
from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "kit" / "manifest.csv"
OUT = ROOT / "data" / "fpvsum"


OFFICIAL_LIST = ROOT / "data" / "fpvsum_repo" / "crawler" / "FPVSum_videolist.csv"


def _official_ids() -> dict:
    """video_name -> youtube_id from the dataset's own list (azuxmioy/fpvsum).

    Suffixed names such as BikePolo_5yf797Lh4Oc_1 point at a *different* upload
    than the name suggests, so the official mapping is the only reliable source.
    """
    if not OFFICIAL_LIST.exists():
        return {}
    return {r["video_name"]: r["youtube_id"] for r in csv.DictReader(OFFICIAL_LIST.open())}


_IDS = _official_ids()


def youtube_id(video_id: str) -> str:
    return _IDS.get(video_id) or video_id.split("_", 1)[1][:11]


def probe_duration(path: Path) -> float | None:
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json", str(path)],
            capture_output=True, text=True, check=True,
        ).stdout
        return float(json.loads(out)["format"]["duration"])
    except Exception:
        return None


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = [r for r in csv.DictReader(MANIFEST.open()) if r["dataset"] == "fpvsum"]
    ok, failed, mismatched = 0, [], []
    for i, row in enumerate(rows, 1):
        vid, want = row["video_id"], float(row["duration_sec"])
        dest = OUT / f"{vid}.mp4"
        if not dest.exists():
            url = f"https://www.youtube.com/watch?v={youtube_id(vid)}"
            print(f"[{i}/{len(rows)}] {vid} <- {url}", flush=True)
            r = subprocess.run(
                [sys.executable, "-m", "yt_dlp", "-q", "--no-warnings", "-f", "bv*[height<=720][ext=mp4]+ba[ext=m4a]/b[height<=720]/b",
                 "--merge-output-format", "mp4", "-o", str(dest), url],
            )
            if r.returncode != 0 or not dest.exists():
                failed.append(vid)
                continue
        got = probe_duration(dest)
        ok += 1
        if got is None or abs(got - want) > 0.5:
            mismatched.append((vid, want, got))
    print(f"\ndownloaded {ok}/{len(rows)}; failed {len(failed)}: {failed}")
    for vid, want, got in mismatched:
        print(f"duration mismatch {vid}: manifest {want}s, file {got}s")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
