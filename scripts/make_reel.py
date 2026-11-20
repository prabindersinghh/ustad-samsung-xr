"""Compile the highlight reel from the original footage, as the track brief asks.

    python scripts/make_reel.py <video.mp4> <out.json> <reel.mp4>

Cuts each highlight segment from the source (re-encoded so cuts are
frame-accurate) and concatenates them in time order. Optional: the scored
output is the JSON; this is the human-viewable summary.
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path


def main(video: str, doc_path: str, reel: str) -> int:
    segs = [h["segment"] for h in json.loads(Path(doc_path).read_text())["highlights"]]
    if not segs:
        print("no highlights", file=sys.stderr)
        return 1
    with tempfile.TemporaryDirectory() as tmp:
        parts = []
        for i, (s, e) in enumerate(segs):
            part = Path(tmp) / f"{i:03d}.mp4"
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{s}", "-i", video, "-t", f"{e - s}",
                            "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-c:a", "aac", str(part)],
                           check=True)
            parts.append(part)
        lst = Path(tmp) / "list.txt"
        lst.write_text("".join(f"file '{p.as_posix()}'\n" for p in parts))
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(lst),
                        "-c", "copy", reel], check=True)
    total = sum(e - s for s, e in segs)
    print(f"{reel}: {len(segs)} segments, {total:.1f} s")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print(__doc__, file=sys.stderr)
        sys.exit(2)
    sys.exit(main(*sys.argv[1:]))
