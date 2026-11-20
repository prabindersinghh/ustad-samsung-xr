"""Run the Track 4 pipeline over every video in a folder, skipping finished ones.

    python scripts/run_dir.py <videos_dir> <outputs_dir>

Calls the same ustad_t4.run.summarize that run.sh calls, one video at a time.
"""
from __future__ import annotations

import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
from ustad_t4 import run, schema  # noqa: E402


def main() -> int:
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    videos = sorted(p for p in src.iterdir() if p.suffix.lower() in (".mp4", ".mov", ".mkv"))
    failed = []
    for i, v in enumerate(videos, 1):
        out = dst / f"{v.stem}.json"
        if out.exists():
            continue
        t = time.time()
        try:
            doc = run.summarize(str(v))
            errors, _ = schema.check(doc, vocab=run.VOCAB)
            if errors:
                raise RuntimeError("; ".join(errors))
            schema.write(out, doc)
            print(f"[{i}/{len(videos)}] {v.stem} ok {time.time() - t:.0f}s label={doc['label']['name']}", flush=True)
        except Exception:
            failed.append(v.stem)
            print(f"[{i}/{len(videos)}] {v.stem} FAILED\n{traceback.format_exc()}", flush=True)
    print(f"done; failed: {failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
