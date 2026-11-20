"""Summary numbers for the README and deck, from outputs/ only.

FPVSum label agreement compares our predicted label with the category prefix of
each FPVSum video name (e.g. Horse_7ZzWccM_5cM -> horse). That prefix is the
public dataset category; it is used here for reporting only, never by the
pipeline (which never reads the video ID).
"""
from __future__ import annotations

import csv
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CAT = {"bikepolo": "bikepolo", "bike": "bike", "boxing": "boxing", "horse": "horse", "jumping": "jumping",
       "longboarding": "longboarding", "motor": "motor", "parkour": "parkour", "plane": "plane",
       "rockclimbing": "rockclimbing", "scuba": "scuba", "skate": "skate", "ski": "ski", "surfing": "surfing"}


def main(out_dir: str = "outputs") -> int:
    rows = list(csv.DictReader((ROOT / "kit" / "manifest.csv").open()))
    out = ROOT / out_dir
    stats = {"epic": 0, "fpvsum": 0, "missing": [], "hl_share": [], "n_actions": [], "label_ok": 0, "label_n": 0}
    for r in rows:
        f = out / f"{r['video_id']}.json"
        if not f.exists():
            stats["missing"].append(r["video_id"])
            continue
        d = json.loads(f.read_text())
        ds = "epic" if r["dataset"].startswith("epic") else "fpvsum"
        stats[ds] += 1
        stats["hl_share"].append(sum(h["segment"][1] - h["segment"][0] for h in d["highlights"]) / d["duration_sec"])
        if ds == "epic":
            stats["n_actions"].append(len(d["actions"]))
        else:
            cat = CAT.get(r["video_id"].split("_")[0].lower())
            if cat:
                stats["label_n"] += 1
                stats["label_ok"] += d["label"]["name"] == cat
    print(json.dumps({
        "epic": stats["epic"], "fpvsum": stats["fpvsum"], "missing": stats["missing"],
        "mean_highlight_share": round(statistics.mean(stats["hl_share"]), 4),
        "max_highlight_share": round(max(stats["hl_share"]), 4),
        "median_actions_epic": statistics.median(stats["n_actions"]) if stats["n_actions"] else 0,
        "fpvsum_label_agreement": f"{stats['label_ok']}/{stats['label_n']}",
    }, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
