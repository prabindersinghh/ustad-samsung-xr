"""Rewrite team_id in every output JSON (and nothing else).

    python scripts/set_team_id.py Thapar_INCOGNITO [outputs_dir]

New runs take it from the TEAM_ID environment variable (run.py, Dockerfile).
"""
import json
import sys
from pathlib import Path

team = sys.argv[1]
out = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(__file__).resolve().parents[1] / "outputs"
n = 0
for f in sorted(out.glob("*.json")):
    d = json.loads(f.read_text())
    d["team_id"] = team
    f.write_text(json.dumps(d, indent=2))
    n += 1
print(f"team_id = {team!r} in {n} files")
