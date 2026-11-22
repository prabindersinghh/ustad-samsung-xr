import sys; from pathlib import Path
sys.argv=[sys.argv[0]]
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT/"scripts"))
import importlib.util
src=(ROOT/"scripts"/"exp_runs.py").read_text().split("eps = load_episodes")[0]
ns={"__file__":str(ROOT/"scripts"/"exp_runs.py")}; exec(src, ns)
pp, select, scoring = ns["pp"], ns["select"], ns["scoring"]
eps = ns["load_episodes"](ROOT/"data"/"train_preds"); gt = ns["load_gt"]()
def per(conf, new):
    pp.CONF_PAIR = conf; out=[]
    for e in eps:
        c = ns["cands_runs"](e["windows"], e["duration"], e["motion"], conf=conf, maxlen=6, pad=0.0, sliding=True) if new else pp.highlight_candidates(e["windows"], e["duration"], e["motion"])
        hl = select.by_coverage(c, e["duration"])
        out.append(scoring.coverage(gt.get(e["video_id"], []), [[w.start, w.end] for w in hl], e["duration"]))
    return out
base = per(0.12, False)
for conf in (0.2, 0.25, 0.3, 0.4):
    new = per(conf, True)
    w = sum(n > b + 1e-9 for n, b in zip(new, base)); l = sum(n < b - 1e-9 for n, b in zip(new, base))
    print(f"conf {conf}: mean {sum(new)/len(new):.3f} vs {sum(base)/len(base):.3f}  wins {w} losses {l} ties {len(new)-w-l}")
