import sys; from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT/"code")); sys.path.insert(0, str(ROOT/"scripts"))
from eval_train import load_episodes
from ustad_t4 import postprocess as pp
def dens(eps, conf):
    runs = mins = 0
    for e in eps:
        runs += sum(r["best"] >= conf for r in pp.action_runs(e["windows"])); mins += e["duration"] / 60
    return runs / mins
tr = load_episodes(ROOT/"data"/"train_preds"); va = load_episodes(ROOT/"outputs_cloud"/"preds")
for c in (0.12, 0.2, 0.25, 0.3, 0.4):
    print(f"conf {c}: confident runs/min train {dens(tr,c):.1f}  val {dens(va,c):.1f}")
