"""[11/25 N4] 시드 앙상블 평가: 같은 모델의 시드 0·1·2 를 매 스텝 평균 (hybrid.Ensemble). evaluate.py 와 같은 시험·지표.

  python experiments/golf/eval_ensemble.py --tag roll25 --models experiments/golf/results/models_1125 --out experiments/golf/results/eval_1125
출력: <out>/ensemble_{tag}.json, .md  (E_<모델> = 앙상블, <모델> = 시드 하나씩의 평균, 비교용)
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from deep_dynamics.golf import data as D  # noqa: E402
from deep_dynamics.golf.configs import ALL, road_of  # noqa: E402
from deep_dynamics.golf.evaluate import DEV, STEPS, TEST_SETS, metrics, predict  # noqa: E402
from deep_dynamics.golf.hybrid import Ensemble  # noqa: E402
from deep_dynamics.golf.ladder import load_model  # noqa: E402
from deep_dynamics.golf.train import windows  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="roll25")
    ap.add_argument("--models", default="experiments/golf/results/models_1125")
    ap.add_argument("--out", default="experiments/golf/results/eval_1125")
    ap.add_argument("--cfg", default="GOLF_THUNDERHILL_G")
    a = ap.parse_args()
    cfg = ALL[a.cfg]
    sp = D.get_split()
    groups = {}
    for p in sorted(glob.glob(os.path.join(a.models, f"[RN]?_s?_{a.tag}.pt"))):
        groups.setdefault(os.path.basename(p).split("_s")[0], []).append(load_model(p, cfg, DEV))
    out, L = {}, [f"# 시드 앙상블 vs 시드 하나 ({a.tag}, 시험 {', '.join(sp['test'])})", ""]
    for key, vstop, title in TEST_SETS:
        X, Y, U = windows(D.split_segs(sp["test"], v_stop=vstop, road=road_of(cfg)), cfg.hist, horizon=max(STEPS), stride=5)
        Xt, Ut = torch.tensor(X, device=DEV), torch.tensor(U, device=DEV)
        res = {}
        for g, ms in groups.items():
            single = [metrics(predict(m, Xt, Ut), Y, cfg.dt) for m in ms]
            res[g] = {h: {q: float(np.mean([s[h][q] for s in single])) for q in ("vx", "vy", "r", "diverged")} for h in single[0]}
            res["E_" + g] = metrics(predict(Ensemble(ms).eval(), Xt, Ut), Y, cfg.dt)
        out[key] = res
        hs = list(next(iter(res.values())))
        L += [f"## {title}, 시작점 {len(X)}개", "", "| 모델 | " + " | ".join(f"{h} vx / vy / r" for h in hs) + " |", "|---|" + "---|" * len(hs)]
        for g in groups:
            for k, name in ((g, f"{g} 시드 {len(groups[g])}개 평균"), ("E_" + g, f"**{g} 앙상블**")):
                L.append(f"| {name} | " + " | ".join(" / ".join(f"{res[k][h][q]:.4f}" for q in ("vx", "vy", "r")) for h in hs) + " |")
        L.append("")
    os.makedirs(a.out, exist_ok=True)
    json.dump(out, open(os.path.join(a.out, f"ensemble_{a.tag}.json"), "w"), indent=1)
    open(os.path.join(a.out, f"ensemble_{a.tag}.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
