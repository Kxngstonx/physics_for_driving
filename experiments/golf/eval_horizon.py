"""길이별 재학습 ablation 의 예측 정확도: 1·2·4초 손실 모델(시드 0)을 같은 시험 파일·같은 시작점에서
0.2·1·2·4초 뒤 vx·vy·r RMSE 로 비교 (학습 때 검증 손실은 손실 길이가 달라 서로 비교할 수 없으므로).

  python experiments/golf/eval_horizon.py   → experiments/golf/results/eval_horizon.json
"""
import json
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from deep_dynamics.golf import data as D  # noqa: E402
from deep_dynamics.golf.configs import GOLF_THUNDERHILL as cfg  # noqa: E402
from deep_dynamics.golf.evaluate import predict, DEV  # noqa: E402
from deep_dynamics.golf.ladder import load_model  # noqa: E402
from deep_dynamics.golf.train import windows  # noqa: E402

RES = os.path.join(ROOT, "experiments", "golf", "results")
M = os.path.join(RES, "models")
STEPS = [5, 25, 50, 100]
sp = D.get_split()
X, Y, U = windows(D.split_segs(sp["test"], v_stop=D.V_STOP), cfg.hist, horizon=max(STEPS), stride=5)
Xt, Ut = torch.tensor(X, device=DEV), torch.tensor(U, device=DEV)
out = dict(n=len(X), test=sp["test"], res={})
for R in ("R1", "R2", "R3", "R4", "R5", "R6"):
    for roll in (25, 50, 100):
        tag = f"{R}_s0_roll{roll}"
        m = load_model(os.path.join(M, tag + ".pt"), cfg, DEV)
        P = predict(m, Xt, Ut)
        row = {}
        for st in STEPS:
            e = P[:, st - 1] - Y[:, st - 1]
            ok = np.abs(e).max(1) <= 100
            rm = np.sqrt(np.mean(e[ok] ** 2, 0))
            row[f"{st * cfg.dt:g}s"] = dict(vx=float(rm[0]), vy=float(rm[1]), r=float(rm[2]), div=float(1 - ok.mean()))
        out["res"][tag] = row
        print(tag, json.dumps({k: [round(v["vx"], 3), round(v["vy"], 3), round(v["r"], 4), v["div"]] for k, v in row.items()}), flush=True)
        del m
        torch.cuda.empty_cache()
json.dump(out, open(os.path.join(RES, "eval_horizon.json"), "w"), indent=1)
