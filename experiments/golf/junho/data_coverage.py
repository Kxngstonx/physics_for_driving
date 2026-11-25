"""(2) 헤어핀 가설 확인 + 실제 데이터에서 속도 예측 편향.
  - 학습 데이터에서 주행 조건(속도 × 곡률)별 비중
  - 시험 파일(실제 주행) 1초 예측: 조건별 RMSE 와 vx 부호 편향 (예측 − 실제), R1 vs R6
사용: python cached_run.py REPO data_coverage   (PYTHONPATH 에 이 폴더)  → coverage.json
"""
import json
import os

import numpy as np
import torch

import deep_dynamics.golf.data as D
from deep_dynamics.golf.configs import GOLF_THUNDERHILL as CFG0, cfg_of_model, road_of
from deep_dynamics.golf.ladder import load_model, rollout
from deep_dynamics.golf.train import windows

HERE = os.path.dirname(os.path.abspath(__file__))
M = "experiments/golf/results/models"
sp = D.get_split()


def cond(vx, r, Fx):
    """vx, r, Fx: (..., T) 창 안 값 → 조건 이름. 헤어핀 = 최저 속도 < 15 m/s 이고 곡률 |r|/vx > 0.04 (반지름 < 25 m)."""
    k = np.abs(r) / np.maximum(vx, 1.0)
    ay = np.abs(vx * r)
    vmin, kmax, aymax = vx.min(-1), k.max(-1), ay.max(-1)
    c = np.full(vmin.shape, "직선·약한 코너", dtype=object)
    c[aymax >= 5] = "빠른 코너(ay≥5)"
    c[(vmin < 20) & (kmax > 0.02)] = "느린 코너(반지름 25~50 m)"
    c[(vmin < 15) & (kmax > 0.04)] = "헤어핀(반지름 <25 m, <15 m/s)"
    return c


out = dict(train_share={}, test={})
pt = f"{M}/R1_s0_roll25.pt"
cfg = cfg_of_model(pt, CFG0)
tr_segs = D.split_segs(sp["train"], road=road_of(cfg))
S = np.concatenate([s for s, _ in tr_segs])
F = np.concatenate([u[:, 0] for _, u in tr_segs])
c = cond(S[:, 0:1], S[:, 2:3], F[:, None])  # 샘플 하나씩
for name in np.unique(c):
    out["train_share"][name] = dict(n=int((c == name).sum()), share=round(float((c == name).mean()), 4))
print("학습 데이터 비중", json.dumps(out["train_share"], ensure_ascii=False), flush=True)

for R in ("R1", "R6"):
    pt = f"{M}/{R}_s0_roll25.pt"
    cfg = cfg_of_model(pt, CFG0)
    m = load_model(pt, cfg).eval()
    X, Y, U = windows(D.split_segs(sp["test"], road=road_of(cfg)), cfg.hist, horizon=25, stride=5)
    with torch.no_grad():
        P = rollout(m, torch.tensor(X), torch.tensor(U)).numpy()
    e = P[:, -1] - Y[:, -1]
    cw = cond(Y[:, :, 0], Y[:, :, 2], U[:, :, 0])
    Fm = U[:, :, 0].mean(1)
    res = {}
    for name, mask in [("전체", np.ones(len(e), bool)), ("제동 중", Fm < -1000), ("구동 중", Fm > 1000)] + [(n, cw == n) for n in np.unique(cw)]:
        ee = e[mask]
        rm = np.sqrt((ee ** 2).mean(0))
        res[name] = dict(n=int(mask.sum()), vx=round(float(rm[0]), 3), vy=round(float(rm[1]), 3), r=round(float(rm[2]), 4), bias_vx=round(float(ee[:, 0].mean()), 3))
    out["test"][R] = res
    print(R, json.dumps(res, ensure_ascii=False), flush=True)
json.dump(out, open(os.path.join(HERE, "coverage.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
