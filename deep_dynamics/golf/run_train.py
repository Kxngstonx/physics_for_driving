"""R1–R6 한 단 학습 (docs/experiment_plan_golf.md 6-3절).

  python -m deep_dynamics.golf.run_train R4 --seed 0 --roll 25              # main (1초 손실)
  python -m deep_dynamics.golf.run_train R4 --seed 0 --roll 1               # ablation: 1스텝 손실
  python -m deep_dynamics.golf.run_train R4 --seed 0 --roll 25 --vmin 6     # ablation: 출발 구간(6 m/s 미만) 제외

학습 파일 분할은 deep_dynamics/data/golf/split.json (파일 단위 14/3/3, 시드 0).
모든 모드가 같은 학습 규칙 (train.fit: 조기 종료 + 검증 손실 기반 학습률 감소 + 검증 최저 모델 저장).
[변경 11/18] 1초 손실도 학습 창을 전부 씀 (이전: 2개 중 1개).
출력: <out>/{rung}_s{seed}_roll{roll}[_v6].pt, .json (에폭별 학습·검증 손실·학습률, 멈춘 에폭, 시간)
"""
import argparse
import json
import os

import torch

from . import data as D
from .configs import ALL, road_of
from .ladder import save_model
from .train import MAX_EPOCHS, fit, windows

ap = argparse.ArgumentParser()
ap.add_argument("rung")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--roll", type=int, default=25)
ap.add_argument("--max_epochs", type=int, default=MAX_EPOCHS)
ap.add_argument("--vmin", type=float, default=D.V_STOP, help=f"이 속도 미만 제외 (학습·검증). {D.V_LAUNCH:g} = 출발 구간 제외")
ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--cfg", default="GOLF_THUNDERHILL")
ap.add_argument("--out", default="experiments/golf/results/models")
a = ap.parse_args()

cfg = ALL[a.cfg]
dev = "cuda" if torch.cuda.is_available() else "cpu"
sp = D.get_split()
tr, va = (windows(D.split_segs(sp[k], v_stop=a.vmin, road=road_of(cfg)), cfg.hist, horizon=a.roll) for k in ("train", "val"))
tag = f"{a.rung}_s{a.seed}_roll{a.roll}" + ("" if a.vmin == D.V_STOP else f"_v{a.vmin:g}")
print(tag, "학습 창", len(tr[0]), "검증 창", len(va[0]), "장치", dev, flush=True)
model, rec = fit(a.rung, cfg, tr, va, max_epochs=a.max_epochs, lr=a.lr, seed=a.seed, device=dev, log_every=10)
os.makedirs(a.out, exist_ok=True)
save_model(model, a.rung, os.path.join(a.out, tag + ".pt"))
info = dict(tag=tag, rung=a.rung, seed=a.seed, roll=a.roll, vmin=a.vmin, max_epochs=a.max_epochs, lr=a.lr, cfg=cfg.name,
            n_train=len(tr[0]), n_val=len(va[0]), **rec)
if hasattr(model, "g"):
    info["global_coeffs"] = dict(zip(cfg.pnames, model.g.values().detach().cpu().tolist()))
json.dump(info, open(os.path.join(a.out, tag + ".json"), "w"), indent=1)
print("done", tag, f"best val {rec['best_val']:.4f}@{rec['best_epoch']} stop {rec['stop_epoch']} ({rec['stop_reason']}) {rec['sec']:.0f}s")
