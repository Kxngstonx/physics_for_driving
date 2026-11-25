"""시드별 1초 예측 (시험 파일): R1·R6 × 시드 0·1·2. 사용: python cached_run.py REPO seed_eval → seed_eval.json"""
import glob, json, os
import numpy as np, torch
import deep_dynamics.golf.data as D
from deep_dynamics.golf.configs import GOLF_THUNDERHILL as CFG0, cfg_of_model, road_of
from deep_dynamics.golf.ladder import load_model, rollout
from deep_dynamics.golf.train import windows
HERE = os.path.dirname(os.path.abspath(__file__))
sp = D.get_split(); out = {}
for pt in sorted(glob.glob("experiments/golf/results/models/R*_s*_roll25.pt")):
    tag = os.path.basename(pt)[:-3]; cfg = cfg_of_model(pt, CFG0); m = load_model(pt, cfg).eval()
    X, Y, U = windows(D.split_segs(sp["test"], road=road_of(cfg)), cfg.hist, horizon=25, stride=5)
    with torch.no_grad():
        e = (rollout(m, torch.tensor(X), torch.tensor(U)).numpy() - Y)[:, -1]
    F = U[:, :, 0].mean(1); rm = np.sqrt((e ** 2).mean(0)); b = e[F < -1000]
    out[tag] = dict(vx=round(float(rm[0]), 3), vy=round(float(rm[1]), 3), r=round(float(rm[2]), 4), bias_vx=round(float(e[:, 0].mean()), 3),
                    brake_vx=round(float(np.sqrt((b[:, 0] ** 2).mean())), 3), brake_bias=round(float(b[:, 0].mean()), 3))
    print(tag, json.dumps(out[tag]), flush=True)
json.dump(out, open(os.path.join(HERE, "seed_eval.json"), "w"), indent=1)
