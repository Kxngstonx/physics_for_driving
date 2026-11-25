"""R1·R6 개루프 평가: 시험 파일 3개(팀 분할), 실제 조작으로 25스텝(1초) 굴림. 시작점 5스텝 간격 (팀 1단계 평가와 같은 간격)."""
import json, os, numpy as np, torch
import deep_dynamics.golf.data as D
from deep_dynamics.golf.configs import ALL, road_of, cfg_of_model, GOLF_THUNDERHILL as CFG0
from deep_dynamics.golf.ladder import load_model, rollout
from deep_dynamics.golf.train import windows
M = "experiments/golf/results/models"
sp = D.get_split()
out = {}
for R in ("R1", "R6"):
    pt = f"{M}/{R}_s0_roll25.pt"; cfg = cfg_of_model(pt, CFG0); m = load_model(pt, cfg)
    X, Y, U = windows(D.split_segs(sp["test"], road=road_of(cfg)), cfg.hist, horizon=25, stride=5)
    with torch.no_grad():
        P = rollout(m, torch.tensor(X), torch.tensor(U)).numpy()
    e = P - Y
    out[R] = {f"{t}s": dict(zip(("vx", "vy", "r"), np.sqrt((e[:, k] ** 2).mean(0)).round(4).tolist())) for t, k in ((0.04, 0), (0.2, 4), (1.0, 24))}
    out[R]["n"] = len(X)
    S = Y; ay = np.abs(S[:, :, 0] * S[:, :, 2]).max(1) / 9.81
    out[R]["1s_by_ay"] = {f"{lo}-{hi}g": dict(n=int(((ay >= lo) & (ay < hi)).sum()), **dict(zip(("vx", "vy", "r"), np.sqrt((e[(ay >= lo) & (ay < hi), 24] ** 2).mean(0)).round(4).tolist())))
                          for lo, hi in ((0, 0.4), (0.4, 0.8), (0.8, 0.95), (0.95, 2))}
    js = json.load(open(pt.replace(".pt", ".json"))); out[R]["train"] = dict(best_epoch=js["best_epoch"], stop_epoch=js["stop_epoch"], best_val=js["best_val"], sec=js["sec"])
    if "global_coeffs" in js: out[R]["coeffs"] = {k: round(v, 3) for k, v in js["global_coeffs"].items()}
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)) if "__file__" in dir() else ".", "openloop.json"), "w"), indent=1)
print(json.dumps(out, ensure_ascii=False, indent=1))
