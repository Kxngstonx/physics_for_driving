"""CommonRoad 구동력 이득 k: 학습 파일 1초 재생에서 vx 편향(시뮬레이터 − 실제)이 0 이 되는 k 를 찾는다 (양수 힘만 ×k).
시험 파일은 맞춘 뒤 확인에만. 사용: python cached_run.py REPO sim_gain REPO SET K1,K2,... → sim_gain_<SET>.json"""
import json
import multiprocessing as mp
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(sys.argv[1])
sys.path[:0] = [REPO, os.path.join(REPO, "experiments", "golf"), HERE]
SET, KS = sys.argv[2], [float(k) for k in sys.argv[3].split(",")]
STRIDE = 100 if SET == "train" else 25
_W = None


def _init(W):
    global _W
    _W = W


def _run(k):
    from deep_dynamics.golf import sim
    w = sim.calibrated_world(delays=None, cap=False)
    e, F = [], []
    for win in _W:
        vx, vy, r = win["s0"]
        w.reset(win["X0"], win["Y0"], win["psi0"], vx, vy, r, delta=win["d0"])
        s = (np.nan,) * 3
        for j in range(len(win["F"])):
            w.set_grade(win["g"][j])
            f = float(win["F"][j])
            s = w.step(f * k if f > 0 else f, float(win["d"][j]))
        e.append(np.array(s) - win["s"][-1])
        F.append(np.mean(win["F"]))
    e, F = np.array(e), np.array(F)
    ok = np.isfinite(e).all(1)
    sd = np.array([4.5, 0.6, 0.15])
    out = dict(k=k)
    for tag, m in (("all", ok), ("brake", ok & (F < -1000)), ("drive", ok & (F > 1000))):
        rm = np.sqrt((e[m] ** 2).mean(0))
        out[tag] = dict(n=int(m.sum()), bias_vx=round(float(e[m, 0].mean()), 4), vx=round(float(rm[0]), 4), vy=round(float(rm[1]), 4), r=round(float(rm[2]), 4))
    return out


if __name__ == "__main__":
    import validate_sim as V
    from deep_dynamics.golf import data as D
    W = V.windows(D.get_split()[SET], 25, stride=STRIDE)
    print(SET, "창", len(W), flush=True)
    ctx = mp.get_context("spawn")
    with ctx.Pool(min(len(KS), 2), initializer=_init, initargs=(W,)) as pool:
        res = pool.map(_run, KS)
    for r in res:
        print(json.dumps(r), flush=True)
    json.dump(res, open(os.path.join(HERE, f"sim_gain_{SET}.json"), "w"), indent=1)
