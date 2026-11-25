"""golf_realsim 계수 맞추기 — 저장소 experiments/golf/fit_sim.py 와 같은 방식.
  학습 파일 14개(팀 분할)의 실제 조작을 1초 재생 → 1초 뒤 vx·vy·r RMSE ÷ 상태별 표준편차 의 합을 최소화 (Nelder–Mead)
  학습 모델(R1–R6)은 쓰지 않음. 검증·시험 파일은 맞춘 뒤 확인에만 씀.
사용: python fit_realsim.py REPO OUT_JSON
"""
import json
import math
import multiprocessing as mp
import os
import sys
import time

import numpy as np
from scipy.optimize import minimize

HERE = os.path.dirname(os.path.abspath(__file__))
NAMES = ("mu_f", "mu_r", "ca_f", "ca_r", "iz", "h_cg", "ds")
BOUNDS = dict(mu_f=(0.8, 1.8), mu_r=(0.8, 1.8), ca_f=(30e3, 250e3), ca_r=(30e3, 250e3), iz=(1000.0, 4000.0), h_cg=(0.30, 0.65), ds=(-1.0, 1.0))
STRIDE, MAXFEV, NPROC = 50, 240, 4
_W = None


def _setup_path(repo):
    for p in (repo, os.path.join(repo, "experiments", "golf"), HERE):
        if p not in sys.path:
            sys.path.insert(0, p)


def _init(repo, W):
    global _W
    _setup_path(repo)
    _W = W


def _replay_part(args):
    import golf_realsim as RS
    params, lo, hi = args
    w = RS.RealWorld(params=params, cap=False)
    out = np.full((hi - lo, len(_W[0]["F"]), 3), np.nan)
    for i in range(lo, hi):
        win = _W[i]
        vx, vy, r = win["s0"]
        w.reset(win["X0"], win["Y0"], win["psi0"], vx, vy, r, delta=win["d0"])
        for k in range(len(win["F"])):
            w.set_grade(win["g"][k])
            s = w.step(float(win["F"][k]), float(win["d"][k]))
            out[i - lo, k] = s
            if not np.isfinite(s[0]):
                break
    return out


def replay(pool, n, params):
    cuts = np.linspace(0, n, NPROC + 1).astype(int)
    return np.concatenate(pool.map(_replay_part, [(params, cuts[i], cuts[i + 1]) for i in range(NPROC)]))


def score(W, P, sd, mask=None):
    S = np.stack([w["s"] for w in W])
    ok = np.isfinite(P).all(axis=(1, 2)) & (True if mask is None else mask)
    e = P[ok, -1] - S[ok, -1]
    rm = np.sqrt((e ** 2).mean(0))
    return dict(n=int(ok.sum()), diverged=int((~np.isfinite(P).all(axis=(1, 2))).sum()), score=float((rm / sd).sum()),
                vx=float(rm[0]), vy=float(rm[1]), r=float(rm[2]))


def categories(W):
    S = np.stack([w["s"] for w in W])
    F = np.stack([w["F"] for w in W]).mean(1)
    ay = np.abs(S[:, :, 0] * S[:, :, 2]).mean(1)
    cat = np.full(len(W), "other", dtype=object)
    cat[ay < 2] = "straight"
    cat[ay >= 5] = "corner"
    cat[(ay >= 5) & (F > 500)] = "drive_corner"
    cat[F < -1000] = "braking"
    return cat


def to_params(z):
    return {k: float(BOUNDS[k][0] + (BOUNDS[k][1] - BOUNDS[k][0]) / (1 + math.exp(-v))) for k, v in zip(NAMES, z)}


def to_z(p):
    return [math.log((p[k] - BOUNDS[k][0]) / (BOUNDS[k][1] - p[k])) for k in NAMES]


def main():
    repo, out_path = os.path.abspath(sys.argv[1]), sys.argv[2]
    _setup_path(repo)
    import validate_sim as V
    import golf_realsim as RS
    from deep_dynamics.golf import data as D
    sp = D.get_split()
    W_tr = V.windows(sp["train"], 25, stride=STRIDE)
    S = np.stack([w["s"] for w in W_tr])
    sd = S.reshape(-1, 3).std(0)
    print(f"학습 창 {len(W_tr)}개, 상태 표준편차 {np.round(sd, 3)}", flush=True)
    hist = []
    with mp.Pool(NPROC, initializer=_init, initargs=(repo, W_tr)) as pool:
        def f(z):
            p = to_params(z)
            sc = score(W_tr, replay(pool, len(W_tr), p), sd)
            val = sc["score"] + 0.5 * sc["diverged"]
            hist.append(dict(p=p, **sc))
            if len(hist) % 10 == 1:
                print(f"평가 {len(hist):3d} {json.dumps({k: round(v, 3) for k, v in p.items()})} 점수 {sc['score']:.4f} vx {sc['vx']:.3f} vy {sc['vy']:.3f} r {sc['r']:.4f} 발산 {sc['diverged']}", flush=True)
            return val
        t0 = time.time()
        p0 = {k: RS.DEFAULT[k] for k in NAMES}
        before = f(to_z(p0))
        z0 = np.array(to_z(p0))
        simplex = np.vstack([z0] + [z0 + 0.6 * np.eye(len(z0))[i] for i in range(len(z0))])  # 시작 탐색 폭을 명시 (z=0 근처에서 기본 폭이 0 이 되는 문제)
        opt = minimize(f, z0, method="Nelder-Mead", options=dict(maxfev=MAXFEV, xatol=1e-3, fatol=1e-4, adaptive=True, initial_simplex=simplex))
        best = to_params(opt.x)
        print(f"맞춤 끝 {time.time() - t0:.0f}s, 점수 {before:.4f} → {opt.fun:.4f}", flush=True)
    res = dict(params=best, default=p0, fixed=dict(load_sens=RS.DEFAULT["load_sens"], steer_rate=RS.DEFAULT["steer_rate"], abs_frac=RS.ABS_FRAC), n_eval=len(hist), sd=sd.tolist())
    for key, names in (("train", sp["train"]), ("val", sp["val"]), ("test", sp["test"])):
        W = V.windows(names, 25, stride=25 if key != "train" else STRIDE)
        cat = categories(W)
        with mp.Pool(NPROC, initializer=_init, initargs=(repo, W)) as pool:
            res[key] = {}
            for tag, p in (("before", p0), ("after", best)):
                P = replay(pool, len(W), p)
                res[key][tag] = dict(all=score(W, P, sd), **{c: score(W, P, sd, cat == c) for c in ("straight", "braking", "corner", "drive_corner")})
        print(key, "before", json.dumps(res[key]["before"]["all"]), "\n      after ", json.dumps(res[key]["after"]["all"]), flush=True)
    json.dump(res, open(out_path, "w"), indent=1)
    print("saved", out_path)


if __name__ == "__main__":
    main()
