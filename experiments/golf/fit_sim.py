"""시스템 식별: 실제 주행 데이터(학습 파일 14개)로 시뮬레이터 계수 4개를 함께 맞춘다 (11/19 사용자 결정).

  python experiments/golf/fit_sim.py   → experiments/golf/results/sim/fit.json, calibration.json 갱신
                                          (이전 calibration.json 은 calibration_before_fit.json 으로 보관)

왜: 500 m 코너에서 실제 조작을 재생하면 시뮬레이터가 실제보다 25% 덜 돌았다(옆 가속도 6.67 vs 8.88 m/s²).
    지금까지는 항목 하나씩(언더스티어·최대 옆 가속도·롤) 맞췄는데, 기록된 조작을 넣었을 때의 움직임 전체를 함께 맞추는 것이
    가장 근거가 충실하다. 학습 모델(R1–R6)은 쓰지 않는다 — 시뮬레이터는 모든 모델에 똑같이 쓰이므로.
맞추는 계수 (물리 의미가 있는 4개만 — 계수가 적어야 데이터의 잡음까지 외우지 않음):
  mu_scale 최대 마찰(앞뒤 공통), ky_f·ky_r 앞·뒤 타이어 옆 강성 배율, rby_f 앞 타이어가 앞으로 미는 동안 옆 힘이 줄어드는 정도
고정: 롤 강성(현가 센서 근거), 질량·무게중심·지연.
목적: 학습 파일 전체(직선·제동·코너·가속 코너 모두)의 1초 재생 오차 = 상태별 데이터 표준편차로 나눈 vx·vy·r RMSE 의 합.
      한 구간만 맞추면 다른 구간이 나빠져 합이 커지므로, 코너 하나에 대한 과적합을 막는다.
확인: 맞추는 데 쓰지 않은 검증·시험 파일에서 보정 전후를 구간별로 비교, 500 m 코너 재측정, 최대 옆 가속도.
"""
import json
import math
import multiprocessing as mp
import os
import shutil
import sys
import time

import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import validate_sim as V  # noqa: E402
from deep_dynamics.golf import data as D  # noqa: E402
from deep_dynamics.golf import sim  # noqa: E402

OUT = os.path.join(V.OUT, "fit.json")
NAMES = ("mu_scale", "ky_f", "ky_r", "rby_f")
BOUNDS = dict(mu_scale=(0.9, 1.6), ky_f=(0.5, 2.0), ky_r=(0.5, 2.0), rby_f=(0.2, 3.0))
STRIDE = 50
MAXFEV = 80

_W = None


def _init(W):
    global _W
    _W = W


def _replay_part(args):
    params, k_roll, lo, hi = args
    w = sim.World(cap=False, delays=None, k_roll=k_roll, **params)
    out = np.full((hi - lo, len(_W[0]["F"]), 5), np.nan)
    for i in range(lo, hi):
        win = _W[i]
        vx, vy, r = win["s0"]
        w.reset(win["X0"], win["Y0"], win["psi0"], vx, vy, r, delta=win["d0"])
        for k in range(len(win["F"])):
            w.set_grade(win["g"][k])
            s = w.step(float(win["F"][k]), float(win["d"][k]))
            out[i - lo, k] = (*s, w.x[0], w.x[1])
            if not np.isfinite(s[0]):
                break
    return out


def replay(pool, W, params, k_roll, nproc=2):
    cuts = np.linspace(0, len(W), nproc + 1).astype(int)
    parts = pool.map(_replay_part, [(params, k_roll, cuts[i], cuts[i + 1]) for i in range(nproc)])
    return np.concatenate(parts)


def categories(W):
    """시작점별 구간 종류: 제동 / 가속하며 도는 코너 / 코너 / 직선 / 그 밖."""
    S = np.stack([w["s"] for w in W])
    F = np.stack([w["F"] for w in W]).mean(1)
    ay = np.abs(S[:, :, 0] * S[:, :, 2]).mean(1)
    cat = np.full(len(W), "other", dtype=object)
    cat[ay < 2] = "straight"
    cat[ay >= 5] = "corner"
    cat[(ay >= 5) & (F > 500)] = "drive_corner"
    cat[F < -1000] = "braking"
    return cat


def score(W, P, sd, mask=None):
    S = np.stack([w["s"] for w in W])
    ok = np.isfinite(P).all(axis=(1, 2))
    if mask is not None:
        ok &= mask
    if not ok.any():
        return dict(n=0)
    e = P[ok, -1, :3] - S[ok, -1]
    rm = np.sqrt((e ** 2).mean(0))
    return dict(n=int(ok.sum()), diverged=int((~np.isfinite(P).all(axis=(1, 2)) & (mask if mask is not None else True)).sum()),
                score=float((rm / sd).sum()), vx=float(rm[0]), vy=float(rm[1]), r=float(rm[2]))


def report_set(pool, W, params, k_roll, sd):
    P = replay(pool, W, params, k_roll)
    cat = categories(W)
    out = dict(all=score(W, P, sd))
    for c in ("straight", "braking", "corner", "drive_corner"):
        out[c] = score(W, P, sd, cat == c)
    return out


def corner500(pool, params, k_roll):
    """검증 랩(2023-10_8)의 500 m 코너(트랙 420~560 m)에서 실제 조작 재생 → 평균 옆 가속도 (실제 vs 시뮬레이터)."""
    from deep_dynamics.golf.track import project
    tr, _, _ = V.recorded_track()
    W = V.windows(["2023-10_8"], 25, stride=5)
    idx = project(np.c_[tr.np["x"], tr.np["y"]], np.array([[w["X0"], w["Y0"]] for w in W]))[0]
    sel = [w for w, i in zip(W, idx) if 420 <= i <= 560]
    _init(sel)
    pool2 = mp.Pool(2, initializer=_init, initargs=(sel,))
    P = replay(pool2, sel, params, k_roll)
    pool2.close()
    S = np.stack([w["s"] for w in sel])
    return dict(n=len(sel), ay_data=float(np.abs(S[:, :, 0] * S[:, :, 2]).mean()), ay_sim=float(np.nanmean(np.abs(P[:, :, 0] * P[:, :, 2]))),
                r_err_1s=float(np.nanmean(P[:, -1, 2] - S[:, -1, 2])))


def max_ay(params, k_roll, v=20.0, T=15.0):
    w = sim.World(cap=False, k_roll=k_roll, **params)
    w.reset(0, 0, 0, v)
    best, n = 0.0, int(T / sim.DT)
    for k in range(n):
        s = w.step(sim.road_load(w.state[0]) + w.p.m * 0.5 * (v - w.state[0]), 0.25 * k / n)
        if not np.isfinite(s[0]) or abs(math.atan2(s[1], s[0])) > 0.15:
            break
        best = max(best, abs(s[0] * s[2]))
    return best


if __name__ == "__main__":
    t_start = time.time()
    sp = D.get_split()
    cal_path = sim.CALIB
    cal0 = sim.load_calibration()
    before = dict(mu_scale=cal0["mu_scale"], ky_f=cal0.get("ky_f", 1.0), ky_r=cal0.get("ky_r", 1.0), rby_f=cal0.get("rby_f", 1.0))
    k_roll = cal0["k_roll"]
    sd = np.concatenate([sg["s"] for n in sp["train"] for sg in D.load_run(n)]).std(0)
    W_tr = V.windows(sp["train"], 25, stride=STRIDE)
    print(f"학습 파일 시작점 {len(W_tr)}개, 시작값 {before}", flush=True)
    pool = mp.Pool(2, initializer=_init, initargs=(W_tr,))
    lo = np.array([BOUNDS[k][0] for k in NAMES])
    hi = np.array([BOUNDS[k][1] for k in NAMES])
    to_p = lambda z: dict(zip(NAMES, np.clip(np.exp(z), lo, hi).tolist()))
    hist = []

    def obj(z):
        p = to_p(z)
        t0 = time.time()
        sc = score(W_tr, replay(pool, W_tr, p, k_roll), sd)
        val = sc["score"] + 0.05 * sc["diverged"]  # 발산한 시작점은 벌점
        hist.append(dict(p, score=sc["score"], diverged=sc["diverged"]))
        print(f"평가 {len(hist):3d} {json.dumps({k: round(v, 4) for k, v in p.items()})} 점수 {sc['score']:.4f} 발산 {sc['diverged']} "
              f"({time.time() - t0:.0f}s)", flush=True)
        return val

    z0 = np.log([before[k] for k in NAMES])
    simplex = np.array([z0] + [z0 + 0.15 * np.eye(4)[i] for i in range(4)])
    sol = minimize(obj, z0, method="Nelder-Mead", options=dict(maxfev=MAXFEV, initial_simplex=simplex, xatol=0.01, fatol=1e-3))
    best = to_p(sol.x)
    pool.close()
    print("최적", best, flush=True)

    res = dict(names=NAMES, bounds=BOUNDS, stride=STRIDE, before=before, after=best, k_roll=k_roll, history=hist,
               nelder_mead=dict(nfev=int(sol.nfev), success=bool(sol.success), message=str(sol.message)), state_sd=sd.tolist())
    for key in ("train", "val", "test"):
        W = V.windows(sp[key], 25, stride=25 if key != "train" else STRIDE)
        pool = mp.Pool(2, initializer=_init, initargs=(W,))
        res[key] = dict(files=sp[key], before=report_set(pool, W, before, k_roll, sd), after=report_set(pool, W, best, k_roll, sd))
        pool.close()
        print(key, json.dumps(res[key]["after"]["all"]), flush=True)
    res["corner500"] = dict(before=corner500(None, before, k_roll), after=corner500(None, best, k_roll))
    res["max_ay"] = dict(before=max_ay(before, k_roll), after=max_ay(best, k_roll), data_p99=cal0.get("ay_target"))
    res["sec"] = time.time() - t_start
    json.dump(res, open(OUT, "w"), indent=1)
    if not os.path.exists(cal_path.replace(".json", "_before_fit.json")):
        shutil.copy(cal_path, cal_path.replace(".json", "_before_fit.json"))
    json.dump(dict(cal0, **{k: round(v, 4) for k, v in best.items()}, fit="experiments/golf/results/sim/fit.json"), open(cal_path, "w"), indent=1)
    print("corner500", res["corner500"], "max_ay", res["max_ay"], f"{res['sec']:.0f}s")
