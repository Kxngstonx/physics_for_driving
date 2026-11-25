"""[저장소 순서판 11/24] 저장소 calibrate_sim 처럼 마찰을 "정상 상태 최대 횡가속도 = 데이터 상위 1% 11.66 m/s²" 에 먼저 묶고,
나머지(앞뒤 마찰 차, Cα, Iz, 센서 위치, 하중 이동 지연)를 재생 오차로 맞춤. 무게중심 0.55 m 고정. 무게중심 높이 0.55 m 고정, 마찰 0.9~1.3·앞뒤 차 ±0.15, Cα 40~160 kN/rad, Iz 1500~2800, 센서 위치 ±0.5 m.
golf_realsim 계수 맞추기 — 저장소 experiments/golf/fit_sim.py 와 같은 방식.
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
NAMES = ("dmu", "ca_f", "ca_r", "iz", "ds", "tau_lt")  # mu_f = mu, mu_r = mu + dmu
BOUNDS = dict(dmu=(-0.15, 0.15), ca_f=(40e3, 160e3), ca_r=(40e3, 160e3), iz=(1500.0, 2800.0), ds=(-0.5, 0.5), tau_lt=(0.05, 0.40))
H_FIXED = 0.55
AY_TARGET = 11.656  # 저장소 calibration.json ay_target (데이터 횡가속도 상위 1%)
V_REF = 25.0        # 정상 원선회 속도 (코너 평균 속도 근처)


def max_ay(mu_f, mu_r, h=H_FIXED, k=0.05):
    """정상 원선회 최대 횡가속도 (golf_realsim._forces 와 같은 하중 이동·하중 민감도·마찰원, 앞바퀴는 주행 저항만큼 구동)."""
    import golf_realsim as RS
    M, L, G = RS.M, RS.L, RS.G
    fx = RS.CR0 + RS.CR2 * V_REF ** 2
    def ok(ay):
        for fz, tw, sh, mu, req, fxa in ((M * G * RS.LR / L, RS.TRACK_F, RS.ROLL_FRONT, mu_f, M * ay * RS.LR / L, fx),
                                         (M * G * RS.LF / L, RS.TRACK_R, 1 - RS.ROLL_FRONT, mu_r, M * ay * RS.LF / L, 0.0)):
            d = min(abs(sh * M * ay * h / tw), 0.49 * fz)
            cap = mu * (fz - 4 * k * d * d / fz)
            if req > math.sqrt(max(cap * cap - fxa * fxa, 0.0)):
                return False
        return True
    lo, hi = 0.0, 25.0
    for _ in range(40):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if ok(mid) else (lo, mid)
    return lo


def mu_for(dmu):
    """앞뒤 마찰 차 dmu 가 주어졌을 때 max_ay = AY_TARGET 이 되는 앞 마찰 (이분법, 0.8~1.6)."""
    lo, hi = 0.8, 1.6
    for _ in range(40):
        mid = (lo + hi) / 2
        lo, hi = (lo, mid) if max_ay(mid, mid + dmu) >= AY_TARGET else (mid, hi)
    return hi
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


def to_raw(z):
    return {k: float(BOUNDS[k][0] + (BOUNDS[k][1] - BOUNDS[k][0]) / (1 + math.exp(-v))) for k, v in zip(NAMES, z)}


def to_params(z):
    q = to_raw(z)
    mu = mu_for(q["dmu"])
    return dict(mu_f=mu, mu_r=mu + q["dmu"], ca_f=q["ca_f"], ca_r=q["ca_r"], iz=q["iz"], ds=q["ds"], tau_lt=q["tau_lt"], h_cg=H_FIXED)


def to_z(q):
    return [math.log((q[k] - BOUNDS[k][0]) / (BOUNDS[k][1] - q[k])) for k in NAMES]


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
        q0 = dict(dmu=0.0, ca_f=85735.0, ca_r=74954.0, iz=1797.0, ds=0.0, tau_lt=0.15)  # 시작: 저장소 specs (Cf_lin, Cr_lin, Iz)
        p0 = to_params(to_z(q0))
        before = f(to_z(q0))
        z0 = np.array(to_z(q0))
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
