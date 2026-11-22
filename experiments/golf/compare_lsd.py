"""앞 LSD 비교 (11/22 사용자 결정: (가)·(나) 모두 구현해 시험). mb_golf 의 lsd_mode 설명 참고.

  python experiments/golf/compare_lsd.py fit              # (가) 수동형 T0·G 를 학습 파일 가속 코너 재생으로 고름 → results/sim/lsd_fit.json
  python experiments/golf/compare_lsd.py run open|passive|active
  python experiments/golf/compare_lsd.py report           # → results/sim/lsd_compare.md

바꾸는 변수: 앞 차동장치 하나 (열림 / (가) 수동 T0 + G·구동 토크 / (나) 전자 u·1,600 Nm).
고정: specs.json, 공식 calibration.json (마찰 0.998·롤 강성, 거울상), 시험 파일, 운전자 설정 9개, 튜닝·검증 랩.
(가) 맞추는 기준: 학습 파일 가속 코너 재생에서 안쪽−바깥 앞바퀴 속도 차이의 평균이 데이터와 같도록 (데이터 = 같은 시점의 실제 바퀴 속도).
(나) u: 재생은 데이터 LSDengage_percent/100 그대로, 폐루프는 학습 파일에서 만든 (앞바퀴 힘, |옆 가속도|) → 평균 작동 비율 표.
"""
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import fit_sim as FS  # noqa: E402
import validate_sim as V  # noqa: E402
from deep_dynamics.golf import data as D  # noqa: E402
from deep_dynamics.golf import sim  # noqa: E402
from deep_dynamics.golf.track import project  # noqa: E402

OUT = V.OUT
G_GRID, T0_GRID = (0.1, 0.25, 0.5, 0.75, 1.0), (0.0, 100.0, 300.0)
F_BINS, AY_BINS = np.array([0, 1000, 2000, 3000, 4500, 1e9]), np.array([0, 2, 4, 6, 8, 1e9])


def windows(names, horizon, stride=25):
    """validate_sim.windows + 같은 시점의 데이터 LSD 작동 비율(0~1)·앞 좌우 바퀴 속도(m/s)."""
    W = V.windows(names, horizon, stride)
    extra = []
    for n in names:
        df = D.read_csv(n)
        t = df["t_s"].to_numpy()
        lsd = df["LSDengage_percent"].to_numpy(float) / 100.0
        wl, wr = df["wheelspeed_fl"].to_numpy(float) * sim.WHEEL_R, df["wheelspeed_fr"].to_numpy(float) * sim.WHEEL_R
        for sg in D.load_run(n):
            ts = sg["t"]
            u, l, r = np.interp(ts, t, lsd), np.interp(ts, t, wl), np.interp(ts, t, wr)
            for k in range(0, len(ts) - horizon - 1, stride):
                extra.append(dict(u=u[k:k + horizon], wl=l[k + 1:k + 1 + horizon], wr=r[k + 1:k + 1 + horizon]))
    assert len(extra) == len(W)
    for w, e in zip(W, extra):
        w.update(e)
    return W


def replay(W, lsd=None, recorded_u=False, table=None):
    """→ 예측 (N, K, 5) [vx, vy, r, X, Y], 앞 좌우 바퀴 속도 (N, K, 2)."""
    out, wh = np.full((len(W), len(W[0]["F"]), 5), np.nan), np.full((len(W), len(W[0]["F"]), 2), np.nan)
    w = sim.calibrated_world(cap=False, delays=None, lsd=lsd, lsd_table=table)
    for i, win in enumerate(W):
        vx, vy, r = win["s0"]
        w.reset(win["X0"], win["Y0"], win["psi0"], vx, vy, r, delta=win["d0"])
        for k in range(len(win["F"])):
            w.set_grade(win["g"][k])
            if recorded_u:
                w.set_lsd(float(win["u"][k]))
            s = w.step(float(win["F"][k]), float(win["d"][k]))
            out[i, k] = (*s, w.x[0], w.x[1])
            wh[i, k] = (w.x[23] * sim.WHEEL_R, w.x[24] * sim.WHEEL_R)
            if not np.isfinite(s[0]):
                break
    return out, wh


def inner_outer(W, P, WH):
    """가속 코너 표본(|vx·r| > 5, 앞바퀴 힘 > 1.7 kN)의 안쪽−바깥 앞바퀴 속도 평균: (시뮬레이터, 데이터), 빠른 앞바퀴/차속 99%."""
    S = np.stack([w["s"] for w in W])
    F = np.stack([w["F"] for w in W])
    dl, dr = np.stack([w["wl"] for w in W]), np.stack([w["wr"] for w in W])
    m = (np.abs(S[:, :, 0] * S[:, :, 2]) > 5) & (F > 1700) & np.isfinite(WH).all(-1)
    left_inner = S[:, :, 2] > 0
    sim_io = np.where(left_inner, WH[..., 0] - WH[..., 1], WH[..., 1] - WH[..., 0])
    dat_io = np.where(left_inner, dl - dr, dr - dl)
    ratio = np.maximum(WH[..., 0], WH[..., 1]) / np.maximum(P[..., 0], 1.0)
    return float(sim_io[m].mean()), float(dat_io[m].mean()), float(np.percentile(ratio[m], 99)), int(m.sum())


def lsd_table_from_train():
    """학습 파일: (앞바퀴 힘 Fxf, |옆 가속도 vx·r|) 구간별 평균 LSD 작동 비율 → 표, 조회 함수."""
    sp = D.get_split()
    F, A, U = [], [], []
    for n in sp["train"]:
        df = D.read_raw(n)
        k = df["Vx_mps"].to_numpy() > 5
        F.append(df["Fxf"].to_numpy()[k]); A.append(np.abs(df["Vx_mps"] * df["yawRate_radps"]).to_numpy()[k])
        U.append(df["LSDengage_percent"].to_numpy(float)[k] / 100.0)
    F, A, U = map(np.concatenate, (F, A, U))
    fi, ai = np.clip(np.digitize(F, F_BINS) - 1, 0, len(F_BINS) - 2), np.clip(np.digitize(A, AY_BINS) - 1, 0, len(AY_BINS) - 2)
    T = np.zeros((len(F_BINS) - 1, len(AY_BINS) - 1))
    for i in range(T.shape[0]):
        for j in range(T.shape[1]):
            s = (fi == i) & (ai == j)
            T[i, j] = U[s].mean() if s.sum() > 50 else 0.0

    def look(F_cmd, v, r):
        i = int(np.clip(np.digitize(F_cmd, F_BINS) - 1, 0, T.shape[0] - 1))
        j = int(np.clip(np.digitize(abs(v * r), AY_BINS) - 1, 0, T.shape[1] - 1))
        return T[i, j]
    return T, look


def fit():
    sp = D.get_split()
    W = windows(sp["train"], 25, stride=50)
    W = [w for w, c in zip(W, FS.categories(W)) if c == "drive_corner"]
    rows = []
    P, WH = replay(W)
    so, do, ra, n = inner_outer(W, P, WH)
    rows.append(dict(T0=None, G=None, sim_io=so, data_io=do, ratio99=ra, n=n))
    print(f"열림: 안쪽−바깥 sim {so:+.3f} / 데이터 {do:+.3f} m/s, 빠른 앞바퀴 99% {ra:.2f} (창 {len(W)}, 표본 {n})", flush=True)
    for T0 in T0_GRID:
        for G in G_GRID:
            P, WH = replay(W, lsd=dict(mode="passive", T0=T0, G=G))
            so, do, ra, n = inner_outer(W, P, WH)
            rows.append(dict(T0=T0, G=G, sim_io=so, data_io=do, ratio99=ra, n=n))
            print(f"T0 {T0:5.0f} G {G:.2f}: 안쪽−바깥 sim {so:+.3f} / 데이터 {do:+.3f}, 빠른 앞바퀴 99% {ra:.2f}", flush=True)
    best = min((r for r in rows if r["T0"] is not None), key=lambda r: abs(r["sim_io"] - r["data_io"]))
    json.dump(dict(rows=rows, best=best, windows=len(W)), open(os.path.join(OUT, "lsd_fit.json"), "w"), indent=1)
    print("선택", best)


def run(mode):
    lsd, table, rec = None, None, False
    if mode == "passive":
        b = json.load(open(os.path.join(OUT, "lsd_fit.json")))["best"]
        lsd = dict(mode="passive", T0=b["T0"], G=b["G"])
    elif mode == "active":
        T, table = lsd_table_from_train()
        lsd, rec = dict(mode="active", Tmax=1600.0), True
    res = dict(mode=mode, lsd=lsd, table=T.tolist() if mode == "active" else None)
    sp = D.get_split()
    sd = np.concatenate([sg["s"] for n in sp["train"] for sg in D.load_run(n)]).std(0)
    for key, K in V.HORIZONS.items():
        W = windows(sp["test"], K)
        P, WH = replay(W, lsd, recorded_u=rec, table=table)
        res[f"replay_{key}"] = V.metrics(W, P)
        if key == "1s":
            cat = FS.categories(W)
            res["by_category_1s"] = {c: FS.score(W, P, sd, cat == c) for c in ("straight", "braking", "corner", "drive_corner")}
            res["by_category_1s"]["all"] = FS.score(W, P, sd)
            so, do, ra, n = inner_outer(W, P, WH)
            res["inner_outer_test"] = dict(sim=so, data=do, ratio99=ra, n=n)
    tr, _, _ = V.recorded_track()
    W5 = windows(["2023-10_8"], 25, stride=5)
    idx = project(np.c_[tr.np["x"], tr.np["y"]], np.array([[w["X0"], w["Y0"]] for w in W5]))[0]
    sel = [w for w, i in zip(W5, idx) if 420 <= i <= 560]
    P, WH = replay(sel, lsd, recorded_u=rec, table=table)
    S = np.stack([w["s"] for w in sel])
    res["corner500"] = dict(ay_data=float(np.abs(S[:, :, 0] * S[:, :, 2]).mean()), ay_sim=float(np.nanmean(np.abs(P[:, :, 0] * P[:, :, 2]))))
    # 폐루프: 모델 없는 운전자 (validate_sim.stage_closed_free 그대로, 시뮬레이터에 LSD 만 더함; (나) 는 표로 u)
    orig = sim.calibrated_world
    sim.calibrated_world = lambda **kw: orig(lsd=lsd, lsd_table=table, **kw)
    try:
        r = {}
        V.stage_closed_free(r)
        res["closed_free"] = r["closed_free"]
    finally:
        sim.calibrated_world = orig
    json.dump(res, open(os.path.join(OUT, f"lsd_{mode}.json"), "w"), indent=1, default=float)
    print(mode, "완료", {k: round(res["replay_1s"][k]["rmse_end"], 3) for k in ("vx", "vy", "r")}, "closed_free", res["closed_free"].get("passed"), flush=True)


def report():
    R = {m: json.load(open(os.path.join(OUT, f"lsd_{m}.json"))) for m in ("open", "passive", "active")}
    f = json.load(open(os.path.join(OUT, "lsd_fit.json")))
    L = ["# 앞 LSD 비교 (11/22)", "", f"(가) 수동형 선택: T0 {f['best']['T0']:.0f} Nm, G {f['best']['G']} (학습 파일 가속 코너 {f['windows']}개 창)", "",
         "| 항목 | 열림 | (가) 수동 | (나) 전자 |", "|---|---|---|---|"]
    row = lambda name, fn: L.append(f"| {name} | " + " | ".join(fn(R[m]) for m in ("open", "passive", "active")) + " |")
    row("가속 코너 안쪽−바깥 앞바퀴 속도 sim (데이터)", lambda r: f"{r['inner_outer_test']['sim']:+.3f} ({r['inner_outer_test']['data']:+.3f})")
    row("가속 코너 빠른 앞바퀴/차속 99%", lambda r: f"{r['inner_outer_test']['ratio99']:.2f}")
    for key in V.HORIZONS:
        row(f"재생 {key} vx / vy / r", lambda r, key=key: " / ".join(f"{r[f'replay_{key}'][k]['rmse_end']:.3f}" for k in ("vx", "vy", "r")))
        row(f"재생 {key} 발산", lambda r, key=key: str(r[f"replay_{key}"]["n_diverged"]))
    for c in ("all", "straight", "braking", "corner", "drive_corner"):
        row(f"1초 점수 ({c})", lambda r, c=c: f"{r['by_category_1s'][c]['score']:.3f}")
    row("500 m 코너 옆 가속도 (실제 8.88)", lambda r: f"{r['corner500']['ay_sim']:.2f}")

    def cf(r):
        c = r["closed_free"]
        if c.get("gains") is None:
            return "튜닝 랩 완주 없음"
        best = min((q for q in c["runs"] if q["finished"]), key=lambda q: abs(q["rel_err"]), default=None)
        return f"{'통과' if c['passed'] else '미달'} ({best['lap_time']:.2f} s, {100 * best['rel_err']:+.1f}%)" if best else "검증 랩 완주 없음"
    row("모델 없는 운전자", cf)
    row("튜닝 랩 완주 설정 수 (9개 중)", lambda r: str(sum(q["finished"] for q in r["closed_free"]["tune_lap"]["grid"])))
    open(os.path.join(OUT, "lsd_compare.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    {"fit": fit, "run": lambda: run(sys.argv[2]), "report": report}[sys.argv[1]]()
