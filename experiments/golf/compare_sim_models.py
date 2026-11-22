"""시뮬레이터 구조 비교 (11/22 사용자 결정 B): 다물체(mb, 지금) vs 이중 궤도(dt, Subosits 2020), 새 검증 기준 ①②③.

  python experiments/golf/compare_sim_models.py run mb|dt_open|dt_lsd
  python experiments/golf/compare_sim_models.py report   # → results/sim/sim_models_compare.md

방식:
  mb      : 지금 기본 (CommonRoad 다물체, 거울상, 열린 차동장치, 공식 calibration.json)
  dt_open : 이중 궤도, LSD 열림 (u_diff = 0)
  dt_lsd  : 이중 궤도, LSD (재생 = 데이터 LSDengage_percent/100, 폐루프 = 학습 파일 표; compare_lsd.lsd_table_from_train)
고정: specs.json, 시험 파일, 운전자 설정 9개, 튜닝·검증 랩. 맞춤(fit_sim) 전 상태.
① 재생: 시험 파일 1·2초 끝 RMSE, 구간별 점수, 발산
② 특성: 속도별 조향 대비 회전 비 (데이터와 같은 조건으로 잼), 정상 선회 최대 옆 가속도 (목표 specs ay_target),
       조향 계단 응답 초과량, 500 m 코너 옆 가속도 (실제 8.88), 가속 코너 빠른 앞바퀴/차속 99% (데이터 1.07)
       (dt 는 바퀴 회전 상태가 없어 1 + κ, κ = Fx/(c8·Fz) 로 계산)
③ 폐루프: 모델 없는 운전자 (validate_sim.stage_closed_free 그대로)
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import compare_lsd as CL  # noqa: E402
import fit_sim as FS  # noqa: E402
import validate_sim as V  # noqa: E402
from deep_dynamics.golf import data as D  # noqa: E402
from deep_dynamics.golf import dt_golf, golf_specs, sim  # noqa: E402
from deep_dynamics.golf import specs as S  # noqa: E402
from deep_dynamics.golf.track import project  # noqa: E402

OUT = V.OUT
MODES = ("mb", "dt_open", "dt_lsd")
L_WB = 2.63


def make(mode, table=None, **kw):
    if mode == "mb":
        return sim.calibrated_world(**kw)
    if mode == "dt_open":
        return sim.calibrated_world(model="dt", **kw)
    return sim.calibrated_world(model="dt", lsd=dict(mode="active"), lsd_table=table, **kw)


def front_ratio(w):
    """빠른 앞바퀴 속도 / 차속. mb: 바퀴 회전 상태, dt: 1 + max κ (κ = Fx/(c8·Fz))."""
    if w.model == "dt":
        Fz, Fx, _ = dt_golf.forces(w.x, w.act.Fe + w.act.Fb, w.p.lsd_u, w.p)
        return 1.0 + max(Fx[0] / (w.p.c8 * max(Fz[0], 1.0)), Fx[1] / (w.p.c8 * max(Fz[1], 1.0)))
    return max(w.x[23], w.x[24]) * sim.WHEEL_R / max(w.state[0], 1.0)


def replay(W, mode, table):
    out, fr = np.full((len(W), len(W[0]["F"]), 5), np.nan), np.full((len(W), len(W[0]["F"])), np.nan)
    w = make(mode, table, cap=False, delays=None)
    for i, win in enumerate(W):
        vx, vy, r = win["s0"]
        w.reset(win["X0"], win["Y0"], win["psi0"], vx, vy, r, delta=win["d0"])
        for k in range(len(win["F"])):
            w.set_grade(win["g"][k])
            if mode == "dt_lsd":
                w.set_lsd(float(win["u"][k]))
            s = w.step(float(win["F"][k]), float(win["d"][k]))
            out[i, k] = (*s, w.pose[0], w.pose[1])
            fr[i, k] = front_ratio(w)
            if not np.isfinite(s[0]):
                break
    return out, fr


def data_steer_ratio():
    """데이터: 속도 구간별 r·L/(vx·δ) 중앙값 (|옆 가속| 1~4, 2 Hz 평활) — 11/21 비교와 같은 조건."""
    V_, R_, D_ = [], [], []
    for n in D.available():
        df = D.read_csv(n)
        V_.append(S.smooth(df["Vx_mps"].to_numpy(float), 2.0, 200)[400:-400:8])
        R_.append(S.smooth(df["yawRate_radps"].to_numpy(float), 2.0, 200)[400:-400:8])
        D_.append(S.smooth(df["roadWheelAngle_rad"].to_numpy(float), 2.0, 200)[400:-400:8])
    v, r, d = map(np.concatenate, (V_, R_, D_))
    out = {}
    for lo in (15, 20, 25, 30):
        k = (v >= lo) & (v < lo + 5) & (np.abs(v * r) > 1) & (np.abs(v * r) < 4) & (np.abs(d) > 0.01)
        out[f"{lo}-{lo + 5}"] = float(np.median(r[k] * L_WB / (v[k] * d[k])))
    return out


def sim_steer_ratio(mode, table):
    """정상 선회: 각 속도에서 옆 가속 약 3 m/s² 가 되는 조향으로 4초 → r·L/(v·δ)."""
    out = {}
    for lo in (15, 20, 25, 30):
        v = lo + 2.5
        d = 3.0 * L_WB / v ** 2 / 0.55  # 대략 옆 가속 3 이 되게 (운동학 대비 0.55 가정)
        w = make(mode, table, cap=False, delays=None)
        w.reset(0, 0, 0, v)
        for _ in range(100):
            s = w.step(sim.road_load(w.state[0]) + w.p.m * 0.5 * (v - w.state[0]), d)
        out[f"{lo}-{lo + 5}"] = float(s[2] * L_WB / (s[0] * d))
    return out


def sim_max_ay(mode, table, v=20.0, T=15.0):
    w = make(mode, table, cap=False, delays=None)
    w.reset(0, 0, 0, v)
    best, n = 0.0, int(T / sim.DT)
    for k in range(n):
        s = w.step(sim.road_load(w.state[0]) + w.p.m * 0.5 * (v - w.state[0]), 0.25 * k / n)
        if not np.isfinite(s[0]) or abs(np.arctan2(s[1], s[0])) > 0.15:
            break
        best = max(best, abs(s[0] * s[2]))
    return best


def sim_step_overshoot(mode, table, v=24.0, d=0.04):
    w = make(mode, table, cap=False, delays=None)
    w.reset(0, 0, 0, v)
    R = [w.step(sim.road_load(w.state[0]) + w.p.m * 0.5 * (v - w.state[0]), d)[2] for _ in range(75)]
    return float(max(R) / np.mean(R[-10:]))


def run(mode):
    table = CL.lsd_table_from_train()[1] if mode == "dt_lsd" else None
    res = dict(mode=mode)
    sp = D.get_split()
    sd = np.concatenate([sg["s"] for n in sp["train"] for sg in D.load_run(n)]).std(0)
    for key, K in V.HORIZONS.items():
        W = CL.windows(sp["test"], K)
        P, FR = replay(W, mode, table)
        res[f"replay_{key}"] = V.metrics(W, P)
        if key == "1s":
            cat = FS.categories(W)
            res["by_category_1s"] = {c: FS.score(W, P, sd, cat == c) for c in ("straight", "braking", "corner", "drive_corner")}
            res["by_category_1s"]["all"] = FS.score(W, P, sd)
            Sx = np.stack([w["s"] for w in W]); F = np.stack([w["F"] for w in W])
            m = (np.abs(Sx[:, :, 0] * Sx[:, :, 2]) > 5) & (F > 1700) & np.isfinite(FR)
            res["front_ratio_drive_corner_p99"] = float(np.percentile(FR[m], 99))
        print(mode, key, {k: round(res[f"replay_{key}"][k]["rmse_end"], 3) for k in ("vx", "vy", "r")}, flush=True)
    tr, _, _ = V.recorded_track()
    W5 = CL.windows(["2023-10_8"], 25, stride=5)
    idx = project(np.c_[tr.np["x"], tr.np["y"]], np.array([[w["X0"], w["Y0"]] for w in W5]))[0]
    sel = [w for w, i in zip(W5, idx) if 420 <= i <= 560]
    P, _ = replay(sel, mode, table)
    S5 = np.stack([w["s"] for w in sel])
    res["corner500"] = dict(ay_data=float(np.abs(S5[:, :, 0] * S5[:, :, 2]).mean()), ay_sim=float(np.nanmean(np.abs(P[:, :, 0] * P[:, :, 2]))))
    res["steer_ratio"] = sim_steer_ratio(mode, table)
    res["max_ay"] = sim_max_ay(mode, table)
    res["step_overshoot"] = sim_step_overshoot(mode, table)
    print(mode, "특성", res["steer_ratio"], round(res["max_ay"], 2), round(res["step_overshoot"], 3), flush=True)
    orig = sim.calibrated_world
    sim.calibrated_world = lambda **kw: (orig(**kw) if mode == "mb" else orig(model="dt", lsd=dict(mode="active") if mode == "dt_lsd" else None,
                                                                          lsd_table=table, **kw))
    try:
        r = {}
        V.stage_closed_free(r)
        res["closed_free"] = r["closed_free"]
    finally:
        sim.calibrated_world = orig
    json.dump(res, open(os.path.join(OUT, f"sim_model_{mode}.json"), "w"), indent=1, default=float)
    print(mode, "완료", flush=True)


def report():
    R = {m: json.load(open(os.path.join(OUT, f"sim_model_{m}.json"))) for m in MODES}
    dsr = data_steer_ratio()
    ayt = golf_specs.value("ay_target")
    L = ["# 시뮬레이터 구조 비교 (11/22): 다물체 vs 이중 궤도 (맞춤 전)", "", "| 기준 | 항목 | 데이터 | 다물체 (지금) | 이중 궤도 LSD 열림 | 이중 궤도 LSD |", "|---|---|---|---|---|---|"]
    row = lambda crit, name, dat, fn: L.append(f"| {crit} | {name} | {dat} | " + " | ".join(fn(R[m]) for m in MODES) + " |")
    for key in V.HORIZONS:
        row("①", f"재생 {key} vx / vy / r", "—", lambda r, key=key: " / ".join(f"{r[f'replay_{key}'][k]['rmse_end']:.3f}" for k in ("vx", "vy", "r")))
        row("①", f"재생 {key} 발산", "—", lambda r, key=key: str(r[f"replay_{key}"]["n_diverged"]))
    for c in ("all", "straight", "braking", "corner", "drive_corner"):
        row("①", f"1초 점수 ({c})", "—", lambda r, c=c: f"{r['by_category_1s'][c]['score']:.3f}")
    for k, v in dsr.items():
        row("②", f"조향 대비 회전 비 {k} m/s", f"{v:.2f}", lambda r, k=k: f"{r['steer_ratio'][k]:.2f}")
    row("②", "정상 선회 최대 옆 가속도", f"{ayt:.2f}", lambda r: f"{r['max_ay']:.2f}")
    row("②", "조향 계단 응답 최대/정상", "—", lambda r: f"{r['step_overshoot']:.3f}")
    row("②", "500 m 코너 옆 가속도", "8.88", lambda r: f"{r['corner500']['ay_sim']:.2f}")
    row("②", "가속 코너 빠른 앞바퀴/차속 99%", "1.07", lambda r: f"{r['front_ratio_drive_corner_p99']:.2f}")

    def cf(r):
        c = r["closed_free"]
        n_t = sum(q["finished"] for q in c["tune_lap"]["grid"])
        if c.get("gains") is None:
            return f"튜닝 랩 완주 0/9"
        fin = [q for q in c["runs"] if q["finished"]]
        best = min(fin, key=lambda q: abs(q["rel_err"])) if fin else None
        return f"튜닝 {n_t}/9, 검증 랩 완주 {len(fin)}/4" + (f" ({best['lap_time']:.2f} s, {100 * best['rel_err']:+.1f}%)" if best else "")
    row("③", "모델 없는 운전자", "완주", cf)
    open(os.path.join(OUT, "sim_models_compare.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    {"run": lambda: run(sys.argv[2]), "report": report}[sys.argv[1]]()
