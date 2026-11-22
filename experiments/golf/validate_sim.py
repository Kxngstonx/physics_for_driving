"""시뮬레이터 검증 (docs/experiment_plan_golf.md 8-3절). 단계별로 실행:

  python experiments/golf/validate_sim.py mu      # 1. (참고) 마찰 배율별 검증 파일 1초 재생 오차. [변경 11/19] 배율은 이것으로
                                                  #    고르지 않고 calibrate_sim.py (데이터가 실제로 낸 횡가속도) 로 정함
  python experiments/golf/validate_sim.py replay  # 2. 개루프 재생: 시험 파일, 1초·2초, Musiu 형식 (최대 오차 / RMSE). 보정값 사용
  python experiments/golf/validate_sim.py closed  # 3. 폐루프 근사: 기록 주행선 + 기록 속도, R1–R6(시드 0) × 조향 지연 4개. 기록 랩처럼 정지 출발
  python experiments/golf/validate_sim.py report  # json → md·그림

재생: 기록된 상태(vx, vy, r, 조향, 위치·방향)로 시작해 기록된 Fxf·조향을 넣는다. 지연·구동력 상한은 넣지 않는다
(기록값은 이미 실제로 작용한 값). 조향은 0.04초 동안 다음 기록값까지 일정한 속도로 움직인다.
경사: [변경 11/21 사용자 결정] OSM 경사 지도 (data.grade_at, 경로는 트랙 npz 의 grade) — 학습 데이터·MPPI 와 같은 원천.
      이전(11/18~11/20)은 관성 센서 Vu / 속력이었다. 데이터의 grade_rad 는 고도 변화와 맞지 않아 쓰지 않음 (sim.py).
출력: experiments/golf/results/sim/validation.json (단계별 키), validation.md, validation_*.png
"""
import json
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from deep_dynamics.golf import data as D  # noqa: E402
from deep_dynamics.golf import sim  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "sim")
JS = os.path.join(OUT, "validation.json")
TRACK = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "tracks", "thunderhill_recorded.npz")
MU_GRID = np.round(np.arange(0.80, 1.401, 0.05), 2)  # 기본 타이어 μ 약 1.05~1.17, Golf 측정 μ 약 1.2 → 1.4 까지
STEER_DELAYS = (0.0, 0.04, 0.08, 0.12)
HORIZONS = {"1s": 25, "2s": 50}


def load_json():
    return json.load(open(JS)) if os.path.exists(JS) else {}


def save_json(d):
    os.makedirs(OUT, exist_ok=True)
    json.dump(d, open(JS, "w"), indent=1)


def windows(names, horizon, stride=25):
    """(파일 단위로 나눈) 구간에서 재생 시작점. 각 항목: 시작 상태, 입력열, 정답열, 경사열."""
    W = []
    for n in names:
        for sg in D.load_run(n):
            s, u, x, y = sg["s"], sg["u"], sg["x"], sg["y"]
            g = D.grade_at(x, y)  # [11/21] OSM 경사 지도 (모델 입력과 같은 원천)
            for k in range(0, len(s) - horizon - 1, stride):
                hd = math.atan2(y[k + 1] - y[k], x[k + 1] - x[k])  # 진행 방향 → 차 방향 = 진행 방향 − 옆미끄럼각
                psi = hd - math.atan2(s[k, 1], s[k, 0])
                W.append(dict(s0=s[k], d0=u[k, 1], X0=x[k], Y0=y[k], psi0=psi, F=u[k:k + horizon, 0], d=u[k + 1:k + 1 + horizon, 1],
                              s=s[k + 1:k + 1 + horizon], x=x[k + 1:k + 1 + horizon], y=y[k + 1:k + 1 + horizon], g=g[k:k + horizon]))
    return W


def replay(W, mu=None):
    """→ 예측 (N, K, 5) [vx, vy, r, X, Y]. 보정된 시뮬레이터 (mu 를 주면 마찰 배율만 바꿈). 지연·힘 상한 없음."""
    out = np.full((len(W), len(W[0]["F"]), 5), np.nan)
    w = sim.calibrated_world(cap=False, delays=None, **({} if mu is None else dict(mu_scale=mu)))
    for i, win in enumerate(W):
        vx, vy, r = win["s0"]
        w.reset(win["X0"], win["Y0"], win["psi0"], vx, vy, r, delta=win["d0"])
        for k in range(len(win["F"])):
            w.set_grade(win["g"][k])
            s = w.step(float(win["F"][k]), float(win["d"][k]))
            out[i, k] = (*s, w.x[0], w.x[1])
            if not np.isfinite(s[0]):
                break
    return out


def metrics(W, P):
    """Musiu 표 형식: 상태별 최대 오차 / RMSE (모든 시작점·스텝), 1초 뒤 RMSE (모델 개루프 평가와 같은 지점)."""
    S = np.stack([w["s"] for w in W])
    XY = np.stack([np.c_[w["x"], w["y"]] for w in W])
    ok = np.isfinite(P).all(axis=(1, 2))
    beta = lambda a: np.arctan2(a[..., 1], a[..., 0])
    errs = dict(vx=P[ok, :, 0] - S[ok, :, 0], vy=P[ok, :, 1] - S[ok, :, 1], beta=beta(P[ok]) - beta(S[ok]),
                r=P[ok, :, 2] - S[ok, :, 2], pos=np.hypot(P[ok, :, 3] - XY[ok, :, 0], P[ok, :, 4] - XY[ok, :, 1]))
    m = {k: dict(max=float(np.abs(e).max()), rmse=float(np.sqrt(np.mean(e ** 2))), rmse_end=float(np.sqrt(np.mean(e[:, -1] ** 2))))
         for k, e in errs.items()}
    m["n_windows"], m["n_diverged"] = int(len(W)), int((~ok).sum())
    return m


def stage_mu(res):
    sp = D.get_split()
    W = windows(sp["val"], HORIZONS["1s"])
    sd = np.concatenate([sg["s"] for n in sp["train"] for sg in D.load_run(n)]).std(0)  # 상태별 크기 맞춤 (학습 손실과 같은 생각)
    rows = []
    for mu in MU_GRID:
        t0 = time.time()
        m = metrics(W, replay(W, mu))
        score = float(sum(m[k]["rmse_end"] / sd[i] for i, k in enumerate(("vx", "vy", "r"))))
        rows.append(dict(mu=float(mu), score=score, vx=m["vx"]["rmse_end"], vy=m["vy"]["rmse_end"], r=m["r"]["rmse_end"],
                         diverged=m["n_diverged"]))
        print(f"mu {mu:.2f} score {score:.4f} vx {m['vx']['rmse_end']:.3f} vy {m['vy']['rmse_end']:.3f} r {m['r']['rmse_end']:.4f} "
              f"발산 {m['n_diverged']}/{m['n_windows']} ({time.time() - t0:.0f}s)", flush=True)
    best = min(rows, key=lambda z: z["score"])
    res["mu"] = dict(files=sp["val"], n_windows=len(W), grid=rows, best=best["mu"], state_sd=sd.tolist())


def stage_replay(res):
    sp = D.get_split()
    mu = sim.load_calibration()["mu_scale"]
    out = {}
    for key, K in HORIZONS.items():
        W = windows(sp["test"], K)
        t0 = time.time()
        out[key] = metrics(W, replay(W, mu))
        print(key, json.dumps({k: v for k, v in out[key].items()}, ensure_ascii=False)[:400], f"{time.time() - t0:.0f}s", flush=True)
    ev = json.load(open(os.path.join(os.path.dirname(OUT), "eval_roll25.json")))
    res["replay"] = dict(files=sp["test"], mu=mu, **out, models_1s=ev.get("openloop", {}).get("all"))


def lap_info(run=None):
    """run=None: 트랙 기준선을 만든 가장 빠른 랩 (2023-10_8, 검증용). run 을 주면 그 파일에서 가장 빠른 완전한 랩."""
    if run is None:
        return json.load(open(TRACK.replace("_recorded.npz", "_info.json")))["best_lap"]
    from deep_dynamics.golf.track import best_lap
    b = best_lap([run])
    return dict(run=run, v_mean=b["v_mean"], lap_time=b["lap_time"], t0=b["t0"])


def recorded_track(info=None):
    """그 랩의 기록 주행선 트랙 + 그 랩의 기록 속도를 위치별로 + 위치별 경사 (랩 info, 기본은 2023-10_8 가장 빠른 랩).
    [변경 11/20] 경로와 목표 속도를 같은 랩에서: 검증 랩은 thunderhill_recorded.npz, 다른 랩은 thunderhill_recorded_<랩>.npz
    (track_osm.build_all 이 만듦, 경로 0번 = 랩 출발점)."""
    from deep_dynamics.golf import mppi
    import torch
    info = info or lap_info()
    df = D.read_raw(info["run"])
    t = df["t_s"].to_numpy()
    m = (t >= info["t0"]) & (t <= info["t0"] + info["lap_time"])
    px, py = df["posE_m"].to_numpy()[m], df["posN_m"].to_numpy()[m]
    v = np.hypot(df["Vx_mps"].to_numpy()[m], df["Vy_mps"].to_numpy()[m])
    tr = mppi.Track(TRACK if info["run"] == lap_info()["run"] else TRACK.replace(".npz", f"_{info['run']}.npz"))
    from deep_dynamics.golf.track import project
    # 표본마다 트랙 위치를 찾고 위치별 평균. 출발선 근처는 랩 처음(출발 중)과 끝(달리는 중)이 같은 위치에 겹치므로,
    # 랩 앞쪽 절반의 표본은 트랙 앞쪽 절반에만, 뒤쪽 절반은 뒤쪽 절반에만 쓴다 (11/19: 이전처럼 트랙 점마다 가장 가까운 표본을
    # 고르면 출발 위치가 기준점보다 앞인 랩(2023-10_5)은 기준점 근처가 랩 끝의 31.9 m/s 로 채워져 '달리며 출발'이 됐다)
    n = len(tr.np["x"])
    idx = project(np.c_[tr.np["x"], tr.np["y"]], np.c_[px, py])[0]
    frac = np.linspace(0.0, 1.0, len(idx))
    keep = ~(((frac < 0.5) & (idx > n / 2)) | ((frac >= 0.5) & (idx < n / 2)))
    cnt = np.bincount(idx[keep], minlength=n)
    has = cnt > 0
    pos = np.arange(n)
    per = lambda val: np.interp(pos, pos[has], (np.bincount(idx[keep], weights=val[keep], minlength=n)[has] / cnt[has]))
    tr.vref = torch.tensor(per(v), dtype=torch.float32, device=mppi.DEV)
    grade = tr.np["grade"]  # [11/21] OSM 경사 지도를 이 경로에 투영한 값 (track_osm.make_path). 이전: 이 랩의 Vu 경사 위치별 평균
    info = dict(info, start_idx=int(idx[0]))  # 이 랩이 실제로 출발한 트랙 위치 (주행도 여기서 시작)
    return tr, grade, info


# ---------------------------------------------------------------- 모델 없는 운전자 (11/19 사용자 결정)
# 시뮬레이터 검증이 학습 모델(R1–R6)에 기대지 않도록, 고전 제어기로 기록 주행선·기록 속도를 따라간다.
#   조향: pure pursuit (앞쪽 경로 한 점을 겨눔) + 언더스티어 보정 + 횡오차 PI.
#         언더스티어 계수 K_us = m/L·(lr/Cf − lf/Cr) 는 축별 코너링 강성(deep_dynamics/golf/specs.json, 11/21 Fiala)에서 — 학습 모델이 아님
#   속도: 앞쪽(0.3초 뒤 도달할 지점)의 기록 속도를 목표로 PI + 목표 속도 변화·저항·경사를 미리 보상
# 운전 실력이 실제 Stanford 제어기와 다르므로 랩타임 외에 속도·횡오차 추종도 함께 기록한다 (계획서 8-3절의 한계)
FREE = dict(look_t=0.8, look_min=6.0, look_max=30.0, k_ey=0.02, ki_ey=0.01, kp_v=1.0, ki_v=0.1, prev_t=0.3,
            d_max=0.30, dd_max=0.02)


def drive_free(world, tr, v_armed=10.0, v_stall=3.0, gains=FREE, i0=0):
    """tr: mppi.Track (vref = 목표 속도, 1 m 간격). world: sim.World (reset·set_pos·step·pose).
    트랙 위치 i0 에서 vref[i0] 속도로 시작해 한 바퀴 (정지 출발이면 vref[i0] ≈ 0.5~1 m/s)."""
    from deep_dynamics.golf import mppi
    P = np.c_[tr.np["x"], tr.np["y"]]
    psi_t, vref, n = tr.np["psi"], tr.vref.detach().cpu().numpy().astype(float), len(P)
    wl, wr = tr.np["width_l"], tr.np["width_r"]
    from deep_dynamics.golf.golf_specs import value
    p = world.p
    L = p.a + p.b
    K_us = p.m / L * (p.b / value("tire_stiffness_f") - p.a / value("tire_stiffness_r"))
    g = gains
    world.reset(P[i0, 0], P[i0, 1], float(psi_t[i0]), max(float(vref[i0]), 0.5), F0=0.0)
    i, prog, last, delta, ie_y, ie_v = i0, 0, i0, 0.0, 0.0, 0.0
    armed, status = False, "timeout"
    ey_log, ev_log = [], []
    max_steps = int(2.0 * float((1.0 / np.maximum(vref, 3.0)).sum()) / sim.DT) + 1000
    for step in range(max_steps):
        X, Y, psi = world.pose
        vx, vy, r = world.state
        if not np.isfinite(vx):
            status = "world_nan"
            break
        cand = (i + np.arange(-10, 90)) % n  # 지난 위치 근처에서만 찾음 (트랙이 가까이 지나는 곳에서 건너뛰지 않게)
        i = int(cand[np.argmin((P[cand, 0] - X) ** 2 + (P[cand, 1] - Y) ** 2)])
        ey = -(X - P[i, 0]) * math.sin(psi_t[i]) + (Y - P[i, 1]) * math.cos(psi_t[i])
        d = (i - last) % n
        if d < n / 2:
            prog += d
        last = i
        if abs(ey) > (wl[i] if ey > 0 else wr[i]) + mppi.OFF_TRACK_M:  # [11/21] 팀 기준 0.9 m (이전 2.0 m), mppi.episode 와 같음
            status = "off_track"
            break
        armed = armed or vx >= v_armed
        if armed and vx < v_stall:
            status = "stalled"
            break
        if prog >= n - 2:
            status = "finished"
            break
        # 조향
        Ld = float(np.clip(g["look_t"] * vx, g["look_min"], g["look_max"]))
        j = (i + int(round(Ld))) % n
        dx, dy = P[j, 0] - X, P[j, 1] - Y
        alpha = math.atan2(dy, dx) - psi
        kap = 2.0 * math.sin(alpha) / max(math.hypot(dx, dy), 1.0)
        ie_y = float(np.clip(ie_y + ey * sim.DT, -5.0, 5.0))
        target = math.atan(L * kap) + K_us * vx * vx * kap - g["k_ey"] * ey - g["ki_ey"] * ie_y
        target = float(np.clip(target, -g["d_max"], g["d_max"]))
        delta += float(np.clip(target - delta, -g["dd_max"], g["dd_max"]))
        # 속도
        jp = (i + max(1, int(round(g["prev_t"] * vx)))) % n
        ev = vref[jp] - vx
        ie_v = float(np.clip(ie_v + ev * sim.DT, -5.0, 5.0))
        dv_ds = (vref[(jp + 1) % n] - vref[jp - 1]) / 2.0
        a_cmd = g["kp_v"] * ev + g["ki_v"] * ie_v + vx * dv_ds
        world.set_pos(i)
        F = p.m * a_cmd + sim.road_load(vx) + world.g_slope * p.m
        world.step(F, delta)
        ey_log.append(ey)
        ev_log.append(vref[i] - vx)
    T = step * sim.DT
    ey_a, ev_a = np.abs(np.array(ey_log or [np.nan])), np.array(ev_log or [np.nan])
    return dict(status=status, finished=status == "finished", progress_frac=prog / n, lap_time=T if status == "finished" else None,
                ey_mean=float(ey_a.mean()), ey_max=float(ey_a.max()), v_err_mean=float(np.mean(np.abs(ev_a))),
                v_err_rms=float(np.sqrt(np.mean(ev_a ** 2))))


TUNE_LAP_RUN = "2023-10_5"  # 운전자 설정을 고르는 랩: 두 번째로 빠른 실제 랩 (108.2 s). 검증 랩(2023-10_8)과 다른 파일
FREE_GRID = dict(prev_t=(0.1, 0.2, 0.3), kp_v=(0.5, 1.0, 2.0))


def stage_closed_free(res):
    """폐루프 근사 검증, 모델 없는 운전자.
    1. 운전자 설정(속도 미리보기 시간, 속도 비례 이득)을 다른 랩(TUNE_LAP_RUN)에서 그 랩의 기록 시간에 가장 가깝게 고른다
       (검증 랩에 맞추면 검증을 저절로 통과시키는 셈이므로). 조향 지연 0
    2. 그 설정으로 검증 랩(2023-10_8)을 조향 지연 4가지로 돈다. 판정: 랩타임이 기록 ±2% (가장 가까운 조향 지연)."""
    tr_t, grade_t, info_t = recorded_track(lap_info(TUNE_LAP_RUN))
    grid = []
    for pt in FREE_GRID["prev_t"]:
        for kp in FREE_GRID["kp_v"]:
            gains = dict(FREE, prev_t=pt, kp_v=kp)
            r = drive_free(sim.calibrated_world(grade_by_idx=grade_t), tr_t, gains=gains, i0=info_t["start_idx"])
            r = dict(prev_t=pt, kp_v=kp, **r, rel_err=((r["lap_time"] - info_t["lap_time"]) / info_t["lap_time"]) if r["finished"] else None)
            grid.append(r)
            print("tune", json.dumps(r, ensure_ascii=False), flush=True)
    fin_t = [r for r in grid if r["finished"]]
    if not fin_t:  # [11/22] 튜닝 랩에서 완주한 설정이 없으면 검증 랩은 돌리지 않고 미달로 기록 (이전에는 여기서 멈춤)
        res["closed_free"] = dict(mu=sim.load_calibration()["mu_scale"], lap_record=None, gains=None, runs=[], tune_lap=dict(info_t, grid=grid),
                                  steer_delay=None, passed=False, note="튜닝 랩에서 완주한 운전자 설정 없음")
        print("튜닝 랩 완주 없음 → 미달", flush=True)
        return
    best = min(fin_t, key=lambda r: abs(r["rel_err"]))
    gains = dict(FREE, prev_t=best["prev_t"], kp_v=best["kp_v"])
    tr, grade, info = recorded_track()
    lap_rec = info["lap_time"]
    rows = []
    for sd in STEER_DELAYS:
        w = sim.calibrated_world(delays=dict(steer=sd), grade_by_idx=grade)
        r = drive_free(w, tr, gains=gains, i0=info["start_idx"])
        r = dict(steer_delay=sd, **r, rel_err=((r["lap_time"] - lap_rec) / lap_rec) if r["finished"] else None)
        rows.append(r)
        print(json.dumps(r, ensure_ascii=False), flush=True)
    fin = [r for r in rows if r["finished"]]
    pick = min(fin, key=lambda r: abs(r["rel_err"])) if fin else None
    res["closed_free"] = dict(mu=sim.load_calibration()["mu_scale"], lap_record=lap_rec, gains=gains, runs=rows,
                              tune_lap=dict(info_t, grid=grid),
                              steer_delay=pick["steer_delay"] if pick else None,
                              passed=bool(pick and abs(pick["rel_err"]) <= 0.02))


def stage_closed(res, key="closed", mppi_kw=None):
    """key="closed": 튜닝 전 MPPI 초기 설정. key="closed_tuned": tune_mppi.py 가 고른 A 방식 설정 (λ, 잡음 배수).
    판정 [변경 11/19, 사용자 결정]: 완주한 모델의 랩타임 중앙값이 기록 ±2% 이내 (모델이 코너에서 실패하는 것은 시뮬레이터 오류가
    아니라 이 프로젝트가 재려는 모델 차이이므로 판정에서 뺌). 조향 지연은 완주 수 → 그 중앙값이 기록에 가까운 순으로 고름.
    처음 기준(6개 모두의 중앙값)도 함께 기록한다."""
    import torch
    from deep_dynamics.golf import mppi
    from deep_dynamics.golf.configs import GOLF_THUNDERHILL as CFG, cfg_of_model
    from deep_dynamics.golf.ladder import load_model
    tr, grade, info = recorded_track()
    mu = sim.load_calibration()["mu_scale"]
    lap_rec = info["lap_time"]
    mdir = os.path.join(os.path.dirname(OUT), "models")
    mppi_kw = mppi_kw or {}
    rows = res.get(key, {}).get("runs", [])
    done = {(r["rung"], r["steer_delay"]) for r in rows}
    for sd in STEER_DELAYS:
        for R in ("R1", "R2", "R3", "R4", "R5", "R6"):
            if (R, sd) in done:
                continue
            pt = os.path.join(mdir, f"{R}_s0_roll25.pt")
            model = load_model(pt, cfg_of_model(pt, CFG), mppi.DEV)  # [11/19] 경사 모델은 json 의 cfg 로
            w = sim.calibrated_world(delays=dict(steer=sd), grade_by_idx=grade)
            r, log = mppi.episode(model, w, tr, seed=0, v_armed=10.0, **mppi_kw)  # 기록 랩은 정지 출발 (시작 속도 0.9 m/s)
            row = dict(rung=R, steer_delay=sd, **{k: r[k] for k in ("status", "finished", "progress_frac", "lap_time", "ey_mean",
                                                                     "ey_max", "v_err_mean", "wall_s")})
            rows.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)
            res[key] = dict(mu=mu, lap_record=lap_rec, track=TRACK, mppi=mppi_kw, runs=rows)
            save_json(res)
            del model
            torch.cuda.empty_cache()
    summ = {}
    for sd in STEER_DELAYS:
        R = [r for r in rows if r["steer_delay"] == sd]
        laps = [r["lap_time"] if r["finished"] else math.inf for r in R]
        fin = [r["lap_time"] for r in R if r["finished"]]
        med = float(np.median(laps))
        fmed = float(np.median(fin)) if fin else math.inf
        summ[str(sd)] = dict(finished=len(fin), median_lap=med, rel_err=(med - lap_rec) / lap_rec,
                             median_lap_finished=fmed, rel_err_finished=(fmed - lap_rec) / lap_rec)
    pick = min(summ, key=lambda k: (-summ[k]["finished"], abs(summ[k]["rel_err_finished"])))
    res[key].update(summary=summ, steer_delay=float(pick), passed=bool(abs(summ[pick]["rel_err_finished"]) <= 0.02),
                    passed_all6=bool(abs(summ[pick]["rel_err"]) <= 0.02))


def stage_closed_tuned(res):
    from deep_dynamics.golf import mppi
    c = json.load(open(os.path.join(os.path.dirname(OUT), "mppi_tune.json")))["chosen"]["A_25"]
    stage_closed(res, key="closed_tuned", mppi_kw=dict(lam=c["lam"], sigma=tuple(c["noise"] * x for x in mppi.SIGMA_FORCE)))


def stage_report(res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    L = ["# 시뮬레이터 검증 (docs/experiment_plan_golf.md 8-3절)", ""]
    if "mu" in res:
        m = res["mu"]
        L += [f"## 1. (참고) 마찰 배율별 재생 오차 — 배율은 calibrate_sim.py 로 정함 (검증 파일 {', '.join(m['files'])}, 1초 재생 {m['n_windows']}개)", "",
              "| mu_scale | 점수 (상태별 표준편차로 나눈 1초 RMSE 합) | vx | vy | r | 발산 |", "|---|---|---|---|---|---|"]
        L += [f"| {g['mu']:.2f}{' (재생 오차 최소, 참고만)' if g['mu'] == m['best'] else ''} | {g['score']:.4f} | {g['vx']:.3f} | {g['vy']:.3f} | {g['r']:.4f} | {g['diverged']} |"
              for g in m["grid"]]
        fig, ax = plt.subplots(figsize=(4.5, 3), layout="constrained")
        ax.plot([g["mu"] for g in m["grid"]], [g["score"] for g in m["grid"]], marker="o", color="#0072B2")
        ax.axvline(m["best"], ls="--", color="#D55E00", label=f"chosen {m['best']:.2f}")
        ax.set(xlabel="tire friction scale (mu_scale)", ylabel="1 s replay error (normalized)", title="Validation files")
        ax.legend()
        fig.savefig(os.path.join(OUT, "validation_mu.png"), dpi=150)
        L += ["", "![](validation_mu.png)", ""]
    if "replay" in res:
        r = res["replay"]
        L += [f"## 2. 개루프 재생 (시험 파일 {', '.join(r['files'])}, mu_scale {r['mu']:.2f})", "",
              "| 구간 | 시작점 | 발산 | vx 최대 / RMSE (m/s) | 옆미끄럼각 (rad) | r (rad/s) | 위치 (m) |", "|---|---|---|---|---|---|---|"]
        for key in HORIZONS:
            q = r[key]
            L.append(f"| {key} | {q['n_windows']} | {q['n_diverged']} | {q['vx']['max']:.2f} / {q['vx']['rmse']:.3f} | "
                     f"{q['beta']['max']:.3f} / {q['beta']['rmse']:.4f} | {q['r']['max']:.3f} / {q['r']['rmse']:.4f} | "
                     f"{q['pos']['max']:.2f} / {q['pos']['rmse']:.3f} |")
        L += ["", "Musiu 외 (한 바퀴 폐루프, 참고): vx 1.30 / 0.46, 옆미끄럼각 0.024 / 0.0035, 횡오차 0.60 / 0.14.", "",
              "1초 뒤 RMSE (vx / vy / r): 시뮬레이터 " + " / ".join(f"{r['1s'][k]['rmse_end']:.3f}" for k in ("vx", "vy", "r")), ""]
    for key, title in (("closed", "3. 폐루프 근사 — 튜닝 전 MPPI"), ("closed_tuned", "4. 폐루프 근사 — 튜닝한 MPPI")):
        if key not in res or "summary" not in res[key]:
            continue
        c = res[key]
        L += [f"## {title} (기록 주행선 + 기록 속도, 기록 랩타임 {c['lap_record']:.2f} s, mu_scale {c['mu']:.3f}, MPPI {c.get('mppi', {})})", "",
              "| 조향 지연 (s) | 완주 | 완주 모델 랩타임 중앙값 (s) | 기록 대비 | 6개 중앙값 기록 대비 |", "|---|---|---|---|---|"]
        for k, s in c["summary"].items():
            L.append(f"| {k} | {s['finished']}/6 | {s.get('median_lap_finished', math.inf):.2f} | {100 * s.get('rel_err_finished', math.inf):+.1f}% | "
                     f"{100 * s['rel_err']:+.1f}% |")
        L += ["", f"선택한 조향 지연 {c['steer_delay']} s. 판정(완주 모델 중앙값이 기록 ±2% 이내): **{'통과' if c['passed'] else '미달'}**"
              f" (처음 기준 6개 중앙값: {'통과' if c.get('passed_all6') else '미달'})", "",
              "| 모델 | 조향 지연 | 결과 | 진행률 | 랩타임 | 횡오차 평균 / 최대 | 속도 오차 평균 |", "|---|---|---|---|---|---|---|"]
        for q in c["runs"]:
            L.append(f"| {q['rung']} | {q['steer_delay']} | {q['status']} | {q['progress_frac']:.2f} | "
                     f"{q['lap_time'] if q['lap_time'] else '-'} | {q['ey_mean']:.2f} / {q['ey_max']:.2f} | {q['v_err_mean'] or float('nan'):.2f} |")
    if "closed_free" in res and res["closed_free"].get("gains") is None:
        c = res["closed_free"]
        L += ["", "## 5. 폐루프 근사 — 모델 없는 운전자", "", f"**미달: {c['note']}** (튜닝 랩 {c['tune_lap']['run']}, 기록 {c['tune_lap']['lap_time']:.2f} s)", "",
              "| 미리보기 (s) | 비례 이득 | 결과 | 진행률 | 횡오차 평균 / 최대 (m) |", "|---|---|---|---|---|"]
        L += [f"| {q['prev_t']} | {q['kp_v']} | {q['status']} | {q['progress_frac']:.3f} | {q['ey_mean']:.2f} / {q['ey_max']:.2f} |" for q in c["tune_lap"]["grid"]]
    elif "closed_free" in res:
        c = res["closed_free"]
        L += ["", f"## 5. 폐루프 근사 — 모델 없는 운전자 (기록 랩타임 {c['lap_record']:.2f} s, mu_scale {c['mu']:.3f})", "",
              "학습 모델을 쓰지 않는 고전 제어기(pure pursuit + 언더스티어 보정 + 속도 PI)로 기록 주행선·기록 속도를 따라감.",
              f"운전자 설정은 다른 랩({c['tune_lap']['run']}, 기록 {c['tune_lap']['lap_time']:.2f} s)에서 고름: 속도 미리보기 {c['gains']['prev_t']} s, 속도 비례 이득 {c['gains']['kp_v']}", "",
              "| 미리보기 (s) | 비례 이득 | 결과 | 랩타임 (s) | 그 랩 기록 대비 |", "|---|---|---|---|---|"]
        for q in c["tune_lap"]["grid"]:
            L.append(f"| {q['prev_t']} | {q['kp_v']} | {q['status']} | {q['lap_time'] or float('nan'):.2f} | "
                     f"{100 * q['rel_err'] if q['rel_err'] is not None else float('nan'):+.1f}% |")
        L += ["", "검증 랩:", "",
              "| 조향 지연 (s) | 결과 | 진행률 | 랩타임 (s) | 기록 대비 | 횡오차 평균 / 최대 (m) | 속도 오차 평균 / RMS (m/s) |", "|---|---|---|---|---|---|---|"]
        for q in c["runs"]:
            L.append(f"| {q['steer_delay']} | {q['status']} | {q['progress_frac']:.3f} | {q['lap_time'] or float('nan'):.2f} | "
                     f"{100 * q['rel_err'] if q['rel_err'] is not None else float('nan'):+.1f}% | {q['ey_mean']:.2f} / {q['ey_max']:.2f} | "
                     f"{q['v_err_mean']:.2f} / {q['v_err_rms']:.2f} |")
        L += ["", f"선택한 조향 지연 {c['steer_delay']} s. 판정(랩타임 기록 ±2%): **{'통과' if c['passed'] else '미달'}**"]
    open(os.path.join(OUT, "validation.md"), "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    stage = sys.argv[1]
    res = load_json()
    {"mu": stage_mu, "replay": stage_replay, "closed": stage_closed, "closed_tuned": stage_closed_tuned, "closed_free": stage_closed_free,
     "report": stage_report}[stage](res)
    if stage != "report":
        save_json(res)
