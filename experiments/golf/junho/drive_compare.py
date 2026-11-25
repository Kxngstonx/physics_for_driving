"""R1 vs R6 코스 주행 비교: 시뮬레이터 2개 (저장소 CommonRoad / 직접 만든 golf_realsim) × 모델 2개.
조건은 저장소 그대로:
  - A 방식(tune_mppi.py): 기록선 + 2023-10_5 랩의 위치별 기록 속도, 그 랩 출발 위치에서 정지 출발, v_armed 10 m/s
  - MPPI: K 1024, 계획 1초(N 25), 저장소가 고른 A 설정 λ 0.3·잡음 ×0.5 (docs/golf_results_phase2.md 8절)
  - 실패: 차 중심이 경계 밖 0.9 m (차 반폭) 이탈, 3 m/s 아래 정지
  - 조향 지연: 각 시뮬레이터의 "모델 없는 운전자 검증"(validate_sim.stage_closed_free 와 같은 절차)으로 고른 값
사용:
  python drive_compare.py REPO free   # 직접 만든 시뮬레이터의 모델 없는 운전자 검증 (CommonRoad 는 저장소 validate_sim 결과 사용)
  python drive_compare.py REPO run    # 4회 주행 (끝난 것은 건너뜀)
결과: realsim_out/drive_compare.json, realsim_out/log_<sim>_<rung>.npz
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(sys.argv[1])
for p in (REPO, os.path.join(REPO, "experiments", "golf"), HERE):
    sys.path.insert(0, p)
import validate_sim as V  # noqa: E402
import golf_realsim as RS  # noqa: E402
from deep_dynamics.golf import mppi, sim  # noqa: E402

OUT = os.path.join(HERE, os.environ.get("REALSIM_OUT", "realsim_out"))  # 결과 폴더 (물리 제한판은 realsim_phys)
JS = os.path.join(OUT, "drive_compare.json")
FIT = os.path.join(OUT, "fit_realsim.json")
DELAYS = os.path.join(REPO, "experiments/golf/results/sim/delays.json")
LAM, NOISE = 0.3, 0.5


def load():
    return json.load(open(JS)) if os.path.exists(JS) else {}


def save(d):
    json.dump(d, open(JS + ".tmp", "w"), indent=1)
    os.replace(JS + ".tmp", JS)


SIM_KIND = os.environ.get("SIM_KIND", "realsim")  # "dt" = 팀원 브랜치 이중 궤도 (dt_world.py)


def real_world(grade, steer, main_units=True):
    """main_units: dt 에서만 의미. 모델(main 데이터 단위 Fxf) 주행은 True, 모델 없는 운전자(실제 N 으로 힘을 계산)는 False."""
    if SIM_KIND == "dt":
        import dt_world
        return dt_world.DTWorld(sim.Actuator, delays=dict(sim.load_delays(DELAYS) or {}, steer=steer), grade_by_idx=grade, main_units=main_units)
    return RS.make_world(FIT, DELAYS, sim.force_cap_table(), grade_by_idx=grade, delays=dict(steer=steer))


def stage_free():
    """validate_sim.stage_closed_free 와 같은 절차를 직접 만든 시뮬레이터에: 운전자 설정은 2023-10_5 랩에서, 검증은 2023-10_8."""
    tr_t, grade_t, info_t = V.recorded_track(V.lap_info(V.TUNE_LAP_RUN))
    grid = []
    for pt in V.FREE_GRID["prev_t"]:
        for kp in V.FREE_GRID["kp_v"]:
            gains = dict(V.FREE, prev_t=pt, kp_v=kp)
            r = V.drive_free(real_world(grade_t, 0.0, main_units=False), tr_t, gains=gains, i0=info_t["start_idx"])
            r = dict(prev_t=pt, kp_v=kp, **r, rel_err=((r["lap_time"] - info_t["lap_time"]) / info_t["lap_time"]) if r["finished"] else None)
            grid.append(r)
            print("tune", json.dumps(r, ensure_ascii=False), flush=True)
    fin = [r for r in grid if r["finished"]]
    if not fin:
        d = load(); d["free"] = dict(grid=grid, passed=False, steer_delay=None); save(d)
        print("운전자 설정 중 완주 없음"); return
    best = min(fin, key=lambda r: abs(r["rel_err"]))
    gains = dict(V.FREE, prev_t=best["prev_t"], kp_v=best["kp_v"])
    tr, grade, info = V.recorded_track()
    rows = []
    for sd in V.STEER_DELAYS:
        r = V.drive_free(real_world(grade, sd, main_units=False), tr, gains=gains, i0=info["start_idx"])
        r = dict(steer_delay=sd, **r, rel_err=((r["lap_time"] - info["lap_time"]) / info["lap_time"]) if r["finished"] else None)
        rows.append(r)
        print(json.dumps(r, ensure_ascii=False), flush=True)
    fin = [r for r in rows if r["finished"]]
    pick = min(fin, key=lambda r: abs(r["rel_err"])) if fin else None
    d = load()
    d["free"] = dict(lap_record=info["lap_time"], gains=gains, grid=grid, runs=rows, steer_delay=pick["steer_delay"] if pick else None,
                     passed=bool(pick and abs(pick["rel_err"]) <= 0.02))
    save(d)
    print("판정", d["free"]["passed"], "조향 지연", d["free"]["steer_delay"])


def stage_run():
    import torch
    from deep_dynamics.golf.configs import GOLF_THUNDERHILL as CFG, cfg_of_model
    from deep_dynamics.golf.ladder import load_model
    tr, grade, info = V.recorded_track(V.lap_info("2023-10_5"))
    d = load()
    which = sys.argv[3].split(",") if len(sys.argv) > 3 else ["realsim", "commonroad"]
    steer = {SIM_KIND: (d.get("free", {}).get("steer_delay") or 0.0)}
    if "commonroad" in which:
        steer["commonroad"] = json.load(open(V.JS))["closed_free"]["steer_delay"] or 0.0
    mk_all = dict(commonroad=lambda: sim.calibrated_world(delays=dict(steer=steer["commonroad"]), grade_by_idx=grade),
                  **{SIM_KIND: lambda: real_world(grade, steer[SIM_KIND])})
    worlds = {k: mk_all[k] for k in which}
    runs = d.get("runs", [])
    for wname, mk in worlds.items():
        for R in ("R1", "R6"):
            if any(r["sim"] == wname and r["rung"] == R for r in runs):
                continue
            pt = os.path.join(REPO, "experiments/golf/results/models", f"{R}_s0_roll25.pt")
            model = load_model(pt, cfg_of_model(pt, CFG), mppi.DEV)
            torch.manual_seed(0)
            res, log = mppi.episode(model, mk(), tr, N=25, seed=0, record=True, i0=info["start_idx"], v_armed=10.0, lam=LAM, mode="A",
                                    sigma=tuple(NOISE * s for s in mppi.SIGMA_FORCE))
            row = dict(sim=wname, rung=R, steer_delay=steer[wname], **{k: res[k] for k in ("status", "finished", "progress_m", "progress_frac",
                       "lap_time", "time_s", "ey_mean", "ey_max", "v_err_mean", "wall_s")})
            np.savez(os.path.join(OUT, f"log_{wname}_{R}.npz"), **{k: np.array(v) for k, v in log.items()})
            print(json.dumps(row, ensure_ascii=False), flush=True)
            d = load(); d.setdefault("runs", []).append(row)
            d.setdefault("setup", {})
            d["setup"].update(track_lap="2023-10_5", lap_record=info["lap_time"], start_idx=info["start_idx"], lam=LAM, noise=NOISE, K=1024, N=25,
                              off_track_m=mppi.OFF_TRACK_M)
            d["setup"].setdefault("steer_delay", {}).update(steer)
            save(d)
            runs = d["runs"]


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    {"free": stage_free, "run": stage_run}[sys.argv[2]]()
