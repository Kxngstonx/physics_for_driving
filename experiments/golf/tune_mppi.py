"""MPPI 모델 중립 튜닝 (docs/experiment_plan_golf.md 9-2절).

  python experiments/golf/tune_mppi.py run A      # A 방식: 9개 설정 × R1–R6 (시드 0)
  python experiments/golf/tune_mppi.py run B      # B 방식: 같은 9개 설정, 계획 길이 1초
  python experiments/golf/tune_mppi.py select     # 설정 고르기 → mppi_tune.json 의 "chosen", mppi_tune.md

설정: 온도 λ {0.1, 0.3, 1.0} × 잡음 배수 {0.5, 1, 2} (기본 잡음 힘 1000 N, 조향 0.06 rad). 후보 수 K 1024.
기준점 [변경 11/19, 사용자 결정]: 기록선 + 다른 실제 랩(2023-10_5, 108.2 s)의 위치별 기록 속도, 그 랩의 실제 출발 위치에서 정지 출발.
  이전 기준점(마찰 계수 1.2 로 계산한 목표 속도, 한 바퀴 82.9 s = 기록보다 23% 빠름)에서는 설정·모델에 상관없이 49회 모두
  초반에 실패해 비교가 되지 않았다 (그 결과는 mppi_tune_mu1.2_aborted.json). 검증 랩(2023-10_8)과 다른 랩이라
  "기록에 맞춰 튜닝해 검증을 통과시키는" 순환을 피한다.
시뮬레이터: calibrated_world (calibration.json, 지연), 조향 지연은 validation.json 의 closed_free 선택값 (모델 없는 운전자 검증).
순위: 6개 모델의 완주 수 → 랩타임 중앙값 → 진행률 중앙값 (모델 하나에 맞추지 않음, 문서 4-5절).
이미 끝난 (방식, 설정, 모델) 은 건너뛴다 → 중단돼도 다시 실행하면 이어서 함.
"""
import itertools
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from deep_dynamics.golf import mppi, sim  # noqa: E402

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
OUT = os.path.join(RES, "mppi_tune.json")
TUNE_RUN = "2023-10_5"
LAMS = (0.1, 0.3, 1.0)
NOISE = (0.5, 1.0, 2.0)
RUNGS = ("R1", "R2", "R3", "R4", "R5", "R6")


def load():
    return json.load(open(OUT)) if os.path.exists(OUT) else {"runs": []}


def save(d):
    tmp = OUT + ".tmp"
    json.dump(d, open(tmp, "w"), indent=1)
    os.replace(tmp, OUT)


def steer_delay():
    v = json.load(open(os.path.join(RES, "sim", "validation.json")))
    return v["closed_free"]["steer_delay"]


def tune_track():
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import validate_sim as V
    return V.recorded_track(V.lap_info(TUNE_RUN))


def run(mode):
    import torch
    from deep_dynamics.golf.configs import GOLF_THUNDERHILL as CFG
    from deep_dynamics.golf.ladder import load_model
    tr, grade, info = tune_track()
    sd = steer_delay()
    for lam, nz in itertools.product(LAMS, NOISE):
        for R in RUNGS:
            d = load()
            if any(r["mode"] == mode and r["lam"] == lam and r["noise"] == nz and r["rung"] == R for r in d["runs"]):
                continue
            model = load_model(os.path.join(RES, "models", f"{R}_s0_roll25.pt"), CFG, mppi.DEV)
            w = sim.calibrated_world(delays=dict(steer=sd), grade_by_idx=grade)
            res, _ = mppi.episode(model, w, tr, seed=0, record=False, i0=info["start_idx"], v_armed=10.0, lam=lam, mode=mode,
                                  sigma=tuple(nz * s for s in mppi.SIGMA_FORCE))
            row = dict(mode=mode, lam=lam, noise=nz, rung=R, **{k: res[k] for k in ("status", "finished", "progress_frac", "lap_time",
                                                                                   "ey_mean", "ey_max", "wall_s")})
            print(json.dumps(row, ensure_ascii=False), flush=True)
            d = load()
            d["runs"].append(row)
            save(d)
            del model
            torch.cuda.empty_cache()


def summarize(runs):
    fin = sum(r["finished"] for r in runs)
    laps = [r["lap_time"] if r["finished"] else math.inf for r in runs]
    return dict(n=len(runs), finished=fin, median_lap=float(np.median(laps)), median_progress=float(np.median([r["progress_frac"] for r in runs])))


def select():
    d = load()
    L = ["# MPPI 튜닝 (계획서 9-2절)", "", f"기준: 기록선 + {TUNE_RUN} 랩의 기록 속도, 정지 출발. 조향 지연 {steer_delay()} s. "
         "순위: 완주 수 → 랩타임 중앙값 → 진행률 중앙값", ""]
    d["chosen"] = {}
    for mode in ("A", "B"):
        rows = []
        for lam, nz in itertools.product(LAMS, NOISE):
            R = [r for r in d["runs"] if r["mode"] == mode and r["lam"] == lam and r["noise"] == nz]
            if len(R) == len(RUNGS):
                rows.append(dict(lam=lam, noise=nz, **summarize(R)))
        if not rows:
            continue
        best = min(rows, key=lambda z: (-z["finished"], z["median_lap"], -z["median_progress"]))
        d["chosen"][mode] = dict(lam=best["lam"], noise=best["noise"])
        L += [f"## {mode} 방식", "", "| λ | 잡음 배수 | 완주 (6개 중) | 랩타임 중앙값 (s) | 진행률 중앙값 |", "|---|---|---|---|---|"]
        for z in rows:
            mark = " **선택**" if z is best else ""
            L.append(f"| {z['lam']}{mark} | {z['noise']} | {z['finished']} | {z['median_lap']:.2f} | {z['median_progress']:.3f} |")
        L.append("")
    save(d)
    open(os.path.join(RES, "mppi_tune.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    {"run": lambda: run(sys.argv[2]), "select": select}[sys.argv[1]]()
