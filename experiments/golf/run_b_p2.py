"""[11/25] MPPI 랩 비교 (A·B 방식) 를 P2 세계에서: "예측이 정확한 모델일수록 주행도 잘하는가".

  python experiments/golf/run_b_p2.py tune                 # MPPI 설정 고르기: λ × 잡음 9개 × 모델 4개 (시드 0)
  python experiments/golf/run_b_p2.py main                 # 고른 설정으로 모델 × 시드 0·1·2
  python experiments/golf/run_b_p2.py report               # → results/mppi_p2_<방식><계획 스텝>.md (개루프 1초 RMSE 와 나란히)
  (환경 변수 B_MODELS="R5" 로 이 프로세스가 돌릴 모델 고르기 → 모델별로 동시에 돌릴 수 있음, 결과 results/mppi_p2_<방식><계획 스텝>_<모델>.json. B_MODE A|B (기본 A), B_N 계획 스텝 (기본 25).
   B_MAXT 로 최대 주행 시간 s)

조건 (run_b.py 와 같은 B 방식, 세계만 P2)
  세계   : p2.P2World.from_fit() (results/sim/p2_fit.json), 브레이크 지연 (fit_delays), 조향 지연 0 (exp_1125_p2_validation.md 4절)
  트랙   : validate_sim.recorded_track() (검증 랩 2023-10_8 기록선, 경계 폭, 위치별 경사), 그 랩의 출발 위치에서 정지 출발
  MPPI   : mode B (경로·목표 속도 없이 경계 안 전진 보상), 후보 1024, 계획 1초 (25스텝), 이탈 0.9 m
  모델   : results/models_1125/<모델>_s<시드>_roll25.pt (11/25 재학습, GOLF_THUNDERHILL_G)
  튜닝   : 모델 하나에 맞추지 않도록 4개 모델 합산 (완주 수 → 랩타임 중앙값 → 진행률 중앙값, tune_mppi.py 와 같은 순위)
이미 끝난 조합은 건너뛴다 → 중단돼도 다시 실행하면 이어서 함.
"""
import itertools
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
from deep_dynamics.golf import mppi  # noqa: E402

RES = os.path.join(HERE, "results")
MODELS = os.path.join(RES, "models_1125")
OUT = os.path.join(RES, "b_p2.json")
LAMS = (0.1, 0.3, 1.0)
NOISE = (0.5, 1.0, 2.0)
RUNGS = tuple(os.environ.get("B_MODELS", "R1,R4,R2,R5").split(","))
N = int(os.environ.get("B_N", 25))
MODE = os.environ.get("B_MODE", "A")  # [11/25] A 기본: 팀원도 B·1초는 54회 모두 실패 (golf_results_phase2.md 8절)
TUNE_RUN = "2023-10_5"
REC = 108.20 if MODE == "A" else 107.28  # A: 튜닝 랩 2023-10_5 기록, B: 검증 랩 2023-10_8 기록
WORLD = os.environ.get("B_WORLD", "p2")  # [11/26] 정답 세계: p2 (기본, 사용자 결정) | p1 (비교용)
PREFIX = f"mppi_{WORLD}_{MODE}{N}"
MINE = os.path.join(RES, f"{PREFIX}_{'_'.join(RUNGS)}.json")  # 이 프로세스가 쓰는 파일 (모델별로 나눠 동시에 돌려도 안 겹침)


def load():
    """같은 (방식, 계획 길이) 의 파일을 모두 합쳐 읽음."""
    import glob
    runs = []
    for p in sorted(glob.glob(os.path.join(RES, f"{PREFIX}_*.json"))):
        runs += json.load(open(p))["runs"]
    return {"runs": runs}


def append(row):
    d = json.load(open(MINE)) if os.path.exists(MINE) else {"runs": []}
    d["runs"].append(row)
    tmp = MINE + ".tmp"
    json.dump(d, open(tmp, "w"), indent=1)
    os.replace(tmp, MINE)


def lap(rung, seed, lam, noise, tag):
    import torch
    import validate_sim as V
    from deep_dynamics.golf.configs import GOLF_THUNDERHILL_G as CFG
    from deep_dynamics.golf.ladder import load_model
    V.WORLD = WORLD
    # A: 기록선 + 튜닝 랩 2023-10_5 의 위치별 기록 속도, 그 랩 출발 위치 (팀원 tune_mppi.py 와 같음). B: 검증 랩, 목표 속도 안 씀
    tr, grade, info = V.recorded_track(V.lap_info(TUNE_RUN)) if MODE == "A" else V.recorded_track()
    model = load_model(os.path.join(MODELS, f"{rung}_s{seed}_roll25.pt"), CFG, mppi.DEV)
    w = V.make_world(delays=dict(steer=0.0), grade_by_idx=grade)
    maxt = float(os.environ.get("B_MAXT", 0)) or None
    res, log = mppi.episode(model, w, tr, N=N, seed=seed, i0=info["start_idx"], v_armed=10.0, mode=MODE, lam=lam,
                            sigma=tuple(noise * s for s in mppi.SIGMA_FORCE), max_time=maxt)
    row = dict(kind=tag, rung=rung, seed=seed, lam=lam, noise=noise,
               fail_track_m=None if res["finished"] else (float(log["idx"][-1]) if log["idx"] else None),
               ms_per_step=1000.0 * res["wall_s"] / max(len(log["vx"]), 1),
               **{k: res[k] for k in ("status", "finished", "progress_m", "progress_frac", "lap_time", "time_s", "ey_mean", "ey_max", "wall_s")})
    print(json.dumps(row, ensure_ascii=False), flush=True)
    del model
    return row


def done(d, **k):
    return any(all(r.get(a) == b for a, b in k.items()) for r in d["runs"])


def run(tag, combos):
    for rung, seed, lam, noise in combos:
        if done(load(), kind=tag, rung=rung, seed=seed, lam=lam, noise=noise):
            continue
        append(lap(rung, seed, lam, noise, tag))


def summarize(runs):
    laps = [r["lap_time"] if r["finished"] else math.inf for r in runs]
    return dict(n=len(runs), finished=sum(r["finished"] for r in runs), median_lap=float(np.median(laps)),
                median_progress=float(np.median([r["progress_frac"] for r in runs])))


TUNE_RUNGS = ("R1", "R4", "R2", "R5")  # 설정 고르기에 쓰는 모델 (고정: 모델별 프로세스에서도 같은 설정을 고르게)


def chosen():
    d = load()
    rows = []
    for lam, nz in itertools.product(LAMS, NOISE):
        R = [r for r in d["runs"] if r["kind"] == "tune" and r["lam"] == lam and r["noise"] == nz and r["rung"] in TUNE_RUNGS]
        if len(R) == len(TUNE_RUNGS):
            rows.append(dict(lam=lam, noise=nz, **summarize(R)))
    best = min(rows, key=lambda z: (-z["finished"], z["median_lap"], -z["median_progress"])) if rows else None
    return best, rows


def report():
    d = load()
    best, rows = chosen()
    ol = json.load(open(os.path.join(RES, "eval_1125", "eval_roll25.json")))["openloop"]["all"]
    L = [f"# MPPI 랩 비교 ({MODE} 방식, {WORLD.upper()} 세계)", "", f"{'튜닝 랩 2023-10_5 기록선·기록 속도' if MODE == 'A' else '검증 랩 2023-10_8'} 출발 위치에서 정지 출발, 계획 {N * mppi.DT:g}초, 후보 1024, 조향 지연 0. 기록 랩타임 {REC} s.", "",
         "## MPPI 설정 고르기 (시드 0, 모델 4개 합산)", "", "| λ | 잡음 배수 | 완주 | 랩타임 중앙값 (s) | 진행률 중앙값 |", "|---|---|---|---|---|"]
    for z in rows:
        L.append(f"| {z['lam']}{' **선택**' if z is best else ''} | {z['noise']} | {z['finished']}/{z['n']} | {z['median_lap']:.2f} | {z['median_progress']:.3f} |")
    L += ["", "## 본 실험: 고른 설정, 시드 0·1·2", "",
          "| 모델 | 개루프 1초 RMSE vx / vy / r | 완주 | 랩타임 (완주한 것) | 진행률 평균 | 횡오차 평균 / 최대 (m) | 실패 위치 | ms/스텝 |",
          "|---|---|---|---|---|---|---|---|"]
    for rung in sorted({r["rung"] for r in d["runs"] if r["kind"] == "main"}):
        R = [r for r in d["runs"] if r["kind"] == "main" and r["rung"] == rung]
        if not R:
            continue
        e = [ol[k]["1.00s"] for k in ol if k.startswith(rung + "_s")]
        rm = " / ".join(f"{np.mean([x[q] for x in e]):.3f}" for q in ("vx", "vy", "r"))
        laps = [r["lap_time"] for r in R if r["finished"]]
        lt = ", ".join(f"{x:.2f}" for x in laps) if laps else "–"
        fails = ", ".join(f"{r['status']} {r['progress_frac']:.0%}" for r in R if not r["finished"]) or "–"
        L.append(f"| {rung} | {rm} | {len(laps)}/{len(R)} | {lt} | {np.mean([r['progress_frac'] for r in R]):.1%} | "
                 f"{np.mean([r['ey_mean'] for r in R]):.2f} / {max(r['ey_max'] for r in R):.2f} | {fails} | {np.mean([r['ms_per_step'] for r in R]):.0f} |")
    open(os.path.join(RES, PREFIX + ".md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "tune":
        run("tune", [(R, 0, lam, nz) for lam, nz in itertools.product(LAMS, NOISE) for R in RUNGS])
    elif cmd == "main":
        best, rows = chosen()
        assert best and len(rows) == len(LAMS) * len(NOISE), "튜닝이 아직 다 안 끝남"
        run("main", [(R, s, best["lam"], best["noise"]) for s in (0, 1, 2) for R in RUNGS])
    elif cmd == "check":  # [11/26] 팀원 11/19 A 방식 선택 설정(λ 0.3, 잡음 ×0.5)으로 시드 0 한 랩씩: 완주하는지
        run("check", [(R, 0, 0.3, 0.5) for R in RUNGS])
    elif cmd == "one":  # 시간 재기: one R5 0.3 1.0
        print(lap(sys.argv[2], 0, float(sys.argv[3]), float(sys.argv[4]), "probe"))
    else:
        report()
