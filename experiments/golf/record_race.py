"""회의 영상(viz2d.py race)용 폐루프 주행 궤적 저장.

run_b.py·tune_mppi.py 는 궤적을 저장하지 않으므로, 같은 설정·시드 0 으로 R1–R6 을 다시 주행해 궤적을 남긴다.
  python experiments/golf/record_race.py A 25          # A 방식, tune_mppi.py 의 A 기준점(기록선 + 2023-10_5 기록 속도), chosen["A_25"]
  python experiments/golf/record_race.py B 25 50 100   # B 방식, run_b.py main 과 같은 준비, chosen["B_N"]
  python experiments/golf/record_race.py check         # 다시 주행한 결과가 원래 결과(mppi_tune.json / b_main.json)와 같은지
결과: results/viz/{A|B}_{N}__R{i}_s0_traj.npz (+ .json). 이미 있으면 건너뜀.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
from deep_dynamics.golf import mppi, sim  # noqa: E402

RES = os.path.join(HERE, "results")
VIZ = os.path.join(RES, "viz")
RUNGS = ("R1", "R2", "R3", "R4", "R5", "R6")
KEYS = ("status", "finished", "progress_frac", "lap_time")


def record(mode, N, seed=0):
    import torch
    import tune_mppi as T
    from deep_dynamics.golf.configs import GOLF_THUNDERHILL as CFG
    from deep_dynamics.golf.ladder import load_model
    os.makedirs(VIZ, exist_ok=True)
    c = json.load(open(os.path.join(RES, "mppi_tune.json")))["chosen"][f"{mode}_{N}"]
    tr, grade, info = T.tune_track() if mode == "A" else T.b_track()
    for R in RUNGS:
        stem = os.path.join(VIZ, f"{mode}_{N}__{R}_s{seed}")
        if os.path.exists(stem + ".json"):
            continue
        model = load_model(os.path.join(RES, "models", f"{R}_s{seed}_roll25.pt"), CFG, mppi.DEV)
        w = sim.calibrated_world(delays=dict(steer=T.steer_delay()), grade_by_idx=grade)
        res, log = mppi.episode(model, w, tr, N=N, seed=seed, record=True, i0=info["start_idx"], v_armed=10.0, mode=mode,
                                lam=c["lam"], sigma=tuple(c["noise"] * s for s in mppi.SIGMA_FORCE))
        np.savez(stem + "_traj.npz", **{k: np.asarray(v) for k, v in log.items()})
        json.dump(dict(res, mode=mode, rung=R, lam=c["lam"], noise=c["noise"]), open(stem + ".json", "w"), indent=1)
        print(mode, N, R, {k: res[k] for k in KEYS}, flush=True)
        del model
        torch.cuda.empty_cache()


def check():
    """원래 결과와 비교. A: mppi_tune.json 의 선택 설정 행, B: b_main.json 의 main 시드 0 행."""
    tune = json.load(open(os.path.join(RES, "mppi_tune.json")))
    bm = json.load(open(os.path.join(RES, "b_main.json")))["runs"] if os.path.exists(os.path.join(RES, "b_main.json")) else []
    for f in sorted(x for x in os.listdir(VIZ) if x.endswith(".json")):
        r = json.load(open(os.path.join(VIZ, f)))
        mode, N, R = r["mode"], r["N"], r["rung"]
        if mode == "A":
            ref = [x for x in tune["runs"] if x["mode"] == "A" and x.get("N", 25) == N and x["rung"] == R
                   and x["lam"] == r["lam"] and x["noise"] == r["noise"]]
        else:
            ref = [x for x in bm if x["kind"] == "main" and x["N"] == N and x["rung"] == R and x["seed"] == 0]
        if not ref:
            print(f, "원래 결과 없음")
            continue
        o = ref[-1]
        same = o["status"] == r["status"] and abs(o["progress_frac"] - r["progress_frac"]) < 0.01 and \
            ((o["lap_time"] is None) == (r["lap_time"] is None)) and (o["lap_time"] is None or abs(o["lap_time"] - r["lap_time"]) < 0.5)
        print(f"{'같음' if same else '다름'} {mode}_{N} {R}: 원래 {o['status']} {o['progress_frac']:.3f} {o['lap_time']} / "
              f"다시 {r['status']} {r['progress_frac']:.3f} {r['lap_time']}")


if __name__ == "__main__":
    if sys.argv[1] == "check":
        check()
    else:
        for n in sys.argv[2:]:
            record(sys.argv[1], int(n))
