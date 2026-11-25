"""목표 속도 배율 올리기 (A 방식, 신영 님 ladder_race 와 같은 생각): vref × 배율, 나머지 조건은 drive_compare 와 같음.
사용: python push_speed.py REPO realsim 1.03,1.06,1.10  → realsim_out/push.json, log_push_<sim>_<rung>_x<배율>.npz"""
import json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.argv.insert(2, "run")
import drive_compare as DC  # noqa: E402  (경로 설정·세계 생성 재사용)
from deep_dynamics.golf import mppi  # noqa: E402
JS = os.path.join(DC.OUT, "push.json")
wname, scales = sys.argv[3], [float(x) for x in sys.argv[4].split(",")]
import torch
from deep_dynamics.golf.configs import GOLF_THUNDERHILL as CFG, cfg_of_model
from deep_dynamics.golf.ladder import load_model
d0 = DC.load()
steer = d0["setup"]["steer_delay"][wname]
for sc in scales:
    for R in ("R1", "R6"):
        res_all = json.load(open(JS)) if os.path.exists(JS) else []
        if any(r["sim"] == wname and r["rung"] == R and r["scale"] == sc for r in res_all):
            continue
        tr, grade, info = DC.V.recorded_track(DC.V.lap_info("2023-10_5"))
        tr.vref = tr.vref * sc
        pt = os.path.join(DC.REPO, "experiments/golf/results/models", f"{R}_s0_roll25.pt")
        model = load_model(pt, cfg_of_model(pt, CFG), mppi.DEV)
        w = DC.real_world(grade, steer) if wname == "realsim" else DC.sim.calibrated_world(delays=dict(steer=steer), grade_by_idx=grade)
        res, log = mppi.episode(model, w, tr, N=25, seed=0, record=True, i0=info["start_idx"], v_armed=10.0, lam=DC.LAM, mode="A",
                                sigma=tuple(DC.NOISE * s for s in mppi.SIGMA_FORCE))
        row = dict(sim=wname, rung=R, scale=sc, **{k: res[k] for k in ("status", "finished", "progress_m", "progress_frac", "lap_time", "ey_mean", "ey_max", "v_err_mean", "wall_s")})
        np.savez(os.path.join(DC.OUT, f"log_push_{wname}_{R}_x{sc}.npz"), **{k: np.array(v) for k, v in log.items()})
        print(json.dumps(row, ensure_ascii=False), flush=True)
        res_all = json.load(open(JS)) if os.path.exists(JS) else []
        res_all.append(row); json.dump(res_all, open(JS, "w"), indent=1)
