"""CommonRoad 조향 지연 민감도: ×1.00 에서 지연 0, 0.08 로 R1·R6 (나머지 조건은 drive_compare 와 같음) → realsim_out/steer_sens.json"""
import json, os, sys
import numpy as np
sys.argv.insert(2, "run")
import drive_compare as DC
from deep_dynamics.golf import mppi
import torch
from deep_dynamics.golf.configs import GOLF_THUNDERHILL as CFG, cfg_of_model
from deep_dynamics.golf.ladder import load_model
JS = os.path.join(DC.OUT, "steer_sens.json")
out = []
for sd in (0.0, 0.08):
    for R in ("R1", "R6"):
        tr, grade, info = DC.V.recorded_track(DC.V.lap_info("2023-10_5"))
        pt = os.path.join(DC.REPO, "experiments/golf/results/models", f"{R}_s0_roll25.pt")
        model = load_model(pt, cfg_of_model(pt, CFG), mppi.DEV)
        torch.manual_seed(0)
        w = DC.sim.calibrated_world(delays=dict(steer=sd), grade_by_idx=grade)
        res, log = mppi.episode(model, w, tr, N=25, seed=0, record=True, i0=info["start_idx"], v_armed=10.0, lam=DC.LAM, mode="A",
                                sigma=tuple(DC.NOISE * s for s in mppi.SIGMA_FORCE))
        row = dict(sim="commonroad", rung=R, steer_delay=sd, **{k: res[k] for k in ("status", "finished", "progress_m", "lap_time", "ey_mean", "ey_max", "v_err_mean")})
        np.savez(os.path.join(DC.OUT, f"log_steer_commonroad_{R}_d{sd}.npz"), **{k: np.array(v) for k, v in log.items()})
        print(json.dumps(row), flush=True); out.append(row)
        json.dump(out, open(JS, "w"), indent=1)
