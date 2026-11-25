"""곡률 속도 계획으로 R1·R6 주행. 사용: python drive_profile.py REPO SIM AY1,AY2,...  → realsim_out/profile.json
나머지 조건(트랙, 출발, MPPI λ·잡음·K·N, 조향 지연, 이탈 0.9 m)은 drive_compare 와 같음."""
import json, os, sys
import numpy as np
sys.argv.insert(2, "run")
import drive_compare as DC, speed_profile as SP
from deep_dynamics.golf import mppi
import torch
from deep_dynamics.golf.configs import GOLF_THUNDERHILL as CFG, cfg_of_model
from deep_dynamics.golf.ladder import load_model
JS = os.path.join(DC.OUT, os.environ.get("PROFILE_JS", "profile.json"))  # 병렬 실행 시 파일을 나눔
SEEDS = [int(a) for a in os.environ.get("SEEDS", "0").split(",")]  # MPPI 무작위 시드
MU_MULT = float(os.environ.get("MU_MULT", "1"))  # CommonRoad 마찰 민감도: 보정 배율 × MU_MULT
MU_KW = dict(mu_scale=DC.sim.load_calibration()["mu_scale"] * MU_MULT) if MU_MULT != 1.0 else {}
START_FIX = int(os.environ.get("START_FIX", "0"))  # 출발 구간 계획 속도 수정 (트랙 인덱스 0~N)
MODEL_SEED = int(os.environ.get("MODEL_SEED", "0"))  # 모델 학습 시드
DRIVE_GAIN = float(os.environ.get("DRIVE_GAIN", "1"))  # 시뮬레이터 구동력 이득 (양수 힘만, 상한 적용 뒤): sim_gain.py 로 맞춤


def gain_world(w, k):
    """World.step 앞에서 힘 범위를 먼저 자르고 양수 힘만 ×k (시뮬레이터 쪽 보정. 모델이 보는 명령은 그대로)."""
    from deep_dynamics.golf import sim as _sim
    orig = w.step
    def step(F, d):
        F = min(max(F, _sim.F_BRAKE_MIN), float(_sim.force_cap(w.x[3])))
        return orig(F * k if F > 0 else F, d)
    w.cap = False
    w.step = step
    return w


wname, ays = sys.argv[3], [float(a) for a in sys.argv[4].split(",")]
steer = DC.load()["setup"]["steer_delay"][wname]
for seed in SEEDS:
  for AY in ays:
    for R in ("R1", "R6"):
        rows = json.load(open(JS)) if os.path.exists(JS) else []
        done = [r for f in os.listdir(DC.OUT) if f.startswith("profile") and f.endswith(".json") for r in json.load(open(os.path.join(DC.OUT, f)))]
        if any(r["sim"] == wname and r["rung"] == R and r["ay_lim"] == AY and r.get("seed", 0) == seed for r in done):
            continue
        tr, grade, info = DC.V.recorded_track(DC.V.lap_info("2023-10_5"))
        x, y, vr = (getattr(tr, a).cpu().numpy() for a in ("x", "y", "vref"))
        v, _ = SP.profile(x, y, vr, AY)
        if START_FIX:  # 기록 랩이 정지 출발이라 0~START_FIX m 의 기록 속도가 5 m/s 부터 → 그 구간은 물리 계획만 (달리며 지나가는 랩처럼)
            vp, _ = SP.profile(x, y, np.full_like(vr, 99.0), AY)
            v = v.copy(); v[:START_FIX] = vp[:START_FIX]
        tr.vref = torch.tensor(v, dtype=tr.vref.dtype, device=tr.vref.device)
        pt = os.path.join(DC.REPO, "experiments/golf/results/models", f"{R}_s{MODEL_SEED}_roll25.pt")
        model = load_model(pt, cfg_of_model(pt, CFG), mppi.DEV)
        torch.manual_seed(seed)
        w = DC.real_world(grade, steer) if wname in ("realsim", "dt") else DC.sim.calibrated_world(delays=dict(steer=steer), grade_by_idx=grade, **MU_KW)
        if DRIVE_GAIN != 1.0:
            w = gain_world(w, DRIVE_GAIN)
        res, log = mppi.episode(model, w, tr, N=25, seed=seed, record=True, i0=info["start_idx"], v_armed=10.0, lam=DC.LAM, mode="A",
                                sigma=tuple(DC.NOISE * s for s in mppi.SIGMA_FORCE))
        row = dict(sim=wname, rung=R, ay_lim=AY, seed=seed, mu_mult=MU_MULT, drive_gain=DRIVE_GAIN, model_seed=MODEL_SEED, start_fix=START_FIX, plan_time=float(np.sum(np.hypot(np.diff(x, append=x[:1]), np.diff(y, append=y[:1])) / np.maximum(v, 1))),
                   **{k: res[k] for k in ("status", "finished", "progress_m", "lap_time", "ey_mean", "ey_max", "v_err_mean", "wall_s")})
        np.savez(os.path.join(DC.OUT, f"log_profile_{wname}_{R}_ay{AY}" + (f"_s{seed}" if seed else "") + ".npz"), vref_plan=v, **{k: np.array(vv) for k, vv in log.items()})
        print(json.dumps(row, ensure_ascii=False), flush=True)
        rows = json.load(open(JS)) if os.path.exists(JS) else []
        rows.append(row); json.dump(rows, open(JS, "w"), indent=1)
