"""팀원 calibrate_sim.calibrate_dt 와 같은 절차 (v 20, 15 s, 조향 0→0.25, 미끄럼각 0.15 에서 멈춤, LSD 없음, 이분 12회)
를 맞춘 타이어 모양(calibration_dt.json 의 mu_dfz, c2)으로. 결과 mu_scale 을 calibration_dt.json 에 넣는다."""
import os, sys, math, json
sys.path[:0]=[os.path.join(os.getcwd(),"physics_for_driving"), os.getcwd()]
import numpy as np
from deep_dynamics.golf import sim
import dt_world as DW

def max_ay(mu, v=20.0, T=15.0, dmax=0.6):  # 팀원 0.25 rad 는 dt 에서 조향이 먼저 막힘 (dt_mu_curve.py) → 0.6
    w = DW.DTWorld(sim.Actuator, delays=None, cap=False, lsd=False, main_units=False)
    w.p.mu_scale = mu
    w.reset(0, 0, 0, v)
    best, n = 0.0, int(T / 0.04)
    for k in range(n):
        s = w.step(sim.road_load(w.state[0]) + w.p.m * 0.5 * (v - w.state[0]), dmax * k / n)
        if not np.isfinite(s[0]) or abs(math.atan2(s[1], s[0])) > 0.15:
            break
        best = max(best, abs(s[0] * s[2]))
    return best

target = DW._v("ay_target")
lo, hi = 0.8, 2.0
for _ in range(12):
    mid = 0.5 * (lo + hi)
    if max_ay(mid) < target: lo = mid
    else: hi = mid
mu = round(0.5 * (lo + hi), 3)
res = dict(mu_scale=mu, max_ay=max_ay(mu), max_ay_mu1=max_ay(1.0), ay_target=target)
print(res)
p = os.path.join(DW.BR, "experiments/golf/results/sim/calibration_dt.json")
cal = json.load(open(p, encoding="utf-8")); cal.update(mu_scale=mu, ay_target=target, max_ay=res["max_ay"], mu_note="dt_calib_mu.py: calibrate_dt 절차, 맞춘 타이어 모양, 조향 램프 0.6 rad")
json.dump(cal, open(p, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
