"""dt 시뮬레이터 정상 선회 최대 옆 가속도: 속도 유지하며 조향을 천천히 늘림."""
import os, sys, math, json
sys.path[:0]=[os.path.join(os.getcwd(),"physics_for_driving"), os.getcwd()]
from deep_dynamics.golf import sim
import dt_world as DW

def max_ay(shape, v=20.0, lsd=True, rate=None):
    w = DW.DTWorld(sim.Actuator, delays=None, cap=False, lsd=lsd, main_units=False)
    w.p = DW.DTG.params(**shape); w.p.mu_scale = 1.0; w.p.lsd_mode = "active" if lsd else None
    if rate is not None: DW.STEER_RATE_MAX = rate
    w.reset(0, 0, 0, v)
    best = 0.0
    for k in range(int(20 / 0.04)):
        d = 0.004 * k * 0.04 * 25 / 10  # 0.01 rad/s
        vx, vy, r = w.state
        F = w.p.m * 2.0 * (v - vx) + sim.road_load(vx)
        w.step(F, d)
        ay = w.state[0] * w.state[2]
        if not math.isfinite(ay): break
        best = max(best, abs(ay))
    return best

for name, sh in (("published", dict(mu_dfz=-0.13, c2=1.14)), ("fitted", dict(mu_dfz=-0.25, c2=1.3065))):
    print(name, {v: round(max_ay(sh, v), 2) for v in (15.0, 25.0)}, "noLSD25", round(max_ay(sh, 25.0, lsd=False), 2), flush=True)
