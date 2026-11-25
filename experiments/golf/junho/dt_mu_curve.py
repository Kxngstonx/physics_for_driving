import os, sys, math
sys.path[:0]=[os.path.join(os.getcwd(),"physics_for_driving"), os.getcwd()]
import numpy as np
from deep_dynamics.golf import sim
import dt_world as DW
def run(mu, dmax, v=20.0, T=15.0):
    w = DW.DTWorld(sim.Actuator, delays=None, cap=False, lsd=False, main_units=False); w.p.mu_scale = mu
    w.reset(0,0,0,v); best=(0,0,0); n=int(T/0.04)
    for k in range(n):
        s = w.step(sim.road_load(w.state[0]) + w.p.m*0.5*(v-w.state[0]), dmax*k/n)
        b = math.atan2(s[1], s[0])
        if not np.isfinite(s[0]) or abs(b) > 0.15: return best + ("spin",)
        if abs(s[0]*s[2]) > best[0]: best = (round(abs(s[0]*s[2]),2), round(dmax*k/n,3), round(b,3))
    return best + ("end",)
for mu in (1.0, 1.2, 1.4, 1.578):
    print(mu, "ramp0.25", run(mu, 0.25), "ramp0.6", run(mu, 0.6), flush=True)
