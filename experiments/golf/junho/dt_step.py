"""dt 조향 계단 응답 (30 m/s, 0.03 rad): 요 속도 초과량·정착. 계수 조합별."""
import os, sys, math
sys.path[:0]=[os.path.join(os.getcwd(),"physics_for_driving"), os.getcwd()]
import numpy as np
from deep_dynamics.golf import sim
import dt_world as DW
def resp(shape, mu, v=30.0, d=0.03, lsd=False):
    w = DW.DTWorld(sim.Actuator, delays=None, cap=False, lsd=lsd, main_units=False)
    w.p = DW.DTG.params(**shape); w.p.mu_scale = mu; w.p.lsd_mode = None
    w.reset(0,0,0,v); rs=[]
    for k in range(75):
        s = w.step(sim.road_load(w.state[0]) + w.p.m*0.5*(v-w.state[0]), d if k>=5 else 0.0); rs.append(s[2])
    rs=np.array(rs); fin=rs[-10:].mean()
    return dict(peak=round(rs.max(),4), final=round(fin,4), overshoot=round(rs.max()/fin-1,3), t_peak=round((rs.argmax()-5)*0.04,2), min_after=round(rs[rs.argmax():].min()/fin,3))
for tag, sh, mu in (("published mu1", dict(mu_dfz=-0.13,c2=1.14), 1.0), ("fitted mu1", dict(mu_dfz=-0.25,c2=1.3065), 1.0), ("fitted mu1.145", dict(mu_dfz=-0.25,c2=1.3065), 1.145)):
    print(tag, resp(sh, mu), "0.06rad", resp(sh, mu, d=0.06), flush=True)
