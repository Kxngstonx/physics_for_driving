import os, sys, math
import numpy as np
sys.argv = [sys.argv[0], "physics_for_driving", "free"]; os.environ.setdefault("SIM_KIND", "dt"); sys.path.insert(0, os.getcwd())
import drive_compare as DC, dt_world as DW
V = DC.V
tr, grade, info = V.recorded_track(V.lap_info(V.TUNE_LAP_RUN))
P = np.c_[tr.np["x"], tr.np["y"]]; psi_t = tr.np["psi"]; vref = tr.vref.cpu().numpy()
w = DW.DTWorld(DC.sim.Actuator, delays=dict(DC.sim.load_delays(DC.DELAYS) or {}, steer=0.0), grade_by_idx=grade, main_units=False)
log = []
orig = w.step
def step(F, d):
    X, Y, psi = w.pose; vx, vy, r = w.state
    i = int(np.argmin((P[:, 0] - X) ** 2 + (P[:, 1] - Y) ** 2))
    ey = -(X - P[i, 0]) * math.sin(psi_t[i]) + (Y - P[i, 1]) * math.cos(psi_t[i])
    log.append((i, ey, vx, vref[i], vy, r, F, d, w.x[8]))
    return orig(F, d)
w.step = step
r = V.drive_free(w, tr, gains=dict(V.FREE, prev_t=0.2, kp_v=1.0), i0=info["start_idx"])
L = np.array(log)
print(r["status"], "steps", len(L), "start idx", info["start_idx"])
for k in range(0, len(L), 50):
    i, ey, vx, vr, vy, rr, F, d, dd = L[k]
    print(f"k{k:4d} i{int(i):5d} ey{ey:6.2f} vx{vx:5.1f} vref{vr:5.1f} vy{vy:5.2f} r{rr:5.2f} F{F:7.0f} dcmd{d:6.3f} delta{dd:6.3f} ay{vx*rr:5.1f}")
for k in range(len(L) - 30, len(L), 3):
    i, ey, vx, vr, vy, rr, F, d, dd = L[k]
    print(f"k{k:4d} i{int(i):5d} ey{ey:6.2f} vx{vx:5.1f} vref{vr:5.1f} vy{vy:5.2f} r{rr:5.2f} F{F:7.0f} dcmd{d:6.3f} delta{dd:6.3f} ay{vx*rr:5.1f}")
