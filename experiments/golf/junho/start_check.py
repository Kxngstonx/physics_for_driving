import os, sys, numpy as np
sys.path[:0] = [os.path.join(os.getcwd(), "physics_for_driving"), os.path.join(os.getcwd(), "physics_for_driving", "experiments", "golf"), os.getcwd()]
os.chdir("physics_for_driving")
import validate_sim as V, speed_profile as SP
tr, grade, info = V.recorded_track(V.lap_info("2023-10_5"))
x, y, vr = (getattr(tr, a).cpu().numpy() for a in ("x", "y", "vref"))
v, k = SP.profile(x, y, vr, 9.0)
vp, _ = SP.profile(x, y, np.full_like(vr, 99.0), 9.0)
n = len(x); print("n", n, "i0", info["start_idx"])
for i in list(range(n - 20, n, 5)) + list(range(0, 400, 20)):
    print(i, "vrec", round(float(vr[i]), 1), "phys", round(float(vp[i]), 1), "plan", round(float(v[i]), 1), "R", round(1 / max(abs(float(k[i])), 1e-4)))
