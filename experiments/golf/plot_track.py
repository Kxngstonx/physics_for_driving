"""썬더힐 경로 3종·경계·목표 속도 그림 → experiments/golf/results/tracks/thunderhill.png"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "tracks")
b = np.load(os.path.join(D, "thunderhill_boundary.npz"))
P = {k: np.load(os.path.join(D, f"thunderhill_{k}.npz")) for k in ["recorded", "mincurv"]}
fig, ax = plt.subplots(1, 2, figsize=(16, 7), gridspec_kw=dict(width_ratios=[1, 1.4]))
r = P["recorded"]
ax[0].plot(*r["inner"].T, color="0.4", lw=0.8, label="boundary (recorded line ±5 m, assumed)")
ax[0].plot(*r["outer"].T, color="0.4", lw=0.8)
n = np.stack([-np.sin(np.arctan2(*np.gradient(b["ref"], axis=0)[:, ::-1].T)), np.cos(np.arctan2(*np.gradient(b["ref"], axis=0)[:, ::-1].T))], 1)
ax[0].fill(*(b["ref"] + b["gps_left"][:, None] * n).T, color="C2", alpha=0.0)
ax[0].plot(*(b["ref"] + b["gps_left"][:, None] * n).T, color="C2", lw=0.5, label="GPS envelope (not used)")
ax[0].plot(*(b["ref"] + b["gps_right"][:, None] * n).T, color="C2", lw=0.5)
ax[0].plot(r["x"], r["y"], color="C0", lw=1, label="recorded (= center)")
ax[0].plot(P["mincurv"]["x"], P["mincurv"]["y"], color="C3", lw=1, ls="--", label="min curvature")
ax[0].plot(r["x"][0], r["y"][0], "ko", label="start")
ax[0].set_aspect("equal")
ax[0].legend(fontsize=8)
ax[0].set_xlabel("E (m)")
ax[0].set_ylabel("N (m)")
for k, c in [("recorded", "C0"), ("mincurv", "C3")]:
    ax[1].plot(P[k]["s"], P[k]["vref"], color=c, lw=1, label=f"{k} vref (mu 1.2, +4.3/-5.1)")
ax[1].set_xlabel("s (m)")
ax[1].set_ylabel("target speed (m/s)")
ax[1].legend(fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(D, "thunderhill.png"), dpi=110)
print("saved")
