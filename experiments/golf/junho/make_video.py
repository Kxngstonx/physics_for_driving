"""주행 기록(npz) 여러 개를 한 트랙 위에 겹쳐 그린 GIF.
사용: python make_video.py REPO OUT.gif "제목" label1=path1.npz label2=path2.npz ...
왼쪽: 전체 트랙과 지나온 길, 가운데 위: 선두 차를 따라가는 확대 화면, 아래: 위치별 속도(계획 vs 실제)."""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager, patches, transforms
from matplotlib.animation import FuncAnimation, PillowWriter

for f in ("Malgun Gothic", "NanumGothic", "AppleGothic"):
    if any(f == x.name for x in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = f; break
plt.rcParams["axes.unicode_minus"] = False

sys.argv.insert(2, "run")
REPO = sys.argv[1]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import drive_compare as DC  # 경로 설정·트랙 재사용
OUT, TITLE = sys.argv[3], sys.argv[4]
runs = [a.split("=", 1) for a in sys.argv[5:]]
DT, EVERY, FPS = 0.04, 10, 10            # 0.4초마다 한 장, 초당 10장 → 실제의 4배속
COL = ["#2563eb", "#ea580c", "#16a34a", "#9333ea"]

tr, grade, info = DC.V.recorded_track(DC.V.lap_info("2023-10_5"))
x, y, psi, wl, wr = (getattr(tr, a).cpu().numpy() for a in ("x", "y", "psi", "wl", "wr"))
nx, ny = -np.sin(psi), np.cos(psi)
L = np.c_[x + wl * nx, y + wl * ny]; R = np.c_[x - wr * nx, y - wr * ny]
ds = np.hypot(np.diff(x, append=x[:1]), np.diff(y, append=y[:1])); S = np.cumsum(ds) - ds[0]
i0 = info["start_idx"]
def prog(idx):  # 출발점부터 진행 거리 (m)
    d = S[np.asarray(idx, int) % len(S)] - S[i0]
    return np.where(d < -50, d + S[-1] + ds[-1], d)

logs = []
for lab, p in runs:
    d = np.load(p)
    pr = prog(d["idx"])
    pr = np.maximum.accumulate(np.where(np.arange(len(pr)) < 50, np.minimum(pr, 60), pr))  # 출발 직후 이음매 튐 방지
    logs.append(dict(lab=lab, X=d["X"], Y=d["Y"], psi=d["psi"], v=np.hypot(d["vx"], d["vy"]), ey=d["ey"], pr=pr,
                     vref=d["vref_plan"] if "vref_plan" in d.files else None, n=len(d["X"])))
T = max(l["n"] for l in logs)
frames = list(range(0, T, EVERY)) + [T - 1] * 15

fig = plt.figure(figsize=(13, 7.2), dpi=80)
gs = fig.add_gridspec(2, 2, width_ratios=[1.15, 1], height_ratios=[1.25, 1], wspace=0.12, hspace=0.25)
axM, axZ, axV = fig.add_subplot(gs[:, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 1])
for ax in (axM, axZ):
    ax.fill(np.r_[L[:, 0], R[::-1, 0]], np.r_[L[:, 1], R[::-1, 1]], color="#d4d4d8", zorder=0)
    ax.plot(*L.T, color="#52525b", lw=0.8); ax.plot(*R.T, color="#52525b", lw=0.8)
    ax.plot(x, y, color="white", lw=0.6, ls="--", zorder=1)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
axM.plot(x[i0], y[i0], "k|", ms=18, mew=3)
for s0 in range(0, 2601, 500):
    j = int(np.argmin(np.abs(prog(np.arange(len(S))) - s0)))
    axM.annotate(f"{s0} m", (x[j], y[j]), xytext=(x[j] + 25 * nx[j], y[j] + 25 * ny[j]), fontsize=8, color="#3f3f46", ha="center")
vmax = max(l["v"].max() for l in logs) + 2
if logs[0]["vref"] is not None:
    vp = logs[0]["vref"]; o = np.argsort(prog(np.arange(len(vp))))
    axV.plot(prog(np.arange(len(vp)))[o], vp[o], color="#a1a1aa", lw=1.2, label="계획 속도")
axV.set_xlim(0, S[-1]); axV.set_ylim(0, vmax); axV.set_xlabel("진행 거리 (m)"); axV.set_ylabel("속도 (m/s)"); axV.grid(alpha=0.3)

art = []
for c, l in zip(COL, logs):
    trail, = axM.plot([], [], color=c, lw=1.6, label=l["lab"])
    ztrail, = axZ.plot([], [], color=c, lw=2)
    car = patches.Rectangle((-2.2, -0.9), 4.4, 1.8, color=c, zorder=5)
    zcar = patches.Rectangle((-2.2, -0.9), 4.4, 1.8, color=c, zorder=5)
    axM.add_patch(car); axZ.add_patch(zcar)
    vline, = axV.plot([], [], color=c, lw=1.4, label=l["lab"])
    art.append((trail, ztrail, car, zcar, vline))
axM.legend(loc="lower left", fontsize=10); axV.legend(loc="lower right", fontsize=8)
txt = axM.set_title("", fontsize=11, loc="left")
fig.suptitle(TITLE, fontsize=13, x=0.02, ha="left")
axZ.set_title("확대 (선두 차 기준 ±45 m), 차 크기 실제 비율", fontsize=9, loc="left")


def draw(k):
    lines = []
    lead = None
    for (trail, ztrail, car, zcar, vline), l in zip(art, logs):
        j = min(k, l["n"] - 1)
        trail.set_data(l["X"][:j + 1], l["Y"][:j + 1]); ztrail.set_data(l["X"][max(0, j - 100):j + 1], l["Y"][max(0, j - 100):j + 1])
        for a, p in ((car, axM), (zcar, axZ)):
            sc = 3.0 if a is car else 1.0
            a.set_width(4.4 * sc); a.set_height(1.8 * sc); a.set_xy((-2.2 * sc, -0.9 * sc))
            a.set_transform(transforms.Affine2D().rotate(l["psi"][j]).translate(l["X"][j], l["Y"][j]) + p.transData)
        vline.set_data(l["pr"][:j + 1], l["v"][:j + 1])
        ended = k >= l["n"] - 1
        st = ("끝" if ended else "")
        lines.append(f"{l['lab']}: {l['pr'][j]:6.0f} m  {l['v'][j] * 3.6:5.1f} km/h  횡오차 {l['ey'][j]:+5.2f} m {st}")
        if lead is None or l["pr"][j] > lead[0]:
            lead = (l["pr"][j], l["X"][j], l["Y"][j])
    axZ.set_xlim(lead[1] - 45, lead[1] + 45); axZ.set_ylim(lead[2] - 30, lead[2] + 30)
    txt.set_text(f"t = {min(k, T - 1) * DT:6.1f} s\n" + "\n".join(lines))
    return []


anim = FuncAnimation(fig, draw, frames=frames, blit=False)
anim.save(OUT, writer=PillowWriter(fps=FPS))
print("saved", OUT, os.path.getsize(OUT) / 1e6, "MB")
