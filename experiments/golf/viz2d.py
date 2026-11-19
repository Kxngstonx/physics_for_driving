"""회의용 2D 탑뷰 영상 — Golf · Thunderhill (11/12 share_source_team/src/iac_viz2d.py 에서 옮김).

mode predict : 시험 파일의 실제 주행 기록을 재생하면서, 매 순간 R1–R6 이 '1초 동안 어디로 갈지' 예측한 선을 차 앞에 그림.
               흰색 = 실제로 간 길. 기록된 조작을 그대로 넣음 (개루프, MPPI 없음).
mode race    : record_race.py 가 저장한 폐루프 주행을 재생. 보정된 시뮬레이터에서 R1–R6 이 각자 MPPI 로 달림.

  python experiments/golf/viz2d.py predict                         # 시험 파일 2023-10_1 한 바퀴, 2배속
  python experiments/golf/viz2d.py race --mode B --N 50            # results/viz/B_50__R*_s0 → docs/videos/race_B_2s.mp4
  --preview 300 : 영상 대신 300번째 프레임만 PNG 로 저장 (확인용)

옮기면서 바꾼 것: 트랙(측정 폭 10.9 m 의 기록선 트랙), 차 크기(Golf), 모델 6개(R1–R6, 1초 손실·시드 0),
조작 표시(앞바퀴 힘 N), 예측 길이 1초. 타이어 한계 사용률 표시는 뺐다 (IAC 의 '물리 세계' 타이어를 쓰던 것이라 Golf 에 없음).
"""
import argparse
import json
import math
import os
import sys

import matplotlib

matplotlib.use("Agg")
import imageio_ffmpeg  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import animation, font_manager  # noqa: E402
from matplotlib.collections import PolyCollection  # noqa: E402
from matplotlib.patches import Circle, Polygon  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
RES = os.path.join(REPO, "experiments", "golf", "results")
TRACK_NPZ = os.path.join(RES, "tracks", "thunderhill_recorded.npz")
VIZ = os.path.join(RES, "viz")
VID = os.path.join(REPO, "docs", "videos")

plt.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
for f in ("/mnt/c/Windows/Fonts/malgun.ttf", "/mnt/c/Windows/Fonts/malgunbd.ttf"):  # 리눅스에 한글 글꼴이 없어 Windows 글꼴을 씀
    if os.path.exists(f):
        font_manager.fontManager.addfont(f)
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

DT = 0.04
H = 8
BG, ASPH, FG, DIM = "#14161a", "#3a3e45", "#e8e8e8", "#8a8f98"
CAR_L, CAR_W = 4.26, 1.80  # Golf 7세대 공개 제원 [가정]
RUNGS = ("R1", "R2", "R3", "R4", "R5", "R6")
RUNG_KO = dict(R1="순수 물리", R2="물리+잔차", R3="신경망 타이어", R4="DD식 계수추정", R5="Stanford식 NN", R6="블랙박스")
COL = dict(R1="#4C8DF6", R2="#2FBF71", R3="#3CC6C6", R4="#B07CF2", R5="#F5883A", R6="#E05A5A")
STAT = {"finished": "완주", "off_track": "트랙 이탈", "stalled": "멈춤", "world_nan": "세계 발산", "timeout": "시간 초과"}
MODE_KO = dict(A="A 방식 · 기록선과 목표 속도 따라가기", B="B 방식 · 경계 안에서 최대한 전진")


# ---------------------------------------------------------------- 공통 그리기
class Track:
    def __init__(self, path=TRACK_NPZ):
        t = np.load(path)
        c = np.stack([t["x"], t["y"]], 1)
        n = np.stack([-np.sin(t["psi"]), np.cos(t["psi"])], 1)
        self.c = c
        self.left = c + n * t["width_l"][:, None]
        self.right = c - n * t["width_r"][:, None]

    def near(self, p, rad):
        return np.where(np.hypot(*(self.c - p).T) < rad)[0]


def to_cam(P, p0, psi0):
    """월드 좌표 → 차 기준(차 머리가 +y)."""
    d = np.asarray(P) - p0
    a = math.pi / 2 - psi0
    ca, sa = math.cos(a), math.sin(a)
    return np.stack([ca * d[..., 0] - sa * d[..., 1], sa * d[..., 0] + ca * d[..., 1]], -1)


def car_poly(x, y, psi):
    pts = np.array([[CAR_L / 2, CAR_W / 2], [CAR_L / 2, -CAR_W / 2], [-CAR_L / 2, -CAR_W / 2], [-CAR_L / 2, CAR_W / 2],
                    [CAR_L / 2 + 0.6, 0]])[[0, 4, 1, 2, 3]]
    c, s = math.cos(psi), math.sin(psi)
    return np.stack([x + c * pts[:, 0] - s * pts[:, 1], y + s * pts[:, 0] + c * pts[:, 1]], 1)


def style_ax(ax):
    ax.set_facecolor(BG)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])


class MainView:
    def __init__(self, fig, track, rect, zoom=(14.0, 1.25)):
        self.zoom = zoom
        self.ax = fig.add_axes(rect)
        style_ax(self.ax)
        self.ax.set_aspect("equal")
        self.track = track
        self.road = PolyCollection([], facecolor=ASPH, edgecolor=ASPH, lw=0.6, zorder=1)
        self.ax.add_collection(self.road)
        self.bl, = self.ax.plot([], [], color=FG, lw=1.6, zorder=2)
        self.br, = self.ax.plot([], [], color=FG, lw=1.6, zorder=2)
        self.cl, = self.ax.plot([], [], color="#6b7079", lw=0.8, ls=(0, (6, 6)), zorder=2)
        self.halfw = None
        fw, fh = fig.get_size_inches()
        self.aspect = (rect[2] * fw) / (rect[3] * fh)

    def frame(self, p0, psi0, vx):
        target = self.zoom[0] + self.zoom[1] * max(vx, 0.0)  # 세로 반폭(m): 1초 동안 갈 거리 + 여유
        self.halfw = target if self.halfw is None else self.halfw + 0.04 * (target - self.halfw)
        hw = self.halfw
        idx = self.track.near(p0, 2.2 * hw * max(1, self.aspect))
        L, R, C = (to_cam(a[idx], p0, psi0) for a in (self.track.left, self.track.right, self.track.c))
        brk = np.where(np.diff(idx) != 1)[0] + 1  # 트랙 시작점을 넘어가는 곳에서 선 끊기

        def with_nan(A):
            return np.insert(A, brk, np.nan, axis=0) if len(brk) else A

        quads = [np.array([L[i], L[i + 1], R[i + 1], R[i]]) for i in range(len(idx) - 1) if idx[i + 1] - idx[i] == 1]
        if len(idx) > 1 and idx[0] == 0 and idx[-1] == len(self.track.c) - 1:
            quads.append(np.array([L[-1], L[0], R[0], R[-1]]))
        self.road.set_verts(quads)
        for ln, A in ((self.bl, L), (self.br, R), (self.cl, C)):
            A = with_nan(A)
            ln.set_data(A[:, 0], A[:, 1])
        cy = 0.45 * hw  # 차를 아래쪽에 두고 앞을 더 넓게
        self.ax.set_xlim(-hw * self.aspect, hw * self.aspect)
        self.ax.set_ylim(cy - hw, cy + hw)
        return hw


class MiniMap:
    def __init__(self, fig, track, rect):
        self.ax = fig.add_axes(rect)
        style_ax(self.ax)
        self.ax.set_aspect("equal")
        self.ax.plot(*np.r_[track.c, track.c[:1]].T, color=DIM, lw=3, solid_capstyle="round")

    def add(self, color, size=7, marker="o"):
        d, = self.ax.plot([], [], marker, color=color, ms=size, mec="k" if marker == "o" else color, mew=0.5 if marker == "o" else 3, zorder=5)
        return d


def save_anim(fig, upd, frames, out, fps, preview):
    os.makedirs(os.path.dirname(out), exist_ok=True)
    if preview is not None:
        upd(preview)
        p = out.replace(".mp4", f"_f{preview}.png")
        fig.savefig(p, facecolor=BG)
        print("preview", p)
        return
    anim = animation.FuncAnimation(fig, upd, frames=frames, blit=False)
    anim.save(out, writer=animation.FFMpegWriter(fps=fps, bitrate=3500))
    print("saved", out, len(frames), "frames", f"{len(frames) / fps:.1f} s")


# ---------------------------------------------------------------- predict 모드
def run_predict(a):
    import torch
    from deep_dynamics.golf import data as D
    from deep_dynamics.golf.configs import GOLF_THUNDERHILL as CFG
    from deep_dynamics.golf.ladder import load_model
    from deep_dynamics.golf.track import best_lap
    torch.set_num_threads(4)
    dev = torch.device("cpu")  # 다른 실험(phase3)이 GPU 를 쓰므로 CPU 로. 모델이 작아 충분히 빠름
    NP = int(round(a.horizon / DT))
    sg = max(D.load_run(a.run), key=lambda s: len(s["t"]))
    su = np.concatenate([sg["s"], sg["u"]], 1)  # [vx, vy, r, Fxf, delta, bank] — 개루프 평가와 같은 입력
    n = len(su)
    t = sg["t"]
    if a.t0 is None:  # 기본: 이 파일의 가장 빠른 완전한 한 바퀴
        b = best_lap([a.run])
        t0, dur = b["t0"], (a.dur or b["lap_time"])
    else:
        t0, dur = a.t0, (a.dur or 60.0)
    k0 = max(H - 1, int(np.searchsorted(t, t0)))
    k1 = min(n - NP - 1, int(np.searchsorted(t, t0 + dur)))
    ks = np.arange(k0, k1)
    hist = np.stack([su[k - H + 1:k + 1] for k in ks])
    fut_u = np.stack([sg["u"][k + 1:k + NP] for k in ks])
    # 차 방향: 기록 yawAngle 은 동쪽 기준이 아니라 90° 어긋나 있음 [확인: 시험 3개 파일 모두 이동 방향 − 미끄럼각과 90.0° 차이]
    # → IAC 때처럼 '이동 방향 − 미끄럼각' 을 씀
    crs = np.unwrap(np.arctan2(np.gradient(sg["y"]), np.gradient(sg["x"])))
    crs = np.convolve(np.pad(crs, 2, mode="edge"), np.ones(5) / 5, mode="valid")
    head = crs - np.arctan2(sg["s"][:, 1], np.maximum(sg["s"][:, 0], 0.5))
    pose0 = np.stack([sg["x"][ks], sg["y"][ks], head[ks]], 1)
    truth = np.stack([np.stack([sg["x"][k + 1:k + 1 + NP], sg["y"][k + 1:k + 1 + NP]], 1) for k in ks])

    preds = {}
    ht0 = torch.tensor(hist, dtype=torch.float32, device=dev)
    ut = torch.tensor(fut_u, dtype=torch.float32, device=dev)
    for tag in RUNGS:
        mdl = load_model(os.path.join(RES, "models", f"{tag}_s{a.seed}_roll25.pt"), CFG, dev)
        hh = ht0.clone()
        pose = torch.tensor(pose0, dtype=torch.float32, device=dev)
        P = []
        with torch.no_grad():  # ladder.rollout 과 같은 굴리기 + 위치 적분 (mppi.kin 과 같은 식)
            for k in range(NP):
                s = mdl(hh)
                psi = pose[:, 2]
                pm = psi + 0.5 * DT * s[:, 2]
                pose = torch.stack([pose[:, 0] + DT * (s[:, 0] * torch.cos(pm) - s[:, 1] * torch.sin(pm)),
                                    pose[:, 1] + DT * (s[:, 0] * torch.sin(pm) + s[:, 1] * torch.cos(pm)), psi + DT * s[:, 2]], 1)
                P.append(pose[:, :2].numpy())
                if k < NP - 1:
                    hh = torch.cat([hh[:, 1:], torch.cat([s, ut[:, k]], 1).unsqueeze(1)], 1)
        preds[tag] = np.nan_to_num(np.stack(P, 1), nan=1e4)
    miss = {tg: np.hypot(*(preds[tg][:, -1] - truth[:, -1]).T) for tg in preds}
    print("mean miss at horizon (m):", {tg: round(float(v.mean()), 2) for tg, v in miss.items()}, flush=True)

    vx, vy, r = sg["s"].T
    sm = lambda z: np.convolve(z, np.ones(5) / 5, mode="same")
    ax_g = (np.gradient(sm(vx), DT) - r * vy) / 9.81
    ay_g = (np.gradient(sm(vy), DT) + r * vx) / 9.81
    fxf, delta = sg["u"][:, 0], sg["u"][:, 1]

    track = Track()
    fig = plt.figure(figsize=(12.8, 7.2), dpi=100, facecolor=BG)
    mv = MainView(fig, track, [0.0, 0.0, 0.62, 1.0])
    mm = MiniMap(fig, track, [0.80, 0.70, 0.19, 0.27])
    mdot = mm.add("#ffffff", 8)

    truth_ln, = mv.ax.plot([], [], color="white", lw=3.2, zorder=6, solid_capstyle="round")
    truth_end, = mv.ax.plot([], [], "o", color="white", ms=9, zorder=7)
    plines, pends = [], []
    for tag in RUNGS:
        plines.append(mv.ax.plot([], [], color=COL[tag], lw=2.2, zorder=5, solid_capstyle="round")[0])
        pends.append(mv.ax.plot([], [], "o", color=COL[tag], ms=7, mec="k", mew=0.6, zorder=5)[0])
    car = Polygon(car_poly(0, 0, math.pi / 2), closed=True, fc="#f2f2f2", ec="k", lw=1, zorder=8)
    mv.ax.add_patch(car)
    trail, = mv.ax.plot([], [], color="white", lw=1, alpha=0.35, zorder=4)
    fig.text(0.015, 0.965, "Thunderhill · Golf 실제 주행 기록 재생 (시험 파일, 학습에 안 씀)", color=FG, fontsize=15, weight="bold", va="top")
    fig.text(0.015, 0.92, f"차 앞의 선 = 실제 조작을 그대로 넣었을 때 모델이 예측한 {a.horizon:g}초 동안의 길 (개루프)", color=DIM, fontsize=11, va="top")
    lx = 0.015
    for lab, col in [("실제", "white")] + [(tag, COL[tag]) for tag in RUNGS]:
        fig.text(lx, 0.03, "━ " + lab, color=col, fontsize=12, weight="bold")
        lx += 0.065

    tx = 0.645
    sp_txt = fig.text(tx, 0.955, "", color=FG, fontsize=30, weight="bold", va="top")
    sp_sub = fig.text(tx, 0.865, "", color=DIM, fontsize=11, va="top")
    t_txt = fig.text(tx, 0.825, "", color=DIM, fontsize=11, va="top")
    axb = fig.add_axes([tx, 0.64, 0.14, 0.12])
    style_ax(axb)
    axb.set_xlim(-1, 1)
    axb.set_ylim(-0.5, 1.9)
    axb.add_patch(plt.Rectangle((-1, 0.95), 2, 0.4, color="#2a2d33"))
    axb.add_patch(plt.Rectangle((-1, -0.25), 2, 0.4, color="#2a2d33"))
    steer_bar = axb.add_patch(plt.Rectangle((0, 0.95), 0, 0.4, color="#cfd3da"))
    pedal_bar = axb.add_patch(plt.Rectangle((0, -0.25), 0, 0.4, color="#2FBF71"))
    for y0 in (0.95, -0.25):  # 가운데 눈금 (막대 높이만)
        axb.plot([0, 0], [y0, y0 + 0.4], color=DIM, lw=0.8)
    axb.text(-1, 1.42, "조향 (좌 ← → 우)", color=DIM, fontsize=9)
    axb.text(-1, 0.22, "제동 ← 앞바퀴 힘 → 구동 (±6 kN)", color=DIM, fontsize=9)
    axf = fig.add_axes([0.645, 0.30, 0.17, 0.33])
    style_ax(axf)
    axf.set_aspect("equal")
    gmax = 1.5
    for rr in (0.5, 1.0, 1.5):
        axf.add_patch(Circle((0, 0), rr, fill=False, ec="#4a4f58", lw=1))
        axf.text(0.05, rr + 0.04, f"{rr:g}g", color=DIM, fontsize=8)
    axf.axhline(0, color="#30343a", lw=0.8)
    axf.axvline(0, color="#30343a", lw=0.8)
    axf.set_xlim(-gmax * 1.12, gmax * 1.12)
    axf.set_ylim(-gmax * 1.12, gmax * 1.12)
    axf.set_title("가속도 (마찰원)", color=FG, fontsize=11)
    g_tail, = axf.plot([], [], color="#F5C542", lw=1.2, alpha=0.5)
    g_dot, = axf.plot([], [], "o", color="#F5C542", ms=9, mec="k")
    axe = fig.add_axes([0.73, 0.075, 0.25, 0.20])
    style_ax(axe)
    ebars = axe.barh(range(len(RUNGS)), [0] * len(RUNGS), color=[COL[tg] for tg in RUNGS], height=0.6)
    axe.set_yticks(range(len(RUNGS)))
    axe.set_yticklabels([f"{tg} {RUNG_KO[tg]}" for tg in RUNGS], color=FG, fontsize=9)
    axe.tick_params(length=0)
    axe.invert_yaxis()
    emax = 1.0 * a.horizon
    axe.set_xlim(0, emax)
    axe.set_title(f"{a.horizon:g}초 뒤 빗나간 거리 (m) · 지금 / 누적 평균", color=FG, fontsize=10.5, loc="left")
    etxt = [axe.text(0, i, "", color=FG, va="center", fontsize=9) for i in range(len(RUNGS))]

    # 끝점 확대 창: 1초 뒤 실제 위치(흰 점)를 가운데 두고 ±ZM m 만 보여 줌 (예측선이 겹쳐 보여 차이를 확대)
    ZM = 1.0 * a.horizon
    axz = fig.add_axes([0.015, 0.50, 0.17, 0.30])
    style_ax(axz)
    axz.set_facecolor("#1d2026")
    axz.set_aspect("equal")
    axz.set_xlim(-ZM, ZM)
    axz.set_ylim(-ZM, ZM)
    for rr in (0.25 * ZM, 0.5 * ZM, ZM):
        axz.add_patch(Circle((0, 0), rr, fill=False, ec="#4a4f58", lw=0.8))
    axz.text(-ZM * 0.97, -ZM * 0.97, f"원 간격 {0.25 * ZM:g} m", color=DIM, fontsize=8, va="bottom")
    axz.set_title(f"{a.horizon:g}초 뒤 끝점 확대 (가운데 = 실제)", color=FG, fontsize=10)
    axz.plot([0], [0], "o", color="white", ms=9, zorder=6)
    zl = [axz.plot([], [], color=COL[tg], lw=1.5, alpha=0.8, zorder=4)[0] for tg in RUNGS]
    zp = [axz.plot([], [], "o", color=COL[tg], ms=8, mec="k", mew=0.6, zorder=5)[0] for tg in RUNGS]
    zt, = axz.plot([], [], color="white", lw=2, zorder=3)

    def upd(i):
        k = ks[i]
        p0, psi0 = pose0[i, :2], pose0[i, 2]
        mv.frame(p0, psi0, vx[k])
        T = to_cam(truth[i], p0, psi0)
        truth_ln.set_data(np.r_[0, T[:, 0]], np.r_[0, T[:, 1]])
        truth_end.set_data([T[-1, 0]], [T[-1, 1]])
        for j, tag in enumerate(RUNGS):
            Q = to_cam(preds[tag][i], p0, psi0)
            plines[j].set_data(np.r_[0, Q[:, 0]], np.r_[0, Q[:, 1]])
            pends[j].set_data([Q[-1, 0]], [Q[-1, 1]])
            e = miss[tag][i]
            ebars[j].set_width(min(e, emax))
            etxt[j].set_text(f" {e:4.2f} / {miss[tag][:i + 1].mean():4.2f}")
            etxt[j].set_x(min(e, emax))
            Z = Q - T[-1]
            zl[j].set_data(Z[-6:, 0], Z[-6:, 1])
            pt = np.clip(Z[-1], -0.97 * ZM, 0.97 * ZM)  # 창 밖이면 가장자리에
            zp[j].set_data([pt[0]], [pt[1]])
        Zt = T - T[-1]
        zt.set_data(Zt[-6:, 0], Zt[-6:, 1])
        tr = to_cam(np.stack([sg["x"][max(0, k - 75):k + 1], sg["y"][max(0, k - 75):k + 1]], 1), p0, psi0)
        trail.set_data(tr[:, 0], tr[:, 1])
        mdot.set_data([p0[0]], [p0[1]])
        sp_txt.set_text(f"{vx[k] * 3.6:5.0f} km/h")
        sp_sub.set_text(f"{vx[k]:4.1f} m/s · 조향 {math.degrees(delta[k]):+5.1f}° · 회전 {math.degrees(r[k]):+5.1f}°/s")
        t_txt.set_text(f"주행 {a.run} · t = {(k - k0) * DT:5.1f} s")
        steer_bar.set_width(-np.clip(delta[k] / 0.25, -1, 1))
        pv = np.clip(fxf[k] / 6000.0, -1, 1)
        pedal_bar.set_width(pv)
        pedal_bar.set_color("#2FBF71" if pv >= 0 else "#E05A5A")
        lo = max(0, k - 25)
        g_tail.set_data(-ay_g[lo:k + 1], ax_g[lo:k + 1])
        g_dot.set_data([-ay_g[k]], [ax_g[k]])
        return []

    out = a.out or os.path.join(VID, "predict_test.mp4")
    save_anim(fig, upd, list(range(0, len(ks), a.every)), out, 25 * a.speed / a.every, a.preview)


# ---------------------------------------------------------------- race 모드
def run_race(a):
    track = Track()
    ntr = len(track.c)
    cars = []
    for tag in RUNGS:
        stem = os.path.join(VIZ, f"{a.mode}_{a.N}__{tag}_s{a.seed}")
        if not os.path.exists(stem + "_traj.npz"):
            print("없음:", stem)
            continue
        L = np.load(stem + "_traj.npz")
        res = json.load(open(stem + ".json"))
        cars.append(dict(tag=tag, name=RUNG_KO[tag], X=L["X"], Y=L["Y"], psi=L["psi"], vx=L["vx"], idx=L["idx"],
                         status=res["status"], lap=res["lap_time"], col=COL[tag]))
    assert cars, "궤적 없음 — record_race.py 를 먼저 실행"
    for c in cars:  # 누적 진행 거리 (중심선 점 = 1 m)
        di = np.diff(c["idx"].astype(int))
        di = np.where(di < -ntr / 2, di + ntr, di)
        di = np.where(di > ntr / 2, di - ntr, di)
        c["prog"] = np.r_[0, np.cumsum(di)]
    T = max(len(c["X"]) for c in cars)
    if a.dur:
        T = min(T, int(a.dur / DT))

    fig = plt.figure(figsize=(12.8, 7.2), dpi=100, facecolor=BG)
    mv = MainView(fig, track, [0.0, 0.0, 0.62, 1.0], zoom=(12.0, 0.55))
    mm = MiniMap(fig, track, [0.70, 0.62, 0.28, 0.33])
    cam = {}
    for c in cars:
        c["patch"] = Polygon(car_poly(0, 0, 0), closed=True, fc=c["col"], ec="k", lw=1, zorder=8)
        mv.ax.add_patch(c["patch"])
        c["trail"], = mv.ax.plot([], [], color=c["col"], lw=1.6, alpha=0.6, zorder=5)
        c["lab"] = mv.ax.text(0, 0, c["tag"], color=c["col"], fontsize=10, weight="bold", zorder=9, ha="left", va="bottom")
        c["mark"], = mv.ax.plot([], [], "x", color=c["col"], ms=16, mew=3.5, zorder=9)
        c["mdot"] = mm.add(c["col"], 8)
        c["mx"] = mm.add(c["col"], 11, "x")
    fig.text(0.015, 0.965, f"Thunderhill · 시뮬레이터에서 제어 주행 · 계획 {a.N * DT:g}초", color=FG, fontsize=15, weight="bold", va="top")
    fig.text(0.015, 0.92, f"MPPI {MODE_KO[a.mode]}. 같은 제어기, 머릿속 차 모델만 다름", color=DIM, fontsize=11, va="top")
    t_txt = fig.text(0.015, 0.03, "", color=DIM, fontsize=12)
    CX = (0.645, 0.835, 0.905)  # 열: 순위·모델 / 속도 / 진행률·상태 (글꼴 폭이 달라 칸을 나눔)
    rows = [[fig.text(x, 0.54 - 0.075 * j, "", fontsize=12, va="top") for x in CX] for j in range(len(cars))]
    for x, h in zip(CX, ("순위 · 모델", "속도", "진행률 / 상태")):
        fig.text(x, 0.585, h, color=DIM, fontsize=10)

    def upd(fi):
        js = [min(fi, len(c["X"]) - 1) for c in cars]
        prog = [c["prog"][j] for c, j in zip(cars, js)]
        alive = [fi < len(c["X"]) - 1 or c["status"] == "finished" for c in cars]
        lead = int(np.argmax([p if ok else -1e9 for p, ok in zip(prog, alive)])) if any(alive) else int(np.argmax(prog))
        if a.follow:
            lead = [c["tag"] for c in cars].index(a.follow)
        cl, jl = cars[lead], js[lead]
        tp, tpsi = np.array([cl["X"][jl], cl["Y"][jl]]), cl["psi"][jl]
        if not cam:
            cam.update(p=tp, psi=tpsi)
        else:  # 선두가 바뀌어도 카메라가 튀지 않게
            cam["p"] = cam["p"] + 0.2 * (tp - cam["p"])
            cam["psi"] = cam["psi"] + 0.2 * ((tpsi - cam["psi"] + math.pi) % (2 * math.pi) - math.pi)
        p0, psi0 = cam["p"], cam["psi"]
        mv.frame(p0, psi0, cl["vx"][jl])
        for c, j in zip(cars, js):
            c["patch"].set_xy(to_cam(car_poly(c["X"][j], c["Y"][j], c["psi"][j]), p0, psi0))
            tr = to_cam(np.stack([c["X"][max(0, j - 50):j + 1], c["Y"][max(0, j - 50):j + 1]], 1), p0, psi0)
            c["trail"].set_data(tr[:, 0], tr[:, 1])
            c["q"] = to_cam(np.array([c["X"][j], c["Y"][j]]), p0, psi0)
            failed = fi >= len(c["X"]) - 1 and c["status"] != "finished"
            c["mark"].set_data(*([[c["q"][0]], [c["q"][1]]] if failed else [[], []]))
            c["mx"].set_data(*([[c["X"][j]], [c["Y"][j]]] if failed else [[], []]))
            c["mdot"].set_data(*([[], []] if failed else [[c["X"][j]], [c["Y"][j]]]))
        ys = sorted(range(len(cars)), key=lambda i: cars[i]["q"][1])  # 이름표 겹침 방지
        gap = mv.halfw * 0.06
        last = -1e9
        for i in ys:
            y = max(cars[i]["q"][1] + 1.0, last + gap)
            cars[i]["lab"].set_position((cars[i]["q"][0] + 2.5, y))
            last = y

        def rank_key(i):
            c = cars[i]
            if c["status"] == "finished" and fi >= len(c["X"]) - 1:
                return (0, c["lap"])  # 완주한 차는 랩타임 순
            return (1, -prog[i])
        for rk, ci in enumerate(sorted(range(len(cars)), key=rank_key)):
            c, j = cars[ci], js[ci]
            ended = fi >= len(c["X"]) - 1
            frac = f"{c['prog'][j] / ntr * 100:3.0f}%"
            st = (STAT[c["status"]] + (f" {c['lap']:.1f}s" if c["lap"] else f" {frac.strip()}")) if ended else frac
            col = c["col"] if not (ended and c["status"] != "finished") else "#777"
            for tx_, val in zip(rows[rk], (f"{rk + 1}.  {c['tag']} {c['name']}", f"{c['vx'][j] * 3.6:3.0f} km/h", st)):
                tx_.set_text(val)
                tx_.set_color(col)
        t_txt.set_text(f"t = {fi * DT:5.1f} s   (카메라: {cl['tag']} {cl['name']})")
        return []

    frames = list(range(0, T, a.every)) + [T - 1] * 50  # 끝에 2초 정지 (최종 순위)
    out = a.out or os.path.join(VID, f"race_{a.mode}_{a.N * DT:g}s.mp4")
    save_anim(fig, upd, frames, out, 25 * a.speed / a.every, a.preview)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("kind", choices=["predict", "race"])
    ap.add_argument("--run", default="2023-10_1")      # predict: 시험 파일
    ap.add_argument("--horizon", type=float, default=1.0)
    ap.add_argument("--t0", type=float, default=None)  # predict: 시작 시각 (기본: 가장 빠른 한 바퀴)
    ap.add_argument("--dur", type=float, default=None)
    ap.add_argument("--mode", default="A", choices=["A", "B"])  # race
    ap.add_argument("--N", type=int, default=25)
    ap.add_argument("--follow", default=None)
    ap.add_argument("--seed", default="0")
    ap.add_argument("--every", type=int, default=2)
    ap.add_argument("--speed", type=float, default=2.0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--preview", type=int, default=None)
    a = ap.parse_args()
    run_predict(a) if a.kind == "predict" else run_race(a)
