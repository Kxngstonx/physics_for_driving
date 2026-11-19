"""MPPI 제어기 + 폐루프 주행 (11/12 share_source_team/src/iac_sim.py 에서 옮김).

옮기면서 바꾼 것은 아래뿐이고 MPPI 계산(시간 상관 잡음, 가중 평균, 조향 변화 제한)은 원본 그대로다.
  - iac_lib 대신 golf 모듈 사용. 트랙 파일은 경로로 받음 (golf/track.py 출력 npz)
  - [결정 11/18, 문서 9-2절] 입력 = [앞바퀴 힘 Fxf (N), 조향 (rad)]
      범위: 힘 −6 kN ~ 속도별 구동력 상한(sim.force_cap_table, 굴리는 매 스텝의 vx 로 보간), 조향 ±0.3 rad
      잡음: 힘 1000 N, 조향 0.06 rad (빠를수록 줄임, 원본과 같음)
      힘 변화 비용 가중치: 원본 5 (스로틀 범위 1.7 기준) → 5·(1.7/13000)² (힘 범위 13 kN 기준, 같은 상대 크기)
  - mode="B": 경로·목표 속도 없이 전진 거리 보상 + 경계 이탈 벌점 (문서 9-4절)
  - episode(): world 에 reset/pose 가 있으면(시뮬레이터) 시작 상태를 넣고 위치도 world 에서 받음.
    정지 판정 3 m/s (문서 9-1절). 출발 속도 v0 를 줄 수 있음 (기본: 트랙 시작점 목표 속도)
  - Track(flat=True, 기본): 횡경사를 0 으로 (시뮬레이터가 평평한 길이므로 모델 입력도 같게)

세계(world): hist (1,H,6) → 다음 [vx, vy, r] 를 내는 함수. 위치·방향은 운동학으로 적분.
제어기(MPPI): 비교 대상 모델을 내부 모델로 써서 K 개 조작 후보를 N 스텝 굴려 보고 가중 평균.
"""
import math
import time

import numpy as np
import torch

from . import sim

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DT = 0.04
H = 8
U_LO_FORCE = (sim.F_BRAKE_MIN, -0.30)  # Golf: [앞바퀴 힘 N, 조향 rad]. 힘 상한은 속도별 표 (U_HI 의 힘 칸은 표의 최댓값)
DELTA_MAX = 0.30
SIGMA_FORCE = (1000.0, 0.06)
W_DU_FORCE = 5.0 * (1.7 / 13000.0) ** 2
F_CRUISE = sim.road_load(20.0)  # 20 m/s 정속 주행 저항 — 첫 계획의 힘 초기값


class Track:
    def __init__(self, path, flat=True):
        """flat: 횡경사를 0 으로 [결정 11/18] — 시뮬레이터가 평평한 길이라 모델 입력도 같게 (데이터 bank_rad 는 센서와 안 맞음)."""
        t = np.load(path)
        self.np = {k: t[k] for k in t.files}
        if flat:
            self.np["bank"] = np.zeros_like(self.np["bank"])
        self.n = len(t["x"])
        f = lambda k: torch.tensor(t[k], dtype=torch.float32, device=DEV)
        self.x, self.y, self.psi, self.vref = f("x"), f("y"), f("psi"), f("vref")
        self.wl, self.wr = f("width_l"), f("width_r")
        self.bank = torch.tensor(self.np["bank"], dtype=torch.float32, device=DEV)

    def project(self, X, Y, idx0, back=10, ahead=90):
        """X, Y (K,), idx0 (K,) long → 가장 가까운 중심선 인덱스, 횡오차(왼쪽 +)."""
        off = torch.arange(-back, ahead, device=DEV)
        cand = (idx0.unsqueeze(1) + off) % self.n
        d2 = (self.x[cand] - X.unsqueeze(1)) ** 2 + (self.y[cand] - Y.unsqueeze(1)) ** 2
        j = d2.argmin(1)
        idx = cand.gather(1, j.unsqueeze(1)).squeeze(1)
        psi = self.psi[idx]
        ey = -(X - self.x[idx]) * torch.sin(psi) + (Y - self.y[idx]) * torch.cos(psi)
        return idx, ey


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def kin(pose, s):
    """pose (B,3) [X,Y,psi], s (B,3) [vx,vy,r] → 다음 pose (DT 동안 일정 속도 가정)."""
    X, Y, psi = pose.unbind(-1)
    vx, vy, r = s.unbind(-1)
    pm = psi + 0.5 * DT * r
    return torch.stack([X + DT * (vx * torch.cos(pm) - vy * torch.sin(pm)),
                        Y + DT * (vx * torch.sin(pm) + vy * torch.cos(pm)), psi + DT * r], -1)


class MPPI:
    def __init__(self, model, track, K=1024, N=25, sigma=SIGMA_FORCE, lam=0.3, seed=0, w=None, use_graph=True, corr=0.85,
                 u_init=F_CRUISE, mode="A"):
        self.corr = corr
        vg, fg = sim.force_cap_table()
        self.cap_v = torch.tensor(vg, dtype=torch.float32, device=DEV)
        self.cap_f = torch.tensor(fg, dtype=torch.float32, device=DEV)
        self.u_lo = torch.tensor(U_LO_FORCE, device=DEV)
        self.u_hi = torch.tensor((float(fg.max()), DELTA_MAX), device=DEV)
        self.use_graph = use_graph
        self.m, self.tr, self.K, self.N = model, track, K, N
        self.sig = torch.tensor(sigma, device=DEV)
        self.lam = lam
        self.U = torch.zeros(N, 2, device=DEV)
        self.U[:, 0] = u_init
        self.g = torch.Generator(device=DEV).manual_seed(seed)
        self.w = dict(ey=1.0, epsi=50.0, v=0.5, du_d=2e4, du_t=W_DU_FORCE, off=1e4, bad=1e6, slow=200.0, prog=0.0)
        if mode == "B":  # 경로·목표 속도 항을 끄고 전진 보상 (1 m 당 10, 출발값 [가정] — 튜닝의 λ 가 비용 중앙값으로 크기를 맞춤)
            self.w.update(ey=0.0, epsi=0.0, v=0.0, prog=10.0)
        if w:
            self.w.update(w)

    @torch.no_grad()
    def _rollout(self):
        """정적 버퍼 → 비용. CUDA 그래프로 캡처해서 반복 실행 (파이썬 오버헤드 제거)."""
        K, N = self.K, self.N
        U = torch.max(torch.min(self.bU, self.u_hi), self.u_lo)
        hist = self.bhist
        s, P, ix, up = self.bs, self.bP, self.bix, self.bup
        cost = torch.zeros(K, device=DEV)
        for k in range(N):
            # 힘 상한: 지금 굴리는 상태의 vx 로 구동력 표 보간 (엔진은 빠를수록 낼 수 있는 힘이 작음)
            j = torch.clamp(torch.searchsorted(self.cap_v, s[:, 0].contiguous()), 1, len(self.cap_v) - 1)
            v0, v1 = self.cap_v[j - 1], self.cap_v[j]
            t = torch.clamp((s[:, 0] - v0) / (v1 - v0), 0.0, 1.0)
            fmax = self.cap_f[j - 1] + t * (self.cap_f[j] - self.cap_f[j - 1])
            u = torch.stack([torch.minimum(U[:, k, 0], fmax), U[:, k, 1]], 1)
            U[:, k] = u
            h = torch.cat([hist, torch.cat([s, u, self.tr.bank[ix].unsqueeze(1)], 1).unsqueeze(1)], 1)
            s_next = self.m(h)
            bad = ~torch.isfinite(s_next).all(1) | (s_next.abs() > 200).any(1)
            s_next = torch.where(bad.unsqueeze(1), s, s_next)
            P = kin(P, s_next)
            ix_prev = ix
            ix, ey = self.tr.project(P[:, 0], P[:, 1], ix)
            dprog = torch.remainder(ix - ix_prev + self.tr.n // 2, self.tr.n) - self.tr.n // 2  # 진행한 중심선 점 수 (1 m 간격)
            epsi = wrap(P[:, 2] - self.tr.psi[ix])
            half = torch.where(ey > 0, self.tr.wl[ix], self.tr.wr[ix])
            off = (ey.abs() > half - 1.5).float()
            du = u - up
            cost = cost + (self.w["ey"] * ey ** 2 + self.w["epsi"] * epsi ** 2 + self.w["v"] * (s_next[:, 0] - self.tr.vref[ix]) ** 2
                           + self.w["du_d"] * du[:, 1] ** 2 + self.w["du_t"] * du[:, 0] ** 2 + self.w["off"] * off + self.w["bad"] * bad.float()
                           + self.w["slow"] * torch.relu(6.0 - s_next[:, 0]) ** 2 - self.w["prog"] * dprog.float())
            up = u
            hist = h[:, 1:]
            s = s_next
        self.bUc.copy_(U)  # 속도별 힘 상한까지 적용한 조작 (가중 평균에 씀)
        self.bcost.copy_(cost)

    def _setup(self, nf):
        K, N = self.K, self.N
        self.bU = torch.zeros(K, N, 2, device=DEV)
        self.bUc = torch.zeros(K, N, 2, device=DEV)
        self.bhist = torch.zeros(K, H - 1, nf, device=DEV)
        self.bs = torch.zeros(K, 3, device=DEV)
        self.bs[:, 0] = 20.0
        self.bP = torch.zeros(K, 3, device=DEV)
        self.bix = torch.zeros(K, dtype=torch.long, device=DEV)
        self.bup = torch.zeros(K, 2, device=DEV)
        self.bcost = torch.zeros(K, device=DEV)
        self.graph = None
        if DEV.type == "cuda" and self.use_graph:
            st = torch.cuda.Stream()
            st.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(st):
                for _ in range(2):
                    self._rollout()
            torch.cuda.current_stream().wait_stream(st)
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                self._rollout()

    @torch.no_grad()
    def act(self, hist_past, pose, idx, u_prev):
        """hist_past (H-1,NF) 과거 [s,u,bank], self.s_now 현재 상태, pose (3,), idx 현재 중심선 인덱스."""
        K, N = self.K, self.N
        if not hasattr(self, "bU"):
            self._setup(hist_past.shape[1])
        # 시간 상관 잡음 (AR(1)), 조향 잡음은 속도가 빠를수록 작게 (고속에서 같은 조향각의 효과가 크므로)
        wn = torch.randn(K, N, 2, device=DEV, generator=self.g)
        eps = torch.empty_like(wn)
        eps[:, 0] = wn[:, 0]
        c = self.corr
        for k in range(1, N):
            eps[:, k] = c * eps[:, k - 1] + (1 - c * c) ** 0.5 * wn[:, k]
        vx = float(self.s_now[0])
        sig = self.sig.clone()
        sig[1] = sig[1] * min(1.0, max(0.1, 6.0 / max(vx, 1.0)))
        eps = eps * sig
        eps[0] = 0.0  # 명목 궤적 하나는 그대로
        self.bU.copy_(self.U.unsqueeze(0) + eps)
        self.bhist.copy_(hist_past.unsqueeze(0).expand(K, -1, -1))
        self.bs.copy_(self.s_now.unsqueeze(0).expand(K, 3))
        self.bP.copy_(pose.unsqueeze(0).expand(K, 3))
        self.bix.fill_(int(idx))
        self.bup.copy_(u_prev.unsqueeze(0).expand(K, 2))
        if self.graph is not None:
            self.graph.replay()
        else:
            self._rollout()
        U, cost = self.bUc, self.bcost
        c = cost - cost.min()
        lam = self.lam * (c.median() + 1e-6)
        wts = torch.softmax(-c / lam, 0)
        self.U = (wts[:, None, None] * U).sum(0)
        u0 = self.U[0].clone()
        # 조향 변화율 제한 (0.5 rad/s → 스텝당 0.02 rad)
        u0[1] = u_prev[1] + torch.clamp(u0[1] - u_prev[1], -0.02, 0.02)
        self.U[0] = u0
        self.U = torch.cat([self.U[1:], self.U[-1:]], 0)
        return u0


@torch.no_grad()
def episode(ctrl, world, track, K=1024, N=25, seed=0, max_time=None, record=True, make_ctrl=None, v_stall=3.0, u0=(F_CRUISE, 0.0),
            v_armed=0.0, v0=None, i0=0, **mppi_kw):
    """한 바퀴 주행. 정지 판정 v_stall 3 m/s [결정 11/18, 문서 9-1]: 목표 속도 최저 14.5 m/s, 실제 랩 중 최저 약 8.3 m/s 보다
    충분히 낮아 이 아래면 주행 실패. (Golf main 데이터는 0.5 m/s 까지 있어 3 m/s 도 모델 학습 범위 안)
    v_armed: 정지 출발일 때 vx 가 이 값을 한 번 넘은 뒤부터 정지 판정 (0 이면 처음부터)"""
    tr = track
    v0 = float(tr.vref[i0]) if v0 is None else float(v0)  # i0: 출발 트랙 위치 (기록 랩의 실제 출발 위치 등)
    pose = torch.tensor([float(tr.x[i0]), float(tr.y[i0]), float(tr.psi[i0])], device=DEV)
    s = torch.tensor([v0, 0.0, 0.0], device=DEV)
    u_prev = torch.tensor(u0, device=DEV)
    hist_past = torch.cat([s, u_prev, tr.bank[i0:i0 + 1]]).repeat(H - 1, 1)  # 정상 주행 중이었다고 가정한 과거
    if hasattr(world, "reset"):  # 시뮬레이터: 같은 시작 상태
        world.reset(float(pose[0]), float(pose[1]), float(pose[2]), v0, F0=u0[0])
    mp = make_ctrl(ctrl, tr) if make_ctrl else MPPI(ctrl, tr, K=K, N=N, seed=seed, **mppi_kw)
    idx = torch.full((1,), i0, device=DEV, dtype=torch.long)
    lap_len = tr.n
    max_time = max_time or 2.0 * float((1.0 / tr.vref).sum())
    steps = int(max_time / DT)
    prog, last_i = 0, i0
    log = dict(X=[], Y=[], psi=[], vx=[], vy=[], r=[], thr=[], delta=[], ey=[], vref=[], idx=[])
    status, t0 = "timeout", time.time()
    armed = v0 >= v_armed
    n_done = 0  # 진행한 스텝 수 (record=False 여도 랩타임을 셀 수 있게. 11/19 이전에는 로그 길이로 세어 record=False 면 0)
    for t in range(steps):
        n_done = t + 1
        mp.s_now = s
        u = mp.act(hist_past, pose, int(idx), u_prev)
        row = torch.cat([s, u, tr.bank[idx]])
        h = torch.cat([hist_past, row.unsqueeze(0)], 0).unsqueeze(0)
        if hasattr(world, "set_pos"):  # 정답 현실 시뮬레이터: 트랙 위치별 노면 마찰 등
            world.set_pos(idx)
        s_next = world(h)[0]
        if not torch.isfinite(s_next).all() or s_next.abs().max() > 200:
            status = "world_nan"
            break
        pose = torch.tensor(world.pose, device=DEV, dtype=torch.float32) if hasattr(world, "pose") else kin(pose.unsqueeze(0), s_next.unsqueeze(0))[0]
        idx, ey = tr.project(pose[0:1], pose[1:2], idx)
        i = int(idx)
        di = (i - last_i) % lap_len
        if di < lap_len / 2:
            prog += di
        last_i = i
        half = float(tr.wl[i] if ey > 0 else tr.wr[i])
        if record:
            for k, v in zip(log, [pose[0], pose[1], pose[2], s_next[0], s_next[1], s_next[2], u[0], u[1], ey[0], tr.vref[i], idx[0]]):
                log[k].append(float(v))
        hist_past = torch.cat([hist_past[1:], row.unsqueeze(0)], 0)
        s, u_prev = s_next, u
        if abs(float(ey)) > half + 2.0:
            status = "off_track"
            break
        armed = armed or float(s_next[0]) >= v_armed
        if armed and float(s_next[0]) < v_stall:
            status = "stalled"
            break
        if prog >= lap_len - 2:
            status = "finished"
            break
    T = n_done * DT
    ey = np.abs(np.array(log["ey"])) if log["ey"] else np.array([np.nan])
    res = dict(status=status, finished=status == "finished", progress_m=float(prog), progress_frac=float(prog / lap_len),
               time_s=float(T), lap_time=float(T) if status == "finished" else None,
               ey_mean=float(ey.mean()), ey_max=float(ey.max()),
               v_err_mean=float(np.mean(np.abs(np.array(log["vx"]) - np.array(log["vref"])))) if log["vx"] else None,
               ddelta_mean=float(np.mean(np.abs(np.diff(log["delta"])))) if len(log["delta"]) > 1 else None,
               wall_s=time.time() - t0, K=K, N=N, seed=seed)
    return res, log
