"""직접 만든 Golf 시뮬레이터 (현실적인 단일 궤도 모델). 저장소 sim.World 와 같은 인터페이스라 mppi.episode() 에 그대로 끼운다.

R1(자전거 + Pacejka, 계수 학습)과 일부러 다르게 만든 점 — 정답 세계가 R1 과 같은 식이면 R1 이 구조적으로 유리하므로:
  1. 타이어: Fiala 브러시 모델 (선형 강성 Cα, 최대 마찰 μ·Fz 에서 포화). 계수는 데이터로 따로 맞춤 (fit_realsim.py)
  2. 마찰원: 축의 종방향 힘이 클수록 쓸 수 있는 횡력이 줄어듦  Fy_max = √((μFz)² − Fx²)
  3. 앞뒤 하중 이동: 가속하면 앞이 가벼워지고 제동하면 무거워짐  ΔFz = m·ax·h/L
  4. 좌우 하중 이동 + 타이어 하중 민감도: 바깥 바퀴로 하중이 몰리면 축 전체 최대 마찰이 줄어듦
  5. 종방향 힘은 축 마찰 한계의 85%까지 (ABS·EBD·구동 미끄럼 방지 근사 [가정]) → 제동 중에도 횡력 여유가 남음
  6. 구동력 상한(저장소 속도별 표), 제동 하한 −6 kN, 앞 제동 몫 0.75 (저장소 값)
  7. 액추에이터 지연 (저장소 delays.json 이 있으면 그 값), 조향 속도 제한
  8. 저속(정지 출발)에서는 운동학 자전거 모델로 부드럽게 넘어감 (미끄럼각 식이 0 m/s 에서 정의되지 않음)
  9. RK4 적분, 0.005 s 간격 (R1 은 오일러 0.01 s)
차량 치수 = 공개 제원 (Golf Mk7: 축거 2.63 m, 윤거 1.541/1.513 m, 폭 1.79 m), 질량·무게 배분은 저장소 값.
무게중심 높이 0.55 m 는 [가정] (해치백 일반값, 공개 제원 없음).
"""
import collections
import json
import math
import os
import types

import numpy as np

G = 9.81
DT = 0.04
M = 1496.9          # 저장소 configs (데이터 추정)
LF, LR = 1.026, 1.604  # 저장소 configs (축거 2.63 m 공개 제원, 앞 무게 몫 0.61 가정)
L = LF + LR
TRACK_F, TRACK_R = 1.541, 1.513  # 공개 제원 (저장소 sim.py 와 같음)
H_CG = 0.55         # [가정]
ROLL_FRONT = 0.61   # 좌우 하중 이동의 앞 몫 (저장소 sim.ROLL_FRONT_SHARE)
BRAKE_FRONT = 0.75  # 저장소 configs
F_BRAKE_MIN = -6000.0
CR0, CR2 = 176.0, 0.40  # 저장소 configs (일반 승용차 값)
ABS_FRAC = 0.85  # [가정] ABS·EBD 근사: 종방향 힘은 축 마찰 한계의 이 비율까지만 (바퀴가 잠기거나 헛돌지 않게, 횡력 여유를 남김)
V_KIN = (1.0, 3.0)  # 이 속도 사이에서 운동학 → 동역학 모델로 넘어감

DEFAULT = dict(mu_f=1.20, mu_r=1.20, ca_f=82700.0, ca_r=72300.0, iz=1797.0, h_cg=H_CG, ds=0.0, load_sens=0.05, steer_rate=1.0, tau_lt=0.0)
# mu_f·mu_r: 앞뒤 축 최대 마찰을 따로 (무게 배분이 가정값이라 R1 의 Df·Dr 처럼 축별로 맞춤). 1.2 = 데이터 최대 횡가속 (팀 보고서).
# h_cg: 무게중심 높이 (맞춤), ds: 센서 위치 보정 (vy_센서 = vy_무게중심 + ds·r, R1·R4 의 ds 와 같은 뜻), Cα: 저장소 제원 추정 회귀값 (specs.json Cf_lin/Cr_lin), Iz: 저장소 configs 초기값.
# tau_lt: [11/24] 하중 이동 지연 시간 상수 s (서스펜션이 눌리는 동안 하중이 늦게 넘어감, 1차 지연 근사). 0 이면 즉시 (v3 와 같음)
# load_sens: 하중 민감도 [가정, 일반 승용 타이어 0.05~0.15]. steer_rate: 조향 속도 상한 rad/s [가정]


def fiala(alpha, ca, fy_max):
    """Fiala 브러시 타이어: 미끄럼각 α (rad) → 횡력 (N). α 와 반대 방향."""
    if fy_max <= 1.0:
        return 0.0
    z = math.tan(alpha)
    a_sl = 3.0 * fy_max / ca  # 완전 미끄럼 시작 tan α
    if abs(z) < a_sl:
        return -ca * z + ca ** 2 / (3 * fy_max) * abs(z) * z - ca ** 3 / (27 * fy_max ** 2) * z ** 3
    return -fy_max * math.copysign(1.0, z)


class RealWorld:
    """mppi.episode() 의 world: world(h) → 다음 [vx, vy, r]. pose = (X, Y, psi). 경사는 set_grade / set_pos."""

    def __init__(self, params=None, delays=None, h=0.005, grade_by_idx=None, force_cap=None, cap=True):
        self.prm = dict(DEFAULT, **(params or {}))
        self.p = types.SimpleNamespace(m=M, a=LF, b=LR)  # 저장소 drive_free 가 world.p.m·a·b 를 읽음 (sim.World 와 같은 이름)
        self.h, self.n_sub = h, int(round(DT / h))
        self.grade_by_idx = grade_by_idx
        self.force_cap = force_cap  # 함수 v → 최대 구동력. None 이면 상한 없음
        self.cap = cap
        d = delays or {}
        self.n_steer = int(round(d.get("steer", 0.0) / DT))
        self.tau_e = d.get("engine", (0.0, 0.0))
        self.tau_b = d.get("brake", (0.0, 0.0))
        self.g_slope = 0.0
        self.x = None

    # ---------------------------------------------------------------- 인터페이스 (sim.World 와 같음)
    def reset(self, X, Y, psi, vx, vy=0.0, r=0.0, delta=0.0, F0=0.0):
        self.x = [X, Y, psi, vx, vy - self.prm["ds"] * r, r]  # 센서 vy → 무게중심 vy
        self.delta = delta
        self.ay = 0.0
        self.ax = 0.0
        self.ax_lt = 0.0  # 하중 이동에 쓰는 (지연된) 가속도
        self.ay_lt = 0.0
        nd = lambda dl: max(int(round(dl[0] / self.h)), 0)
        self.ne, self.nb = nd(self.tau_e), nd(self.tau_b)
        self.ke = 1.0 if self.tau_e[1] <= self.h else self.h / self.tau_e[1]
        self.kb = 1.0 if self.tau_b[1] <= self.h else self.h / self.tau_b[1]
        n = max(self.ne, self.nb) + 1
        self.Fq = collections.deque([F0] * n, maxlen=n)
        self.Fe, self.Fb = max(F0, 0.0), min(F0, 0.0)
        self.dq = collections.deque([delta] * (self.n_steer + 1), maxlen=self.n_steer + 1)

    def set_grade(self, grade):
        self.g_slope = G * math.sin(float(grade))

    def set_pos(self, idx):
        if self.grade_by_idx is not None:
            self.set_grade(self.grade_by_idx[int(idx)])

    @property
    def pose(self):
        return self.x[0], self.x[1], self.x[2]

    @property
    def state(self):
        return self.x[3], self.x[4] + self.prm["ds"] * self.x[5], self.x[5]  # 무게중심 vy → 센서 vy (데이터·모델과 같은 기준)

    def __call__(self, hist):
        import torch
        row = hist[0, -1].detach().cpu().double().tolist()
        s = self.step(row[3], row[4])
        return torch.tensor([s], dtype=hist.dtype, device=hist.device)

    # ---------------------------------------------------------------- 물리
    def _forces(self, vx, vy, r, delta, Fxf_cmd):
        p = self.prm
        # 앞뒤 축 하중 (앞뒤 하중 이동: 직전 가속도 사용)
        h = p["h_cg"]
        fz_f = M * (G * LR - self.ax_lt * h) / L
        fz_r = M * (G * LF + self.ax_lt * h) / L
        fz_f, fz_r = max(fz_f, 100.0), max(fz_r, 100.0)
        # 좌우 하중 이동 → 축 최대 마찰 감소 (하중 민감도: μ_wheel = μ (1 − k·(Fz_wheel/Fz_static_wheel − 1)))
        dfz_tot = M * self.ay_lt * h
        mu_ax = []
        for fz, tw, share, mu in ((fz_f, TRACK_F, ROLL_FRONT, p["mu_f"]), (fz_r, TRACK_R, 1 - ROLL_FRONT, p["mu_r"])):
            d = min(abs(share * dfz_tot / tw), 0.49 * fz)
            fo, fi, f0 = fz / 2 + d, fz / 2 - d, fz / 2
            k = p["load_sens"]
            cap = mu * (fo * (1 - k * (fo / f0 - 1)) + fi * (1 - k * (fi / f0 - 1)))
            mu_ax.append(cap)
        cap_f, cap_r = mu_ax
        # 종방향 힘: 앞 = 명령 (제동이면 뒤에도 몫), 마찰로 제한
        if Fxf_cmd >= 0:
            fx_f, fx_r = Fxf_cmd, 0.0
        else:
            fx_f, fx_r = Fxf_cmd, Fxf_cmd * (1 - BRAKE_FRONT) / BRAKE_FRONT
        fx_f = max(min(fx_f, ABS_FRAC * cap_f), -ABS_FRAC * cap_f)
        fx_r = max(min(fx_r, ABS_FRAC * cap_r), -ABS_FRAC * cap_r)
        # 마찰원: 남은 횡 마찰
        fy_max_f = math.sqrt(max(cap_f ** 2 - fx_f ** 2, (0.05 * cap_f) ** 2))
        fy_max_r = math.sqrt(max(cap_r ** 2 - fx_r ** 2, (0.05 * cap_r) ** 2))
        # 축 강성도 하중에 비례해 변함 (정적 하중 기준)
        ca_f = p["ca_f"] * fz_f / (M * G * LR / L)
        ca_r = p["ca_r"] * fz_r / (M * G * LF / L)
        avx = max(vx, 0.5)
        a_f = math.atan2(vy + LF * r, avx) - delta
        a_r = math.atan2(vy - LR * r, avx)
        fy_f = fiala(a_f, ca_f, fy_max_f)
        fy_r = fiala(a_r, ca_r, fy_max_r)
        return fx_f, fx_r, fy_f, fy_r

    def _rhs(self, x, delta, Fxf):
        X, Y, psi, vx, vy, r = x
        p = self.prm
        fx_f, fx_r, fy_f, fy_r = self._forces(vx, vy, r, delta, Fxf)
        res = CR0 * math.copysign(1.0, vx) + CR2 * vx * vx if vx > 0.1 else 0.0
        c, s = math.cos(delta), math.sin(delta)
        dvx = (fx_f * c - fy_f * s + fx_r - res) / M + r * vy - self.g_slope
        dvy = (fy_f * c + fx_f * s + fy_r) / M - r * vx
        dr = (LF * (fy_f * c + fx_f * s) - LR * fy_r) / p["iz"]
        # 저속: 운동학 자전거로 넘어감 (r = vx·tanδ/L, β = atan(lr/L·tanδ))
        w = min(max((vx - V_KIN[0]) / (V_KIN[1] - V_KIN[0]), 0.0), 1.0)
        if w < 1.0:
            ax_k = (fx_f + fx_r - res) / M - self.g_slope
            r_k = vx * math.tan(delta) / L
            vy_k = LR * r_k
            dvx = w * dvx + (1 - w) * ax_k
            dvy = w * dvy + (1 - w) * (vy_k - vy) / 0.05
            dr = w * dr + (1 - w) * (r_k - r) / 0.05
        dX = vx * math.cos(psi) - vy * math.sin(psi)
        dY = vx * math.sin(psi) + vy * math.cos(psi)
        return [dX, dY, r, dvx, dvy, dr], (dvx - r * vy, dvy + r * vx)

    def step(self, F_cmd, delta_cmd):
        """한 제어 간격 (0.04 s)."""
        if self.cap:
            F_cmd = max(F_cmd, F_BRAKE_MIN)
            if self.force_cap is not None:
                F_cmd = min(F_cmd, float(self.force_cap(self.x[3])))
        target = self.dq[0]
        self.dq.append(delta_cmd)
        dmax = self.prm["steer_rate"] * self.h
        x = self.x
        try:
            for _ in range(self.n_sub):
                self.delta += max(min(target - self.delta, dmax), -dmax)
                self.Fq.append(F_cmd)
                self.Fe += (max(self.Fq[-1 - self.ne], 0.0) - self.Fe) * self.ke
                self.Fb += (min(self.Fq[-1 - self.nb], 0.0) - self.Fb) * self.kb
                F = self.Fe + self.Fb
                h = self.h
                k1, acc = self._rhs(x, self.delta, F)
                k2, _ = self._rhs([a + 0.5 * h * b for a, b in zip(x, k1)], self.delta, F)
                k3, _ = self._rhs([a + 0.5 * h * b for a, b in zip(x, k2)], self.delta, F)
                k4, _ = self._rhs([a + h * b for a, b in zip(x, k3)], self.delta, F)
                x = [a + h / 6 * (b + 2 * c + 2 * d + e) for a, b, c, d, e in zip(x, k1, k2, k3, k4)]
                self.ax, self.ay = acc
                tl = self.prm["tau_lt"]
                kl = 1.0 if tl <= h else h / tl
                self.ax_lt += (self.ax - self.ax_lt) * kl
                self.ay_lt += (self.ay - self.ay_lt) * kl
                if x[3] < 0.0:  # 뒤로 가지 않음 (정지)
                    x[3], x[4], x[5] = 0.0, 0.0, 0.0
        except (ValueError, OverflowError, ZeroDivisionError):
            x = [math.nan] * 6
        self.x = x
        return self.state


def make_world(fit_path=None, delays_path=None, force_table=None, **kw):
    """맞춘 계수(fit_realsim.py 출력)·저장소 지연·구동력 표를 넣은 RealWorld."""
    params = json.load(open(fit_path))["params"] if fit_path and os.path.exists(fit_path) else {}
    delays = None
    if delays_path and os.path.exists(delays_path):
        d = json.load(open(delays_path))
        delays = dict(brake=tuple(d["brake"]["use"]), engine=tuple(d["engine"]["use"]))
    if "delays" in kw:
        delays = dict(delays or {}, **(kw.pop("delays") or {}))
    cap = None
    if force_table is not None:
        vg, fg = force_table
        cap = lambda v: np.interp(v, vg, fg)
    return RealWorld(params=params, delays=delays, force_cap=cap, **kw)
