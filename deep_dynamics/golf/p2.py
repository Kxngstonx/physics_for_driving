"""P2 자전거 시뮬레이터 (11/25 결정: 정답 세계로 쓴다). 우리 저장소 physics/bicycle.py P2 를 옮김.

  from deep_dynamics.golf import p2
  w = p2.P2World(grade_by_idx=grade)           # mppi.episode(model, w, track, ...) 의 world
  w = p2.P2World.from_fit()                     # experiments/golf/results/sim/p2_fit.json (fit_p2.py) 값

R1(사다리 순수 물리)과 다른 점 — 세계가 R1 식과 같으면 R1 이 구조적으로 유리해지므로:
  - 타이어 최대 힘 = μ·수직 하중 (R1 은 축마다 고정 D)
  - 앞뒤 하중 이동: 가속하면 뒤로, 제동하면 앞으로 (무게중심 높이 h)
  - 결합 슬립(마찰원 derate): 앞뒤로 쓴 마찰만큼 옆으로 낼 수 있는 최대 힘이 줄어듦
  - RK4 (R1 은 오일러)
같은 점: 단일 트랙(자전거), Pacejka 옆 힘, 굴림·공기 저항, 종경사 −g·sin(경사).

상태 x = [X, Y, psi, ux, uy, r] (무게중심 기준). 입력 = 조향각 δ, 전체 종방향 힘 Fx.
  앞바퀴 힘 Fxf(모델·MPPI 입력) → Fx: 구동 Fx = Fxf (앞바퀴굴림), 제동 Fx = Fxf / brake_front (나머지는 뒷바퀴 브레이크).
측정 vy 는 무게중심보다 ds 앞에서 잰 값: vy_측정 = uy + ds·r (D7 데이터와 같은 기준으로 내보낸다).
타이어 부호: 슬립각이 양수면 옆 힘은 음수 (Spielberg 코드와 같음). B 는 코너링 강성이 c·Fz 가 되도록 B = c/(C·μ).
"""
import json
import math
import os
from dataclasses import asdict, dataclass

import torch

from .configs import GOLF_THUNDERHILL as CFG

G = 9.81
DT = CFG.dt
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FIT = os.path.join(REPO, "experiments", "golf", "results", "sim", "p2_fit.json")
DELTA_MAX = 0.61                     # D7 최대 조향각 [rad]
# 액추에이터 지연 기본값 = 팀원 fit_delays.py 결과 (브레이크 순수 지연 35 ms + 1차 지연 20 ms, 엔진 0). results/sim/delays.json 이 있으면 그것을 씀
DEFAULT_DELAYS = dict(brake=(0.035, 0.02), engine=(0.0, 0.0))


@dataclass
class P2Params:
    m: float = CFG.m                 # 질량 [kg] (팀 기준 configs.GOLF_THUNDERHILL)
    L: float = CFG.lf + CFG.lr       # 축거 [m]
    a: float = CFG.lf                # 무게중심 ~ 앞 차축 [m]
    Iz: float = 2500.0               # 요 관성 [kg m²]
    h: float = 0.5                   # 무게중심 높이 [m]
    cf: float = 12.0                 # 앞 코너링 강성 / 수직 하중 [1/rad]
    cr: float = 17.0                 # 뒤
    Cf: float = 1.3                  # Pacejka 형상 계수 앞
    Cr: float = 1.8                  # 뒤
    mu: float = 1.09                 # 노면 마찰
    c0: float = 200.0                # 굴림 저항 [N]
    c2: float = 0.5                  # 공기 저항 [N/(m/s)²]
    ds: float = 0.3                  # 측정 vy 위치 (무게중심 앞) [m]
    brake_front: float = CFG.brake_front_share   # 제동 중 앞바퀴 몫 (0.75)
    drive_front: float = 1.0         # 앞바퀴굴림

    @property
    def b(self):
        return self.L - self.a


def _pacejka(alpha, fz, mu, c, C):
    B = c / (C * mu)
    return -mu * fz * torch.sin(C * torch.atan(B * alpha))


def _derate(fx, fz, mu, floor=0.02):
    """종방향 힘을 μFz 안으로 자르고, 옆으로 남은 마찰계수 μ_y = sqrt(μ² − (Fx/Fz)²) (마찰원)."""
    fmax = mu * fz * math.sqrt(1.0 - floor ** 2)
    fx_c = torch.maximum(torch.minimum(fx, fmax), -fmax)
    mu_y = torch.sqrt(torch.clamp(mu ** 2 - (fx_c / fz) ** 2, min=(floor * mu) ** 2))
    return fx_c, mu_y


def derivatives(x, delta, fx, p, grade=None):
    """x (..., 6), delta·fx·grade (...) → dx/dt (..., 6). p 의 값은 float 또는 텐서(학습용)."""
    psi, ux, uy, r = x[..., 2], x[..., 3], x[..., 4], x[..., 5]
    m, a, L = p.m, p.a, p.L
    b = L - a
    fxf = torch.where(fx > 0, p.drive_front * fx, p.brake_front * fx)
    fxr = fx - fxf
    ax = fx / m                                          # 하중 이동 (앞뒤)
    fzf = torch.clamp(m * G * b / L - p.h * m * ax / L, min=0.05 * m * G)
    fzr = torch.clamp(m * G * a / L + p.h * m * ax / L, min=0.05 * m * G)
    uxs = torch.clamp(ux, min=1.0)
    alpha_f = torch.atan((uy + a * r) / uxs) - delta
    alpha_r = torch.atan((uy - b * r) / uxs)
    fxf, muf = _derate(fxf, fzf, p.mu)                   # 결합 슬립
    fxr, mur = _derate(fxr, fzr, p.mu)
    fyf = _pacejka(alpha_f, fzf, muf, p.cf, p.Cf)
    fyr = _pacejka(alpha_r, fzr, mur, p.cr, p.Cr)
    cd, sd = torch.cos(delta), torch.sin(delta)
    res = p.c0 + p.c2 * ux * ux
    ux_dot = (fxf * cd - fyf * sd + fxr - res) / m + r * uy
    uy_dot = (fyf * cd + fxf * sd + fyr) / m - r * ux
    r_dot = (a * (fyf * cd + fxf * sd) - b * fyr) / p.Iz
    if grade is not None:
        ux_dot = ux_dot - G * torch.sin(grade)
    return torch.stack([ux * torch.cos(psi) - uy * torch.sin(psi), ux * torch.sin(psi) + uy * torch.cos(psi),
                        r, ux_dot, uy_dot, r_dot], -1)


def rk4(x, delta, fx, p, h, grade=None):
    k1 = derivatives(x, delta, fx, p, grade)
    k2 = derivatives(x + 0.5 * h * k1, delta, fx, p, grade)
    k3 = derivatives(x + 0.5 * h * k2, delta, fx, p, grade)
    k4 = derivatives(x + h * k3, delta, fx, p, grade)
    return x + h / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)


def fx_total(Fxf, brake_front):
    """앞바퀴 힘 (모델·MPPI 입력) → 전체 종방향 힘."""
    return Fxf if Fxf >= 0 else Fxf / brake_front


def load_fit(path=FIT):
    assert os.path.exists(path), f"{path} 없음: python experiments/golf/fit_p2.py 를 먼저 실행"
    d = json.load(open(path))
    return P2Params(**{k: v for k, v in d["params"].items() if k in P2Params.__dataclass_fields__})


class P2World:
    """mppi.episode() 의 world (sim.World 와 같은 형식). world(h) → 다음 [vx, vy_측정, r].
    h[0, -1] = [vx, vy, r, Fxf 명령, 조향 명령, bank(, grade)]. pose = (X, Y, psi). 경사는 set_grade / set_pos 로.
    지연(delays)·구동력 상한·제동 하한은 sim.World 와 같은 코드(sim.Actuator, sim.force_cap, sim.F_BRAKE_MIN)."""

    def __init__(self, params=None, delays=None, h=0.01, grade_by_idx=None, cap=True):
        from . import sim
        self._sim = sim
        self.p = params or P2Params()
        self.act = sim.Actuator(delays, h)
        self.h, self.cap = h, cap
        self.n_sub = int(round(DT / h))
        self.grade_by_idx = grade_by_idx
        self.grade = 0.0
        self.x, self.delta = None, 0.0

    @classmethod
    def from_fit(cls, path=FIT, **kw):
        if "delays" not in kw:
            from . import sim
            kw["delays"] = sim.load_delays() or DEFAULT_DELAYS
        return cls(params=load_fit(path), **kw)

    def reset(self, X, Y, psi, vx, vy=0.0, r=0.0, delta=0.0, F0=0.0):
        """vy 는 측정 기준 (무게중심 앞 ds) → 무게중심 기준으로 바꿔 저장."""
        self.x = torch.tensor([X, Y, psi, vx, vy - self.p.ds * r, r], dtype=torch.float64)
        self.delta = float(delta)
        self.act.reset(F0, delta)

    def set_grade(self, grade):
        self.grade = float(grade)

    @property
    def g_slope(self):
        """g·sin(경사) (sim.World 와 같은 이름, 운전자 앞먹임용)."""
        return G * math.sin(self.grade)

    def lin_stiffness(self):
        """선형 구간 축 코너링 강성 [N/rad] (정적 하중 × 정규화 강성). 모델 없는 운전자의 언더스티어 보정용."""
        p = self.p
        return p.cf * p.m * G * p.b / p.L, p.cr * p.m * G * p.a / p.L

    def set_pos(self, idx):
        if self.grade_by_idx is not None:
            self.set_grade(self.grade_by_idx[int(idx)])

    @property
    def pose(self):
        return float(self.x[0]), float(self.x[1]), float(self.x[2])

    @property
    def state(self):
        ux, uy, r = (float(v) for v in self.x[3:6])
        return ux, uy + self.p.ds * r, r

    def step(self, F_cmd, delta_cmd):
        """한 제어 간격 (0.04 s). 조향은 지연된 목표에 0.04 s 동안 일정한 속도로 도달 (sim.World 와 같음)."""
        sim = self._sim
        if self.cap:
            F_cmd = min(max(F_cmd, sim.F_BRAKE_MIN), float(sim.force_cap(float(self.x[3]))))
        target = max(-DELTA_MAX, min(DELTA_MAX, self.act.steer_target(delta_cmd)))
        dv = (target - self.delta) / DT
        g = torch.tensor(self.grade, dtype=torch.float64)
        x = self.x
        for _ in range(self.n_sub):
            self.delta += dv * self.h
            fx = fx_total(self.act.force_step(F_cmd), self.p.brake_front)
            x = rk4(x, torch.tensor(self.delta, dtype=torch.float64), torch.tensor(fx, dtype=torch.float64), self.p, self.h, g)
        self.x = x if torch.isfinite(x).all() else torch.full_like(x, math.nan)
        return self.state

    def __call__(self, hist):
        row = hist[0, -1].detach().cpu().double().tolist()
        s = self.step(row[3], row[4])
        return torch.tensor([s], dtype=hist.dtype, device=hist.device)


def save_fit(params, path=FIT, **extra):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump(dict(params=asdict(params), **extra), open(path, "w"), indent=1)
