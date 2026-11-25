"""새 모델 N1·N2 (11/25, shinyoung). R1–R6 과 같은 형식 (hist (B, H, nf) → 다음 [vx, vy, r]) 이라
train.fit · evaluate.py · mppi 를 그대로 쓴다. ladder.build("N1", cfg) 로 만든다 (ladder 가 이 파일을 불러옴).

  N1 P2 물리      : R1 대신 P2 식 (타이어 최대 힘 μ·Fz, 앞뒤 하중 이동, 마찰원, 경사, RK4). 계수는 전역 하나 (학습)
  N2 P2 + 힘 보정 : N1 의 미분 식 안에 MLP(이력) 보정 가속도 [Δax, Δay, Δṙ] 를 더함. 시작은 N1 과 같음 (마지막 층 0)
                    R2 는 다음 상태에 직접 더하지만, N2 는 힘(가속도) 자리에 더해 적분을 거친다

물리 계수 초기값 = experiments/golf/fit_p2.py 결과 (11/25, D7 학습 파일, 1초 롤아웃). 범위는 fit_p2.SPEC 과 같음.
입력 lon = 앞바퀴 힘 Fxf (팀 정의). 전체 Fx = Fxf (구동, 앞바퀴굴림), Fxf / 앞 제동 몫 (제동). 좌우 기울기(bank)는 쓰지 않음 (팀 결정 0).
"""
import torch
import torch.nn as nn

from .ladder import Base, G, mlp

# 이름: (초기값 = 11/25 P2 피팅, 하한, 상한)
P2_SPEC = [
    ("a", 1.420, 0.85, 1.45), ("Iz", 4099.0, 1500.0, 4500.0), ("cf", 12.73, 5.0, 40.0), ("cr", 18.32, 5.0, 50.0),
    ("Cf", 1.248, 1.05, 1.95), ("Cr", 1.856, 1.05, 1.95), ("mu", 1.144, 0.7, 1.5), ("h", 0.273, 0.25, 0.75),
    ("c0", 30.4, 0.0, 800.0), ("c2", 0.585, 0.0, 1.5), ("ds", 0.349, -1.5, 1.5),
]
NSUB = 2          # 0.04 s 를 RK4 2번 (0.02 s). P2 피팅은 0.01 s — 비교는 tests/golf/test_hybrid.py
FLOOR = 0.02      # 마찰원: 옆으로 남기는 최소 마찰 비율 (p2._derate 와 같음)


class P2Phys(nn.Module):
    """전역 P2 계수. sigmoid 로 [하한, 상한] 안에. 이름이 'g.raw' 로 끝나 train.fit 에서 학습률 3배 (R1 계수와 같은 취급)."""

    def __init__(self, spec=P2_SPEC):
        super().__init__()
        self.names = [s[0] for s in spec]
        lo = torch.tensor([s[2] for s in spec])
        hi = torch.tensor([s[3] for s in spec])
        z = ((torch.tensor([s[1] for s in spec]) - lo) / (hi - lo)).clamp(1e-3, 1 - 1e-3)
        self.register_buffer("lo", lo)
        self.register_buffer("rng", hi - lo)
        self.raw = nn.Parameter(torch.log(z / (1 - z)))

    def values(self):
        return torch.sigmoid(self.raw) * self.rng + self.lo

    def P(self):
        return dict(zip(self.names, self.values().unbind()))


def _derate(fx, fz, mu):
    fmax = mu * fz * (1.0 - FLOOR ** 2) ** 0.5
    fx_c = torch.maximum(torch.minimum(fx, fmax), -fmax)
    mu_y = torch.sqrt(torch.maximum(mu ** 2 - (fx_c / fz) ** 2, (FLOOR * mu) ** 2))
    return fx_c, mu_y


def p2_deriv(cfg, P, ux, uy, r, fx, delta, grade):
    """무게중심 기준 [ux, uy, r] 미분 (p2.derivatives 와 같은 식, 위치 없이). fx = 전체 종방향 힘."""
    m, L = cfg.m, cfg.lf + cfg.lr
    a = P["a"]
    b = L - a
    fxf = torch.where(fx > 0, fx, cfg.brake_front_share * fx)
    fxr = fx - fxf
    ax = fx / m
    fzf = torch.clamp(m * G * b / L - P["h"] * m * ax / L, min=0.05 * m * G)
    fzr = torch.clamp(m * G * a / L + P["h"] * m * ax / L, min=0.05 * m * G)
    uxs = torch.clamp(ux, min=1.0)
    alpha_f = torch.atan((uy + a * r) / uxs) - delta
    alpha_r = torch.atan((uy - b * r) / uxs)
    fxf, muf = _derate(fxf, fzf, P["mu"])
    fxr, mur = _derate(fxr, fzr, P["mu"])
    fyf = -muf * fzf * torch.sin(P["Cf"] * torch.atan(P["cf"] / (P["Cf"] * muf) * alpha_f))
    fyr = -mur * fzr * torch.sin(P["Cr"] * torch.atan(P["cr"] / (P["Cr"] * mur) * alpha_r))
    c, s = torch.cos(delta), torch.sin(delta)
    dux = (fxf * c - fyf * s + fxr - P["c0"] - P["c2"] * ux * ux) / m + r * uy - G * torch.sin(grade)
    duy = (fyf * c + fxf * s + fyr) / m - r * ux
    dr = (a * (fyf * c + fxf * s) - b * fyr) / P["Iz"]
    return torch.stack([dux, duy, dr], -1)


def p2_step(cfg, P, s, u, extra=None, nsub=NSUB):
    """s (B,3) [vx, vy_측정, r], u (B, ≥2) [Fxf, δ, (bank), (grade)] → cfg.dt 뒤 상태. extra (B,3): 더할 가속도 (N2)."""
    lon, delta = u[:, 0], u[:, 1]
    grade = u[:, 3] if u.shape[1] > 3 else torch.zeros_like(lon)
    fx = torch.where(lon >= 0, lon, lon / cfg.brake_front_share)
    ds = P["ds"]
    x = torch.stack([s[:, 0], s[:, 1] - ds * s[:, 2], s[:, 2]], -1)
    f = lambda z: p2_deriv(cfg, P, z[:, 0], z[:, 1], z[:, 2], fx, delta, grade) + (0.0 if extra is None else extra)
    h = cfg.dt / nsub
    for _ in range(nsub):
        k1 = f(x)
        k2 = f(x + 0.5 * h * k1)
        k3 = f(x + 0.5 * h * k2)
        k4 = f(x + h * k3)
        x = x + h / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
    return torch.stack([x[:, 0], x[:, 1] + ds * x[:, 2], x[:, 2]], -1)


class N1(Base):
    uses_hist = False

    def __init__(self, cfg, norm):
        super().__init__(cfg, norm)
        self.g = P2Phys()

    def forward(self, hist):
        return p2_step(self.cfg, self.g.P(), hist[:, -1, :3], hist[:, -1, 3:])


class N2(Base):
    """보정 가속도 크기: 출력 × ACC_SCALE (m/s², m/s², rad/s²). 이력 8스텝 MLP 는 R2 와 같은 크기 [64, 64]."""
    ACC_SCALE = (1.0, 1.0, 1.0)

    def __init__(self, cfg, norm):
        super().__init__(cfg, norm)
        self.g = P2Phys()
        self.res = mlp(cfg.nf * cfg.hist, [64, 64], 3)
        nn.init.zeros_(self.res[-1].weight)
        self.register_buffer("acc_scale", torch.tensor(self.ACC_SCALE))

    def forward(self, hist):
        extra = self.res(self.norm.x(hist)) * self.acc_scale
        return p2_step(self.cfg, self.g.P(), hist[:, -1, :3], hist[:, -1, 3:], extra=extra)


MODELS = dict(N1=N1, N2=N2)
MODEL_KO = dict(N1="P2 물리", N2="P2 + 힘 보정")
