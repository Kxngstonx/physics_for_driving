"""[11/25] 새 모델 N1·N2 (deep_dynamics/golf/hybrid.py)."""
import os
import tempfile

import torch

from deep_dynamics.golf import hybrid, ladder, p2
from deep_dynamics.golf.configs import GOLF_THUNDERHILL_G as CFG


def _batch(B=64, seed=0):
    g = torch.Generator().manual_seed(seed)
    s = torch.stack([torch.rand(B, generator=g) * 30 + 5, torch.randn(B, generator=g) * 0.5, torch.randn(B, generator=g) * 0.3], -1)
    u = torch.stack([torch.rand(B, generator=g) * 12000 - 8000, torch.randn(B, generator=g) * 0.05, torch.zeros(B),
                     torch.randn(B, generator=g) * 0.03], -1)
    return s.double(), u.double()


def test_n1_matches_p2_simulator():
    """같은 계수·같은 적분 간격(0.01 s)이면 P2 시뮬레이터(p2.rk4)와 같은 답."""
    s, u = _batch()
    P = {k: v.double() for k, v in ladder.build("N1", CFG).g.P().items()}
    pp = p2.P2Params(**{k: float(v) for k, v in P.items()})
    a = hybrid.p2_step(CFG, P, s, u, nsub=4)
    z = torch.zeros(len(s), dtype=torch.float64)
    x = torch.stack([z, z, z, s[:, 0], s[:, 1] - pp.ds * s[:, 2], s[:, 2]], -1)
    fx = torch.where(u[:, 0] >= 0, u[:, 0], u[:, 0] / CFG.brake_front_share)
    for _ in range(4):
        x = p2.rk4(x, u[:, 1], fx, pp, 0.01, u[:, 3])
    b = torch.stack([x[:, 3], x[:, 4] + pp.ds * x[:, 5], x[:, 5]], -1)
    assert float((a - b).abs().max()) < 1e-8


def test_n2_starts_as_n1_and_learns():
    torch.manual_seed(0)
    hist = torch.randn(32, CFG.hist, CFG.nf)
    hist[..., 0] = hist[..., 0].abs() * 5 + 15
    hist[..., 3] *= 2000
    hist[..., 4] *= 0.03
    hist[..., 5] = 0.0
    hist[..., 6] *= 0.02
    n1, n2 = ladder.build("N1", CFG), ladder.build("N2", CFG)
    assert torch.allclose(n1(hist), n2(hist), atol=1e-6)               # 보정 마지막 층 0 → 처음엔 N1 과 같음
    n2(hist).pow(2).sum().backward()
    assert n2.res[-1].weight.grad.abs().sum() > 0 and n2.g.raw.grad.abs().sum() > 0


def test_save_load_and_grade_effect():
    m = ladder.build("N2", CFG)
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "N2_s0_roll25.pt")
        ladder.save_model(m, "N2", p)
        m2 = ladder.load_model(p, CFG)
    s, u = _batch(8)
    hist = torch.cat([s, u], 1).float().unsqueeze(1).repeat(1, CFG.hist, 1)
    assert torch.allclose(m.eval()(hist), m2(hist))
    up = hist.clone()
    up[:, -1, 6] = 0.05
    flat = hist.clone()
    flat[:, -1, 6] = 0.0
    with torch.no_grad():
        assert (m2(up)[:, 0] < m2(flat)[:, 0]).all()                    # 오르막이면 vx 가 덜 늘어남


def test_n5_starts_as_n1_and_ensemble():
    torch.manual_seed(1)
    hist = torch.randn(16, CFG.hist, CFG.nf)
    hist[..., 0] = hist[..., 0].abs() * 5 + 15
    hist[..., 3] *= 2000
    hist[..., 4] *= 0.03
    hist[..., 5] = 0.0
    hist[..., 6] *= 0.02
    n1, n5 = ladder.build("N1", CFG), ladder.build("N5", CFG)
    assert torch.allclose(n1(hist), n5(hist), atol=1e-6)
    n5(hist).pow(2).sum().backward()
    assert n5.net[-1].weight.grad.abs().sum() > 0 and n5.g.raw.grad.abs().sum() > 0
    e = hybrid.Ensemble([n1, n5])
    with torch.no_grad():
        assert torch.allclose(e(hist), (n1(hist) + n5(hist)) / 2)
