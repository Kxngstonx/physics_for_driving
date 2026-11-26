"""[11/25] P2 자전거 시뮬레이터 (deep_dynamics/golf/p2.py) 검증. 맞춘 계수 파일 없이 기본값으로."""
import math
import os

import torch

from deep_dynamics.golf import p2

TRACK = "experiments/golf/results/tracks/thunderhill_recorded.npz"


def _x(vx, vy=0.0, r=0.0):
    return torch.tensor([[0.0, 0.0, 0.0, vx, vy, r]], dtype=torch.float64)


def _t(v):
    return torch.tensor([v], dtype=torch.float64)


def test_grade_slows_uphill():
    p = p2.P2Params()
    flat = p2.derivatives(_x(25.0), _t(0.0), _t(0.0), p, _t(0.0))[0, 3]
    up = p2.derivatives(_x(25.0), _t(0.0), _t(0.0), p, _t(0.05))[0, 3]
    assert abs(float(flat - up) - 9.81 * math.sin(0.05)) < 1e-9


def test_braking_reduces_cornering_force():
    """같은 미끄럼각에서 세게 제동하면 (마찰원 + 뒤 하중 감소) 회전 가속도가 달라진다 — R1 에는 없는 성질."""
    p = p2.P2Params()
    coast = p2.derivatives(_x(25.0), _t(0.05), _t(0.0), p)[0]
    brake = p2.derivatives(_x(25.0), _t(0.05), _t(-12000.0), p)[0]
    assert float(brake[3]) < -5.0                                   # 감속
    assert abs(float(brake[4] - coast[4])) > 0.1 or abs(float(brake[5] - coast[5])) > 0.01


def test_friction_circle_respected():
    p = p2.P2Params()
    fx_c, mu_y = p2._derate(_t(1e5), _t(5000.0), p.mu)
    assert float(fx_c) <= p.mu * 5000.0 and float(mu_y) > 0


def test_world_interface_and_mppi_step():
    w = p2.P2World(grade_by_idx=[0.02] * 10, delays=p2.DEFAULT_DELAYS)
    w.reset(0.0, 0.0, 0.0, 20.0, vy=0.1, r=0.0, F0=500.0)
    w.set_pos(3)
    assert abs(w.g_slope - 9.81 * math.sin(0.02)) < 1e-12
    for _ in range(25):
        s = w.step(500.0, 0.02)
    assert all(math.isfinite(v) for v in s) and w.pose[0] > 15.0
    cf, cr = w.lin_stiffness()
    assert cf > 0 and cr > 0
    if os.path.exists(TRACK):
        from deep_dynamics.golf import ladder, mppi
        from deep_dynamics.golf.configs import GOLF_THUNDERHILL_G
        tr = mppi.Track(TRACK)
        res, log = mppi.episode(ladder.build("R1", GOLF_THUNDERHILL_G).to(mppi.DEV), p2.P2World(), tr, K=64, N=10, max_time=2.0, use_graph=False)
        assert len(log["vx"]) > 0
