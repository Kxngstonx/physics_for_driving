"""이중 궤도 시뮬레이터 (dt_golf, Subosits 2020 4.2절) 점검 (11/22)."""
import math

import numpy as np

from deep_dynamics.golf import dt_golf, sim


def _run(F, d=0.0, v0=20.0, steps=50, **kw):
    w = sim.World(cap=False, model="dt", **kw)
    w.reset(0.0, 0.0, 0.0, v0)
    for _ in range(steps):
        s = w.step(F(w) if callable(F) else F, d)
    return w, s


def test_straight_cruise_no_lateral_motion():
    w, s = _run(lambda w: sim.road_load(w.state[0]), steps=50)
    assert abs(s[0] - 20.0) < 0.01 and abs(s[1]) < 1e-9 and abs(s[2]) < 1e-9, s


def test_loads_sum_and_outer_wheels_loaded():
    w, s = _run(lambda w: sim.road_load(w.state[0]) + w.p.m * 0.5 * (25 - w.state[0]), d=0.03, v0=25.0, steps=100)
    fl, fr, rl, rr = dt_golf.loads(w.x, w.p)
    assert abs(fl + fr + rl + rr - w.p.m * 9.81) < 1e-6 * w.p.m
    assert s[2] > 0 and fr > fl and rr > rl  # 왼쪽 회전 → 바깥(오른쪽, 물리 위치 그대로)이 무거움


def test_left_right_symmetric():
    r = [_run(lambda w: sim.road_load(w.state[0]), d=d, v0=25.0, steps=100)[1][2] for d in (0.03, -0.03)]
    assert abs(r[0] + r[1]) < 1e-9 * abs(r[0]), r


def test_drive_and_brake_acceleration():
    w, s = _run(3000.0, v0=20.0, steps=25)
    assert abs((s[0] - 20.0) - (3000.0 - sim.road_load(20.7)) / w.p.m) < 0.01
    w, s = _run(-3000.0, v0=25.0, steps=25)
    assert abs((s[0] - 25.0) - (-3000.0 / w.p.T_sb - sim.road_load(24.4)) / w.p.m) < 0.02


def test_lsd_open_splits_equally_and_locked_has_no_effect_straight():
    p = dt_golf.params()
    x = dt_golf.init(0, 0, 0, 20.0)
    Fz = dt_golf.loads(x, p)
    kin = dt_golf.wheel_kin(x, p)
    open_ = dt_golf.long_forces(4000.0, 0.0, Fz, kin, x, p)
    locked = dt_golf.long_forces(4000.0, 1.0, Fz, kin, x, p)
    assert abs(open_[0] - open_[1]) < 1e-9 and np.allclose(open_, locked)  # 논문: 직진·같은 하중이면 잠가도 효과 없음
    x[5] = 0.3  # 왼쪽 회전: 잠기면 좌우 힘이 달라짐
    kin = dt_golf.wheel_kin(x, p)
    lk = dt_golf.long_forces(4000.0, 1.0, Fz, kin, x, p)
    assert abs(lk[0] + lk[1] - 4000.0) < 1e-6 and abs(lk[0] - lk[1]) > 1.0


def test_tire_forces_inside_friction_circle():
    p = dt_golf.params()
    rng = np.random.default_rng(0)
    for _ in range(300):
        Fz = rng.uniform(500, 9000)
        mu = dt_golf._mu(Fz, "f", p)
        Fx = rng.uniform(-0.99, 0.99) * mu * Fz
        a = rng.uniform(-0.3, 0.3)
        Fy = dt_golf.tire_fy(a, Fz, Fx, "f", p)
        assert math.hypot(Fx, Fy) <= mu * Fz * (1 + 1e-9), (Fz, Fx, a, Fy)


def test_tire_shape_override_keeps_axle_stiffness_and_friction():
    """[11/22] μd_fz·c2 를 바꿔도 (맞춤) 축 강성(정적 하중)·축 실효 마찰(한계 하중 상태)이 데이터(specs)와 같게 유지된다."""
    from deep_dynamics.golf import golf_specs
    v = golf_specs.value
    for mu_dfz, c2 in ((-0.13, 1.14), (-0.05, 2.0), (-0.2, 1.0)):
        p = dt_golf.params(mu_dfz=mu_dfz, c2=c2)
        L, dW = p.a + p.b, 0.5 * (p.df + p.dr)
        for ax, Fz_s, C, mu, share in (("f", p.m * 9.81 * p.b / L / 2, v("tire_stiffness_f"), v("tire_mu_f"), p.gamma),
                                       ("r", p.m * 9.81 * p.a / L / 2, v("tire_stiffness_r"), v("tire_mu_r"), 1 - p.gamma)):
            c0 = getattr(p, "c1_" + ax) * p.Fz0 * math.sin(2 * math.atan(Fz_s / (p.c2 * p.Fz0)))
            assert abs(2 * c0 - C) < 1e-6 * C
            dla = p.h * p.m * mu * 9.81 / dW
            cap = sum(dt_golf._mu(z, ax, p) * z for z in (Fz_s - share * dla, Fz_s + share * dla))
            assert abs(cap - mu * 2 * Fz_s) < 1e-6 * cap
