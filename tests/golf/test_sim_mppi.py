"""시뮬레이터(sim.py)·MPPI 힘 입력(mppi.py) 점검 (docs/experiment_plan_golf.md 4-4, 8-2, 9-2절)."""
import math

import numpy as np
import torch

from deep_dynamics.golf import data as D, golf_specs, mppi, sim
from deep_dynamics.golf.configs import GOLF_THUNDERHILL as CFG

TRACK = "experiments/golf/results/tracks/thunderhill_recorded.npz"
# 바퀴 4개의 회전 관성이 더하는 겉보기 질량 (가속할 때 바퀴도 돌려야 함)
M_EQ = CFG.m + 4 * sim.golf_params().I_y_w / sim.WHEEL_R ** 2


def _run(v0, F, delta=0.0, steps=25, **kw):
    w = sim.World(cap=False, **kw)
    w.reset(0.0, 0.0, 0.0, v0)
    for _ in range(steps):
        s = w.step(F, delta)
    return w, s


_res = sim.road_load


def test_params_are_golf_values():
    p = sim.golf_params()
    v = golf_specs.value
    assert (p.m, p.a, p.b, p.R_w, p.T_se, p.T_sb) == (CFG.m, CFG.lf, CFG.lr, sim.WHEEL_R, 1.0, CFG.brake_front_share)
    assert (p.m, p.a, p.b, p.R_w, p.T_sb) == (v("mass"), v("cg_a"), v("cg_b"), v("wheel_radius"), v("brake_front_share"))
    assert abs(p.m_s + p.m_uf + p.m_ur - p.m) < 1e-9
    Fz_f = p.m * sim.G * p.b / (p.a + p.b)
    assert abs(-p.tire_f.p_ky1 * Fz_f - v("tire_stiffness_f")) < 1e-6 * v("tire_stiffness_f")
    assert (p.steering.v_max, p.steering.v_min) == (v("steer_rate_max"), -v("steer_rate_max"))  # [11/21] 데이터 범위
    assert p.longitudinal.a_max * p.longitudinal.v_switch / 30.0 > 100  # Escort 가속 제한(30 m/s 에서 1.8 m/s²)이 풀림
    p, p2 = sim.golf_params(1.0), sim.golf_params(0.9)
    assert abs(p2.tire_f.p_dy1 - 0.9 * p.tire_f.p_dy1) < 1e-12 and abs(p2.tire_r.p_dx1 - 0.9 * p.tire_r.p_dx1) < 1e-12


def test_calibrated_track_and_roll():
    """[결정 11/19] Golf 윤거, 롤 기울기 0.30 도/(m/s²) (Escort 그대로면 약 1.0 → 첫 큰 코너에서 전복)."""
    p = sim.golf_params()
    assert (p.T_f, p.T_r) == (sim.TRACK_F, sim.TRACK_R)
    cal = sim.load_calibration()
    grad, _ = sim.roll_gradient(cal["mu_scale"], cal["k_roll"])
    assert abs(grad - 0.30) < 0.02, grad
    assert sim.roll_gradient(cal["mu_scale"], None)[0] > 0.8  # Escort 서스펜션 그대로면 약 1.0


def test_drive_force_gives_expected_acceleration_and_is_not_clipped():
    """Fxf → 가속도 변환 + Escort 가속 제한 해제: 30 m/s 에서 4.3 kN (약 2.4 m/s²) 이 잘리지 않음.
    [11/21 (가)] 구동은 바퀴 관성 보정으로 가속 = Fxf/m − 저항/M_EQ (데이터 효율을 추정한 식과 같은 Fxf/m)."""
    for v0, F in ((20.0, 2000.0), (30.0, 4300.0)):
        _, s = _run(v0, F)
        a = s[0] - v0
        exp = F / CFG.m - _res(v0 + 0.5 * a) / M_EQ
        assert abs(a - exp) < 0.03 * abs(exp), (v0, a, exp)


def test_brake_force_is_front_share_of_total():
    """Fxf < 0 은 앞바퀴 제동력 → 전체 제동력 = Fxf / 앞 제동 몫(0.75)."""
    v0, F = 25.0, -3000.0
    _, s = _run(v0, F)
    a = s[0] - v0
    exp = (F / CFG.brake_front_share - _res(v0 + 0.5 * a)) / M_EQ
    assert abs(a - exp) < 0.03 * abs(exp), (a, exp)


def test_straight_cruise_has_no_lateral_motion():
    """직진 정속에서 옆 흐름이 없다 (기본 설정 = 표준 타이어 좌우 거울상, 타이어 밀림 항 그대로).
    [11/22] CommonRoad 원본 식(tire_mirror=False)은 밀림 항을 sign(캠버)·하중의 3.7% 로 둬서 2초 뒤 vy −0.013 m/s 로 흘렀다.
    거울상은 좌우가 상쇄된다 (mb_golf 모듈 설명, docs/code_map.md 4절 #9)."""
    v0 = 20.0
    _, s = _run(v0, _res(v0), steps=50)
    assert abs(s[0] - v0) < 0.05 and abs(s[1]) < 1e-3 and abs(s[2]) < 1e-3, s


def test_mirrored_tires_turn_symmetrically():
    """왼쪽·오른쪽 같은 조향의 회전 응답이 대칭 (기본 = 거울상. 원본 식은 약 1% 비대칭)."""
    r = [_run(25.0, _res(25.0), delta=d, steps=100)[1][2] for d in (0.03, -0.03)]
    assert abs(r[0] + r[1]) < 1e-3 * abs(r[0]), r


def test_understeer_like_golf_data():
    """25 m/s 정상 선회의 회전 반응 / 운동학 값: Golf 데이터 0.44~0.48 (20~35 m/s). 원본 타이어면 약 0.88."""
    _, s = _run(25.0, _res(25.0), delta=0.03, steps=100)
    ratio = s[2] / (s[0] * 0.03 / (CFG.lf + CFG.lr))
    assert 0.35 < ratio < 0.65, ratio


def test_abs_keeps_wheels_rolling_when_braking_in_a_turn():
    w, s = _run(30.0, -5000.0, delta=0.01, steps=50)
    wheel = np.array(w.x[23:27]) * sim.WHEEL_R
    assert np.isfinite(s).all() and (wheel > 0.5 * s[0]).all(), wheel


def test_force_cap_table():
    vg, fg = sim.force_cap_table()
    t = {k: np.asarray(x) for k, x in golf_specs.value("drive_force_cap").items()}
    assert abs(sim.force_cap(9.0) - t["F_p99_N"][t["v_lo"] == 8][0]) < 1.0  # 8~10 m/s 구간 상위 1% (구간 가운데)
    hi = sim.force_cap(np.array([34.0, 40.0, 50.0]))
    assert (np.diff(hi) < 0).all()  # 32 m/s 이상은 일정 출력 → 속도에 반비례
    assert abs(hi[1] * 40.0 - hi[2] * 50.0) < 0.01 * hi[2] * 50.0  # 격자 사이는 선형 보간이라 1% 이내


class _Toy(torch.nn.Module):
    """다음 vx = 지금 vx + 힘/1000 (조향 무시). MPPI 비용 계산만 보려는 가짜 모델."""
    def forward(self, h):
        last = h[:, -1]
        return torch.stack([last[:, 0] + last[:, 3] / 1000.0, torch.zeros_like(last[:, 0]), torch.zeros_like(last[:, 0])], 1)


def _one_act(mode):
    tr = mppi.Track(TRACK)
    m = mppi.MPPI(_Toy(), tr, K=256, N=10, seed=0, mode=mode, use_graph=False)
    m.s_now = torch.tensor([30.0, 0.0, 0.0], device=mppi.DEV)
    pose = torch.tensor([float(tr.x[0]), float(tr.y[0]), float(tr.psi[0])], device=mppi.DEV)
    hist = torch.tensor([30.0, 0.0, 0.0, 300.0, 0.0, 0.0], device=mppi.DEV).repeat(mppi.H - 1, 1)
    m.act(hist, pose, 0, torch.tensor([300.0, 0.0], device=mppi.DEV))
    return m


def test_mppi_force_samples_respect_speed_dependent_cap():
    m = _one_act("A")
    F = m.bUc[:, :, 0]
    vx = 30.0 + torch.cumsum(F, 1) / 1000.0  # 가짜 모델이 굴린 속도 (k 번째 입력은 k 스텝 전 속도에서 적용)
    v_before = torch.cat([torch.full_like(vx[:, :1], 30.0), vx[:, :-1]], 1)
    cap = torch.tensor(sim.force_cap(v_before.cpu().numpy()), device=F.device, dtype=F.dtype)
    assert (F <= cap + 1.0).all() and (F >= mppi.U_LO_FORCE[0] - 1e-3).all()


def test_mode_b_rewards_progress():
    m = _one_act("B")
    progress = m.bUc[:, :, 0].mean(1)  # 가짜 모델에서는 힘이 클수록 더 멀리 감
    c = np.corrcoef(progress.cpu().numpy(), m.bcost.cpu().numpy())[0, 1]
    assert c < -0.5, c


def test_actuator_delays_per_path():
    """브레이크 → 구동으로 바뀔 때: 엔진 경로(지연 0)는 바로, 브레이크 경로는 자기 지연(35 ms)만큼 남았다가 풀림."""
    h = 1e-3
    a = sim.Actuator(dict(brake=(0.035, 0.0), engine=(0.0, 0.0)), h)
    a.reset(-3000.0)
    out = [a.force_step(2000.0) for _ in range(60)]
    assert abs(out[0] - (2000.0 - 3000.0)) < 1e-6  # 처음 35 ms: 엔진 2000 + 아직 남은 브레이크 −3000
    assert abs(out[-1] - 2000.0) < 1e-6  # 35 ms 뒤: 브레이크 풀림
    b = sim.Actuator(dict(brake=(0.035, 0.0), engine=(0.0, 0.0)), h)
    b.reset(2000.0)
    out = [b.force_step(-3000.0) for _ in range(60)]
    assert abs(out[0] - 0.0) < 1e-6 and abs(out[-1] + 3000.0) < 1e-6  # 엔진은 바로 0, 브레이크는 35 ms 뒤


def test_fit_coefficients_reach_params():
    """[11/19] 데이터로 맞추는 배율(ky_f, ky_r, rby_f)이 축별 타이어 계수에 들어가는지."""
    p0, p1 = sim.golf_params(), sim.golf_params(ky_f=1.3, ky_r=0.8, rby_f=2.0)
    assert abs(p1.tire_f.p_ky1 - 1.3 * p0.tire_f.p_ky1) < 1e-9 and abs(p1.tire_r.p_ky1 - 0.8 * p0.tire_r.p_ky1) < 1e-9
    assert abs(p1.tire_f.r_by1 - 2.0 * p0.tire_f.r_by1) < 1e-9 and p1.tire_r.r_by1 == p0.tire_r.r_by1


def test_episode_starts_at_i0():
    tr = mppi.Track(TRACK)
    i0 = 300
    res, log = mppi.episode(_Toy(), sim.World(), tr, K=64, N=5, max_time=0.2, use_graph=False, i0=i0)
    assert abs(log["idx"][0] - i0) < 5, log["idx"][:3]


def test_episode_time_counted_without_record():
    """11/19 결함 재발 방지: record=False 일 때 랩타임(time_s)이 0 으로 나오던 문제."""
    tr = mppi.Track(TRACK)
    res, _ = mppi.episode(_Toy(), sim.World(), tr, K=64, N=5, max_time=0.4, use_graph=False, record=False)
    assert abs(res["time_s"] - 0.4) < 0.05, res["time_s"]


def test_mode_b_reward_follows_forward_progress():
    """[11/19] B 보상 = 진행 방향 속도. 트랙 반대 방향을 보고 달리면(거꾸로) 보상 대신 벌점."""
    import math as _m
    tr = mppi.Track(TRACK)
    costs = []
    for flip in (0.0, _m.pi):
        m = mppi.MPPI(_Toy(), tr, K=128, N=10, seed=0, mode="B", use_graph=False)
        m.s_now = torch.tensor([20.0, 0.0, 0.0], device=mppi.DEV)
        pose = torch.tensor([float(tr.x[0]), float(tr.y[0]), float(tr.psi[0]) + flip], device=mppi.DEV)
        hist = torch.tensor([20.0, 0.0, 0.0, 300.0, 0.0, 0.0], device=mppi.DEV).repeat(mppi.H - 1, 1)
        m.act(hist, pose, 0, torch.tensor([300.0, 0.0], device=mppi.DEV))
        costs.append(float(m.bcost.median()))
    assert m.w["prog"] == 2.0 / mppi.DT and m.w["off"] == 1e6 and m.off_margin == 0.9
    assert costs[0] < 0 < costs[1], costs


def test_track_width_is_measured_value():
    # [11/20] OSM 트랙: 실측 14곳 위치별 폭 10.1~11.6 m, 측정 없는 곳 평균 10.9 m (track_osm.md)
    t = np.load(TRACK)
    w = t["width_l"] + t["width_r"]
    assert w.min() > 10.0 and w.max() < 11.7
    assert abs(np.median(w) - 10.9) < 0.05


def test_tuning_select_groups_by_horizon(tmp_path=None):
    import importlib.util, os, tempfile
    spec = importlib.util.spec_from_file_location("tune_mppi", "experiments/golf/tune_mppi.py")
    T = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(T)
    d = tempfile.mkdtemp()
    T.OUT, T.RES = os.path.join(d, "t.json"), d
    T.steer_delay = lambda: 0.0
    runs = []
    for N, good in ((25, (0.3, 0.5)), (50, (1.0, 2.0))):
        for lam in T.LAMS:
            for nz in T.NOISE:
                for R in T.RUNGS:
                    fin = (lam, nz) == good
                    runs.append(dict(mode="B", N=N, lam=lam, noise=nz, rung=R, finished=fin, lap_time=110.0 if fin else None,
                                     progress_frac=1.0 if fin else 0.1))
    T.save({"runs": runs})
    T.select()
    ch = T.load()["chosen"]
    assert ch["B_25"] == dict(lam=0.3, noise=0.5) and ch["B_50"] == dict(lam=1.0, noise=2.0)


def test_specs_items_have_source_and_basis():
    """[11/21] specs.json 의 모든 항목에 출처(published/data/assumed)와 근거가 있다. 데이터 값은 20개 파일."""
    items = golf_specs.load()["items"]
    for k, it in items.items():
        assert it["source"] in ("published", "data", "assumed") and len(it["basis"]) > 10, k
        if it["source"] == "data":
            assert it.get("files") == len(D.available()) == 20, k


def test_model_data_sim_read_the_same_specs():
    """[11/21] 모델(configs)·데이터(data)·시뮬레이터(sim)가 같은 specs 값을 쓴다 (손으로 옮겨 적은 곳 없음)."""
    v = golf_specs.value
    assert (CFG.m, CFG.lf, CFG.lr, CFG.brake_front_share) == (v("mass"), v("cg_a"), v("cg_b"), v("brake_front_share"))
    P = dict((n, x) for n, x, *_ in CFG.phys)
    assert (P["Iz"], P["Cr0"], P["Cr2"]) == (v("yaw_inertia"), v("resist_cr0"), v("resist_cr2"))
    assert (sim.WHEEL_R, sim.F_BRAKE_MIN, (sim.TRACK_F, sim.TRACK_R)) == (v("wheel_radius"), v("brake_force_min"), (v("track_f"), v("track_r")))
    assert sim.load_delays() == dict(brake=tuple(v("actuator_delays")["brake"]["use"]), engine=tuple(v("actuator_delays")["engine"]["use"]))
    df = D.read_raw("2023-11_9")
    assert np.allclose(df["Fxf"], v("drive_eff") * df["F_drive"] - df["F_brake_f"])
    assert np.allclose(df["F_drive"], D.drive_force(df, v("gear_ratios"), v("wheel_radius"))[0])
