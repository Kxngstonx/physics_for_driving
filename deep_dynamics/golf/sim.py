"""정답 역할의 시뮬레이터: CommonRoad 다물체(4륜) 모델 + 변환 단계 (docs/experiment_plan_golf.md 4-4, 8절).

기본 차량은 Ford Escort(CommonRoad ID 1, 앞바퀴굴림, 앞 무게 몫 0.63)이고, Golf 에서 알거나 가정한 값은 덮어쓴다
(golf_params). 서스펜션 스프링·무게중심 높이·바퀴 쪽 질량 등은 Escort 값을 빌린다.
  - 타이어: CommonRoad 의 모든 차량이 공유하는 한 벌(ADAMS handbook)에서 앞뒤 축 횡강성만 Golf 데이터 추정값에 맞춤.
    원본은 거의 중립 조향(25 m/s 에서 회전 반응 = 운동학 값의 0.88배)인데 실제 Golf 는 0.44~0.48배 [확인 11/18].
    그래서 다물체 식을 축별 타이어·ABS 를 받게 복사했다 (mb_golf.py). 패키지 원본은 고치지 않음
  - [결정 11/19] 윤거 Golf 공개 제원, 롤 강성·마찰 배율은 calibrate_sim.py 로 보정 (calibration.json). calibrated_world() 로 씀
이 파일의 rhs() 가 다물체 식 위에 더하는 것:
  - 굴림·공기 저항 road_load(v) = Cr0 + Cr2·v² (원본 다물체 모델에는 없음). 값은 질량 추정에 쓴 가정값 (configs)
  - 종경사의 중력 성분 −g·sin(경사). 경사 값은 부르는 쪽이 set_grade / set_pos(grade_by_idx) 로 넣는다.
    [변경 11/21] 원천은 OSM 경사 지도 (validate_sim: data.grade_at, 트랙 npz 의 grade) — 학습 데이터·MPPI 와 같음.
    이전은 관성 센서 고도 변화(Vu/속도). 데이터의 grade_rad 열은 고도 변화와 상관 −0.09 로 맞지 않아 쓰지 않음 [확인 11/18].
    횡경사는 넣지 않음 (평평한 길)

입력 변환 (패키지 식: 입력 가속도 a → 엔진 토크 m·R_w·a·T_se 또는 브레이크 토크 m·R_w·a·T_sb):
  앞바퀴 힘 Fxf ≥ 0 → a = Fxf·k_w / m, k_w = 1 + 4·I_w/(R²·m)  (T_se = 1, 앞바퀴굴림)
       [변경 11/21, 사용자 결정 (가)] 데이터 효율은 바퀴·구동계 회전 관성을 흡수한 '실효' 값인데 이 모델은 바퀴 관성을 따로 계산한다.
       구동 입력을 k_w 만큼 키워, 구동 시 가속 = Fxf / m (데이터 효율을 추정한 식과 같음)이 되게 한다. 제동은 그대로.
  앞바퀴 힘 Fxf < 0 → a = Fxf / (T_sb·m)  (T_sb = 주행 중 앞 제동 몫, specs. 나머지는 뒷바퀴 브레이크)
[변경 11/21] 차량 값(바퀴 반지름, 윤거, 롤 앞 몫, ABS, 제동 하한, 코너링 강성, 구동력 상한표, 지연, 조향 속도 한계)은
  deep_dynamics/golf/specs.json 에서 읽는다 (estimate_specs.py 총괄). 무게중심 높이: 공개 0.56 m ≈ Escort 전체 무게중심 0.558 m 라
  Escort 값(스프링 위 질량 높이 h_s 0.594)을 그대로 둔다.
Escort 의 가속·최고 속도·조향 속도 제한은 Golf 보다 약해서 푼다. 엔진 한계는 속도별 구동력 표(force_cap)로 준다.
"""
import collections
import copy
import functools
import json
import math
import os

import numpy as np
from vehiclemodels.init_mb import init_mb
from vehiclemodels.parameters_vehicle1 import parameters_vehicle1

from . import golf_specs
from .configs import GOLF_THUNDERHILL as CFG
from .mb_golf import vehicle_dynamics_mb

G = 9.81
DT = CFG.dt
_v = golf_specs.value
WHEEL_R = _v("wheel_radius")
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CALIB = os.path.join(REPO, "experiments/golf/results/sim/calibration.json")  # experiments/golf/calibrate_sim.py 출력
TRACK_F, TRACK_R = _v("track_f"), _v("track_r")  # 윤거 (공개, Subosits 2020). Escort(1.39/1.42 m)는 15 cm 좁아 전복에 가까웠다
ROLL_FRONT_SHARE = _v("roll_front_share")  # 롤 강성 앞 몫 (가정: Escort 원래 값 유지)
ABS_SLIP = tuple(_v("abs_slip"))  # 이 미끄럼률 사이에서 브레이크 토크를 1 → 0 (가정)
F_BRAKE_MIN = _v("brake_force_min")  # 제동 하한 (N, 데이터 20개 파일). 이전 −6000 (2개 파일)
STEER_RATE_MAX = _v("steer_rate_max")  # 조향 속도 한계 (rad/s, 자동 조향 99.9%). 이전 ±10 (사실상 없음)
POWER_BINS = (28.0, 32.0)  # 이 속도 구간의 출력(힘×속도) 평균으로 32 m/s 이상을 이어 씀 (32~34 구간은 표본 적음)
_PHYS = {n: v for n, v, *_ in CFG.phys}
CR0, CR2 = _PHYS["Cr0"], _PHYS["Cr2"]


def road_load(v):
    """굴림·공기 저항 (N). 시뮬레이터·MPPI 초기값·보정·테스트가 모두 이것을 씀."""
    return CR0 * math.copysign(1.0, v) + CR2 * v * v


def load_calibration(path=CALIB):
    return json.load(open(path)) if os.path.exists(path) else {}


def load_delays():
    d = _v("actuator_delays")
    return dict(brake=tuple(d["brake"]["use"]), engine=tuple(d["engine"]["use"]))


def roll_stiffness(p):
    """전체 롤 강성 (N m/rad): 스프링 K_s·T²/2 − 안티롤바 K_ts (패키지 부호 규약상 음수가 복원) [확인: mb_golf.py 의 F_S** 식]."""
    return p.K_sf * p.T_f ** 2 / 2 + p.K_sr * p.T_r ** 2 / 2 - p.K_tsf - p.K_tsr


def set_roll_stiffness(p, k_roll):
    """전체 롤 강성 k_roll 을 앞 ROLL_FRONT_SHARE 로 나눠 안티롤바로 맞춘다 (스프링은 Escort 그대로)."""
    p.K_tsf = -(ROLL_FRONT_SHARE * k_roll - p.K_sf * p.T_f ** 2 / 2)
    p.K_tsr = -((1 - ROLL_FRONT_SHARE) * k_roll - p.K_sr * p.T_r ** 2 / 2)
    return p


def _tire_stiffness():
    """앞·뒤 축 코너링 강성 (N/rad): Fiala 맞춤 (specs, 11/21). 이전: 직선 회귀 85.7 / 75.0 kN/rad."""
    return _v("tire_stiffness_f"), _v("tire_stiffness_r")


def golf_params(mu_scale=1.0, k_roll=None, ky_f=1.0, ky_r=1.0, rby_f=1.0, tire_mirror=True):
    """CommonRoad 파라미터 (Escort 위에 Golf 값). k_roll None = Escort 서스펜션 그대로 (보정값은 calibrated_world 가 넣음).
    [11/19] 실제 주행 데이터로 함께 맞추는 배율 (experiments/golf/fit_sim.py): ky_f·ky_r = 앞·뒤 타이어 옆 강성 배율,
    rby_f = 앞 타이어가 앞으로 미는 동안 옆 힘이 줄어드는 정도(PAC2002 결합 미끄럼 r_by1) 배율."""
    p = parameters_vehicle1()
    p.m = CFG.m
    p.m_s = CFG.m - p.m_uf - p.m_ur
    p.a, p.b = CFG.lf, CFG.lr
    p.I_z = _PHYS["Iz"]
    p.R_w = WHEEL_R
    p.T_f, p.T_r = TRACK_F, TRACK_R
    if k_roll is not None:
        set_roll_stiffness(p, k_roll)
    p.T_se, p.T_sb = 1.0, CFG.brake_front_share
    p.abs_slip = ABS_SLIP
    p.tire_mirror = tire_mirror  # 타이어 좌우 거울상 (mb_golf 모듈 설명). [11/22 채택] 기본 True. False = CommonRoad 원본 식 (비교 재현용)
    # 제한 풀기: 가속 상한 a_max·v_switch/v (Escort: 30 m/s 에서 1.8 m/s², Golf 데이터 2.9), 최고 속도 45.8, 조향 속도 0.4 rad/s
    p.longitudinal.a_max = 1e3
    p.longitudinal.v_switch = 1e3
    p.longitudinal.v_max = 1e3
    p.longitudinal.v_min = -1e3
    p.steering.v_max, p.steering.v_min = STEER_RATE_MAX, -STEER_RATE_MAX  # [변경 11/21] 데이터 범위 (이전 ±10)
    # 축별 타이어 (mb_golf 는 tire_f/tire_r 만 씀): 횡강성 (축 강성 = p_ky1 × 축 정적 하중) 을 데이터 추정 코너링 강성에 맞춤
    cf, cr = _tire_stiffness()
    p.tire_f, p.tire_r = copy.deepcopy(p.tire), copy.deepcopy(p.tire)
    p.tire_f.p_ky1 = -cf / (p.m * G * p.b / (p.a + p.b)) * ky_f
    p.tire_r.p_ky1 = -cr / (p.m * G * p.a / (p.a + p.b)) * ky_r
    p.tire_f.r_by1 *= rby_f
    for t in (p.tire_f, p.tire_r):
        t.p_dx1 *= mu_scale
        t.p_dy1 *= mu_scale
    return p


def force_cap_table():
    """속도별 구동력 상한 (상위 1%, 효율 포함, specs 의 drive_force_cap) → (속도 격자, 힘). 구간 가운데 속도에 값을 두고,
    POWER_BINS 이상은 그 구간들의 평균 출력 P 로 F = P / v."""
    t = {k: np.asarray(v, dtype=float) for k, v in _v("drive_force_cap").items()}
    vc = 0.5 * (t["v_lo"] + t["v_hi"])
    keep = t["v_hi"] <= POWER_BINS[1]
    pw = (t["F_p99_N"] * vc)[(t["v_lo"] >= POWER_BINS[0]) & keep].mean()
    v_ext = np.arange(POWER_BINS[1] + 1.0, 80.0, 2.0)
    return np.r_[vc[keep], v_ext], np.r_[t["F_p99_N"][keep], pw / v_ext]


@functools.lru_cache(maxsize=None)
def _cap_table():
    return force_cap_table()


def force_cap(v):
    return np.interp(v, *_cap_table())


def accel_input(Fxf, p):
    """앞바퀴 힘 (N) → CommonRoad 종방향 입력 (가속도)."""
    if Fxf >= 0:
        return Fxf * (1.0 + 4.0 * p.I_y_w / (p.R_w ** 2 * p.m)) / p.m  # 바퀴 관성 보정 (모듈 설명, 11/21 (가))
    return Fxf / (p.T_sb * p.m)


def rhs(x, u, p, g_slope=0.0):
    """g_slope = g·sin(종경사)."""
    f = vehicle_dynamics_mb(x, u, p)
    f[3] -= road_load(x[3]) / p.m + g_slope
    return f


def rk4(x, u, p, h, g_slope=0.0):
    k1 = rhs(x, u, p, g_slope)
    k2 = rhs([a + 0.5 * h * b for a, b in zip(x, k1)], u, p, g_slope)
    k3 = rhs([a + 0.5 * h * b for a, b in zip(x, k2)], u, p, g_slope)
    k4 = rhs([a + h * b for a, b in zip(x, k3)], u, p, g_slope)
    x = [a + h / 6.0 * (b + 2 * c + 2 * d + e) for a, b, c, d, e in zip(x, k1, k2, k3, k4)]
    for i in range(23, 27):  # 바퀴 역회전 금지 (패키지와 같은 처리)
        x[i] = max(x[i], 0.0)
    return x


class Actuator:
    """MPPI 명령 → 실제 조향각 목표·앞바퀴 힘. 지연은 순수 지연 + 1차 지연 (계획서 8-2).
    delays = dict(steer=s, brake=(d, tau), engine=(d, tau)) 초 단위. 조향 지연은 0.04 의 배수만.
    힘은 두 경로: 엔진 = 명령의 양수 부분, 브레이크 = 음수 부분. 각자의 지연을 거친 뒤 더한다
    (명령 부호가 바뀌어도 한쪽 지연이 다른 쪽 이력에 섞이지 않음)."""

    def __init__(self, delays, h):
        d = delays or {}
        self.n_steer = int(round(d.get("steer", 0.0) / DT))
        # (이력에서 몇 칸 전 명령을 볼지, 1차 지연 한 칸의 비율): 구동(엔진)·제동(브레이크)
        lag = lambda dt: (int(round(dt[0] / h)), 1.0 if dt[1] <= h else h / dt[1])
        self.lag_engine, self.lag_brake = lag(d.get("engine", (0.0, 0.0))), lag(d.get("brake", (0.0, 0.0)))
        self.n_hist = max(self.lag_engine[0], self.lag_brake[0]) + 1
        self.reset()

    def reset(self, F0=0.0, delta0=0.0):
        self.Fq = collections.deque([F0] * self.n_hist, maxlen=self.n_hist)
        self.dq = collections.deque([delta0] * (self.n_steer + 1), maxlen=self.n_steer + 1)
        self.Fe, self.Fb = max(F0, 0.0), min(F0, 0.0)

    def steer_target(self, delta_cmd):
        target = self.dq[0]
        self.dq.append(delta_cmd)
        return target

    def force_step(self, F_cmd):
        """한 적분 간격 h 만큼 진행 → 실제 앞바퀴 힘. F_cmd 는 이미 범위 안으로 자른 값."""
        self.Fq.append(F_cmd)
        (ne, ke), (nb, kb) = self.lag_engine, self.lag_brake
        self.Fe += (max(self.Fq[-1 - ne], 0.0) - self.Fe) * ke
        self.Fb += (min(self.Fq[-1 - nb], 0.0) - self.Fb) * kb
        return self.Fe + self.Fb


class World:
    """mppi.episode() 의 world: world(h) → 다음 [vx, vy, r] (torch, h 와 같은 장치).
    h[0, -1] = [vx, vy, r, Fxf 명령, 조향 명령, bank]. pose = (X, Y, psi). 경사는 set_grade / set_pos 로."""

    def __init__(self, mu_scale=1.0, delays=None, h=2e-3, grade_by_idx=None, cap=True, k_roll=None, ky_f=1.0, ky_r=1.0, rby_f=1.0,
                 tire_mirror=True, lsd=None, lsd_table=None):
        """lsd: [11/22] 앞 LSD (mb_golf 설명). None = 열린 차동장치. dict(mode='passive', T0, G) 또는 dict(mode='active', Tmax).
        lsd_table(F_cmd, vx, r) → 작동 비율 0~1: 'active' 폐루프에서 매 스텝 u 를 정하는 함수 (재생은 set_lsd 로 기록값을 넣음)."""
        self.p = golf_params(mu_scale, k_roll, ky_f, ky_r, rby_f, tire_mirror)
        self.p.lsd_mode = None
        if lsd:
            self.p.lsd_mode, self.p.lsd_eps = lsd["mode"], lsd.get("eps", 1.0)
            self.p.lsd_T0, self.p.lsd_G = lsd.get("T0", 0.0), lsd.get("G", 0.0)
            self.p.lsd_Tmax, self.p.lsd_u = lsd.get("Tmax", 1600.0), 0.0
        self.lsd_table, self.lsd_override = lsd_table, None
        self.act = Actuator(delays, h)
        self.h, self.cap = h, cap
        self.n_sub = int(round(DT / h))
        self.grade_by_idx = grade_by_idx
        self.g_slope = 0.0
        self.x = None

    def reset(self, X, Y, psi, vx, vy=0.0, r=0.0, delta=0.0, F0=0.0):
        v = math.hypot(vx, vy)
        self.x = init_mb([X, Y, delta, v, psi, r, math.atan2(vy, vx)], self.p)
        self.act.reset(F0, delta)

    def set_grade(self, grade):
        self.g_slope = G * math.sin(float(grade))

    def set_lsd(self, u):
        """'active' LSD 작동 비율 (0~1) 을 직접 넣음 (재생: 데이터 기록값). None 이면 lsd_table 을 씀."""
        self.lsd_override = u

    def set_pos(self, idx):
        if self.grade_by_idx is not None:
            self.set_grade(self.grade_by_idx[int(idx)])

    @property
    def pose(self):
        return self.x[0], self.x[1], self.x[4]

    @property
    def state(self):
        return self.x[3], self.x[10], self.x[5]

    def step(self, F_cmd, delta_cmd):
        """한 제어 간격 (0.04 s). 조향은 지연된 목표에 0.04 s 동안 일정한 조향 속도로 도달.
        힘 범위(제동 하한, 속도별 구동력 상한)는 간격 시작의 속도로 한 번 정함."""
        if self.cap:
            F_cmd = min(max(F_cmd, F_BRAKE_MIN), float(force_cap(self.x[3])))
        dv = (self.act.steer_target(delta_cmd) - self.x[2]) / DT
        if self.p.lsd_mode == "active":  # [11/22] 이 간격의 LSD 작동 비율 (기록값 또는 표)
            self.p.lsd_u = self.lsd_override if self.lsd_override is not None else (
                float(self.lsd_table(F_cmd, self.x[3], self.x[5])) if self.lsd_table else 0.0)
        x = self.x
        try:
            for _ in range(self.n_sub):
                x = rk4(x, [dv, accel_input(self.act.force_step(F_cmd), self.p)], self.p, self.h, self.g_slope)
        except (ZeroDivisionError, ValueError, OverflowError):  # 완전히 미끄러져 식이 계산되지 않음 (예: 바퀴 속도 0)
            x = [math.nan] * len(x)  # → episode() 가 world_nan 으로 끝냄
        self.x = x
        return self.state

    def __call__(self, hist):
        import torch
        row = hist[0, -1].detach().cpu().double().tolist()
        s = self.step(row[3], row[4])
        return torch.tensor([s], dtype=hist.dtype, device=hist.device)


def calibrated_world(**kw):
    """보정값(calibration.json: mu_scale, k_roll)과 지연(delays.json)을 넣은 World. kw 로 덮어쓸 수 있음."""
    cal = load_calibration()
    base = dict(mu_scale=cal.get("mu_scale", 1.0), k_roll=cal.get("k_roll"), delays=load_delays(), tire_mirror=cal.get("tire_mirror", True),
                **{k: cal[k] for k in ("ky_f", "ky_r", "rby_f") if k in cal})
    if "delays" in kw and kw["delays"] is not None and base["delays"]:
        kw["delays"] = dict(base["delays"], **kw["delays"])
    return World(**dict(base, **kw))


def roll_gradient(mu_scale=1.0, k_roll=None, v=25.0, delta=0.05, steps=150, tire_mirror=True):
    """정상 원선회의 롤 기울기 (도 / (m/s²)) 와 그때의 횡가속도."""
    w = World(mu_scale=mu_scale, k_roll=k_roll, cap=False, tire_mirror=tire_mirror)
    w.reset(0, 0, 0, v)
    for _ in range(steps):
        s = w.step(road_load(w.state[0]) + w.p.m * 0.5 * (v - w.state[0]), delta)
    return abs(math.degrees(w.x[6])) / abs(s[0] * s[2]), s[0] * s[2]
