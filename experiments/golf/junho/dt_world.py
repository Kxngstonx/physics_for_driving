"""팀원 브랜치(archive/sim-multibody-4wheel-1122)의 이중 궤도 시뮬레이터 dt_golf 를 main 환경의 mppi.episode() 에서 쓰는 어댑터.

- 차량 계수·타이어·하중 이동·LSD 는 브랜치 그대로 (specs.json, calibration_dt.json, LSD 작동 표).
  브랜치 World.step(model='dt') 의 처리 순서를 그대로 옮김: 힘 범위 자름 → 조향 지연·조향 속도 한계 → LSD 작동 비율 → RK4 2 ms.
- 단위 환산 (유일한 추가): 모델(R1·R6)은 main 데이터로 학습했고, main 의 Fxf = 0.699·구동력 − 앞 제동력,
  브랜치의 Fxf = 0.852·구동력 − 앞 제동력 이다. 같은 실제 조작이 두 시뮬레이터에 같은 입력으로 들어가도록
  양수(구동) 명령만 0.852/0.699 배 한다. 제동력(N)은 두 쪽이 같은 측정값이라 그대로.
저장소 코드는 고치지 않는다 (dt_golf.py 를 읽어 import 경로만 바꿔 실행).
"""
import importlib.util
import json
import math
import os
import sys
import types
from types import SimpleNamespace

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BR = os.path.join(HERE, "pfd_dt")
DT_OUT = os.path.join(HERE, "dt_out")
G = 9.81
DT = 0.04


def _load_branch():
    gs_spec = importlib.util.spec_from_file_location("dtb.golf_specs", os.path.join(BR, "deep_dynamics/golf/golf_specs.py"))
    gs = importlib.util.module_from_spec(gs_spec)
    gs_spec.loader.exec_module(gs)
    v = gs.value
    pkg = types.ModuleType("dtb")
    pkg.__path__ = []
    cfg = types.ModuleType("dtb.configs")
    cfg.GOLF_THUNDERHILL = SimpleNamespace(m=v("mass"), lf=v("cg_a"), lr=v("cg_b"), brake_front_share=v("brake_front_share"), dt=DT)
    sys.modules.update({"dtb": pkg, "dtb.golf_specs": gs, "dtb.configs": cfg})
    pkg.golf_specs, pkg.configs = gs, cfg
    sp = importlib.util.spec_from_file_location("dtb.dt_golf", os.path.join(BR, "deep_dynamics/golf/dt_golf.py"))
    dt = importlib.util.module_from_spec(sp)
    sys.modules["dtb.dt_golf"] = dt
    sp.loader.exec_module(dt)
    return gs, dt


GS, DTG = _load_branch()
_v = GS.value
F_BRAKE_MIN = _v("brake_force_min")
STEER_RATE_MAX = _v("steer_rate_max")
DRIVE_SCALE = _v("drive_eff") / 0.699  # main 데이터 Fxf(구동) → 브랜치 Fxf(구동)
POWER_BINS = (28.0, 32.0)


def _cap_table():  # 브랜치 sim.force_cap_table 과 같은 식
    t = {k: np.asarray(x, dtype=float) for k, x in _v("drive_force_cap").items()}
    vc = 0.5 * (t["v_lo"] + t["v_hi"])
    keep = t["v_hi"] <= POWER_BINS[1]
    pw = (t["F_p99_N"] * vc)[(t["v_lo"] >= POWER_BINS[0]) & keep].mean()
    v_ext = np.arange(POWER_BINS[1] + 1.0, 80.0, 2.0)
    return np.r_[vc[keep], v_ext], np.r_[t["F_p99_N"][keep], pw / v_ext]


CAP = _cap_table()


def lsd_lookup(path=os.path.join(DT_OUT, "lsd_table.json")):
    d = json.load(open(path))
    FB, AB, T = np.array(d["F_BINS"]), np.array(d["AY_BINS"]), np.array(d["T"])

    def look(F_cmd, v, r):  # 브랜치 compare_lsd.lsd_table_from_train 의 look 과 같음 (F 는 브랜치 단위)
        i = int(np.clip(np.digitize(F_cmd, FB) - 1, 0, T.shape[0] - 1))
        j = int(np.clip(np.digitize(abs(v * r), AB) - 1, 0, T.shape[1] - 1))
        return T[i, j]
    return look


def calibration(path=os.path.join(BR, "experiments/golf/results/sim/calibration_dt.json")):
    return json.load(open(path)) if os.path.exists(path) else {}


def _rk4(x, u, p, h, g):
    f = DTG.rhs
    k1 = f(x, u, p, g)
    k2 = f([a + 0.5 * h * b for a, b in zip(x, k1)], u, p, g)
    k3 = f([a + 0.5 * h * b for a, b in zip(x, k2)], u, p, g)
    k4 = f([a + h * b for a, b in zip(x, k3)], u, p, g)
    return [a + h / 6.0 * (b + 2 * c + 2 * d + e) for a, b, c, d, e in zip(x, k1, k2, k3, k4)]


class DTWorld:
    """mppi.episode() 의 world. 입력 명령은 main 단위 (모델이 학습한 Fxf)."""

    def __init__(self, Actuator, delays=None, h=2e-3, grade_by_idx=None, cap=True, lsd=True, main_units=True):
        cal = calibration()
        self.p = DTG.params(**{k: cal[k] for k in ("mu_dfz", "c2") if k in cal})
        self.p.mu_scale = cal.get("mu_scale", 1.0)
        self.p.lsd_mode = "active" if lsd else None
        self.lsd_table = lsd_lookup() if lsd else None
        self.act = Actuator(delays, h)
        self.h, self.cap, self.main_units = h, cap, main_units
        self.n_sub = int(round(DT / h))
        self.grade_by_idx = grade_by_idx
        self.g_slope = 0.0
        self.x = None

    def reset(self, X, Y, psi, vx, vy=0.0, r=0.0, delta=0.0, F0=0.0):
        self.x = DTG.init(X, Y, psi, vx, vy, r, delta)
        self.act.reset(F0, delta)

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
        return self.x[3], self.x[4], self.x[5]

    def step(self, F_cmd, delta_cmd):
        if self.main_units and F_cmd > 0:
            F_cmd = F_cmd * DRIVE_SCALE
        if self.cap:
            F_cmd = min(max(F_cmd, F_BRAKE_MIN), float(np.interp(self.x[3], *CAP)))
        dv = (self.act.steer_target(delta_cmd) - self.x[DTG.I_DELTA]) / DT
        dv = max(-STEER_RATE_MAX, min(STEER_RATE_MAX, dv))
        self.p.lsd_u = float(self.lsd_table(F_cmd, self.x[3], self.x[5])) if self.lsd_table else 0.0
        x = self.x
        try:
            for _ in range(self.n_sub):
                x = _rk4(x, [dv, self.act.force_step(F_cmd), self.p.lsd_u], self.p, self.h, self.g_slope)
        except (ZeroDivisionError, ValueError, OverflowError):
            x = [math.nan] * len(x)
        self.x = x
        return self.state

    def __call__(self, hist):
        import torch
        row = hist[0, -1].detach().cpu().double().tolist()
        s = self.step(row[3], row[4])
        return torch.tensor([s], dtype=hist.dtype, device=hist.device)
