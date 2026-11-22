"""이중 궤도(4륜, 서스펜션 없음) 차량 모델 — Subosits 박사논문 2020 (Stanford DDL) 4.2절, 같은 썬더힐 GTI (11/22 사용자 결정 B).

식 번호는 논문 기준. 평평한 길(p = q = uz = 0)이고, 종경사는 gx = −g·sin(경사) 로만 넣는다.
  운동 (4.4~4.6), 하중 이동 1차 지연 (4.10~4.11), 바퀴 하중 (4.13~4.16), 확장 Fiala 타이어 (4.17~4.21),
  전자식 LSD (4.22~4.29: 열림 / 잠김 을 u_diff 로 섞음).
바퀴 회전 상태 없음: 논문의 "바퀴 속도 평형" 가정 (ABS 같은 장치가 잠김을 막음). 앞뒤 방향 힘은 입력이다.
  [내 결정] 바퀴별 앞뒤 힘이 마찰 한계를 넘으면 0.99·μ(Fz)·Fz 에서 자른다 (이상적인 트랙션·ABS. 논문은 최적화 제약으로 처리).
  [논문 식 4.13 의 b/L·Fz 는 축 하중이라 바퀴당은 b/(2L)·Fz 로 씀 — 네 바퀴 합이 Fz 가 되도록.]
좌표: 차체 x 앞, y 왼쪽, 회전 r 반시계 +. 바퀴 이름 L/R 은 물리 위치 그대로 (CommonRoad 와 달리).
계수 출처는 deep_dynamics/golf/specs.json (params() 설명).

상태 x = [X, Y, ψ, ux, uy, r, ΔFz_long, ΔFz_lat, δ]
입력 u = [조향 속도 (rad/s), 앞바퀴 힘 Fxf (N), u_diff (0~1)], g_slope = g·sin(종경사)
"""
import math
from types import SimpleNamespace

from . import golf_specs
from .configs import GOLF_THUNDERHILL as CFG

G = 9.81
N_STATE = 9
I_DELTA = 8


def params(mu_dfz=None, c2=None):
    """계수: 차체는 specs (공개값), 하중 이동·타이어 모양은 Subosits 공개값, 타이어 크기는 데이터 (축별 Fiala 강성은 정적 하중에서, 마찰은 그 축이 한계에 닿은 하중 상태에서 같게; 아래 설명).
    c1_f/c1_r, mu0_f/mu0_r 는 축마다 따로 (논문은 네 바퀴 같은 타이어. 우리 데이터는 앞 138 / 뒤 161 kN/rad 로, 논문 하중 의존 모양만으로는
    재현 안 됨 → 크기는 축별 데이터, 모양 c2·μd_fz·n 은 공개값)."""
    v = golf_specs.value
    # [11/22] mu_dfz·c2 를 주면 공개값 대신 씀 (fit_sim --model dt 의 맞춤 값, calibration_dt.json). c1·μ0 환산은 아래에서 다시 계산해
    # 축 강성(작은 옆 가속)·축 실효 마찰(한계)이 데이터와 계속 같게 유지된다.
    p = SimpleNamespace(m=CFG.m, a=CFG.lf, b=CFG.lr, Iz=v("yaw_inertia"), h=v("cg_height"), df=v("track_f"), dr=v("track_r"),
                        gamma=v("lt_gamma"), k_lat=v("lt_k_lat"), k_long=v("lt_k_long"),
                        c2=v("tire_c2") if c2 is None else c2, mu_dfz=v("tire_mu_dfz") if mu_dfz is None else mu_dfz,
                        n=v("tire_n"), Fz0=v("tire_Fz0"), c8=v("tire_c8"),
                        T_sb=CFG.brake_front_share, Cr0=v("resist_cr0"), Cr2=v("resist_cr2"), mu_scale=1.0)
    L = p.a + p.b
    dW = 0.5 * (p.df + p.dr)
    for ax, Fz_s, C, mu, share in (("f", p.m * G * p.b / L / 2, v("tire_stiffness_f"), v("tire_mu_f"), p.gamma),
                                   ("r", p.m * G * p.a / L / 2, v("tire_stiffness_r"), v("tire_mu_r"), 1 - p.gamma)):
        setattr(p, "c1_" + ax, C / (2.0 * p.Fz0 * math.sin(2.0 * math.atan(Fz_s / (p.c2 * p.Fz0)))))  # 2 바퀴 × Cα(정적) = 축 강성 (작은 옆 가속에서 잰 처음 기울기)
        # [변경 11/22] 마찰: 데이터 Fiala 축 마찰은 그 축이 한계에 닿은 상태(옆 가속 = μ·g, 하중이 이미 바깥으로 옮겨짐)의 실효값이다.
        # 그 상태의 두 바퀴 하중 Fz_s ∓ share·ΔFz_lat (ΔFz_lat = h·m·μ·g/d) 에서 Σ μ(Fz_i)·Fz_i = μ_data·(2·Fz_s) 가 되게 μ0 를 정한다.
        # (이전: 정적 하중에서 μ(Fz_s) = μ_data → 하중 감도가 이동 손실을 한 번 더 빼 앞 축 실효 마찰 약 0.98, 최대 옆 가속 8.26)
        dla = p.h * p.m * mu * G / dW
        Fz_i = (Fz_s - share * dla, Fz_s + share * dla)
        sens = sum((z - p.Fz0) * z for z in Fz_i) * p.mu_dfz / p.Fz0
        setattr(p, "mu0_" + ax, (mu * 2 * Fz_s - sens) / (2 * Fz_s))
    p.lsd_u = 0.0
    p.lsd_mode = None
    return p


def init(X, Y, psi, vx, vy=0.0, r=0.0, delta=0.0):
    return [X, Y, psi, max(vx, 0.5), vy, r, 0.0, 0.0, delta]


def _mu(Fz, ax, p):
    return p.mu_scale * (getattr(p, "mu0_" + ax) + p.mu_dfz * (Fz - p.Fz0) / p.Fz0)  # 식 4.19


def _c_alpha(Fz, Fx, mu, ax, p):
    """식 4.20~4.21: 하중·앞뒤 힘에 따른 코너링 강성."""
    c0 = getattr(p, "c1_" + ax) * p.Fz0 * math.sin(2.0 * math.atan(Fz / (p.c2 * p.Fz0)))
    ratio = min(abs(Fx) / (mu * Fz), 1.0)
    return 0.5 * (mu * Fz - Fx) + (1.0 - ratio ** p.n) ** (1.0 / p.n) * (c0 - 0.5 * mu * Fz)


def tire_fy(alpha, Fz, Fx, ax, p):
    """확장 Fiala 옆 힘 (식 4.17, 4.18, 마지막 식). Fx 는 이미 마찰 한계 안으로 자른 값."""
    Fz = max(Fz, 1.0)
    mu = _mu(Fz, ax, p)
    fmax = math.sqrt(max((mu * Fz) ** 2 - Fx ** 2, 0.0))
    if fmax <= 1e-6:
        return 0.0
    Ca = max(_c_alpha(Fz, Fx, mu, ax, p), 1.0)
    z = math.tan(alpha)
    if abs(z) <= 3.0 * fmax / Ca:  # α_slide = 3·Fy,max/Cα (tan 기준)
        return -Ca * z + Ca ** 2 / (3.0 * fmax) * z * abs(z) - Ca ** 3 / (27.0 * fmax ** 2) * z ** 3
    return -fmax * math.copysign(1.0, z)


def loads(x, p):
    """바퀴 하중 (식 4.13~4.16, 바퀴당 b/(2L)·Fz). 반환 (fl, fr, rl, rr)."""
    L = p.a + p.b
    Fz = p.m * G
    dlo, dla = x[6], x[7]
    return (p.b / L * Fz / 2 - dlo / 2 - p.gamma * dla, p.b / L * Fz / 2 - dlo / 2 + p.gamma * dla,
            p.a / L * Fz / 2 + dlo / 2 - (1 - p.gamma) * dla, p.a / L * Fz / 2 + dlo / 2 + (1 - p.gamma) * dla)


def wheel_kin(x, p):
    """바퀴 접지점 속도 (차체 좌표) → (바퀴 방향 속도 vx_w, 미끄럼각). 반환 dict."""
    ux, uy, r, d = x[3], x[4], x[5], x[I_DELTA]
    ux = max(ux, 0.5)
    pts = dict(fl=(p.a, p.df / 2, d), fr=(p.a, -p.df / 2, d), rl=(-p.b, p.dr / 2, 0.0), rr=(-p.b, -p.dr / 2, 0.0))
    out = {}
    for k, (px, py, dk) in pts.items():
        vx, vy = ux - r * py, uy + r * px
        out[k] = (vx * math.cos(dk) + vy * math.sin(dk), math.atan2(vy, vx) - dk)
    return out


def long_forces(Fxf, u_diff, Fz, kin, x, p):
    """바퀴별 앞뒤 힘. 구동: 열림(4.22~4.23)과 잠김(4.28~4.29)을 u_diff 로 섞음. 제동: 앞 반반, 뒤 = 앞 제동 몫으로 계산해 반반."""
    fl, fr, rl, rr = Fz
    if Fxf >= 0:
        vl, vr = max(kin["fl"][0], 0.5), max(kin["fr"][0], 0.5)
        den = fr * vl + fl * vr
        lock_l = (fl * Fxf * vr + p.df * p.c8 * x[5] * fl * fr) / den
        lock_r = (fr * Fxf * vl - p.df * p.c8 * x[5] * fl * fr) / den
        Fx = [(1 - u_diff) * Fxf / 2 + u_diff * lock_l, (1 - u_diff) * Fxf / 2 + u_diff * lock_r, 0.0, 0.0]
    else:
        rear = Fxf * (1 - p.T_sb) / p.T_sb
        Fx = [Fxf / 2, Fxf / 2, rear / 2, rear / 2]
    out = []
    for f, z, ax in zip(Fx, Fz, ("f", "f", "r", "r")):  # [내 결정] 마찰 한계에서 자름 (이상적인 트랙션·ABS)
        lim = 0.99 * _mu(max(z, 1.0), ax, p) * max(z, 1.0)
        out.append(max(-lim, min(lim, f)))
    return out


def forces(x, Fxf, u_diff, p):
    """→ 바퀴 하중, 앞뒤 힘, 옆 힘 (fl, fr, rl, rr)."""
    Fz = loads(x, p)
    kin = wheel_kin(x, p)
    Fx = long_forces(Fxf, u_diff, Fz, kin, x, p)
    Fy = [tire_fy(kin[k][1], z, fx, ax, p) for k, z, fx, ax in zip(("fl", "fr", "rl", "rr"), Fz, Fx, ("f", "f", "r", "r"))]
    return Fz, Fx, Fy


def rhs(x, u, p, g_slope=0.0):
    """dx/dt. u = [조향 속도, Fxf, u_diff]."""
    X, Y, psi, ux, uy, r, dlo, dla, d = x
    Fz, Fx, Fy = forces(x, u[1], u[2], p)
    Fxf, Fxr, Fyf, Fyr = Fx[0] + Fx[1], Fx[2] + Fx[3], Fy[0] + Fy[1], Fy[2] + Fy[3]
    cd, sd = math.cos(d), math.sin(d)
    # 앞 바퀴 힘은 바퀴 좌표 → 차체 좌표 (조향각만큼 회전)
    FX_f, FY_f = Fxf * cd - Fyf * sd, Fyf * cd + Fxf * sd
    drag = p.Cr0 * math.copysign(1.0, ux) + p.Cr2 * ux * ux
    duy = (FY_f + Fyr) / p.m - r * ux  # 4.4
    dr = (p.a * FY_f - p.b * Fyr + p.dr / 2 * (Fx[3] - Fx[2]) + p.df / 2 * (Fx[1] - Fx[0]) * cd + p.df / 2 * (Fy[0] - Fy[1]) * sd) / p.Iz  # 4.5
    dux = (FX_f + Fxr - drag) / p.m + r * uy - g_slope  # 4.6
    L = p.a + p.b
    dW = 0.5 * (p.df + p.dr)
    ddlo = -p.k_long * (dlo - p.h * (FX_f + Fxr) / L)  # 4.10 (p = q = 0)
    ddla = -p.k_lat * (dla - p.h * (FY_f + Fyr) / dW)  # 4.11 (p = q = 0)
    return [ux * math.cos(psi) - uy * math.sin(psi), ux * math.sin(psi) + uy * math.cos(psi), r, dux, duy, dr, ddlo, ddla, u[0]]
