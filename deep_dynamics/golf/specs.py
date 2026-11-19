"""차량 제원 추정 (docs/experiment_plan_golf.md 7절).

1. 종방향: 질량 m, 구동계 효율 eta, 굴림저항 Cr0, 공기저항 Cr2
     m*(dvx/dt - vy*r + g*sin(grade)) = eta*F_drive - F_brake - Cr0 - Cr2*vx^2
   직진 구간(|조향| 작음, |r| 작음)만 쓴다. 일괄 최소제곱 + 망각 계수 RLS 두 가지.
2. 횡·요: 무게중심 위치 lf (축거 L 고정), 요 관성 Iz, 선형 코너링 강성 Cf, Cr
     m*(dvy/dt + vx*r) = Cf*af*cos(delta) + Cr*ar
     Iz*dr/dt          = lf*Cf*af*cos(delta) - (L-lf)*Cr*ar
   af = delta - (vy + lf*r)/vx, ar = (lr*r - vy)/vx. 선형 타이어 구간(|횡가속| 작음)만 쓴다.
   lf 가 미끄럼각에 들어가므로 비선형 최소제곱(scipy least_squares)으로 네 값을 함께 푼다.

같은 함수를 제원을 아는 차(GTI, AV-21)에 먼저 돌려 방법을 검증한다 (7-2절).
"""
import numpy as np
from scipy.optimize import least_squares
from scipy.signal import butter, filtfilt

G = 9.81


def smooth(x, cutoff, fs, order=4):
    b, a = butter(order, cutoff, fs=fs)
    return filtfilt(b, a, x, axis=0)


def deriv(x, fs):
    return np.gradient(x, 1.0 / fs, axis=0)


# ---------------------------------------------------------------- 종방향
def lstsq(A, y):
    th, *_ = np.linalg.lstsq(A, y, rcond=None)
    res = y - A @ th
    # 계수 표준오차 (잔차가 독립이라는 가정. 시계열이라 실제 불확실성은 더 큼)
    s2 = res @ res / max(len(y) - A.shape[1], 1)
    cov = s2 * np.linalg.pinv(A.T @ A)
    return th, np.sqrt(np.diag(cov)), float(np.sqrt(np.mean(res ** 2)))


def rls(A, y, lam=0.999, theta0=None, P0=1e6, record_every=200):
    """망각 계수 lam 의 재귀 최소제곱. 반환 (최종 theta, 기록 (k, theta) 배열)."""
    n = A.shape[1]
    th = np.zeros(n) if theta0 is None else np.asarray(theta0, float).copy()
    # 열마다 크기가 크게 달라서 표준화한 공간에서 갱신
    sc = np.maximum(np.abs(A).std(0), 1e-9)
    Ah = A / sc
    th = th * sc
    P = np.eye(n) * P0
    hist = []
    for k in range(len(y)):
        a = Ah[k]
        Pa = P @ a
        g = Pa / (lam + a @ Pa)
        th = th + g * (y[k] - a @ th)
        P = (P - np.outer(g, Pa)) / lam
        if k % record_every == 0 or k == len(y) - 1:
            hist.append(np.r_[k, th / sc])
    return th / sc, np.array(hist)


def longitudinal_fit(vx, vy, r, ax_dot, F_target, F_drive=None, grade=None, lam=0.999):
    """m*(ax_dot - vy*r + g*sin(grade)) = eta*F_drive - F_brake - Cr0 - Cr2*vx^2.
    F_drive 가 None 이면 eta 없이 (F_target 이 이미 바퀴 힘의 합인 경우: GTI).
    부호 정리: F_target 은 '타이어가 차를 미는 힘의 합'을 넣는다. 반환 dict."""
    a_eff = ax_dot - vy * r + (G * np.sin(grade) if grade is not None else 0.0)
    if F_drive is None:
        # F_target = m*a_eff + Cr0 + Cr2*vx^2
        A = np.stack([a_eff, np.ones_like(vx), vx ** 2], 1)
        th, se, rms = lstsq(A, F_target)
        names = ["m", "Cr0", "Cr2"]
        thr, hist = rls(A, F_target, lam)
    else:
        # 뉴턴: m a_eff = eta F_drive - F_brake - Cr0 - Cr2 vx^2
        #   →  -F_brake = -eta F_drive + Cr0 + Cr2 vx^2 + m a_eff.  여기서 F_target = -F_brake (음수)
        A = np.stack([-F_drive, np.ones_like(vx), vx ** 2, a_eff], 1)
        th, se, rms = lstsq(A, F_target)
        names = ["eta", "Cr0", "Cr2", "m"]
        thr, hist = rls(A, F_target, lam)
    return dict(batch={n: float(v) for n, v in zip(names, th)}, batch_se={n: float(v) for n, v in zip(names, se)},
                rls={n: float(v) for n, v in zip(names, thr)}, rls_hist=hist.tolist(), rls_names=names,
                rms_N=rms, n=int(len(vx)))


# ---------------------------------------------------------------- 횡·요
def _alphas(lf, L, vx, vy, r, delta):
    lr = L - lf
    af = delta - np.arctan2(vy + lf * r, vx)
    ar = np.arctan2(lr * r - vy, vx)
    return af, ar


def lateral_fit(m, L, vx, vy, r, delta, vy_dot, r_dot, lf0=None, Iz0=None, fix_lf=False):
    """선형 타이어 구간 데이터로 lf, Iz, Cf, Cr 을 함께 추정. fix_lf=True 면 lf 를 lf0 로 고정. 반환 dict."""
    ay = vy_dot + vx * r
    s1 = max(np.std(m * ay), 1.0)
    lf0 = L / 2 if lf0 is None else lf0
    Iz0 = m * 1.5 if Iz0 is None else Iz0

    def resid(p):
        lf, Iz, Cf, Cr = p[0], p[1] * 1e3, p[2] * 1e5, p[3] * 1e5
        af, ar = _alphas(lf, L, vx, vy, r, delta)
        Ff, Fr = Cf * af * np.cos(delta), Cr * ar
        e1 = (Ff + Fr - m * ay) / s1
        mom = lf * Ff - (L - lf) * Fr
        e2 = (mom - Iz * r_dot) / max(np.std(mom), 1.0)
        return np.r_[e1, e2]

    p0 = np.array([lf0, Iz0 / 1e3, 1.0, 1.0])
    lo, hi = [0.2 * L, 0.1, 0.05, 0.05], [0.8 * L, 20.0, 10.0, 10.0]
    if fix_lf:
        lo[0], hi[0] = lf0 - 1e-9, lf0 + 1e-9
    sol = least_squares(resid, p0, bounds=(lo, hi), x_scale="jac")
    lf, Iz, Cf, Cr = sol.x[0], sol.x[1] * 1e3, sol.x[2] * 1e5, sol.x[3] * 1e5
    # 근사 표준오차 (야코비안 기반)
    J = sol.jac
    try:
        cov = np.linalg.inv(J.T @ J) * (2 * sol.cost / max(len(sol.fun) - 4, 1))
        se = np.sqrt(np.diag(cov)) * np.array([1, 1e3, 1e5, 1e5])
    except np.linalg.LinAlgError:
        se = np.full(4, np.nan)
    # lf 를 조금씩 바꿔 가며 비용 곡선 (lf 가 얼마나 잘 정해지는지 보기 위함)
    prof = []
    for lfi in np.linspace(0.3 * L, 0.7 * L, 21):
        s = least_squares(lambda q: resid(np.r_[lfi, q]), sol.x[1:], bounds=([0.1, 0.05, 0.05], [20.0, 10.0, 10.0]), x_scale="jac")
        prof.append((float(lfi), float(2 * s.cost / len(s.fun))))
    return dict(lf=float(lf), lr=float(L - lf), Iz=float(Iz), Cf=float(Cf), Cr=float(Cr),
                se=dict(lf=float(se[0]), Iz=float(se[1]), Cf=float(se[2]), Cr=float(se[3])),
                rms_norm=float(np.sqrt(2 * sol.cost / len(sol.fun))), n=int(len(vx)), lf_profile=prof)
