"""썬더힐 West 트랙: 근사 경계, 경로 3종, 목표 속도 (docs/experiment_plan_golf.md 9-3, 9-5절).

  기준선   : 20개 주행 중 평균 속도가 가장 빠른 완전한 한 바퀴의 GPS 궤적 (1 m 재표본, 평활)
  경계    : 기준선 ± 고정 폭. [변경 11/19] 기본 총 10.9 m = 사용자가 위성 지도로 잰 14곳 평균
            (docs/track_width_measurement_thunderhill.pdf, 최소 10.1 / 최대 11.6 m). 이전은 근거 없는 가정 10 m.
             문서 9-5 의 'GPS 궤적 범위' 경계는 폭이 중앙값 1.2 m (최소 0.15 m) 로 나와 쓰지 않음:
             자동 운전이 같은 선을 반복하고 사람 구간도 95% 가 기준선 0.85 m 안이라 궤적으로는 트랙 폭을 알 수 없음.
             궤적 범위는 참고용으로 thunderhill_info.json 의 gps_envelope 에 남긴다
  경로 3종 : center   = 경계의 가운데 선 (고정 폭이면 기준선과 같음)
             mincurv  = 경계 안(가장자리에서 margin 여유)에서 곡률 제곱 합이 가장 작은 선
             recorded = 기록된 가장 빠른 주행선 (= 기준선)
  목표 속도: speed_profile(kappa, mu, ax_acc, ax_dec)
출력 npz 키는 mppi.Track 이 읽는 형식 (x, y, psi, kappa, s, width_l, width_r, vref, bank, inner, outer).

python -m deep_dynamics.golf.track [--out experiments/golf/results/tracks]
"""
import argparse
import json
import os

import numpy as np
from scipy.ndimage import maximum_filter1d, minimum_filter1d, uniform_filter1d
from scipy.optimize import minimize
from scipy.spatial import cKDTree

from . import data as D

G = 9.81
MU0, AX_ACC0, AX_DEC0 = 1.2, 4.3, -5.1  # 데이터에서 잰 기준값 (문서 5-5)


# ---------------------------------------------------------------- 기하
def resample_closed(p, ds=1.0):
    p = np.vstack([p, p[:1]])
    d = np.r_[0, np.cumsum(np.hypot(*np.diff(p, axis=0).T))]
    s = np.arange(0, d[-1] - 0.5 * ds, ds)
    return np.stack([np.interp(s, d, p[:, 0]), np.interp(s, d, p[:, 1])], 1)


def heading_curv(p, smooth=15):
    d = np.gradient(p, axis=0)
    psi = np.unwrap(np.arctan2(d[:, 1], d[:, 0]))
    dd = np.gradient(d, axis=0)
    kappa = (d[:, 0] * dd[:, 1] - d[:, 1] * dd[:, 0]) / np.hypot(*d.T) ** 3
    if smooth:
        kappa = uniform_filter1d(kappa, smooth, mode="wrap")
    return psi, kappa


def closed_grad(p):
    """닫힌 곡선의 중앙 차분 (wrap)."""
    return 0.5 * (np.roll(p, -1, 0) - np.roll(p, 1, 0))


def normals(p):
    d = closed_grad(p)
    t = d / np.linalg.norm(d, axis=1, keepdims=True)
    return np.stack([-t[:, 1], t[:, 0]], 1)  # 왼쪽 법선


def project(path, pts):
    """pts (M,2) → 가장 가까운 경로 점 인덱스, 횡위치 (왼쪽 +)."""
    idx = cKDTree(path).query(pts)[1]
    nrm = normals(path)
    ey = np.einsum("ij,ij->i", pts - path[idx], nrm[idx])
    return idx, ey


# ---------------------------------------------------------------- 목표 속도
def speed_profile(kappa, ds=1.0, mu=MU0, ax_acc=AX_ACC0, ax_dec=AX_DEC0, vmax=None, closed=True):
    """코너 한계 √(μg/|κ|) → 앞→뒤로 가속 한계 → 뒤→앞으로 감속 한계. 닫힌 트랙은 두 바퀴 돌려 이음매를 맞춤."""
    vlim = np.sqrt(mu * G / np.maximum(np.abs(kappa), 1e-6))
    if vmax is not None:
        vlim = np.minimum(vlim, vmax)
    n = len(vlim)
    reps = 3 if closed else 1
    v = np.tile(vlim, reps)
    for i in range(1, len(v)):
        v[i] = min(v[i], np.sqrt(v[i - 1] ** 2 + 2 * ax_acc * ds))
    for i in range(len(v) - 2, -1, -1):
        v[i] = min(v[i], np.sqrt(v[i + 1] ** 2 + 2 * abs(ax_dec) * ds))
    return v[(reps // 2) * n:(reps // 2 + 1) * n] if closed else v


# ---------------------------------------------------------------- 데이터에서 트랙 만들기
def complete_laps(sg, min_len=2000.0, close_tol=5.0):
    """한 구간에서 출발점으로 돌아오는 완전한 바퀴들 [(평균속도, i0, k)]."""
    x, y = sg["x"], sg["y"]
    d = np.r_[0, np.cumsum(np.hypot(np.diff(x), np.diff(y)))]
    laps, i0 = [], 0
    while True:
        far = np.where(d - d[i0] > min_len)[0]
        if len(far) == 0:
            break
        j = far[0]
        k = np.where(np.hypot(x[j:] - x[i0], y[j:] - y[i0]) < close_tol)[0]
        if len(k) == 0:
            break
        k = j + k[0]
        laps.append((float(sg["s"][i0:k, 0].mean()), i0, k))
        i0 = k
    return laps


def best_lap(runs):
    best = None
    for r in runs:
        for sg in D.load_run(r):
            for v, i0, k in complete_laps(sg):
                if best is None or v > best[0]:
                    best = (v, r, sg, i0, k)
    v, r, sg, i0, k = best
    return dict(run=r, v_mean=v, x=sg["x"][i0:k], y=sg["y"][i0:k], s=sg["s"][i0:k], u=sg["u"][i0:k],
                t0=float(sg["t"][i0]), t1=float(sg["t"][k]), lap_time=float(sg["t"][k] - sg["t"][i0]))


def build_boundary(ref, runs, max_off=15.0, win=5):
    """모든 주행 점을 기준선에 투영 → 칸별 횡위치 최대(왼쪽)·최소(오른쪽)."""
    n = len(ref)
    left, right = np.full(n, -np.inf), np.full(n, np.inf)
    bank = [[] for _ in range(n)]
    npts = nfar = 0
    for r in runs:
        for sg in D.load_run(r):
            pts = np.stack([sg["x"], sg["y"]], 1)
            idx, ey = project(ref, pts)
            ok = np.abs(ey) < max_off  # 피트 등 트랙 밖 점 제외
            npts += len(ey)
            nfar += int((~ok).sum())
            np.maximum.at(left, idx[ok], ey[ok])
            np.minimum.at(right, idx[ok], ey[ok])
            for i, b in zip(idx[ok][::5], sg["u"][ok, 2][::5]):
                bank[i].append(b)
    have = np.isfinite(left) & np.isfinite(right)
    ii = np.arange(n)
    left = np.interp(ii, ii[have], left[have], period=n)
    right = np.interp(ii, ii[have], right[have], period=n)
    # 칸 사이 들쭉날쭉함을 줄이되 바깥선은 유지 (구간 최대/최소 후 평활)
    left = uniform_filter1d(maximum_filter1d(left, win, mode="wrap"), win, mode="wrap")
    right = uniform_filter1d(minimum_filter1d(right, win, mode="wrap"), win, mode="wrap")
    bk = np.array([np.median(b) if len(b) else np.nan for b in bank])
    ok = np.isfinite(bk)
    bk = uniform_filter1d(np.interp(ii, ii[ok], bk[ok], period=n), 31, mode="wrap")
    return left, right, bk, dict(points=npts, excluded_far=nfar, empty_bins=int((~have).sum()))


def min_curvature_offsets(ref, left, right, margin=1.0):
    """경계 안 횡오프셋 e_i 로 경로 p_i = ref_i + e_i n_i 를 만들고 Σ|p_{i+1} - 2p_i + p_{i-1}|² 최소화.
    (곡률 제곱 합의 표준 근사, 2차 형식 → L-BFGS-B 로 상·하한과 함께 풂)"""
    nrm = normals(ref)
    lo, hi = right + margin, left - margin
    bad = lo > hi
    mid = 0.5 * (lo + hi)
    lo, hi = np.where(bad, mid, lo), np.where(bad, mid, hi)

    def f(e):
        p = ref + e[:, None] * nrm
        dd = np.roll(p, -1, 0) - 2 * p + np.roll(p, 1, 0)
        # grad wrt p: 2*(D^T dd) where D is the second-difference operator (self-adjoint for closed curves)
        gp = 2 * (np.roll(dd, -1, 0) - 2 * dd + np.roll(dd, 1, 0))
        return float((dd ** 2).sum()), np.einsum("ij,ij->i", gp, nrm)

    e0 = np.clip(np.zeros(len(ref)), lo, hi)
    sol = minimize(f, e0, jac=True, method="L-BFGS-B", bounds=list(zip(lo, hi)), options=dict(maxiter=5000))
    return sol.x, dict(success=bool(sol.success), nit=int(sol.nit), cost0=float(f(e0)[0]), cost=float(sol.fun))


def make_path(ref, left, right, bank, offs, name, ds=1.0, mu=MU0, ax_acc=AX_ACC0, ax_dec=AX_DEC0):
    """기준선 + 횡오프셋 → 경로. 경계는 기준선 좌표계에서 정의돼 있으므로 새 경로에 다시 투영해 폭을 구한다."""
    nrm = normals(ref)
    inner_l = ref + left[:, None] * nrm   # 왼쪽 경계선
    outer_r = ref + right[:, None] * nrm  # 오른쪽 경계선
    p = ref + offs[:, None] * nrm
    p = resample_closed(p, ds)
    psi, kappa = heading_curv(p)
    # 새 경로의 각 점에서 기준선으로 투영해 오프셋과 경계 보간
    idx, ey = project(ref, p)
    wl = left[idx] - ey
    wr = ey - right[idx]
    vref = speed_profile(kappa, ds, mu, ax_acc, ax_dec)
    return dict(name=name, x=p[:, 0], y=p[:, 1], psi=psi, kappa=kappa, s=np.arange(len(p)) * ds, width_l=wl,
                width_r=wr, vref=vref, bank=bank[idx], inner=inner_l, outer=outer_r,
                mu=mu, ax_acc=ax_acc, ax_dec=ax_dec)


def build_all(runs=None, out_dir=None, mu=MU0, ax_acc=AX_ACC0, ax_dec=AX_DEC0, width=10.9):
    runs = runs or D.available()
    lap = best_lap(runs)
    ref = resample_closed(np.stack([lap["x"], lap["y"]], 1), 1.0)
    for _ in range(2):
        ref = np.stack([uniform_filter1d(ref[:, k], 5, mode="wrap") for k in range(2)], 1)
    ref = resample_closed(ref, 1.0)
    gl, gr, bank, bstat = build_boundary(ref, runs)  # GPS 궤적 범위 (참고용)
    bstat.update(width_total_pct5_50_95=[float(np.percentile(gl - gr, q)) for q in (5, 50, 95)])
    left, right = np.full(len(ref), width / 2), np.full(len(ref), -width / 2)
    center = 0.5 * (left + right)
    mc, mcinfo = min_curvature_offsets(ref, left, right)
    paths = {
        "recorded": make_path(ref, left, right, bank, np.zeros(len(ref)), "recorded", mu=mu, ax_acc=ax_acc, ax_dec=ax_dec),
        "center": make_path(ref, left, right, bank, center, "center", mu=mu, ax_acc=ax_acc, ax_dec=ax_dec),
        "mincurv": make_path(ref, left, right, bank, mc, "mincurv", mu=mu, ax_acc=ax_acc, ax_dec=ax_dec),
    }
    info = dict(best_lap=dict(run=lap["run"], v_mean=lap["v_mean"], lap_time=lap["lap_time"], t0=lap["t0"]),
                boundary=dict(kind="recorded line ± fixed", width_total_m=width), gps_envelope=bstat, mincurv=mcinfo, length_m=len(ref),
                width_total=dict(min=float((left - right).min()), median=float(np.median(left - right)), max=float((left - right).max())))
    for k, p in paths.items():
        lt = float(np.sum(1.0 / p["vref"]))
        info[k] = dict(length_m=float(len(p["x"])), kappa_absmax=float(np.abs(p["kappa"]).max()),
                       sum_kappa2=float(np.sum(p["kappa"] ** 2)), vref_min=float(p["vref"].min()), vref_max=float(p["vref"].max()),
                       lap_time_profile=lt, width_min=float(np.minimum(p["width_l"], p["width_r"]).min()))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        for k, p in paths.items():
            np.savez(os.path.join(out_dir, f"thunderhill_{k}.npz"), **{kk: np.asarray(v) for kk, v in p.items() if kk != "name"})
        np.savez(os.path.join(out_dir, "thunderhill_boundary.npz"), ref=ref, left=left, right=right, bank=bank,
                 gps_left=gl, gps_right=gr)
        json.dump(info, open(os.path.join(out_dir, "thunderhill_info.json"), "w"), indent=1)
    return paths, info, (ref, left, right)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="experiments/golf/results/tracks")
    ap.add_argument("--width", type=float, default=10.9, help="트랙 총 폭 (m). 기본 = 위성 지도 측정 평균")
    a = ap.parse_args()
    _, info, _ = build_all(out_dir=a.out, width=a.width)
    print(json.dumps(info, indent=1, ensure_ascii=False))
