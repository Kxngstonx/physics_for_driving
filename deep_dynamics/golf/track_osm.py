"""썬더힐 West 트랙 (팀 결정 11/19): OSM 노면 중앙선 기반 경계 + 실측 폭 + GPS 고도 경사 지도.

  python -m deep_dynamics.golf.track_osm                     # 위치별 실측 폭 (기본)
  python -m deep_dynamics.golf.track_osm --uniform-width     # 실측 평균 10.9 m 균일
  → experiments/golf/results/tracks/thunderhill_{recorded,center,mincurv}.npz, thunderhill_boundary.npz, thunderhill_info.json
    (track.py 와 같은 파일 이름·키라 mppi.Track, validate_sim, tune_mppi, run_b, viz2d, plot_track 이 그대로 읽는다)

이전 방식(track.py build_all)과 다른 점
  경계의 위치 : 가장 빠른 주행선 ± 5.45 m → OSM 노면 중앙선 ± 실측 폭/2.
                레이싱 라인은 코너 안·바깥 끝을 지나므로(OSM 중앙선에서 중앙값 2.1 m, 최대 4.4 m) 주행선 기준 경계는
                코너 안쪽이 실제 잔디·연석 쪽으로 최대 약 4 m 열린다. B 방식(경계 안 최대 전진)과 최소 곡률 경로에 직접 영향.
  폭         : 실측 14곳(docs/track_width_measurement_thunderhill.pdf)을 트랙 거리로 옮겨 보간 (thunderhill_width_points.csv).
                측정점이 먼 곳(600~2,350 m)은 평균 10.9 m.
  경사 지도   : 학습 파일의 GPS 고도(posU_m)를 트랙 위치별 중앙값 → 거리로 미분. 경로 npz 의 grade 키 (rad, 오르막 +).
                시험 파일 3개의 GPS 상하 속도(Vu/속력)와 RMS 0.09° 차이 (data 의 grade_rad 열은 고도 변화와 상관 −0.07이라 안 씀).
그대로 둔 것 (track.py 재사용)
  recorded 경로 = track.best_lap (20개 주행 중 가장 빠른 완전한 한 바퀴, 2023-10_8). validate_sim 의 검증 랩과 같다.
  [11/20] 경로 0번 = 랩 출발점. 튜닝 랩(TUNE_LAP)의 선은 thunderhill_recorded_<랩>.npz 에 따로 저장한다.
  목표 속도 vref = track.speed_profile (μ 1.2, +4.3 / −5.1 m/s²). 경로 3종 이름과 npz 키.

중앙선 만드는 법
  1. Overpass API 로 썬더힐 반경 3.5 km 의 highway=raceway 선을 받는다(처음 한 번, results/tracks/thunderhill_osm.json 에 캐시).
     © OpenStreetMap contributors, ODbL.
  2. D7 이 달린 West 구간 선 3개만 쓴다(OSM_WEST_WAYS). 위경도 → ECEF → D7 지역 좌표(REF_ECEF 기준 동·북).
  3. 학습 파일의 닫힌 한 바퀴를 따라가며 가장 가까운 OSM 점을 모은다(갈림길에서 D7 이 간 쪽만).
  4. OSM 선은 점이 듬성한 곳에서 꺾이므로, D7 레이싱 라인에서 OSM 선까지의 좌우 거리를 10 m 로 평활해 레이싱 라인에 더한다
     (모양은 RTK-GPS 레이싱 라인, 위치는 OSM). 다듬은 선은 OSM 원래 선과 중앙값 0.9 m, 최대 2.8 m 다르다.
오차 (docs/old_ver/track_osm.md): 경사 RMS 0.09°, 기록선 0.013 m, 폭 ±0.2 m(측정점)·±0.5~0.8 m(측정 없는 곳),
  경계 좌우 위치 보통 ±1 m, 헤어핀(약 1,850·2,400 m) 안쪽 최대 +2 m.
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import minimize
from scipy.spatial import cKDTree

from . import data as D
from .track import G, MU0, AX_ACC0, AX_DEC0, best_lap, heading_curv, normals, project, speed_profile

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(REPO, "experiments", "golf", "results", "tracks")
WIDTH_POINTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "thunderhill_width_points.csv")
WIDTH = 10.9                      # 실측 14곳 평균 (m)
CAR_HALF = 0.9                    # Golf 폭 1.79 m 의 절반
# D1(Spielberg 2019) 트랙 자료의 지역 좌표 기준점(ECEF). D7 위치(posE, posN)도 같은 기준이다 (OSM 선과 2 m 안으로 겹침)
REF_ECEF = np.array([-2634230.47253972, -4161936.65368287, 4038639.74184999])   # 북위 39.538493°, 서경 122.331103°
LAT0, LON0 = 39.538493, -122.331103
OSM_WEST_WAYS = (756822999, 756823000, 1016088342)   # D7 경로와 겹치는 OSM highway=raceway 선 (West 짧은 판)
TUNE_LAP = "2023-10_5"   # 튜닝 랩 (validate_sim.TUNE_LAP_RUN, tune_mppi.TUNE_RUN 과 같음). 경로 파일 thunderhill_recorded_<랩>.npz


# ---------------------------------------------------------------- 좌표·OSM
def to_local(lat, lon):
    """위도·경도 → D7 지역 좌표 [동, 북] (m)."""
    a, f = 6378137.0, 1 / 298.257223563
    e2 = f * (2 - f)
    la, lo = np.radians(lat), np.radians(lon)
    N = a / np.sqrt(1 - e2 * np.sin(la) ** 2)
    X = np.c_[N * np.cos(la) * np.cos(lo), N * np.cos(la) * np.sin(lo), N * (1 - e2) * np.sin(la)] - REF_ECEF
    la0, lo0 = np.radians(LAT0), np.radians(LON0)
    R = np.array([[-np.sin(lo0), np.cos(lo0), 0], [-np.sin(la0) * np.cos(lo0), -np.sin(la0) * np.sin(lo0), np.cos(la0)]])
    return X @ R.T


def fetch_osm(cache):
    """OpenStreetMap(ODbL) 썬더힐 주변 highway=raceway 선. 처음 한 번 Overpass API 로 받아 cache 에 저장."""
    if not os.path.exists(cache):
        import urllib.parse
        import urllib.request
        q = '[out:json][timeout:60];(way["highway"="raceway"](around:3500,39.5385,-122.3311););out tags geom;'
        req = urllib.request.Request("https://overpass-api.de/api/interpreter", data=urllib.parse.urlencode({"data": q}).encode(),
                                     headers={"User-Agent": "VehicleDynamicsResearch/0.1 (academic project)", "Accept": "application/json"})
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        open(cache, "wb").write(urllib.request.urlopen(req, timeout=90).read())
    return json.load(open(cache, encoding="utf-8"))


# ---------------------------------------------------------------- 기하
def resample_closed(p, ds=1.0, sigma=0.0):
    """닫힌 곡선을 균일 간격으로 (점 수 = round(길이/ds), 이음매까지 같은 간격). sigma>0 이면 가우스 평활 [점]."""
    p = np.vstack([p, p[:1]])
    d = np.r_[0, np.cumsum(np.hypot(*np.diff(p, axis=0).T))]
    n = int(round(d[-1] / ds))
    s = np.arange(n) * (d[-1] / n)
    q = np.stack([np.interp(s, d, p[:, 0]), np.interp(s, d, p[:, 1])], 1)
    return gaussian_filter1d(q, sigma, axis=0, mode="wrap") if sigma > 0 else q


def find_lap(xy, min_len=1000.0, close=1.0):
    """닫힌 한 바퀴 (시작·끝 1 m 안). 5 m 로 하면 시작·끝이 1.4 m 어긋난 채 이어져 이음매가 꺾였다."""
    s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))]
    for i0 in range(0, len(xy) // 3, 200):
        far = np.flatnonzero(s - s[i0] > min_len)
        if not len(far):
            break
        j = far[0] + np.argmin(np.linalg.norm(xy[far[0]:] - xy[i0], axis=1)[:20000])
        if np.linalg.norm(xy[j] - xy[i0]) < close:
            return xy[i0:j]
    return None


def osm_centerline(ref, osm, max_dist=8.0):
    """D7 한 바퀴(ref)를 따라가며 가장 가까운 OSM 노면 중앙선 점(ref 와 같은 순서)."""
    dense = []
    for e in osm["elements"]:
        if e["id"] not in OSM_WEST_WAYS:
            continue
        Q = to_local(np.array([p["lat"] for p in e["geometry"]]), np.array([p["lon"] for p in e["geometry"]]))
        for i in range(len(Q) - 1):
            n = max(2, int(np.linalg.norm(Q[i + 1] - Q[i]) / 0.25))
            dense.append(np.linspace(Q[i], Q[i + 1], n))
    dense = np.vstack(dense)
    dist, k = cKDTree(dense).query(ref)
    assert dist.max() < max_dist, f"OSM 선이 D7 경로에서 {dist.max():.1f} m 떨어진 곳이 있다"
    return dense[k]


def smooth_onto_line(ref, pts, sigma=10.0):
    """꺾인 지도 중앙선 다듬기: 모양은 레이싱 라인(ref)에서, 위치는 지도 선(pts)에서. 좌우 거리를 sigma [m] 로 평활해 더한다."""
    t = np.gradient(ref, axis=0)
    t /= np.linalg.norm(t, axis=1, keepdims=True)
    n = np.stack([-t[:, 1], t[:, 0]], 1)
    e = np.sum((pts - ref) * n, 1)
    return resample_closed(ref + n * gaussian_filter1d(e, sigma, mode="wrap")[:, None], 1.0, sigma=1.0)


def surface_profile(center, samples, iters=8):
    """위치별 고도·경사·좌우 기울기. samples (M,5) [동, 북, 고도, 롤, 횡가속도].
    고도: 위치별 중앙값(5 m 평활). 경사: 고도를 트랙 거리로 미분 (차체 피치를 쓰지 않아 가감속 영향이 없다).
    좌우 기울기: 롤 = 노면 기울기 + k·횡가속도 로 보고 번갈아 푼다 (분리가 깔끔하지 않아 참고용, 시뮬레이터는 평평한 길)."""
    j = cKDTree(center).query(samples[:, :2])[1]
    n = len(center)
    idx = np.arange(n)

    def per_point(v):
        m = pd.Series(v).groupby(j).median().reindex(idx).to_numpy()
        ok = np.isfinite(m)
        return gaussian_filter1d(np.interp(idx, idx[ok], m[ok], period=n), 5, mode="wrap")

    z = per_point(samples[:, 2])
    ds = np.hypot(*(np.roll(center, -1, 0) - np.roll(center, 1, 0)).T)
    grade = np.arctan((np.roll(z, -1) - np.roll(z, 1)) / ds)        # 오르막 +
    roll, ay = samples[:, 3], samples[:, 4]
    bank, k = np.zeros(n), 0.0
    for _ in range(iters):
        k = float(np.linalg.lstsq(ay[:, None], roll - bank[j], rcond=None)[0][0])
        bank = per_point(roll - k * ay)
    return z, grade, bank, k


def load_positions(names):
    """주행 파일들 → [동, 북, 고도, 롤, 횡가속도] (vx > 5 m/s), 파일별 [동, 북] 목록."""
    cols = ["posE_m", "posN_m", "posU_m", "rollAngle_rad", "Vx_mps", "yawRate_radps"]
    P = []
    for nm in names:
        d = pd.read_csv(os.path.join(D.DATA_DIR, nm + ".csv"), comment="#", usecols=cols)[cols]
        d = d[d.Vx_mps > 5]
        P.append(np.c_[d.posE_m, d.posN_m, d.posU_m, d.rollAngle_rad, d.Vx_mps * d.yawRate_radps])
    return P


def min_curvature_offsets(ref, lo, hi, e0):
    """track.min_curvature_offsets 와 같은 문제(Σ|p_{i+1} − 2p_i + p_{i−1}|² 최소, lo ≤ e ≤ hi).
    4차 미분 꼴이라 수렴이 느려 기록선 오프셋(e0)에서 시작하고 반복을 늘렸다."""
    nrm = normals(ref)

    def f(e):
        p = ref + e[:, None] * nrm
        dd = np.roll(p, -1, 0) - 2 * p + np.roll(p, 1, 0)
        gp = 2 * (np.roll(dd, -1, 0) - 2 * dd + np.roll(dd, 1, 0))
        return float((dd ** 2).sum()), np.einsum("ij,ij->i", gp, nrm)

    e0 = np.clip(e0, lo, hi)
    sol = minimize(f, e0, jac=True, method="L-BFGS-B", bounds=list(zip(lo, hi)),
                   options=dict(maxiter=50000, maxfun=100000, ftol=1e-14, gtol=1e-10))
    return sol.x, dict(success=bool(sol.success), nit=int(sol.nit), cost_center=float(f(np.zeros(len(ref)))[0]),
                       cost_start=float(f(e0)[0]), cost=float(sol.fun))


def widths(n, width_points=None, mean=WIDTH, reach=30.0, gap_max=150.0):
    """트랙 총 폭 (n,). width_points csv (no, s, width, ...): 이웃 측정점 사이가 gap_max 이하면 선형 보간,
    더 멀면 양 끝에서 reach [m] 에 걸쳐 평균 폭으로 넘어간다. None 이면 평균 폭 균일."""
    w = np.full(n, mean)
    if not width_points:
        return w
    W = np.loadtxt(width_points, delimiter=",", comments="#", usecols=(1, 2), encoding="utf-8")
    W = W[np.argsort(W[:, 0] % n)]
    S, V = W[:, 0] % n, W[:, 1]
    for i in range(n):
        k = np.searchsorted(S, i, side="right")
        s0, w0 = (S[k - 1], V[k - 1]) if k > 0 else (S[-1] - n, V[-1])
        s1, w1 = (S[k], V[k]) if k < len(S) else (S[0] + n, V[0])
        if s1 - s0 <= gap_max:
            w[i] = w0 + (w1 - w0) * (i - s0) / max(s1 - s0, 1e-9)
        else:
            t0, t1 = np.exp(-0.5 * ((i - s0) / reach) ** 2), np.exp(-0.5 * ((s1 - i) / reach) ** 2)
            w[i] = (t0 * w0 + t1 * w1 + 0.05 * mean) / (t0 + t1 + 0.05)
    return w


def make_path(center, half_l, half_r, road, p, ds=1.0, mu=MU0, ax_acc=AX_ACC0, ax_dec=AX_DEC0):
    """경로 점 → mppi.Track 형식. 경계·노면은 중앙선 좌표계에서 정의돼 있으므로 새 경로에서 다시 투영한다."""
    nrm = normals(center)
    p = resample_closed(p, ds)
    psi, kappa = heading_curv(p)
    idx, ey = project(center, p)
    return dict(x=p[:, 0], y=p[:, 1], psi=psi, kappa=kappa, s=np.arange(len(p)) * ds,
                width_l=half_l[idx] - ey, width_r=half_r[idx] + ey, vref=speed_profile(kappa, ds, mu, ax_acc, ax_dec),
                bank=road["bank"][idx], grade=road["grade"][idx], z=road["z"][idx],
                inner=center + half_l[:, None] * nrm, outer=center - half_r[:, None] * nrm, mu=mu, ax_acc=ax_acc, ax_dec=ax_dec)


# ---------------------------------------------------------------- 전체
def build_all(out_dir=OUT, width_points=WIDTH_POINTS, margin=1.0, smooth=10.0):
    train = D.get_split()["train"]                                   # 지도·중앙선은 학습 파일만 (시험 주행 정보가 섞이지 않게)
    P = load_positions(train)
    samples = np.vstack(P)
    lap0 = next(l for l in (find_lap(p[:, :2]) for p in P) if l is not None)
    ref = resample_closed(lap0, 1.0, sigma=2.0)
    raw = osm_centerline(ref, fetch_osm(os.path.join(out_dir, "thunderhill_osm.json")))
    center = smooth_onto_line(ref, raw, smooth)
    n = len(center)
    z, grade, bank, k_roll = surface_profile(center, samples)
    road = dict(z=z, grade=grade, bank=bank)
    w = widths(n, width_points)
    half_l = half_r = w / 2

    lap = best_lap(D.available())                                     # track.py 와 같은 기준 랩 (2023-10_8)
    lap_xy = np.c_[lap["x"], lap["y"]]
    li, le = project(center, lap_xy)
    off = pd.Series(le).groupby(li).median()
    rec_off = np.interp(np.arange(n), off.index.to_numpy(), off.to_numpy(), period=n)
    mc, mcinfo = min_curvature_offsets(center, -half_r + margin, half_l - margin, rec_off)
    nrm = normals(center)
    # [변경 11/20] 기록 경로는 그 랩의 실제 출발점에서 시작한다 (경로 0번 = 랩 출발점). 이전에는 0번을 중앙선 시작점에 맞춰
    # 출발점이 경로 끝(2602번)에 놓였고, validate_sim.recorded_track 의 '랩 앞쪽 절반 = 경로 앞쪽 절반' 가정이 깨져
    # 출발 지점 목표 속도가 랩 끝의 32 m/s 가 됐다. 튜닝 랩도 자기 선을 따로 만든다 (경로와 목표 속도를 같은 랩에서)
    tune = best_lap([TUNE_LAP])
    paths = {"recorded": make_path(center, half_l, half_r, road, lap_xy),
             f"recorded_{TUNE_LAP}": make_path(center, half_l, half_r, road, np.c_[tune["x"], tune["y"]]),
             "center": make_path(center, half_l, half_r, road, center),
             "mincurv": make_path(center, half_l, half_r, road, center + mc[:, None] * nrm)}

    # 점검: 모든 주행 위치가 경계 안에 드는가 (바퀴 기준 = 차 중심 + 차 반폭)
    allp = np.vstack([p[::10, :2] for p in load_positions(D.available())])
    idx, ey = project(center, allp)
    room = w[idx] / 2 - CAR_HALF - np.abs(ey)
    # 주행 범위 (track.py 의 gps_envelope 와 같은 뜻, 중앙선 기준, 참고용)
    gl = pd.Series(ey).groupby(idx).max().reindex(range(n)).interpolate(limit_direction="both").to_numpy()
    gr = pd.Series(ey).groupby(idx).min().reindex(range(n)).interpolate(limit_direction="both").to_numpy()
    info = dict(method="osm", best_lap=dict(run=lap["run"], v_mean=lap["v_mean"], lap_time=lap["lap_time"], t0=lap["t0"]),
                boundary=dict(kind="OSM 노면 중앙선 ± 실측 폭/2", width_points=os.path.basename(width_points) if width_points else None,
                              width_total_min_median_max=[float(w.min()), float(np.median(w)), float(w.max())]),
                map_files=train, length_m=n, roll_gain_deg_per_mps2=float(np.degrees(k_roll)),
                z_range_m=float(z.max() - z.min()), grade_p1_p99_deg=np.degrees(np.percentile(grade, [1, 99])).round(2).tolist(),
                inside_check=dict(points=int(len(ey)), offset_abs_p50_p99_max=[float(np.percentile(np.abs(ey), 50)), float(np.percentile(np.abs(ey), 99)), float(np.abs(ey).max())],
                                  min_wheel_room_m=float(room.min()), track_m_with_wheel_outside=int(len(np.unique(idx[room < 0])))),
                mincurv=mcinfo)
    for k, p in paths.items():
        info[k] = dict(length_m=float(len(p["x"])), kappa_absmax=float(np.abs(p["kappa"]).max()), sum_kappa2=float(np.sum(p["kappa"] ** 2)),
                       vref_min=float(p["vref"].min()), vref_max=float(p["vref"].max()), lap_time_profile=float(np.sum(1.0 / p["vref"])),
                       width_min=float(np.minimum(p["width_l"], p["width_r"]).min()))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        for k, p in paths.items():
            np.savez(os.path.join(out_dir, f"thunderhill_{k}.npz"), **{kk: np.asarray(v) for kk, v in p.items()})
        np.savez(os.path.join(out_dir, "thunderhill_boundary.npz"), ref=center, left=half_l, right=-half_r, bank=bank,
                 gps_left=gl, gps_right=gr, z=z, grade=grade, osm_raw=raw)
        json.dump(info, open(os.path.join(out_dir, "thunderhill_info.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    return paths, info, (center, half_l, half_r)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--uniform-width", action="store_true", help="위치별 실측 대신 평균 10.9 m 균일")
    ap.add_argument("--margin", type=float, default=1.0, help="mincurv 경로의 가장자리 여유 [m]")
    a = ap.parse_args()
    _, info, _ = build_all(out_dir=a.out, width_points=None if a.uniform_width else WIDTH_POINTS, margin=a.margin)
    print(json.dumps(info, indent=1, ensure_ascii=False))
