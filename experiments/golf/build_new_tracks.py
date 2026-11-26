"""[11/26] 학습에 안 쓴 새 트랙 (일반화 시험용): Thunderhill East (직접 제작) + TUM racetrack-database 저·중속 코스 4개 (Norisring 은 폭 자료 문제로 제외).

  python experiments/golf/build_new_tracks.py east          # Thunderhill East (OSM 중앙선 + USGS 고도 경사, 폭 가정)
  python experiments/golf/build_new_tracks.py tum           # TUM 4개 (tum_raw/ 의 CSV, Norisring 제외)
  python experiments/golf/build_new_tracks.py check_usgs    # USGS 고도로 West 경사를 다시 만들어 D7 GPS 고도 경사와 비교
  python experiments/golf/build_new_tracks.py plot          # → results/tracks/new_tracks.png
→ experiments/golf/results/tracks/<이름>_{center,mincurv[,raceline]}.npz, <이름>_boundary.npz, <이름>_info.json
  (thunderhill_*.npz 와 같은 키라 mppi.Track 이 그대로 읽음. 기존 thunderhill_* 파일은 건드리지 않음)

Thunderhill East
  중앙선: OSM way 28825115 (highway=raceway, "Thunderhill Raceway", 닫힌 고리 4.6 km, oneway → 점 순서 = 주행 방향).
          © OpenStreetMap contributors, ODbL. results/tracks/thunderhill_osm.json (West 만들 때 받은 캐시) 그대로.
          OSM 점 간격이 약 18 m 라 주기 스플라인으로 잇고 1 m 로 다시 나눔.
  고도:   USGS 3DEP (epqs.nationalmap.gov, 1 m 라이다) 를 중앙선 10 m 마다 → 보간 → 5 m 평활 → 거리로 미분 = 경사 (West 와 같은 처리).
          캐시 results/tracks/usgs_elev_cache.json. 방법 확인은 check_usgs (West 에서 D7 GPS 고도와 비교).
  폭:     실측 없음 → West 실측 평균 10.9 m 균일 [가정]. 좌우 기울기 0 (West 도 MPPI 에서는 0).
TUM racetrack-database (LGPL-3.0, github.com/TUMFTM/racetrack-database)
  중앙선 = OSM 점 평활, 좌우 폭 = 위성 이미지 자동 추출 (README: 정밀도는 곳마다 다름). 레이싱 라인 = TUM 최소 곡률.
  고도 없음 → 경사 0. 원점은 트랙마다 지역 좌표 (위경도 아님).
공통: 경로 center·mincurv (우리 min_curvature_offsets, 가장자리 여유 1 m).
  목표 속도 vref = track.speed_profile 과 같은 방식 (코너 √(μg/|κ|), μ 1.2, 감속 −5.1 m/s²) 에 가속 한계만 속도별로:
  min(4.3, (P2 구동력 상한(v) − 굴림·공기 저항(v)) / m) (p2_fit.json 의 c0·c2). 4.3 고정이면 East 긴 직선에서 64 m/s 가 나와 비현실적.
"""
import json
import os
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from scipy.interpolate import splev, splprep
from scipy.ndimage import gaussian_filter1d

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
from deep_dynamics.golf import track_osm as TO  # noqa: E402
from deep_dynamics.golf.track import G, MU0, AX_ACC0, AX_DEC0, normals  # noqa: E402

OUT = TO.OUT
EAST_WAY = 28825115
TUM_RAW = os.path.join(OUT, "tum_raw")
TUM_TRACKS = ("Budapest", "Zandvoort", "Oschersleben", "BrandsHatch")  # [11/27] Norisring 제외 (build_tum 참고)
USGS_CACHE = os.path.join(OUT, "usgs_elev_cache.json")
UA = "VehicleDynamicsResearch/0.1 (academic project)"
DATA_VMAX = 34.0  # D7 학습 데이터 최고 속도 근처 (m/s). 목표 속도가 이보다 크면 모델 외삽 구간


# ---------------------------------------------------------------- 고도 (USGS 3DEP)
def to_latlon(e, n):
    """D7 지역 좌표 [동, 북] → 위경도 (to_local 의 역, 뉴턴 2번이면 1 mm 안)."""
    lat = TO.LAT0 + n / 111_000.0
    lon = TO.LON0 + e / (111_000.0 * np.cos(np.radians(TO.LAT0)))
    for _ in range(3):
        q = TO.to_local(np.atleast_1d(lat), np.atleast_1d(lon))
        q = np.asarray(q).reshape(-1, 2)
        lat = lat + (n - q[:, 1]) / 111_000.0
        lon = lon + (e - q[:, 0]) / (111_000.0 * np.cos(np.radians(lat)))
    return lat, lon


def usgs_elev(lat, lon):
    cache = json.load(open(USGS_CACHE)) if os.path.exists(USGS_CACHE) else {}
    keys = [f"{a:.6f},{b:.6f}" for a, b in zip(lat, lon)]
    todo = [k for k in dict.fromkeys(keys) if k not in cache]

    def get(k):
        a, b = k.split(",")
        url = f"https://epqs.nationalmap.gov/v1/json?x={b}&y={a}&wkid=4326&units=Meters&includeDate=false"
        for t in range(5):
            try:
                r = json.loads(urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": UA}), timeout=30).read())
                return k, float(r["value"])
            except Exception:
                time.sleep(1.0 + t)
        return k, None

    with ThreadPoolExecutor(4) as ex:
        for i, (k, v) in enumerate(ex.map(get, todo)):
            if v is not None:
                cache[k] = v
            if i % 100 == 99:
                json.dump(cache, open(USGS_CACHE, "w"))
    json.dump(cache, open(USGS_CACHE, "w"))
    return np.array([cache.get(k, np.nan) for k in keys])


def elevation_profile(center, step=10):
    """중앙선 (n,2) 지역 좌표 → z (n,), 경사 (n,) [rad, 오르막 +]. step m 마다 USGS, 주기 보간, 5 m 평활 (West surface_profile 과 같은 평활)."""
    n = len(center)
    idx = np.arange(0, n, step)
    lat, lon = to_latlon(center[idx, 0], center[idx, 1])
    zs = usgs_elev(lat, lon)
    ok = np.isfinite(zs)
    z = np.interp(np.arange(n), idx[ok], zs[ok], period=n)
    z = gaussian_filter1d(z, 5, mode="wrap")
    ds = np.hypot(*(np.roll(center, -1, 0) - np.roll(center, 1, 0)).T)
    grade = np.arctan((np.roll(z, -1) - np.roll(z, 1)) / ds)
    return z, grade, int((~ok).sum())


# ---------------------------------------------------------------- 공통
def speed_profile_power(kappa, ds=1.0, mu=MU0, ax_acc=AX_ACC0, ax_dec=AX_DEC0):
    """track.speed_profile 과 같고, 가속 한계만 속도별 (P2 구동력 상한 − 저항) / m 과 ax_acc 중 작은 값."""
    from deep_dynamics.golf import p2
    P = p2.load_fit()
    v = np.tile(np.sqrt(mu * G / np.maximum(np.abs(kappa), 1e-6)), 3)
    for i in range(1, len(v)):
        u = v[i - 1]
        cap = float(p2.force_cap(u)) if hasattr(p2, "force_cap") else float(__import__("deep_dynamics.golf.sim", fromlist=["sim"]).force_cap(u))
        a = min(ax_acc, (cap - P.c0 - P.c2 * u * u) / P.m)
        v[i] = min(v[i], np.sqrt(max(u * u + 2 * a * ds, 0.0)))
    for i in range(len(v) - 2, -1, -1):
        v[i] = min(v[i], np.sqrt(v[i + 1] ** 2 + 2 * abs(ax_dec) * ds))
    n = len(kappa)
    return v[n:2 * n]


def spline_closed(p, ds=1.0, smooth=0.0):
    p = p[:-1] if np.linalg.norm(p[0] - p[-1]) < 1e-6 else p
    tck, _ = splprep([p[:, 0], p[:, 1]], s=smooth * len(p), per=1)
    u = np.linspace(0, 1, 20 * len(p), endpoint=False)
    return TO.resample_closed(np.stack(splev(u, tck), 1), ds)


def build(name, center, half_l, half_r, road, extra_paths=None, margin=1.0, meta=None, mincurv=True):
    n = len(center)
    paths, mcinfo = {"center": TO.make_path(center, half_l, half_r, road, center)}, None
    if mincurv:
        mc, mcinfo = TO.min_curvature_offsets(center, -half_r + margin, half_l - margin, np.zeros(n))
        paths["mincurv"] = TO.make_path(center, half_l, half_r, road, center + mc[:, None] * normals(center))
    for k, p in (extra_paths or {}).items():
        paths[k] = TO.make_path(center, half_l, half_r, road, p)
    for p in paths.values():
        p["vref"] = speed_profile_power(p["kappa"])
    info = dict(name=name, length_m=n, mincurv_opt=mcinfo, width_total_min_median_max=[float(x) for x in np.percentile(half_l + half_r, [0, 50, 100])],
                z_range_m=float(np.ptp(road["z"])), grade_p1_p99_deg=np.degrees(np.percentile(road["grade"], [1, 99])).round(2).tolist(), **(meta or {}))
    for k, p in paths.items():
        info[k] = dict(length_m=float(len(p["x"])), kappa_absmax=float(np.abs(p["kappa"]).max()), min_radius_m=float(1 / np.abs(p["kappa"]).max()),
                       vref_min=float(p["vref"].min()), vref_max=float(p["vref"].max()), lap_time_profile=float(np.sum(1.0 / p["vref"])),
                       frac_above_data_vmax=float(np.mean(p["vref"] > DATA_VMAX)), width_min=float(np.minimum(p["width_l"], p["width_r"]).min()))
    for k, p in paths.items():
        f = os.path.join(OUT, f"{name}_{k}.npz")
        assert not os.path.basename(f).startswith("thunderhill_") or name != "thunderhill", f
        np.savez(f, **{kk: np.asarray(v) for kk, v in p.items()})
    np.savez(os.path.join(OUT, f"{name}_boundary.npz"), ref=center, left=half_l, right=-half_r, bank=road["bank"], z=road["z"], grade=road["grade"])
    json.dump(info, open(os.path.join(OUT, f"{name}_info.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    return paths, info


# ---------------------------------------------------------------- Thunderhill East
def east_centerline():
    osm = TO.fetch_osm(os.path.join(OUT, "thunderhill_osm.json"))
    e = next(x for x in osm["elements"] if x["id"] == EAST_WAY)
    Q = np.asarray(TO.to_local(np.array([p["lat"] for p in e["geometry"]]), np.array([p["lon"] for p in e["geometry"]]))).reshape(-1, 2)
    return spline_closed(Q, 1.0, smooth=0.05), Q


def build_east():
    center, raw = east_centerline()
    z, grade, n_miss = elevation_profile(center)
    road = dict(z=z, grade=grade, bank=np.zeros(len(center)))
    w = np.full(len(center), TO.WIDTH)
    paths, info = build("thunderhill_east", center, w / 2, w / 2, road,
                        meta=dict(source="OSM way 28825115 (ODbL) + USGS 3DEP 1 m (10 m 간격)", width="10.9 m 균일 [가정: West 실측 평균]",
                                  usgs_missing=n_miss, osm_points=len(raw),
                                  osm_to_center_max_m=float(np.max(np.min(np.linalg.norm(raw[:, None] - center[None], axis=2), 1)))))
    print(json.dumps({k: v for k, v in info.items() if k != "mincurv_opt"}, indent=1, ensure_ascii=False))


def check_usgs():
    """West: USGS 로 만든 경사 vs D7 GPS 고도로 만든 경사 (thunderhill_boundary.npz)."""
    b = np.load(os.path.join(OUT, "thunderhill_boundary.npz"))
    z_u, g_u, n_miss = elevation_profile(b["ref"])
    g_d = b["grade"]
    d = np.degrees(g_u - g_d)
    off = float(np.median(z_u - b["z"]))
    r = dict(points=len(g_d), usgs_missing=n_miss, grade_rms_diff_deg=float(np.sqrt(np.mean(d ** 2))), grade_p99_absdiff_deg=float(np.percentile(np.abs(d), 99)),
             grade_corr=float(np.corrcoef(g_u, g_d)[0, 1]), z_offset_m=off, z_rms_after_offset_m=float(np.sqrt(np.mean((z_u - b["z"] - off) ** 2))),
             grade_range_d7_deg=np.degrees([g_d.min(), g_d.max()]).round(2).tolist(), grade_range_usgs_deg=np.degrees([g_u.min(), g_u.max()]).round(2).tolist())
    json.dump(r, open(os.path.join(OUT, "usgs_check_west.json"), "w"), indent=1)
    print(json.dumps(r, indent=1))


# ---------------------------------------------------------------- TUM
def tum_center(a, ds=1.0):
    """TUM 중앙선 (5 m 점, 폭은 점마다) → 점 사이 직선, 1 m 간격 + 폭 (호 길이로 선형 보간).
    [11/27 확인] TUM 식 경계 (원본 점 ± 폭 × 스플라인 법선) 와 경계선 거리 차이 중앙값 0.00, 최대 0.04~0.08 m.
    주기 스플라인으로 이어 보니 최대 0.08~0.12 m 로 오히려 멀어져 직선을 유지한다."""
    p = a[:, :2]
    d = np.r_[0, np.cumsum(np.hypot(*np.diff(np.vstack([p, p[:1]]), axis=0).T))]
    center = TO.resample_closed(p, ds)
    s = np.arange(len(center)) * d[-1] / len(center)  # resample_closed 와 같은 호 길이 위치
    return center, np.interp(s, d, np.r_[a[:, 3], a[0, 3]]), np.interp(s, d, np.r_[a[:, 2], a[0, 2]])


def raceline_margin_raw(a, rl):
    """원본 자료 기준 레이싱 라인 여유 (꺾은선 중앙선에 정확 투영, 점 사이 폭 선형). 변환 오차 비교용."""
    C, wr, wl = a[:, :2], a[:, 2], a[:, 3]
    A, B = C, np.roll(C, -1, 0)
    AB = B - A
    L2 = (AB ** 2).sum(1)
    m = []
    for q in rl:
        t = np.clip(((q - A) * AB).sum(1) / L2, 0, 1)
        dist = np.linalg.norm(q - (A + t[:, None] * AB), axis=1)
        i = int(np.argmin(dist))
        j = (i + 1) % len(C)
        side = np.sign(AB[i, 0] * (q - A[i])[1] - AB[i, 1] * (q - A[i])[0])
        WL, WR = wl[i] + (wl[j] - wl[i]) * t[i], wr[i] + (wr[j] - wr[i]) * t[i]
        m.append(WL - dist[i] if side > 0 else WR - dist[i])
    return np.array(m)


def build_tum():
    """[11/27] Norisring 제외 (헤어핀 안쪽 폭 8~9 m 가 중앙선 회전 반경 약 11 m 와 맞먹어 경계가 접힘 — 원본 폭 자료 문제).
    우리 최소 곡률 경로는 만들지 않음 (TUM 레이싱 라인이 더 좋음: 이상적 랩타임 10~13% vs 우리 2~3% 단축)."""
    for t in TUM_TRACKS:
        a = np.loadtxt(os.path.join(TUM_RAW, "tracks", f"{t}.csv"), delimiter=",", comments="#")
        rl = np.loadtxt(os.path.join(TUM_RAW, "racelines", f"{t}.csv"), delimiter=",", comments="#")
        center, half_l, half_r = tum_center(a)
        road = dict(z=np.zeros(len(center)), grade=np.zeros(len(center)), bank=np.zeros(len(center)))
        paths, info = build(f"tum_{t.lower()}", center, half_l, half_r, road, extra_paths={"raceline": rl}, mincurv=False,
                            meta=dict(source="TUMFTM racetrack-database (LGPL-3.0): 중앙선 OSM 평활, 폭 위성 이미지, raceline TUM 최소 곡률",
                                      grade="없음 (0)", center="원본 5 m 점을 지나는 주기 스플라인, 1 m 간격"))
        mo = np.minimum(paths["raceline"]["width_l"], paths["raceline"]["width_r"])
        mr = raceline_margin_raw(a, rl)
        info["raceline_margin"] = dict(ours_min=float(mo.min()), ours_p1=float(np.percentile(mo, 1)), raw_min=float(mr.min()), raw_p1=float(np.percentile(mr, 1)),
                                       ours_frac_below_0p9=float(np.mean(mo < 0.9)), raw_frac_below_0p9=float(np.mean(mr < 0.9)))
        json.dump(info, open(os.path.join(OUT, f"tum_{t.lower()}_info.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
        print(t, f"{info['length_m'] / 1000:.2f} km", "여유 우리/원본: 최소 %.2f/%.2f 1%% %.2f/%.2f 0.9m 미만 %.1f%%/%.1f%%" % (
            mo.min(), mr.min(), np.percentile(mo, 1), np.percentile(mr, 1), 100 * np.mean(mo < 0.9), 100 * np.mean(mr < 0.9)),
              {k: {q: round(info[k][q], 1) for q in ("min_radius_m", "vref_max", "lap_time_profile")} for k in ("center", "raceline")})


def plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = ["thunderhill", "thunderhill_east"] + [f"tum_{t.lower()}" for t in TUM_TRACKS]
    names = [n for n in names if os.path.exists(os.path.join(OUT, f"{n}_boundary.npz"))]
    fig, axs = plt.subplots(2, 4, figsize=(18, 9), layout="constrained")
    for ax, nm in zip(axs.flat, names):
        b = np.load(os.path.join(OUT, f"{nm}_boundary.npz"))
        c, nrm = b["ref"], normals(b["ref"])
        for side in (b["left"], b["right"]):
            q = c + side[:, None] * nrm
            ax.plot(np.r_[q[:, 0], q[0, 0]], np.r_[q[:, 1], q[0, 1]], color="0.3", lw=0.8)
        key = "raceline" if os.path.exists(os.path.join(OUT, f"{nm}_raceline.npz")) else "mincurv"
        p = np.load(os.path.join(OUT, f"{nm}_{key}.npz"))
        sc = ax.scatter(p["x"], p["y"], c=p["vref"], s=1.5, cmap="viridis", vmin=5, vmax=45)
        info = json.load(open(os.path.join(OUT, f"{nm}_info.json"), encoding="utf-8")) if os.path.exists(os.path.join(OUT, f"{nm}_info.json")) else {}
        L = info.get("length_m", len(c))
        ax.set_title(f"{nm} ({L / 1000:.2f} km)")
        ax.set_aspect("equal")
        ax.set_xticks([]), ax.set_yticks([])
    for ax in list(axs.flat)[len(names):]:
        ax.axis("off")
    fig.colorbar(sc, ax=axs, shrink=0.6, label="vref on min-curvature path (m/s)")
    fig.suptitle("Tracks: grey = boundaries, color = target speed on min-curvature path (mu 1.2)")
    fig.savefig(os.path.join(OUT, "new_tracks.png"), dpi=110)


if __name__ == "__main__":
    {"east": build_east, "tum": build_tum, "check_usgs": check_usgs, "plot": plot}[sys.argv[1]]()
