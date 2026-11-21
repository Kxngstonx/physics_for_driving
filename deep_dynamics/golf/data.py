"""Golf (Stanford hh613qz0317, Thunderhill West) 데이터 → 사다리 모델 입력.

docs/experiment_plan_golf.md 5-6절, 6-2절을 그대로 구현한다.
  1. CSV 읽기 (200 Hz, '#' 머리 2줄)
  2. 앞바퀴 종방향 힘 Fxf (N) 계산
       실측 감속비 = 엔진 회전 속도 ÷ 앞바퀴 평균 회전 속도 → 가장 가까운 기어 값으로 맞춤
       구동력 = 엔진 토크 × 감속비 ÷ 바퀴 반지름   (감속비가 어느 기어에도 안 맞으면 변속 중 → 0)
       출발 중(1단, 클러치 미끄러짐)은 엔진 토크 × 1단 감속비 (drive_force 주석)
       [변경 11/19] 클러치 신호(clutchEngage_int)는 쓰지 않는다 (drive_force 주석)
       Fxf = 구동계 효율 × 구동력 − (앞왼쪽 + 앞오른쪽 제동력)   (제동력 열은 양수로 기록돼 있음 [확인])
       구동계 효율·바퀴 반지름·감속비는 deep_dynamics/golf/specs.json (estimate_specs.py 총괄, 11/21). force 모드 물리 식(R1, R4)에는
       구동 이득 계수가 없어 효율을 학습으로 흡수할 수 없으므로 입력 계산에 넣는다
  3. [vx, vy, r, Fxf, 조향각, 뱅크] 에 10 Hz 4차 버터워스 양방향 필터 → 8개마다 1개 (dt 0.04 s)
  4. 5 m/s 기준으로 버리지 않음 (그 기준은 AV-21 것). 단, 정지 구간(vx < V_STOP)은 잘라 낸다:
     정지 중 기록된 제동력은 차를 붙잡아 두는 힘(중앙값 −3.1 kN)이라 운동을 만드는 힘이 아니고,
     물리 모델에 넣으면 '0 m/s 에서 2 m/s² 감속'이 된다. 전체의 3.1% (144 s) [확인: 20개 파일]
  5. 파일(주행) 단위 분할 14/3/3, 배정 시드 0
"""
import functools
import glob
import json
import os

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data", "golf")

FS = 200.0          # 원본 샘플링 (Hz)
EVERY = 8           # 200 Hz → 25 Hz
DT = EVERY / FS     # 0.04 s
CUTOFF = 10.0       # 저역 통과 차단 주파수 (Hz)
GEAR_TOL = 0.10     # 어느 기어와도 10% 넘게 다르면 변속 중으로 보고 구동력 0
# [변경 11/21] 바퀴 반지름(wheel_radius)·감속비(gear_ratios)·구동계 효율(drive_eff)은 손으로 적지 않고 specs.json 에서 읽는다
#   (golf_specs.value). 이전 상수: 0.319 m, [8.8, 5.65, 3.93, 3.0] (2개 파일 탐색값), 0.699
V_STOP = 0.5        # 정지 판정 (m/s, 필터 후 25 Hz 값 기준)
V_LAUNCH = 6.0      # 출발 구간 제외 ablation 의 기준 속도 (계획서 10절)
MIN_LEN = 40        # 구간 최소 길이 (25 Hz 샘플 수)
DUPLICATES = ("2023-10_2", "2023-10_3")  # 2023-10, 2023-10_1 과 MD5 같음


def available(data_dir=DATA_DIR):
    names = sorted(os.path.splitext(os.path.basename(f))[0] for f in glob.glob(os.path.join(data_dir, "*.csv")))
    return [n for n in names if n not in DUPLICATES]


def drive_force(df, gears=None, wheel_r=None):
    """엔진 토크 → 앞바퀴 구동력 (N). 반환 (구동력, 기어 감속비 또는 nan).
    gears·wheel_r 를 주지 않으면 specs.json 값 (estimate_specs 는 새로 추정한 값을 직접 넘긴다)."""
    from .golf_specs import value
    GEARS = np.asarray(value("gear_ratios") if gears is None else gears, dtype=float)
    WHEEL_R = value("wheel_radius") if wheel_r is None else wheel_r
    wf = 0.5 * (df["wheelspeed_fl"].to_numpy() + df["wheelspeed_fr"].to_numpy())
    eng = df["engine_rpm"].to_numpy() * 2.0 * np.pi / 60.0
    with np.errstate(divide="ignore", invalid="ignore"):
        raw = np.where(wf > 1.0, eng / wf, np.nan)
    k = np.nanargmin(np.abs(np.nan_to_num(raw, nan=1e9)[:, None] / GEARS - 1.0), axis=1)
    gear = GEARS[k]
    ok = np.abs(raw / gear - 1.0) < GEAR_TOL
    # [변경 11/19] 클러치 신호를 곱하지 않는다. 엔진과 바퀴가 기어비대로 묶여 돌면(ok) 클러치는 물리적으로 붙어 있다.
    # 실제로 ok 인데 clutchEngage_int = 0 인 표본(10 m/s 이상·스로틀 30% 이상의 약 15%)도 체결(4)일 때와 거의 같은
    # 토크·가속도를 보인다 (1단 216 Nm·2.2 m/s² vs 231 Nm·2.6 m/s², 2단 195 Nm·0.95 vs 198 Nm·1.24) [확인 11/19].
    # 이전에는 '클러치 신호 ÷ 4'를 곱해 이 구간의 구동력을 0 으로 계산했다.
    # 출발: 엔진이 1단 감속비보다 빨리 돎 = 1단에서 클러치가 미끄러지는 중. 미끄러지는 클러치는 엔진 토크를
    # 그대로 넘기므로 1단 감속비로 본다. (이 처리가 없으면 6 m/s 아래 출발 구간의 구동력이 0 이 되는데,
    # 실제로는 2~4 m/s² 로 가속함 [확인: 2023-11.csv t 5.9~8.9 s])
    launch = (raw > GEARS[0] * (1.0 + GEAR_TOL)) & (df["throttleCmd_percent"].to_numpy() > 1.0)
    gear = np.where(launch, GEARS[0], np.where(ok, gear, np.nan))
    F = np.where(ok | launch, df["engineTorque_Nm"].to_numpy() * np.nan_to_num(gear) / WHEEL_R, 0.0)
    return F, gear


def read_csv(name, data_dir=DATA_DIR):
    """CSV 하나 → 200 Hz DataFrame, 기록된 열만 (specs.json 이 필요 없음: estimate_specs 가 specs 를 처음 만들 때 씀)."""
    return pd.read_csv(os.path.join(data_dir, name + ".csv"), comment="#")


def read_raw(name, data_dir=DATA_DIR, gears=None, wheel_r=None, drive_eff=None):
    """CSV 하나 → 200 Hz DataFrame + 계산 열 (F_drive, F_brake_f, F_brake_r, Fxf, gear).
    gears·wheel_r·drive_eff 를 주지 않으면 specs.json 값."""
    from .golf_specs import value
    df = read_csv(name, data_dir)
    F, gear = drive_force(df, gears, wheel_r)
    df["F_drive"] = F
    df["gear"] = gear
    df["F_brake_f"] = df["brake_fl_N"] + df["brake_fr_N"]
    df["F_brake_r"] = df["brake_rl_N"] + df["brake_rr_N"]
    df["Fxf"] = (value("drive_eff") if drive_eff is None else drive_eff) * df["F_drive"] - df["F_brake_f"]
    return df


def lowpass(x, cutoff=CUTOFF, fs=FS, order=4, axis=0):
    b, a = butter(order, cutoff, fs=fs)
    return filtfilt(b, a, x, axis=axis)


MODEL_COLS = ["Vx_mps", "Vy_mps", "yawRate_radps", "Fxf", "roadWheelAngle_rad", "bank_rad"]
EXTRA_COLS = ["t_s", "posE_m", "posN_m", "yawAngle_rad", "s_m", "grade_rad", "F_drive", "F_brake_f", "F_brake_r",
              "auto_steer_bool", "auto_throttle_bool"]


@functools.lru_cache(maxsize=None)
def _filtered(name, data_dir=DATA_DIR):
    """한 주행 → 시간 끊김 없는 조각마다 (필터·솎아낸 모델 열 (T,6), 보조 열 dict). 속도 기준과 무관해 한 번만 계산."""
    df = read_raw(name, data_dir)
    t = df["t_s"].to_numpy()
    gaps = np.where(np.abs(np.diff(t) - 1.0 / FS) > 1e-4)[0]
    bounds = np.r_[0, gaps + 1, len(t)]
    out = []
    for i0, i1 in zip(bounds[:-1], bounds[1:]):
        if (i1 - i0) // EVERY < MIN_LEN:  # 너무 짧은 조각은 필터를 걸 수 없고 창도 못 만듦
            continue
        sub = df.iloc[i0:i1]
        filt = lowpass(sub[MODEL_COLS].to_numpy(dtype=np.float64))[::EVERY]
        ex = {c: sub[c].to_numpy(dtype=np.float64)[::EVERY] for c in EXTRA_COLS}
        out.append((filt, ex))
    return out


TRACK_MAP = os.path.join(ROOT, "..", "experiments", "golf", "results", "tracks", "thunderhill_boundary.npz")


@functools.lru_cache(maxsize=None)
def _grade_map(path=TRACK_MAP):
    """[11/19] 트랙 경사 지도 (track_osm.py 출력: 중앙선 ref, grade rad). 위치 (동, 북) → 경사."""
    from scipy.spatial import cKDTree
    path = os.path.normpath(path)
    assert os.path.exists(path), f"{path} 없음: python -m deep_dynamics.golf.track 을 먼저 실행"
    b = np.load(path)
    assert "grade" in b.files, "경사가 없는 이전 트랙 파일: python -m deep_dynamics.golf.track (osm) 으로 다시 만들 것"
    return cKDTree(b["ref"]), b["grade"].astype(np.float64)


def grade_at(x, y):
    """위치 (동, 북) [m] → 트랙 지도 경사 [rad, 오르막 +] (가장 가까운 중앙선 점)."""
    tree, g = _grade_map()
    return g[tree.query(np.c_[x, y])[1]]


@functools.lru_cache(maxsize=None)
def load_run(name, data_dir=DATA_DIR, min_len=MIN_LEN, v_stop=V_STOP, road="bank"):
    """한 주행 → 끊김 없는 구간 리스트 (25 Hz). 같은 프로세스 안에서는 한 번만 만든다 (반환값을 고치지 말 것).
    구간 dict: run, t, x, y, psi, s_m, s (T,3) [vx,vy,r], u (T,3) [Fxf, delta, bank], extra(dict).
    vx < v_stop 인 곳에서 자른다 (필터를 먼저 걸고 자름: 자른 조각마다 걸면 끝부분이 흔들림).
    road: 'bank' (이전, 데이터 bank_rad 열) | 'grade' ([11/19] u (T,4) [Fxf, delta, bank=0, grade], 경사는 트랙 지도에서)"""
    assert road in ("bank", "grade")
    segs = []
    for filt, ex in _filtered(name, data_dir):
        edges = np.flatnonzero(np.diff(np.r_[0, (filt[:, 0] >= v_stop).astype(int), 0]))
        for k, j in zip(edges[::2], edges[1::2]):  # [k, j) = 연속으로 v_stop 이상인 구간
            if j - k < min_len:
                continue
            exs = {c: v[k:j] for c, v in ex.items()}
            u = filt[k:j, 3:].copy()
            if road == "grade":  # 좌우 기울기 0 (시뮬레이터·MPPI 와 같은 평평한 길), 경사 열 추가
                u[:, 2] = 0.0
                u = np.c_[u, grade_at(exs["posE_m"], exs["posN_m"])]
            segs.append(dict(run=name, t=exs["t_s"], x=exs["posE_m"], y=exs["posN_m"], psi=exs["yawAngle_rad"],
                             s_m=exs["s_m"], s=filt[k:j, :3].copy(), u=u, extra=exs))
    return segs


def make_split(names, seed=0, n=(14, 3, 3)):
    names = sorted(names)
    assert len(names) == sum(n), f"파일 {len(names)}개, 분할 {n}"
    p = np.random.default_rng(seed).permutation(len(names))
    pick = [names[i] for i in p]
    return dict(seed=seed, train=sorted(pick[:n[0]]), val=sorted(pick[n[0]:n[0] + n[1]]), test=sorted(pick[n[0] + n[1]:]))


def get_split(data_dir=DATA_DIR):
    path = os.path.join(data_dir, "split.json")
    if os.path.exists(path):
        return json.load(open(path))
    sp = make_split(available(data_dir))
    json.dump(sp, open(path, "w"), indent=1)
    return sp


def split_segs(names, data_dir=DATA_DIR, v_stop=V_STOP, road="bank"):
    """[(s, u), ...] — ladder train.windows() 입력 형식.
    v_stop: 이 속도 미만 구간을 잘라 냄. 기본 V_STOP (정지만 제외). V_LAUNCH = 출발 구간 제외 ablation.
    road: 'bank' (이전) | 'grade' (경사 채널, configs.road_of(cfg))"""
    return [(sg["s"], sg["u"]) for n in names for sg in load_run(n, data_dir, v_stop=v_stop, road=road)]
