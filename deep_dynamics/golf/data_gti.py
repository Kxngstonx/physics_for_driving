"""GTI(D1, Spielberg 2019 Stanford VW GTI "Niki", Thunderhill 본 코스) → Golf 사다리 입력 형식 (차량 간 실험용, 11/19).

  from deep_dynamics.golf import data_gti as G
  segs = G.split_segs("train", road="grade")      # [(s (T,3), u (T,nu)), ...] — train.windows() 입력

데이터: deep_dynamics/reference_vehicles/gti/exp_*.npz (11개, 100 Hz, ux·uy·r·delta·fxf). 받는 곳은 docs/experiment_plan_golf.md 13·14절.
전처리: Golf(data.py)와 같게 — 10 Hz 4차 버터워스 양방향 필터 → 4개마다 1개 (0.04 s) → 정지(vx < 0.5 m/s)에서 자름, 최소 40 스텝.
종방향 입력: GTI 앞바퀴 힘 fxf 를 "같은 가속도"가 나오게 Golf 기준으로 환산한다 (모델의 물리 식은 Golf 질량·제동 배분을 씀).
  구동: lon = fxf · m_Golf / m_GTI
  제동: lon = fxf · (b_Golf · m_Golf) / (b_GTI · m_GTI)   (b = 제동 중 앞바퀴 몫: Golf 0.75, GTI 0.66)
노면: 위치 정보가 없어 bank = 0, 경사(road='grade')도 0.
분할 (주행 단위, 우리 저장소 실험과 같음): 검증 exp_11·12·15, 시험 exp_5·9, 나머지 6개 학습.
"""
import glob
import os

import numpy as np

from .configs import GOLF_THUNDERHILL, GTI_THUNDERHILL
from .data import MIN_LEN, V_STOP, lowpass

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GTI_DIR = os.path.join(ROOT, "reference_vehicles", "gti")
FS_GTI = 100.0
EVERY_GTI = 4                      # 100 Hz → 25 Hz
SPLIT = {"val": ["exp_11", "exp_12", "exp_15"], "test": ["exp_5", "exp_9"]}


def to_golf_force(fxf, golf=GOLF_THUNDERHILL, gti=GTI_THUNDERHILL):
    """GTI 앞바퀴 힘 → 같은 가속도를 내는 Golf 앞바퀴 힘 (모델 입력 단위)."""
    drive = fxf * golf.m / gti.m
    brake = fxf * (golf.brake_front_share * golf.m) / (gti.brake_front_share * gti.m)
    return np.where(fxf >= 0, drive, brake)


def names(split):
    allv = sorted(os.path.basename(f)[:-4] for f in glob.glob(os.path.join(GTI_DIR, "exp_*.npz")))
    assert allv, f"{GTI_DIR} 에 exp_*.npz 없음 (docs/experiment_plan_golf.md 13·14절)"
    if split == "all":
        return allv
    if split == "train":
        return [n for n in allv if n not in SPLIT["val"] + SPLIT["test"]]
    return [n for n in allv if n in SPLIT[split]]


def load_run(name, road="bank", min_len=MIN_LEN, v_stop=V_STOP):
    """한 실험 파일 → 구간 리스트 [(s (T,3), u (T,3|4))]. road: 'bank'·'none' → u=[lon, δ, 0], 'grade' → u=[lon, δ, 0, 0]."""
    z = np.load(os.path.join(GTI_DIR, name + ".npz"))
    a = np.stack([z["ux"], z["uy"], z["r"], z["fxf"], z["delta"]], 1).astype(np.float64)
    f = lowpass(a, fs=FS_GTI)[::EVERY_GTI]
    s = f[:, :3]
    u = np.c_[to_golf_force(f[:, 3]), f[:, 4], np.zeros(len(f))]
    if road == "grade":
        u = np.c_[u, np.zeros(len(f))]
    out = []
    edges = np.flatnonzero(np.diff(np.r_[0, (s[:, 0] >= v_stop).astype(int), 0]))
    for k, j in zip(edges[::2], edges[1::2]):
        if j - k >= min_len:
            out.append((s[k:j].copy(), u[k:j].copy()))
    return out


def split_segs(split, road="bank"):
    return [sg for n in names(split) for sg in load_run(n, road)]
