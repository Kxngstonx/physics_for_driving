"""모델–세계 일치도: 각 시뮬레이터에서 실제로 달린 궤적 위에서 R1·R6 의 1초(25스텝) 예측 오차.
같은 궤적(R1 이 몬 것 + R6 가 몬 것)을 두 모델이 모두 예측 → 공정 비교.
입력 줄은 mppi.episode 와 같게: [vx, vy, r, Fxf 명령, 조향 명령, bank, grade] (트랙 위치는 그 스텝 직전 위치).
사용: python model_vs_world.py REPO  → mvw.json
"""
import glob
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(sys.argv[1])
sys.path[:0] = [REPO, os.path.join(REPO, "experiments", "golf")]
import validate_sim as V  # noqa: E402
from deep_dynamics.golf import mppi  # noqa: E402
from deep_dynamics.golf.configs import GOLF_THUNDERHILL as CFG, cfg_of_model  # noqa: E402
from deep_dynamics.golf.ladder import load_model, rollout  # noqa: E402

HOR, STRIDE = 25, 5
tr, grade, info = V.recorded_track(V.lap_info("2023-10_5"))
I0, NTR = info["start_idx"], tr.n
bank, grd = tr.bank.cpu().numpy(), tr.grade.cpu().numpy()
SD = np.array(json.load(open(os.path.join(HERE, "realsim_out", "fit_realsim.json")))["sd"])  # 학습 데이터 상태 표준편차 (점수 정규화)

SIMS = {  # 이름: (폴더, 파일 접두)
    "CommonRoad": ("realsim_out", "log_profile_commonroad_"),
    "v3(데이터 맞춤)": ("realsim_out", "log_profile_realsim_"),
    "횡가속도 고정판": ("realsim_phys3", "log_profile_realsim_"),
}
models = {}
for R in ("R1", "R6"):
    pt = os.path.join(REPO, "experiments/golf/results/models", f"{R}_s0_roll25.pt")
    models[R] = load_model(pt, cfg_of_model(pt, CFG), mppi.DEV).eval()


def windows_from_log(d):
    vx, vy, r, thr, de, idx = (d[k] for k in ("vx", "vy", "r", "thr", "delta", "idx"))
    T = len(vx)
    v0 = float(d["vref_plan"][I0])  # 출발 속도 = 계획 속도의 출발 위치 값 (episode 의 v0)
    s = np.c_[np.r_[v0, vx[:-1]], np.r_[0.0, vy[:-1]], np.r_[0.0, r[:-1]]]  # s_t (스텝 전 상태)
    ix = np.r_[I0, idx[:-1]].astype(int)  # 스텝 t 의 입력 줄에 쓰인 트랙 위치
    rows = np.c_[s, thr, de, bank[ix], grd[ix]].astype(np.float32)
    nxt = np.c_[vx, vy, r].astype(np.float32)  # s_{t+1}
    dist = ((idx.astype(int) - I0) % NTR).astype(float)  # 랩 출발부터 거리 (m, 1 m 간격 트랙)
    X, Y, U, D, AY = [], [], [], [], []
    for t in range(8, T - HOR, STRIDE):
        X.append(rows[t - 7:t + 1]); Y.append(nxt[t:t + HOR]); U.append(rows[t + 1:t + HOR, 3:])
        D.append(dist[t]); AY.append(np.abs(nxt[t:t + HOR, 0] * nxt[t:t + HOR, 2]).max())
    return np.array(X), np.array(Y), np.array(U), np.array(D), np.array(AY)


def err(m, X, Y, U):
    with torch.no_grad():
        P = rollout(m, torch.tensor(X, device=mppi.DEV), torch.tensor(U, device=mppi.DEV)).cpu().numpy()
    return P[:, -1] - Y[:, -1]


def corner_bucket(D):
    return (D >= 170) & (D <= 318)


def summ(e):
    if len(e) == 0:
        return None
    rm = np.sqrt((e ** 2).mean(0))
    b = e.mean(0)  # 부호 있는 평균 (예측 − 실제): vx 가 + 면 모델이 1초 뒤 속도를 실제보다 높게 봄
    return dict(n=int(len(e)), vx=round(float(rm[0]), 3), vy=round(float(rm[1]), 3), r=round(float(rm[2]), 4), score=round(float((rm / SD).sum()), 3),
                bias_vx=round(float(b[0]), 3), bias_absr=None)


out = {}
for sname, (folder, pre) in SIMS.items():
    files = sorted(f for f in glob.glob(os.path.join(HERE, folder, pre + "R*_ay9.0*.npz")))
    if not files:
        continue
    acc = {R: dict(all=[], corner=[], hi_ay=[], brake=[], drive=[]) for R in models}
    for f in files:
        d = dict(np.load(f))
        X, Y, U, D, AY = windows_from_log(d)
        if len(X) == 0:
            continue
        for R, m in models.items():
            e = err(m, X, Y, U)
            acc[R]["all"].append(e)
            acc[R]["corner"].append(e[(D >= 170) & (D <= 318)])
            acc[R]["hi_ay"].append(e[AY >= 8.0])
            F = U[:, :, 0].mean(1)
            acc[R]["brake"].append(e[F < -1000]); acc[R]["drive"].append(e[F > 1000])
    out[sname] = dict(files=[os.path.basename(f) for f in files],
                      **{R: {k: summ(np.concatenate(v)) for k, v in a.items()} for R, a in acc.items()})
    print(sname, json.dumps({R: out[sname][R] for R in models}, ensure_ascii=False), flush=True)
json.dump(out, open(os.path.join(HERE, "mvw.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
