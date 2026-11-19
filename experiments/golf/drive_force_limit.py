"""구동력 상한표 (docs/experiment_plan_golf.md 12절): 속도 2 m/s 구간별 앞바퀴 구동력의 최댓값·상위 1%.
구동력 = 구동계 효율 × 엔진 토크 × 실측 감속비 ÷ 바퀴 반지름 (golf/data.py 와 같은 계산). 20개 파일 전체, 200 Hz.
python experiments/golf/drive_force_limit.py → experiments/golf/results/drive_force_limit.csv
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from deep_dynamics.golf import data as D  # noqa: E402
from deep_dynamics.golf.configs import GOLF_THUNDERHILL as CFG  # noqa: E402

v, F = [], []
for n in D.available():
    df = D.read_raw(n)
    v.append(df["Vx_mps"].to_numpy())
    F.append(D.DRIVE_EFF * df["F_drive"].to_numpy())
v, F = np.concatenate(v), np.concatenate(F)
rows = ["v_lo,v_hi,n,F_max_N,F_p99_N,a_p99_at_m_mps2"]
m = CFG.m
for lo in range(0, 36, 2):
    k = (v >= lo) & (v < lo + 2)
    if k.sum() < 200:
        continue
    p99 = np.percentile(F[k], 99)
    rows.append(f"{lo},{lo + 2},{int(k.sum())},{F[k].max():.0f},{p99:.0f},{p99 / m:.2f}")
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "drive_force_limit.csv")
open(out, "w").write("\n".join(rows) + "\n")
print("\n".join(rows))
