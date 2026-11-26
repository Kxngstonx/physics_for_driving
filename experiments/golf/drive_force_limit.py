"""구동력 상한표 (docs/experiment_plan_golf.md 12절): 속도 2 m/s 구간별 앞바퀴 구동력의 최댓값·상위 1%.
구동력 = 구동계 효율 × 엔진 토크 × 실측 감속비 ÷ 바퀴 반지름 (golf/data.py 와 같은 계산). 20개 파일 전체, 200 Hz.
[변경 11/21] 표는 estimate_specs 가 table() 을 불러 specs.json 의 drive_force_cap 으로 넣는다 (이전 results/drive_force_limit.csv).
python experiments/golf/drive_force_limit.py  → 확인용 출력 (specs.json 값으로 계산)
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from deep_dynamics.golf import data as D  # noqa: E402


def table(v, F, m):
    """v (m/s), F (N, 효율 포함 구동력) → dict 열 목록. 표본 200개 미만 구간은 뺀다. m 은 참고 가속도 열에만."""
    out = dict(v_lo=[], v_hi=[], n=[], F_max_N=[], F_p99_N=[], a_p99_at_m_mps2=[])
    for lo in range(0, 36, 2):
        k = (v >= lo) & (v < lo + 2)
        if k.sum() < 200:
            continue
        p99 = float(np.percentile(F[k], 99))
        for key, val in zip(out, (lo, lo + 2, int(k.sum()), float(F[k].max()), p99, p99 / m)):
            out[key].append(val)
    return out


if __name__ == "__main__":
    from deep_dynamics.golf.golf_specs import value
    v = np.concatenate([D.read_raw(n)["Vx_mps"].to_numpy() for n in D.available()])
    F = np.concatenate([value("drive_eff") * D.read_raw(n)["F_drive"].to_numpy() for n in D.available()])
    t = table(v, F, value("mass"))
    for row in zip(*t.values()):
        print(row)
