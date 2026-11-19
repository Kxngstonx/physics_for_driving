"""시뮬레이터 보정 두 가지 (docs/experiment_plan_golf.md 4-4절, [결정 11/19]).

  python experiments/golf/calibrate_sim.py   → experiments/golf/results/sim/calibration.json

1. 롤 강성 k_roll: 정상 원선회(25 m/s, 횡가속도 약 4~6 m/s²)의 롤 기울기가 ROLL_TARGET 이 되도록 안티롤바를 맞춤
   (스프링은 Escort 그대로, 앞뒤 몫 0.61 유지). 롤 기울기 = 차체 기울기(도) ÷ 횡가속도(m/s²)
2. 마찰 배율 mu_scale: 시뮬레이터가 정상 선회로 낼 수 있는 최대 횡가속도가 데이터 상위 1% 값이 되도록 맞춤
   (20 m/s 등속, 조향을 천천히 늘리며 vx·r 의 최댓값을 잼)
두 값은 서로 영향을 주므로 번갈아 3번 맞춘다.
"""
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from deep_dynamics.golf import data as D  # noqa: E402
from deep_dynamics.golf import sim  # noqa: E402

ROLL_TARGET = 0.30  # 도 / (m/s²) [가정]: 현가 센서 추정 0.23(단위 cm 가정)과 일반 승용차 0.3~0.6 이 겹치는 값
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "sim", "calibration.json")


def ay_p99():
    s = np.concatenate([sg["s"] for n in D.available() for sg in D.load_run(n)])
    return float(np.percentile(np.abs(s[:, 0] * s[:, 2]), 99))


def hold_speed(w, v):
    return sim.road_load(w.state[0]) + w.p.m * 0.5 * (v - w.state[0])


def max_ay(mu, k_roll, v=20.0, T=15.0):
    w = sim.World(mu_scale=mu, k_roll=k_roll, cap=False)
    w.reset(0, 0, 0, v)
    best, n = 0.0, int(T / sim.DT)
    for k in range(n):
        s = w.step(hold_speed(w, v), 0.25 * k / n)
        if not np.isfinite(s[0]) or abs(math.atan2(s[1], s[0])) > 0.15:  # 미끄러져 돌기 시작하면 멈춤
            break
        best = max(best, abs(s[0] * s[2]))
    return best


def bisect(f, lo, hi, target, n=12, increasing=True):
    for _ in range(n):
        mid = 0.5 * (lo + hi)
        if (f(mid) < target) == increasing:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


if __name__ == "__main__":
    target_ay = ay_p99()
    k_escort = sim.roll_stiffness(sim.golf_params(1.0, None))  # Golf 윤거에서의 Escort 롤 강성
    g0, _ = sim.roll_gradient(1.0, None)
    print(f"목표 최대 횡가속도 {target_ay:.2f} m/s² (데이터 상위 1%), 목표 롤 기울기 {ROLL_TARGET} 도/(m/s²)")
    print(f"보정 전 (Escort 서스펜션, Golf 윤거): 롤 강성 {k_escort:.0f} N m/rad, 롤 기울기 {g0:.3f}")
    mu, k = 1.0, k_escort
    for it in range(3):
        k = bisect(lambda kk: -sim.roll_gradient(mu, kk)[0], k_escort, 10 * k_escort, -ROLL_TARGET)
        mu = bisect(lambda m: max_ay(m, k), 0.8, 1.6, target_ay)
        g, ay = sim.roll_gradient(mu, k)
        ay_max = max_ay(mu, k)
        print(f"반복 {it}: k_roll {k:.0f} (Escort 의 {k / k_escort:.2f}배) → 롤 기울기 {g:.3f} (횡가속도 {ay:.1f}), "
              f"mu_scale {mu:.3f} → 최대 횡가속도 {ay_max:.2f}", flush=True)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(dict(mu_scale=round(mu, 3), k_roll=round(k, 0), roll_target_deg_per_mps2=ROLL_TARGET, ay_target=target_ay,
                   roll_gradient=g, max_ay=ay_max, k_roll_escort=k_escort, roll_gradient_escort=g0,
                   track=[sim.TRACK_F, sim.TRACK_R]), open(OUT, "w"), indent=1)
