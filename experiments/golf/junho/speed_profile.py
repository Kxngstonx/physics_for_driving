"""곡률 기반 속도 계획 (준정상 상태 g-g 방법): 코너만 횡한계에 맞춰 깎고, 직선은 기록 속도 유지.
  1) v_lat(s) = sqrt(AY / |κ(s)|)         코너 최고 속도
  2) 뒤로 훑기: 다음 지점까지 감속 가능하게, 감속 한계 × √(1-(ay/AY)²) (마찰 타원)
  3) 앞으로 훑기: 코너 탈출 가속 한계 (속도별, 마찰 타원 적용)
  4) vref = min(기록 속도, 위 결과)  → 기록보다 빠르게 하지는 않음 (A 방식 유지)
가속·감속 한계: 기본은 학습 파일 포락선 realsim_out/accel_env.json (accel_env.py)
  - 가속: 속도별 상위 0.5% (엔진 한계라 속도에 따라 줄어듦)
  - 감속: 속도 구간 하위 0.5% 중 가장 큰 값 하나 (마찰 한계라 속도에 거의 무관, 고속 구간은 제동 표본이 적음)
"""
import json
import os

import numpy as np

ENV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "realsim_out", "accel_env.json")


def data_limits(path=ENV):
    rows = json.load(open(path))
    v = np.array([r["v"] for r in rows]); acc = np.array([r["acc"] for r in rows])
    axb = float(-min(r["brk"] for r in rows))
    return (lambda s: float(np.interp(s, v, acc, right=0.0))), axb


def curvature(x, y, win_m=15.0):
    ds = np.hypot(np.diff(x, append=x[:1]), np.diff(y, append=y[:1]))
    psi = np.unwrap(np.arctan2(np.gradient(y), np.gradient(x)))
    k = np.gradient(psi) / np.maximum(ds, 1e-3)
    n = max(1, int(round(win_m / np.median(ds))))
    ker = np.ones(n) / n
    kp = np.concatenate([k[-n:], k, k[:n]])  # 닫힌 트랙
    return np.convolve(kp, ker, "same")[n:-n], ds


def profile(x, y, vrec, AY, AXB=None, AXA=None, win_m=15.0):
    """AXB: 감속 한계 (상수), AXA: 가속 한계 (상수 또는 속도→가속 함수). None 이면 데이터 포락선."""
    if AXB is None or AXA is None:
        fa, fb = data_limits()
        AXA = fa if AXA is None else AXA
        AXB = fb if AXB is None else AXB
    acc = AXA if callable(AXA) else (lambda s, c=AXA: c)
    k, ds = curvature(x, y, win_m)
    # 물리 한계만으로 계획한 뒤 마지막에 기록 속도로 덮음 (기록의 정지 출발 저속 구간이 앞뒤로 번지지 않게)
    v = np.minimum(45.0, np.sqrt(AY / np.maximum(np.abs(k), 1e-6)))
    n = len(v)
    for _ in range(2):  # 닫힌 트랙이라 두 바퀴 훑기
        for i in range(n - 1, -1, -1):
            j = (i + 1) % n
            use = min(1.0, v[j] ** 2 * abs(k[j]) / AY)
            v[i] = min(v[i], np.sqrt(v[j] ** 2 + 2 * AXB * np.sqrt(1 - use ** 2) * ds[i]))
        for i in range(n):
            j = (i - 1) % n
            use = min(1.0, v[j] ** 2 * abs(k[j]) / AY)
            v[i] = min(v[i], np.sqrt(v[j] ** 2 + 2 * acc(v[j]) * np.sqrt(1 - use ** 2) * ds[j]))
    return np.minimum(vrec, v), k
