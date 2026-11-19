"""변환 단계의 액추에이터 지연 맞추기 (docs/experiment_plan_golf.md 4-4, 8-2절).

자동 운전 구간(auto_throttle_bool)에서 명령 → 실제 값을 "순수 지연 d + 1차 지연 tau" 로 맞춘다.
  브레이크: brakeCmd_{fl,fr}_bar → brake_{fl,fr}_bar   (앞 두 바퀴 평균)
  엔진   : throttleCmd_percent   → engineTorque_Nm     (y = k·f + c 로 배율·치우침도 함께 맞춤. 토크는 rpm 에도 달려 있어 근사)
격자: d 0~300 ms (5 ms), tau 0~300 ms (10 ms). 200 Hz 원본, 20개 파일 전체.
쓰는 값(use): 지연 없음보다 RMS 가 1% 이상 줄 때만 맞춘 값, 아니면 (0, 0) — 데이터로 정해지지 않은 지연을 넣지 않음.
python experiments/golf/fit_delays.py → experiments/golf/results/sim/delays.json
"""
import json
import os
import sys

import numpy as np
from scipy.signal import lfilter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from deep_dynamics.golf import data as D  # noqa: E402

FS = D.FS
DS = np.arange(0, 0.301, 0.005)
TAUS = np.arange(0, 0.301, 0.01)


def runs(cmd_cols, act_cols):
    out = []
    for n in D.available():
        df = D.read_raw(n)
        auto = df["auto_throttle_bool"].to_numpy() > 0.5
        c = df[cmd_cols].to_numpy().mean(1)
        a = df[act_cols].to_numpy().mean(1)
        edges = np.flatnonzero(np.diff(np.r_[0, auto.astype(int), 0]))
        for k, j in zip(edges[::2], edges[1::2]):
            if j - k > 2 * FS:
                out.append((c[k:j], a[k:j]))
    return out


def fit(pairs, active):
    """pairs: [(명령, 실제)]. active(명령, 실제) → 평가에 쓸 표본 표시 (움직이는 구간만)."""
    best, grid = None, []
    for d in DS:
        n = int(round(d * FS))
        for tau in TAUS:
            al = 1.0 if tau == 0 else 1.0 - np.exp(-1.0 / (FS * tau))
            F, Y = [], []
            for c, a in pairs:
                cd = np.r_[np.full(n, c[0]), c[:len(c) - n]]
                f = lfilter([al], [1, al - 1], cd, zi=[c[0] * (1 - al)])[0]
                m = active(c, a)
                F.append(f[m])
                Y.append(a[m])
            F, Y = np.concatenate(F), np.concatenate(Y)
            A = np.c_[F, np.ones_like(F)]
            coef = np.linalg.lstsq(A, Y, rcond=None)[0]
            rms = float(np.sqrt(np.mean((A @ coef - Y) ** 2)))
            grid.append((d, tau, rms))
            if best is None or rms < best[2]:
                best = (float(d), float(tau), rms, coef.tolist())
    return best, grid


if __name__ == "__main__":
    res = {}
    br = runs(["brakeCmd_fl_bar", "brakeCmd_fr_bar"], ["brake_fl_bar", "brake_fr_bar"])
    b, gb = fit(br, lambda c, a: (c > 1.0) | (a > 1.0))
    en = runs(["throttleCmd_percent"], ["engineTorque_Nm"])
    e, ge = fit(en, lambda c, a: (c > 1.0) | (a > 5.0))
    for name, (bst, grid), pairs in (("brake", (b, gb), br), ("engine", (e, ge), en)):
        g = np.array(grid)
        zero = g[(g[:, 0] == 0) & (g[:, 1] == 0)][0, 2]
        use = [bst[0], bst[1]] if bst[2] <= 0.99 * zero else [0.0, 0.0]
        res[name] = dict(use=use, fit=[bst[0], bst[1]], rms=bst[2], rms_no_delay=float(zero), gain_offset=bst[3],
                         segments=len(pairs), samples=int(sum(len(c) for c, _ in pairs)),
                         near_best=[(float(d), float(t)) for d, t, r in grid if r <= bst[2] * 1.02][:20])
        print(name, f"d {bst[0] * 1e3:.0f} ms tau {bst[1] * 1e3:.0f} ms rms {bst[2]:.3f} (지연 없음 {zero:.3f}) → 사용 {use}")
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "sim")
    os.makedirs(out, exist_ok=True)
    json.dump(res, open(os.path.join(out, "delays.json"), "w"), indent=1)
