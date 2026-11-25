"""P2 자전거 시뮬레이터 계수를 D7 학습 파일로 맞춘다 (시스템 식별, 11/25). 우리 저장소 physics/golf_sysid.py 를 팀 기준으로 옮김.

  python experiments/golf/fit_p2.py                  # → experiments/golf/results/sim/p2_fit.json, 화면에 시험 결과
  python experiments/golf/fit_p2.py --eval-only      # 저장된 값으로 시험만

팀 기준 (11/19 결정)
  - 분할: deep_dynamics/data/golf/split.json (학습으로 맞추고, 검증으로 가장 좋은 상태를 고르고, 시험으로 보고)
  - 입력: 기록된 조향각, 전체 종방향 힘 Fx = 0.699·구동력 − (네 바퀴 브레이크 힘) (data.read_raw 의 F_drive, F_brake_f/r)
  - 경사: 트랙 지도 (data.grade_at, track_osm.py). 좌우 기울기 0
  - 고정: 질량 m·축거 L·앞 제동 몫 (configs.GOLF_THUNDERHILL), 앞바퀴굴림
맞추는 값: 무게중심 a, Iz, 코너링 강성(앞뒤, 하중으로 정규화), Pacejka 형상 C(앞뒤), μ, 무게중심 높이 h, 저항 c0·c2, vy 센서 위치 ds
손실: 1초(100 스텝, 0.01 s RK4) 롤아웃 [r, vy, vx] 오차를 상태별 표준편차로 나눈 제곱 평균. 한 스텝 손실은 물리 계수가 비현실적으로 나온다.
주의: a 와 ds 는 서로 보완해 데이터로 따로 정해지지 않는다 (앞 차축 ~ 센서 거리 a − ds 만 정해짐). a 를 무게 배분으로 읽지 말 것.
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
from deep_dynamics.golf import data as D  # noqa: E402
from deep_dynamics.golf import p2  # noqa: E402

DT = 0.01
# 이름: (초기값, 하한, 상한). 초기값은 우리 저장소 1초 손실 재피팅(11/19) 값
SPEC = {
    "a": (1.29, 0.85, 1.45), "Iz": (3400.0, 1500.0, 4500.0), "cf": (12.5, 5.0, 40.0), "cr": (17.3, 5.0, 50.0),
    "Cf": (1.25, 1.05, 1.95), "Cr": (1.8, 1.05, 1.95), "mu": (1.09, 0.7, 1.5), "h": (0.4, 0.25, 0.75),
    "c0": (200.0, 0.0, 800.0), "c2": (0.5, 0.0, 1.5), "ds": (0.4, -1.5, 1.5),
}


def load_runs(names):
    """주행 파일 → 연속 구간 리스트, 각 (N, 6) [r, vy_측정, vx, δ, Fx 전체, 경사] (100 Hz, vx > 5 m/s, 3 s 이상)."""
    segs = []
    for n in names:
        df = D.read_raw(n)
        fx = D.DRIVE_EFF * df["F_drive"].to_numpy() - (df["F_brake_f"].to_numpy() + df["F_brake_r"].to_numpy())
        a = np.c_[df["yawRate_radps"], df["Vy_mps"], df["Vx_mps"], df["roadWheelAngle_rad"], fx, df["posE_m"], df["posN_m"]].astype(np.float64)
        a = a[:len(a) // 2 * 2].reshape(-1, 2, a.shape[1]).mean(1)          # 200 → 100 Hz
        F = np.c_[a[:, :5], D.grade_at(a[:, 5], a[:, 6])]
        ok = a[:, 2] > 5.0
        e = np.flatnonzero(np.diff(np.r_[0, ok.astype(int), 0]))
        segs += [F[i:j] for i, j in zip(e[::2], e[1::2]) if j - i > 300]
    return segs


def windows(segs, H, stride):
    return torch.as_tensor(np.stack([F[s:s + H + 1] for F in segs for s in range(0, len(F) - H - 1, stride)]), dtype=torch.float64)


class Params(torch.nn.Module):
    def __init__(self, init=None):
        super().__init__()
        self.raw = torch.nn.ParameterDict()
        for k, (v0, lo, hi) in SPEC.items():
            v = (init or {}).get(k, v0)
            u = min(max((v - lo) / (hi - lo), 1e-4), 1 - 1e-4)
            self.raw[k] = torch.nn.Parameter(torch.tensor(math.log(u / (1 - u)), dtype=torch.float64))

    def values(self):
        return {k: SPEC[k][1] + (SPEC[k][2] - SPEC[k][1]) * torch.sigmoid(self.raw[k]) for k in SPEC}

    def params(self):
        return p2.P2Params(**self.values())


def simulate(P, W, H):
    """W (B, H+1, 6). 처음 상태에서 기록된 조향·Fx·경사로 H 스텝 → (B, H, 3) [r, vy_측정, vx]."""
    p = P.params()
    r0, vy0, vx0 = W[:, 0, 0], W[:, 0, 1], W[:, 0, 2]
    z = torch.zeros_like(r0)
    x = torch.stack([z, z, z, vx0, vy0 - p.ds * r0, r0], -1)
    out = []
    for k in range(H):
        x = p2.rk4(x, W[:, k, 3], W[:, k, 4], p, DT, W[:, k, 5])
        out.append(torch.stack([x[:, 5], x[:, 4] + p.ds * x[:, 5], x[:, 3]], -1))
    return torch.stack(out, 1)


def errors(P, segs, horizons=(50, 100, 200), stride=50):
    """기록 입력으로 굴린 오차 (원단위 RMSE: r rad/s, vy·vx m/s). P=None 이면 현재값 유지."""
    res = {}
    with torch.no_grad():
        for H in horizons:
            W = windows(segs, H, stride)
            pr = simulate(P, W, H) if P is not None else W[:, :1, :3].expand(-1, H, -1)
            e = pr[:, -1] - W[:, H, :3]
            ok = torch.isfinite(e).all(1)
            hi = ((W[:, :, 2] * W[:, :, 0]).abs().amax(1) / 9.81 > 0.8) & ok
            rm = lambda m: dict(zip(["r", "vy", "vx"], torch.sqrt((e[m] ** 2).mean(0)).tolist()))
            res[f"{H * DT:.1f}s"] = dict(**rm(ok), diverged=int((~ok).sum()), hi=dict(**rm(hi), n=int(hi.sum())))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=400)
    ap.add_argument("--H", type=int, default=100, help="학습 롤아웃 길이 [스텝, 0.01 s]. 100 = 1초")
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=0.03)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--out", default=p2.FIT)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    torch.manual_seed(0)
    sp = D.get_split()
    tr, va, te = (load_runs(sp[k]) for k in ("train", "val", "test"))
    print(f"구간: 학습 {len(tr)}개 {sum(map(len, tr)) / 6000:.1f}분, 검증 {len(va)}개, 시험 {len(te)}개 {sum(map(len, te)) / 6000:.1f}분", flush=True)
    if a.eval_only:
        P = Params({k: v for k, v in json.load(open(a.out))["params"].items() if k in SPEC})
    else:
        Wtr, Wva = windows(tr, a.H, 10), windows(va, a.H, 25)
        std = Wtr[:, 1:, :3].reshape(-1, 3).std(0)
        P = Params()
        opt = torch.optim.Adam(P.parameters(), lr=a.lr)
        best, best_state, t0 = np.inf, None, time.time()
        for it in range(a.iters):
            W = Wtr[torch.randint(len(Wtr), (a.batch,))]
            loss = (((simulate(P, W, a.H) - W[:, 1:, :3]) / std) ** 2).mean((1, 2))
            loss = loss[torch.isfinite(loss)].mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            if it % 20 == 0 or it == a.iters - 1:
                with torch.no_grad():
                    lv = (((simulate(P, Wva, a.H) - Wva[:, 1:, :3]) / std) ** 2)
                    lv = float(lv[torch.isfinite(lv).all(-1).all(-1)].mean())
                if lv < best:
                    best, best_state = lv, {k: t.clone() for k, t in P.state_dict().items()}
                print(f"it {it:4d} {time.time() - t0:5.0f}s train {float(loss):.4f} val {lv:.4f}  "
                      f"{ {k: round(float(t), 3) for k, t in P.values().items()} }", flush=True)
        P.load_state_dict(best_state)
        test = errors(P, te)
        p2.save_fit(p2.P2Params(**{k: float(t) for k, t in P.values().items()}), a.out, best_val=best, H=a.H, iters=a.iters, split=dict(train=sp["train"], val=sp["val"]),
                    fixed=dict(m=p2.P2Params().m, L=p2.P2Params().L, brake_front=p2.P2Params().brake_front),
                    test=test, note="experiments/golf/fit_p2.py (팀 기준 힘 정의·분할·경사 지도)")
        print("saved", os.path.relpath(a.out, REPO))
    print("\n시험 주행 롤아웃 RMSE (r rad/s, vy·vx m/s)")
    for name, model in (("P2", P), ("현재값 유지", None)):
        res = errors(model, te)
        print(f"{name:8s} " + "  ".join(f"{h}: r {v['r']:.3f} vy {v['vy']:.3f} vx {v['vx']:.3f}" for h, v in res.items()))
        print(f"{'└0.8g+':8s} " + "  ".join(f"{h}: r {v['hi']['r']:.3f} vy {v['hi']['vy']:.3f} (창 {v['hi']['n']})" for h, v in res.items()))
    print("값:", {k: round(float(t), 3) for k, t in P.values().items()})


if __name__ == "__main__":
    main()
