"""타이어 좌우 처리 비교 (11/21 사용자 결정): CommonRoad 원본 식 vs 표준 좌우 거울상 (mb_golf 모듈 설명).

  python experiments/golf/compare_tire_mirror.py run orig     # 원본 식 (tire_mirror=False)
  python experiments/golf/compare_tire_mirror.py run mirror   # 표준 거울상 (tire_mirror=True)
  python experiments/golf/compare_tire_mirror.py report       # → results/sim/tire_mirror_compare.md

바꾸는 변수: 타이어 좌우 처리 하나. 고정: specs.json, 보정 절차(calibrate_sim.calibrate: 롤 기울기 0.30·최대 옆 가속도 = specs ay_target),
옆 강성·rby_f 배율 1 (fit_sim 전, 중립), 시험 파일 3개, 재생 조건(지연·힘 상한 없음, validate_sim.replay 와 같음).
각 방식은 자기 보정값을 results/sim/tire_mirror_<방식>.json 에 따로 저장한다 (공식 calibration.json 은 건드리지 않음).
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import calibrate_sim as CS  # noqa: E402
import fit_sim as FS  # noqa: E402
import validate_sim as V  # noqa: E402
from deep_dynamics.golf import data as D  # noqa: E402
from deep_dynamics.golf import sim  # noqa: E402

OUT = V.OUT
MODES = dict(orig=False, mirror=True)


def replay(W, params):
    out = np.full((len(W), len(W[0]["F"]), 5), np.nan)
    w = sim.World(cap=False, delays=None, **params)
    for i, win in enumerate(W):
        vx, vy, r = win["s0"]
        w.reset(win["X0"], win["Y0"], win["psi0"], vx, vy, r, delta=win["d0"])
        for k in range(len(win["F"])):
            w.set_grade(win["g"][k])
            s = w.step(float(win["F"][k]), float(win["d"][k]))
            out[i, k] = (*s, w.x[0], w.x[1])
            if not np.isfinite(s[0]):
                break
    return out


def run(mode):
    tm = MODES[mode]
    cal = CS.calibrate(tm)
    params = dict(mu_scale=cal["mu_scale"], k_roll=cal["k_roll"], tire_mirror=tm)
    res = dict(mode=mode, calibration=cal)
    w = sim.World(cap=False, **params)
    w.reset(0, 0, 0, 20.0)
    for _ in range(50):
        s = w.step(sim.road_load(20.0), 0.0)
    res["straight_2s"] = dict(vy=float(s[1]), r=float(s[2]))
    sp = D.get_split()
    sd = np.concatenate([sg["s"] for n in sp["train"] for sg in D.load_run(n)]).std(0)
    for key, K in V.HORIZONS.items():
        W = V.windows(sp["test"], K)
        P = replay(W, params)
        m = V.metrics(W, P)
        res[f"replay_{key}"] = m
        if key == "1s":
            cat = FS.categories(W)
            res["by_category_1s"] = {c: FS.score(W, P, sd, cat == c) for c in ("straight", "braking", "corner", "drive_corner")}
            res["by_category_1s"]["all"] = FS.score(W, P, sd)
        print(mode, key, {k: round(m[k]["rmse_end"], 3) for k in ("vx", "vy", "r")}, "발산", m["n_diverged"], flush=True)
    res["corner500"] = _corner500(params)
    json.dump(res, open(os.path.join(OUT, f"tire_mirror_{mode}.json"), "w"), indent=1, default=float)


def _corner500(params):
    """fit_sim.corner500 과 같은 측정 (검증 랩 500 m 코너, 실제 조작 재생 → 평균 옆 가속도). 여기서는 한 프로세스로."""
    from deep_dynamics.golf.track import project
    tr, _, _ = V.recorded_track()
    W = V.windows(["2023-10_8"], 25, stride=5)
    idx = project(np.c_[tr.np["x"], tr.np["y"]], np.array([[w["X0"], w["Y0"]] for w in W]))[0]
    sel = [w for w, i in zip(W, idx) if 420 <= i <= 560]
    P = replay(sel, params)
    S = np.stack([w["s"] for w in sel])
    return dict(n=len(sel), ay_data=float(np.abs(S[:, :, 0] * S[:, :, 2]).mean()), ay_sim=float(np.nanmean(np.abs(P[:, :, 0] * P[:, :, 2]))),
                r_err_1s=float(np.nanmean(P[:, -1, 2] - S[:, -1, 2])))


def report():
    R = {m: json.load(open(os.path.join(OUT, f"tire_mirror_{m}.json"))) for m in MODES}
    L = ["# 타이어 좌우 처리 비교 (11/21)", "", "바꾼 변수: 타이어 좌우 처리. 고정: specs.json, 보정 절차, 옆 강성 배율 1, 시험 파일 3개.", "",
         "| 항목 | 원본 식 (orig) | 표준 거울상 (mirror) |", "|---|---|---|"]
    row = lambda name, f: L.append(f"| {name} | {f(R['orig'])} | {f(R['mirror'])} |")
    row("마찰 배율 / 롤 강성", lambda r: f"{r['calibration']['mu_scale']} / {r['calibration']['k_roll']:.0f}")
    row("롤 기울기 (목표 0.30)", lambda r: f"{r['calibration']['roll_gradient']:.3f}")
    row(f"최대 옆 가속도 (목표 {R['orig']['calibration']['ay_target']:.2f})", lambda r: f"{r['calibration']['max_ay']:.2f}")
    row("직진 2초 vy / r", lambda r: f"{r['straight_2s']['vy']:+.4f} / {r['straight_2s']['r']:+.4f}")
    for key in V.HORIZONS:
        row(f"재생 {key} 끝 RMSE vx / vy / r", lambda r, key=key: " / ".join(f"{r[f'replay_{key}'][k]['rmse_end']:.3f}" for k in ("vx", "vy", "r")))
        row(f"재생 {key} 최대 vx / vy / r", lambda r, key=key: " / ".join(f"{r[f'replay_{key}'][k]['max']:.2f}" for k in ("vx", "vy", "r")))
        row(f"재생 {key} 발산", lambda r, key=key: f"{r[f'replay_{key}']['n_diverged']} / {r[f'replay_{key}']['n_windows']}")
    for c in ("all", "straight", "braking", "corner", "drive_corner"):
        row(f"1초 점수·vy RMSE ({c})", lambda r, c=c: f"{r['by_category_1s'][c].get('score', float('nan')):.3f} · {r['by_category_1s'][c].get('vy', float('nan')):.3f}")
    row("500 m 코너 옆 가속도 (실제 / 시뮬)", lambda r: f"{r['corner500']['ay_data']:.2f} / {r['corner500']['ay_sim']:.2f}")
    open(os.path.join(OUT, "tire_mirror_compare.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    if sys.argv[1] == "run":
        run(sys.argv[2])
    else:
        report()
