"""B 방식 본 실험 (docs/experiment_plan_golf.md 9-4, 10절, 11/19 사용자 결정).

  python experiments/golf/run_b.py main 25        # 계획 길이 N 스텝(25·50·100 = 1·2·4초) × R1–R6 × 시드 0·1·2 (1초 손실 모델)
  python experiments/golf/run_b.py ablation 50    # 같은 N 에 맞춘 손실로 학습한 모델(roll N, 시드 0) × R1–R6
  python experiments/golf/run_b.py report         # → results/b_main.md, b_main.png

B 방식: 경로·목표 속도 없이 트랙 경계(기록선 ± 5.45 m, 위성 지도 측정 평균 폭 10.9 m) 안에서 최대한 전진.
시뮬레이터는 calibrated_world (데이터로 맞춘 계수, 지연), 조향 지연 0 [가정, 고려 필요]. 검증 랩(2023-10_8) 출발 위치에서 정지 출발.
MPPI 설정은 tune_mppi.py 가 계획 길이별로 고른 것 (chosen["B_N"]). 이미 끝난 조합은 건너뛴다 → 중단돼도 이어서 함.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from deep_dynamics.golf import mppi, sim  # noqa: E402

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
OUT = os.path.join(RES, "b_main.json")
RUNGS = ("R1", "R2", "R3", "R4", "R5", "R6")
NS = (25, 50, 100)


def load():
    return json.load(open(OUT)) if os.path.exists(OUT) else {"runs": []}


def save(d):
    tmp = OUT + ".tmp"
    json.dump(d, open(tmp, "w"), indent=1)
    os.replace(tmp, OUT)


def drive(kind, N, tag_of, seeds):
    import torch
    import tune_mppi as T
    from deep_dynamics.golf.configs import GOLF_THUNDERHILL as CFG
    from deep_dynamics.golf.ladder import load_model
    c = json.load(open(os.path.join(RES, "mppi_tune.json")))["chosen"][f"B_{N}"]
    tr, grade, info = T.b_track()
    for seed in seeds:
        for R in RUNGS:
            tag = tag_of(R, seed)
            d = load()
            if any(r["kind"] == kind and r["N"] == N and r["tag"] == tag for r in d["runs"]):
                continue
            model = load_model(os.path.join(RES, "models", tag + ".pt"), CFG, mppi.DEV)
            w = sim.calibrated_world(delays=dict(steer=T.steer_delay()), grade_by_idx=grade)
            res, log = mppi.episode(model, w, tr, N=N, seed=seed, i0=info["start_idx"], v_armed=10.0, mode="B", lam=c["lam"],
                                    sigma=tuple(c["noise"] * s for s in mppi.SIGMA_FORCE))
            fail_m = None if res["finished"] else float(log["idx"][-1]) if log["idx"] else None
            row = dict(kind=kind, N=N, rung=R, seed=seed, tag=tag, lam=c["lam"], noise=c["noise"], fail_track_m=fail_m,
                       ms_per_step=1000.0 * res["wall_s"] / max(len(log["vx"]), 1),
                       **{k: res[k] for k in ("status", "finished", "progress_m", "progress_frac", "lap_time", "ey_mean", "ey_max", "wall_s")})
            print(json.dumps(row, ensure_ascii=False), flush=True)
            d = load()
            d["runs"].append(row)
            save(d)
            del model
            torch.cuda.empty_cache()


def report():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d = load()
    rec = 107.28
    L = ["# B 방식 결과 (경계 안에서 최대한 전진)", "", f"기록 랩타임 {rec} s (2023-10_8). 시드 평균, 괄호는 완주 수.", ""]
    for kind, title in (("main", "본 실험: 1초 손실 모델, 시드 0·1·2"), ("ablation", "길이별 재학습 ablation: 계획 길이에 맞춘 손실, 시드 0")):
        R = [r for r in d["runs"] if r["kind"] == kind]
        if not R:
            continue
        Ns = sorted({r["N"] for r in R})
        L += [f"## {title}", "", "| 모델 | " + " | ".join(f"{n * mppi.DT:g}초: 진행률 (완주) / 랩타임" for n in Ns) + " |",
              "|---|" + "---|" * len(Ns)]
        for rung in RUNGS:
            cells = []
            for n in Ns:
                q = [r for r in R if r["rung"] == rung and r["N"] == n]
                if not q:
                    cells.append("—")
                    continue
                laps = [r["lap_time"] for r in q if r["finished"]]
                cells.append(f"{np.mean([r['progress_frac'] for r in q]):.2f} ({len(laps)}/{len(q)}) / "
                             + (f"{np.mean(laps):.1f} s" if laps else "—"))
            L.append(f"| {rung} | " + " | ".join(cells) + " |")
        L += ["", "| 계획 길이 | 1스텝 계산 (ms, 중앙값) | 실패 위치 (트랙 m, 중앙값) |", "|---|---|---|"]
        for n in Ns:
            q = [r for r in R if r["N"] == n]
            fm = [r["fail_track_m"] for r in q if r["fail_track_m"] is not None]
            L.append(f"| {n * mppi.DT:g}초 | {np.median([r['ms_per_step'] for r in q]):.0f} | {np.median(fm) if fm else float('nan'):.0f} |")
        L.append("")
    R = [r for r in d["runs"] if r["kind"] == "main"]
    if R:
        Ns = sorted({r["N"] for r in R})
        fig, ax = plt.subplots(figsize=(6, 3.4), layout="constrained")
        colors = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9"]
        markers = ["o", "s", "^", "D", "v", "P"]
        for i, rung in enumerate(RUNGS):
            y = [np.mean([r["progress_frac"] for r in R if r["rung"] == rung and r["N"] == n]) for n in Ns]
            ax.plot([n * mppi.DT for n in Ns], y, marker=markers[i], color=colors[i], label=rung)
        ax.set(xlabel="planning horizon (s)", ylabel="lap progress (mean of 3 seeds)", ylim=(0, 1.05), xticks=[n * mppi.DT for n in Ns],
               title="B mode: progress vs planning horizon")
        ax.legend(ncol=3, fontsize=8)
        fig.savefig(os.path.join(RES, "b_main.png"), dpi=150)
        L += ["![](b_main.png)", ""]
    open(os.path.join(RES, "b_main.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "main":
        drive("main", int(sys.argv[2]), lambda R, s: f"{R}_s{s}_roll25", (0, 1, 2))
    elif cmd == "ablation":
        n = int(sys.argv[2])
        drive("ablation", n, lambda R, s: f"{R}_s{s}_roll{n}", (0,))
    else:
        report()
