"""B 방식 보상 ablation (11/23 사용자 결정, docs/experiment_plan_golf.md [결정 11/21] 10).

  python experiments/golf/run_b_reward.py run 25        # 보상 후보 4개 × R1–R6 (시드 0), 계획 길이 N 스텝
  python experiments/golf/run_b_reward.py report        # → results/b_reward.md

바꾸는 것: 기본 B 보상에 항목 하나 (mppi.B_REWARDS)
  vlim ① 중심선 곡률 한계 속도 초과 벌점   term ② 계획 끝 제동 가능 속도 초과 벌점
  slip ③ 옆 미끄럼각이 데이터 99% 초과 벌점   wall ④ 경계 연속 벌점
고정: 모델 R1–R6 시드 0 (1초 손실), 시뮬레이터 calibrated_world (기본 st), 검증 랩 경로·경사·정지 출발, K 1024,
  조향 지연 (validation.json), 기본 B 항목, λ·잡음 = tune_mppi 가 계획 길이별로 고른 기본 B 값 (후보마다 다시 튜닝하지 않음 → 후보가 조금 불리할 수 있음).
기준(기본 B) = mppi_tune.json 의 같은 설정·시드 0 주행 (같은 조건이라 다시 돌리지 않음).
가중치·문턱은 실행 전에 고정 (결과로 맞추지 않음): 단위 = 진행 1 m 보상 (mppi.W_PROG_M).
  ay_max = specs ay_target (데이터 참 옆 가속도 99%), a_brk = 데이터 앞뒤 가속도 1% (20개 파일, 10 Hz), beta_max = 데이터 |옆 미끄럼각| 99%.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from deep_dynamics.golf import golf_specs, mppi, sim  # noqa: E402

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
OUT = os.path.join(RES, "b_reward.json")
RUNGS = ("R1", "R2", "R3", "R4", "R5", "R6")


def thresholds():
    """데이터 20개 파일 (10 Hz 평활, vx > 5 m/s): 앞뒤 가속도 1% → a_brk, |atan(vy/vx)| 99% → beta_max."""
    from deep_dynamics.golf import data as D
    from deep_dynamics.golf import specs as S
    AX, BE = [], []
    for n in D.available():
        df = D.read_raw(n)
        sm = lambda c: S.smooth(df[c].to_numpy(float), 10.0, 200)[400:-400]
        vx, vy, r = sm("Vx_mps"), sm("Vy_mps"), sm("yawRate_radps")
        k = vx > 5
        AX.append((np.gradient(vx, 0.005) - r * vy)[k])
        BE.append(np.abs(np.arctan2(vy[k], vx[k])))
    return dict(ay_max=float(golf_specs.value("ay_target")), a_brk=float(-np.percentile(np.concatenate(AX), 1)),
                beta_max=float(np.percentile(np.concatenate(BE), 99)), n_files=len(D.available()))


def load():
    return json.load(open(OUT)) if os.path.exists(OUT) else {"runs": []}


def save(d):
    tmp = OUT + ".tmp"
    json.dump(d, open(tmp, "w"), indent=1)
    os.replace(tmp, OUT)


def run(N):
    import torch
    import tune_mppi as T
    from deep_dynamics.golf.configs import GOLF_THUNDERHILL as CFG, cfg_of_model
    from deep_dynamics.golf.ladder import load_model
    d = load()
    if "thresholds" not in d:
        d["thresholds"] = thresholds()
        save(d)
    th = {k: d["thresholds"][k] for k in ("ay_max", "a_brk", "beta_max")}
    print("문턱", th, flush=True)
    c = json.load(open(os.path.join(RES, "mppi_tune.json")))["chosen"][f"B_{N}"]
    tr, grade, info = T.b_track()
    for kind in mppi.B_REWARDS:
        for R in RUNGS:
            d = load()
            if any(r["kind"] == kind and r["N"] == N and r["rung"] == R for r in d["runs"]):
                continue
            pt = os.path.join(RES, "models", f"{R}_s0_roll25.pt")
            model = load_model(pt, cfg_of_model(pt, CFG), mppi.DEV)
            w = sim.calibrated_world(delays=dict(steer=T.steer_delay()), grade_by_idx=grade)
            res, log = mppi.episode(model, w, tr, N=N, seed=0, i0=info["start_idx"], v_armed=10.0, mode="B", lam=c["lam"],
                                    sigma=tuple(c["noise"] * s for s in mppi.SIGMA_FORCE), rew=dict(kind=kind, **th))
            row = dict(kind=kind, N=N, rung=R, lam=c["lam"], noise=c["noise"], sim_model=w.model,
                       fail_track_m=None if res["finished"] else (float(log["idx"][-1]) if log["idx"] else None),
                       vx_max=float(max(log["vx"])) if log["vx"] else None,
                       **{k: res[k] for k in ("status", "finished", "progress_m", "progress_frac", "lap_time", "ey_mean", "ey_max", "wall_s")})
            print(json.dumps(row, ensure_ascii=False), flush=True)
            d = load()
            d["runs"].append(row)
            save(d)
            del model
            torch.cuda.empty_cache()


def report():
    d = load()
    tune = json.load(open(os.path.join(RES, "mppi_tune.json")))
    names = dict(base="기본 B", vlim="① 곡률 속도 상한", term="② 계획 끝 제동 가능", slip="③ 미끄러짐", wall="④ 경계 연속")
    th = d.get("thresholds", {})
    L = ["# B 방식 보상 ablation (11/23)", "",
         f"문턱: 최대 옆 가속도 {th.get('ay_max', float('nan')):.2f} m/s², 최대 감속 {th.get('a_brk', float('nan')):.2f} m/s², "
         f"옆 미끄럼각 {th.get('beta_max', float('nan')):.4f} rad (데이터 {th.get('n_files')}개 파일). 시드 0, 검증 랩 정지 출발 (기록 107.28 s).",
         "칸: 완주면 랩타임(s), 아니면 이탈까지 진행률.", ""]
    for N in sorted({r["N"] for r in d["runs"]}):
        c = tune["chosen"][f"B_{N}"]
        base = {r["rung"]: r for r in tune["runs"] if r["mode"] == "B" and r.get("N", 25) == N and r["lam"] == c["lam"] and r["noise"] == c["noise"]}
        rows = dict(base=base, **{k: {r["rung"]: r for r in d["runs"] if r["kind"] == k and r["N"] == N} for k in mppi.B_REWARDS})
        L += [f"## 계획 길이 {N * mppi.DT:g}초 (λ {c['lam']}, 잡음 {c['noise']})", "",
              "| 보상 | " + " | ".join(RUNGS) + " | 완주 | 진행률 중앙값 |", "|---|" + "---|" * (len(RUNGS) + 2)]
        for k, R in rows.items():
            if not R:
                continue
            cell = lambda r: "—" if r is None else (f"**{r['lap_time']:.1f}**" if r["finished"] else f"{r['progress_frac']:.2f}")
            fin = sum(r["finished"] for r in R.values())
            L.append(f"| {names[k]} | " + " | ".join(cell(R.get(g)) for g in RUNGS) +
                     f" | {fin}/{len(R)} | {np.median([r['progress_frac'] for r in R.values()]):.2f} |")
        L.append("")
    open(os.path.join(RES, "b_reward.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    {"run": lambda: run(int(sys.argv[2])), "report": report}[sys.argv[1]]()
