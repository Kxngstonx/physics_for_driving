"""학습한 R1–R6 평가 (docs/experiment_plan_golf.md 11절, 개루프·모델 진단·학습 곡선).

11/12 의 iac_eval.py(개루프), iac_sensitivity.py(조작 민감도), iac_coeffs.py(R4 계수) 를 Golf 데이터에 맞춰 합친 것.
  개루프 : 시험 파일 3개, 시작점 5스텝(0.2 s) 간격, 기록된 입력으로 25스텝 굴림.
           0.04 / 0.2 / 0.48 / 1.0 s 뒤 vx, vy, r RMSE, 발산 비율. 기준선: 현재값 유지, 직선 외삽.
  민감도 : 시험 파일 상태에서 조향 +0.01 rad, 앞바퀴 힘 +500 N 을 1초 유지했을 때 반응 차이
           (원본은 스로틀 +0.1. force 입력이라 +500 N 으로 바꿈: 약 1.5 t 차에 0.3 m/s²)
  R4 계수: 시험 파일에서 R4 가 고른 계수의 평균·표준편차·vx 와의 상관. R1 전역 계수 최종값.
  학습 곡선: 에폭별 학습·검증 손실 그림.

python -m deep_dynamics.golf.evaluate --tag roll25    (roll1, roll25_v6 은 ablation)
"""
import argparse
import glob
import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from . import data as D
from .configs import ALL, road_of
from .ladder import RUNG_KO, load_model, rollout
from .train import windows

STEPS = [1, 5, 12, 25]
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def predict(model, X, U):
    """기록된 입력으로 U.shape[1]+1 스텝 굴림 (ladder.rollout). 발산한 값은 ±1e6 으로 잘라 지표에서 '발산'으로 셈."""
    return torch.nan_to_num(rollout(model, X, U), nan=1e6, posinf=1e6, neginf=-1e6).clamp(-1e6, 1e6).cpu().numpy()


def metrics(P, Y, dt):
    out = {}
    for st in STEPS:
        e = P[:, st - 1] - Y[:, st - 1]
        rm = np.sqrt(np.mean(e ** 2, 0))
        out[f"{st * dt:.2f}s"] = dict(vx=float(rm[0]), vy=float(rm[1]), r=float(rm[2]),
                                       diverged=float(np.mean(np.abs(e).max(1) > 100)))
    return out


@torch.no_grad()
def sensitivity(model, X, U, n=25, dd=0.01, dF=500.0, dt=0.04):
    def roll(ddel, dfx):  # 첫 스텝부터 n 스텝 동안 조향 +ddel, 앞바퀴 힘 +dfx 를 유지
        hist, u = X.clone(), U[:, :n - 1].clone()
        hist[:, -1, 4] += ddel
        hist[:, -1, 3] += dfx
        u[:, :, 1] += ddel
        u[:, :, 0] += dfx
        P = rollout(model, hist, u)
        return P[:, -1], dt * P[:, :, 2].sum(1)
    s0, y0 = roll(0.0, 0.0)
    _, y1 = roll(dd, 0.0)
    s2, _ = roll(0.0, dF)
    dyaw = (y1 - y0).cpu().numpy()
    dvx = (s2[:, 0] - s0[:, 0]).cpu().numpy()
    q = lambda z: dict(med=float(np.median(z)), p10=float(np.percentile(z, 10)), p90=float(np.percentile(z, 90)))
    return dict(dyaw=q(dyaw), dvx=q(dvx))


TEST_SETS = [("all", D.V_STOP, "시험 전체 (정지만 제외)"), ("v6", D.V_LAUNCH, f"시험 {D.V_LAUNCH:g} m/s 이상 (출발 구간 제외)")]


def openloop_table(res, title):
    L = [f"## {title}", "",
         "| 모델 | " + " | ".join(f"{h} vx / vy / r" for h in res["B0_persist"]) + " |", "|---|" + "---|" * len(res["B0_persist"])]
    groups = {"B0_persist": ["B0_persist"], "B1_linear": ["B1_linear"]}
    for k in res:
        if k.startswith("R"):
            groups.setdefault(k[:2], []).append(k)
    for g, mem in groups.items():
        name = {"B0_persist": "기준선: 현재값 유지", "B1_linear": "기준선: 직선 외삽"}.get(g, f"{g} {RUNG_KO.get(g, '')} (시드 {len(mem)}개)")
        cells = []
        for h in res["B0_persist"]:
            v = [np.mean([res[x][h][q] for x in mem]) for q in ("vx", "vy", "r")]
            sd = [np.std([res[x][h][q] for x in mem]) for q in ("vx", "vy", "r")]
            dv = np.mean([res[x][h]["diverged"] for x in mem])
            cells.append(" / ".join(f"{x:.3g}" + (f"±{s:.2g}" if len(mem) > 1 else "") for x, s in zip(v, sd)) + (f" (발산 {dv:.0%})" if dv > 0 else ""))
        L.append(f"| {name} | " + " | ".join(cells) + " |")
    return L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="roll25", help="모델 이름 끝부분: roll25 (main), roll1 (1스텝 손실), roll25_v6 (출발 구간 제외)")
    ap.add_argument("--models", default="experiments/golf/results/models")
    ap.add_argument("--out", default="experiments/golf/results")
    ap.add_argument("--cfg", default="GOLF_THUNDERHILL_G",
                    help="[변경 11/20] 기본 = 종경사 입력 포함")
    a = ap.parse_args()
    cfg = ALL[a.cfg]
    sp = D.get_split()
    paths = sorted(glob.glob(os.path.join(a.models, f"[RN]?_s?_{a.tag}.pt")))
    models = {os.path.basename(p)[:-3]: load_model(p, cfg, DEV) for p in paths}
    out = dict(tag=a.tag, test=sp["test"], openloop={}, n_starts={})
    L = [f"# 개루프 평가: 모델 {a.tag} (시험 파일 {', '.join(sp['test'])})", ""]
    tensors = {}
    for key, vstop, title in TEST_SETS:
        X, Y, U = windows(D.split_segs(sp["test"], v_stop=vstop, road=road_of(cfg)), cfg.hist, horizon=max(STEPS), stride=5)
        Xt, Ut = torch.tensor(X, device=DEV), torch.tensor(U, device=DEV)
        tensors[key] = (Xt, Ut)
        res = {"B0_persist": metrics(np.repeat(X[:, -1:, :3], max(STEPS), 1), Y, cfg.dt)}
        lin = X[:, -1:, :3] + (X[:, -1:, :3] - X[:, -2:-1, :3]) * np.arange(1, max(STEPS) + 1)[None, :, None]
        res["B1_linear"] = metrics(lin, Y, cfg.dt)
        for tag, m in models.items():
            res[tag] = metrics(predict(m, Xt, Ut), Y, cfg.dt)
        out["openloop"][key], out["n_starts"][key] = res, len(X)
        L += openloop_table(res, f"연속 예측 RMSE — {title}, 시작점 {len(X)}개") + [""]
    Xa, Ua = tensors["all"]  # 민감도·계수는 시험 전체에서
    near = lambda v, lo, hi, frac: (v - lo < frac * (hi - lo)) | (hi - v < frac * (hi - lo))
    sens, coef, curves, training = {}, {}, {}, {}
    for tag, m in models.items():
        sens[tag] = sensitivity(m, Xa[::2], Ua[::2], dt=cfg.dt)
        if tag.startswith("R4"):
            with torch.no_grad():
                c = m.coeffs(Xa).cpu().numpy()
            vx = Xa[:, -1, 0].cpu().numpy()
            coef[tag] = {n: dict(mean=float(c[:, i].mean()), std=float(c[:, i].std()),
                                 corr_vx=float(np.corrcoef(c[:, i], vx)[0, 1]) if c[:, i].std() > 0 else 0.0,
                                 at_bound=float(np.mean(near(c[:, i], *cfg.phys[i][2:4], 1e-3))))
                         for i, n in enumerate(cfg.pnames)}
        if hasattr(m, "g"):
            v = m.g.values().detach().cpu().numpy()
            coef[tag] = {n: dict(value=float(v[i]), lo=cfg.phys[i][2], hi=cfg.phys[i][3],
                                 at_bound=bool(near(v[i], *cfg.phys[i][2:4], 0.01)))
                         for i, n in enumerate(cfg.pnames)}
        info = json.load(open(os.path.join(a.models, tag + ".json")))
        training[tag] = {k: info[k] for k in ("sec", "best_epoch", "ref_epoch", "stop_epoch", "stop_reason")}
        curves[tag] = dict(hist=info["hist"], **training[tag])
    out.update(sensitivity=sens, coeffs=coef, training=training)
    json.dump(out, open(os.path.join(a.out, f"eval_{a.tag}.json"), "w"), indent=1)

    L += ["## 조작 민감도 (시험 전체, 1초 뒤 반응 차이, 중앙값 [10~90 백분위])", "",
          "| 모델 | 조향 +0.01 rad → 방향 변화 (rad) | 앞바퀴 힘 +500 N → vx 변화 (m/s) |", "|---|---|---|"]
    for t, s_ in sens.items():
        L.append(f"| {t} | {s_['dyaw']['med']:.4f} [{s_['dyaw']['p10']:.4f}, {s_['dyaw']['p90']:.4f}] | "
                 f"{s_['dvx']['med']:.3f} [{s_['dvx']['p10']:.3f}, {s_['dvx']['p90']:.3f}] |")
    L += ["", "## 학습 (같은 규칙: 조기 종료 + 검증 손실 기반 학습률 감소, 검증 최저 모델 사용)", "",
          "| 모델 | 초 | 검증 최저 에폭 | 멈춘 에폭 | 멈춘 이유 |", "|---|---|---|---|---|"]
    for t, c in training.items():
        L.append(f"| {t} | {c['sec']:.0f} | {c['best_epoch']} | {c['stop_epoch']} | {c['stop_reason']} |")
    open(os.path.join(a.out, f"eval_{a.tag}.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L))
    plot_curves(curves, os.path.join(a.out, f"learning_curves_{a.tag}.png"))


SEED_STYLE = {0: ("#0072B2", "o"), 1: ("#D55E00", "s"), 2: ("#009E73", "^")}  # Okabe-Ito, 색 + 표식 함께


def plot_curves(curves, path):
    """모델별 학습 곡선. 위: 학습(실선)·검증(점선) 손실 (로그 축), ●■▲ = 검증 최저 에폭, | = 멈춘 에폭.
    아래: 그 에폭의 기본 학습률 (로그 축). 같은 모델의 두 패널은 x 축을 공유한다."""
    if not curves:
        return
    rungs = list(RUNG_KO)
    fig = plt.figure(figsize=(15, 9.5), layout="constrained")
    gs = fig.add_gridspec(4, 3, height_ratios=[3, 1, 3, 1])
    for i, rung in enumerate(rungs):
        r0, c0 = 2 * (i // 3), i % 3
        ax = fig.add_subplot(gs[r0, c0])
        axl = fig.add_subplot(gs[r0 + 1, c0], sharex=ax)
        for t, c in sorted(curves.items()):
            if not t.startswith(rung):
                continue
            seed = int(re.match(r"R\d_s(\d+)_", t).group(1))
            col, mk = SEED_STYLE.get(seed, ("0.3", "x"))
            h = np.array(c["hist"])
            ax.plot(h[:, 0], h[:, 1], color=col, lw=1, ls="-", label=f"seed {seed} train")
            ax.plot(h[:, 0], h[:, 2], color=col, lw=1, ls="--", label=f"seed {seed} val")
            b = int(c["best_epoch"])
            ax.plot(h[b, 0], h[b, 2], marker=mk, color=col, ms=6, ls="none")
            ax.axvline(c["stop_epoch"], color=col, lw=0.8, ls=":")
            axl.plot(h[:, 0], h[:, 3], color=col, lw=1, marker=mk, markevery=max(1, len(h) // 6), ms=3)
        ax.set_yscale("log")
        ax.set_title(rung)  # 이 환경에 한글 글꼴이 없어 그림 글자는 영어로
        ax.set_ylabel("loss (log10)")
        ax.tick_params(labelbottom=False)
        ax.legend(fontsize=6, ncol=2)
        axl.set_yscale("log")
        axl.set_ylabel("lr (log10)")
        axl.set_xlabel("epoch")
    fig.suptitle("Learning curves: solid = train, dashed = val loss; marker = best-val epoch (model used); dotted line = stop epoch")
    fig.savefig(path, dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    main()
