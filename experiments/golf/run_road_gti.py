"""노면 비교 실험 + GTI(D1) 섞기·차량 간 평가 (11/19, 우리 저장소 실험을 팀 기준으로 옮김. 학습은 나중에).

  python experiments/golf/run_road_gti.py list                                   # 실험·모델 현황
  python experiments/golf/run_road_gti.py train road_grade R2 --seed 0           # 하나 학습
  python experiments/golf/run_road_gti.py train-all                              # 전부 (이미 있는 것은 건너뜀)
  python experiments/golf/run_road_gti.py train-all --exps road_none road_grade --rungs R1 R2 --seeds 0
  python experiments/golf/run_road_gti.py eval                                   # → results/road_gti/eval.json, eval.md

실험 (모두 팀 기준: 분할 split.json, 학습 규칙 train.fit, 1초 손실(25스텝), 전처리 data.py, 차량 GOLF_THUNDERHILL)
  road_none  : 노면 없음 (bank 0)                                   cfg GOLF_THUNDERHILL
  road_bank  : 데이터 bank_rad 열 (= 팀원 main 과 같은 조건)          cfg GOLF_THUNDERHILL  (results/models 의 main 모델이 있으면 eval 이 그것을 씀)
  road_grade : 경사 (우리 방식, bank 0) — 노면 비교의 기준             cfg GOLF_THUNDERHILL_G
  grade_gti  : road_grade + GTI 학습 데이터 (data_gti, Golf 기준 힘 환산)  cfg GOLF_THUNDERHILL_G
질문
  노면: road_none / road_bank / road_grade 의 Golf 시험 1초 오차 (우리 저장소 결과: 경사를 빼면 vx 오차 약 50% 증가)
  차량 간: road_grade(Golf 만 학습)의 GTI 시험 오차 = 처음 보는 차. Golf 시험 대비 나빠지는 비율이 물리 비중에 따라 다른가
           (우리 저장소 결과: R1 +29%, R4 +56%, R6 +57%, R2 +68%, R5 +101%). grade_gti 는 섞었을 때 두 차 모두에서 어떤가
  GTI 시험은 노면 정보가 없어(0) 차 차이와 코스(본 코스)·노면 차이가 섞여 있다.
출력: experiments/golf/results/road_gti/{실험}/{단}_s{시드}.pt, .json
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
from deep_dynamics.golf import data as D  # noqa: E402
from deep_dynamics.golf import data_gti as G  # noqa: E402
from deep_dynamics.golf.configs import ALL, cfg_of_model  # noqa: E402
from deep_dynamics.golf.evaluate import STEPS, metrics, predict  # noqa: E402
from deep_dynamics.golf.ladder import RUNGS, load_model, save_model  # noqa: E402
from deep_dynamics.golf.train import MAX_EPOCHS, fit, windows  # noqa: E402

RES = os.path.join(REPO, "experiments", "golf", "results")
OUT = os.path.join(RES, "road_gti")
MAIN = os.path.join(RES, "models")  # 팀원 main 모델 (road_bank 와 같은 조건)
ROLL = 25
EXPS = {
    "road_none": dict(cfg="GOLF_THUNDERHILL", road="none", gti=False, note="노면 없음 (bank 0)"),
    "road_bank": dict(cfg="GOLF_THUNDERHILL", road="bank", gti=False, note="데이터 bank_rad 열 (= 팀원 main)"),
    "road_grade": dict(cfg="GOLF_THUNDERHILL_G", road="grade", gti=False, note="경사 (우리 방식, bank 0)"),
    "grade_gti": dict(cfg="GOLF_THUNDERHILL_G", road="grade", gti=True, note="경사 + GTI 학습 데이터"),
}
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def path_of(exp, rung, seed):
    return os.path.join(OUT, exp, f"{rung}_s{seed}.pt")


def model_path(exp, rung, seed):
    """학습한 모델 경로. road_bank 는 따로 학습한 것이 없으면 팀원 main 모델(같은 조건)을 쓴다."""
    p = path_of(exp, rung, seed)
    if os.path.exists(p):
        return p
    if exp == "road_bank":
        q = os.path.join(MAIN, f"{rung}_s{seed}_roll{ROLL}.pt")
        if os.path.exists(q):
            return q
    return None


def train_data(e, split):
    sp = D.get_split()
    segs = D.split_segs(sp[split], road=e["road"])
    if e["gti"] and split == "train":  # 검증은 Golf 만 (모델 선택 기준을 실험끼리 같게)
        segs += G.split_segs("train", road=e["road"])
    return segs


def train_one(exp, rung, seed, max_epochs=MAX_EPOCHS, lr=1e-3):
    e = EXPS[exp]
    cfg = ALL[e["cfg"]]
    out = path_of(exp, rung, seed)
    if os.path.exists(out):
        print("skip (있음)", out)
        return
    tr, va = (windows(train_data(e, k), cfg.hist, horizon=ROLL) for k in ("train", "val"))
    print(exp, rung, seed, "학습 창", len(tr[0]), "검증 창", len(va[0]), "장치", DEV, flush=True)
    model, rec = fit(rung, cfg, tr, va, max_epochs=max_epochs, lr=lr, seed=seed, device=DEV, log_every=10)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    save_model(model, rung, out)
    info = dict(exp=exp, rung=rung, seed=seed, roll=ROLL, cfg=cfg.name, road=e["road"], gti=e["gti"], max_epochs=max_epochs, lr=lr,
                n_train=len(tr[0]), n_val=len(va[0]), **rec)
    if hasattr(model, "g"):
        info["global_coeffs"] = dict(zip(cfg.pnames, model.g.values().detach().cpu().tolist()))
    json.dump(info, open(out[:-3] + ".json", "w"), indent=1)


def test_sets(road, cfg):
    """시험 창: Golf 시험(팀원 분할), GTI 시험. (X, Y, U, Y 표준편차)"""
    out = {}
    for name, segs in (("golf", D.split_segs(D.get_split()["test"], road=road)), ("gti", G.split_segs("test", road=road))):
        X, Y, U = windows(segs, cfg.hist, horizon=max(STEPS), stride=5)
        out[name] = (X, Y, U, Y.reshape(-1, 3).std(0))
    return out


def evaluate(seeds):
    cache, res = {}, {}
    for exp, e in EXPS.items():
        cfg = ALL[e["cfg"]]
        if e["road"] not in cache:
            cache[e["road"]] = test_sets(e["road"], cfg)
        T = cache[e["road"]]
        for rung in RUNGS:
            for seed in seeds:
                p = model_path(exp, rung, seed)
                if p is None:
                    continue
                m = load_model(p, cfg_of_model(p, cfg), DEV)
                row = dict(exp=exp, rung=rung, seed=seed, model=os.path.relpath(p, REPO))
                for name, (X, Y, U, sd) in T.items():
                    P = predict(m, torch.tensor(X, device=DEV), torch.tensor(U, device=DEV))
                    mt = metrics(P, Y, cfg.dt)
                    one = mt[f"{ROLL * cfg.dt:.2f}s"]
                    one["n_vyr"] = float((one["vy"] / sd[1] + one["r"] / sd[2]) / 2)  # 우리 저장소 표와 같은 정규화 (vy·r)
                    row[name] = mt
                res[f"{exp}/{rung}_s{seed}"] = row
    for name, (X, Y, U, sd) in next(iter(cache.values())).items():  # 현재값 유지 기준선
        P = np.repeat(X[:, -1:, :3], max(STEPS), 1)
        one = metrics(P, Y, 0.04)[f"{ROLL * 0.04:.2f}s"]
        one["n_vyr"] = float((one["vy"] / sd[1] + one["r"] / sd[2]) / 2)
        res[f"_hold/{name}"] = one
    os.makedirs(OUT, exist_ok=True)
    json.dump(res, open(os.path.join(OUT, "eval.json"), "w"), indent=1)
    write_md(res)


def write_md(res):
    key = f"{ROLL * 0.04:.2f}s"
    rows = [v for k, v in res.items() if not k.startswith("_")]

    def cell(exp, rung, test):
        v = [r[test][key] for r in rows if r["exp"] == exp and r["rung"] == rung]
        if not v:
            return "–"
        n = np.array([x["n_vyr"] for x in v])
        return f"{n.mean():.3f}±{n.std():.3f} / {np.mean([x['vx'] for x in v]):.2f}"

    L = ["# 노면 비교 + GTI 차량 간 평가 (1초 롤아웃)", "",
         "셀 = vy·r 정규화 오차 (시드 평균 ± 표준편차) / vx RMSE [m/s]. 낮을수록 좋다.", ""]
    for test, title in (("golf", "Golf 시험 (팀원 분할)"), ("gti", "GTI 시험 (처음 보는 차, 노면 0)")):
        h = res.get(f"_hold/{test}", {})
        L += [f"## {title}", "", f"현재값 유지: {h.get('n_vyr', float('nan')):.3f} / {h.get('vx', float('nan')):.2f}", "",
              "| 실험 | " + " | ".join(RUNGS) + " |", "|---|" + "---|" * len(RUNGS)]
        for exp, e in EXPS.items():
            L.append(f"| {exp} ({e['note']}) | " + " | ".join(cell(exp, r, test) for r in RUNGS) + " |")
        L.append("")
    L += ["## 처음 보는 차에서 나빠지는 비율 (road_grade: GTI 시험 / Golf 시험 − 1, vy·r 정규화 오차)", "", "| " + " | ".join(RUNGS) + " |",
          "|" + "---|" * len(RUNGS)]
    deg = []
    for rung in RUNGS:
        a = [r["golf"][key]["n_vyr"] for r in rows if r["exp"] == "road_grade" and r["rung"] == rung]
        b = [r["gti"][key]["n_vyr"] for r in rows if r["exp"] == "road_grade" and r["rung"] == rung]
        deg.append(f"{(np.mean(b) / np.mean(a) - 1) * 100:+.0f}%" if a and b else "–")
    L += ["| " + " | ".join(deg) + " |", ""]
    open(os.path.join(OUT, "eval.md"), "w", encoding="utf-8").write("\n".join(L))
    print("\n".join(L))


def listing(seeds):
    for exp, e in EXPS.items():
        have = [f"{r}_s{s}" for r in RUNGS for s in seeds if model_path(exp, r, s)]
        print(f"{exp:11s} {e['cfg']:20s} road={e['road']:5s} gti={int(e['gti'])}  모델 {len(have)}/{len(RUNGS) * len(seeds)}  — {e['note']}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["list", "train", "train-all", "eval"])
    ap.add_argument("exp", nargs="?")
    ap.add_argument("rung", nargs="?")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--exps", nargs="+", default=list(EXPS))
    ap.add_argument("--rungs", nargs="+", default=list(RUNGS))
    ap.add_argument("--max_epochs", type=int, default=MAX_EPOCHS)
    a = ap.parse_args()
    if a.cmd == "list":
        listing(a.seeds)
    elif a.cmd == "train":
        train_one(a.exp, a.rung, a.seed, a.max_epochs)
    elif a.cmd == "train-all":
        for exp in a.exps:
            for seed in a.seeds:
                for rung in a.rungs:
                    train_one(exp, rung, seed, a.max_epochs)
    else:
        evaluate(a.seeds)
