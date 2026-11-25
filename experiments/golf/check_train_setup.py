"""R1–R6 재학습 전 점검: 학습 코드가 팀 결정(11/19·11/20)대로 도는지 확인한다. 하나라도 어긋나면 멈춘다.

  python experiments/golf/check_train_setup.py            # 로컬·Colab 공통, 학습 직전에

점검 항목
  1. D7 분할 = 팀원 분할 (검증 11_4·11_7·11_8, 시험 10_1·11_1·11_5), 원본 CSV 20개가 다 있음
  2. 기본 설정 = GOLF_THUNDERHILL_G (11/20): m 1496.9, 앞 제동 몫 0.75, dt 0.04, hist 8, 경사 채널 켬 → 입력 7열
  3. 힘 정의: Fxf = 0.699·구동력 − (앞 두 바퀴 제동력) — CSV 한 개에서 다시 계산해 비교
  4. 노면: 학습 입력 u = [Fxf, δ, bank = 0, grade], 경사는 OSM 트랙 지도(track_osm.py)에서, 크기가 상식 범위
  5. 정지 구간 제외 (vx ≥ 0.5 m/s), 학습 규칙 (최대 300 에폭, 조기 종료 25, 학습률 감소 10, 1초 롤아웃 손실)
  6. 이탈 기준 0.9 m (학습과 무관, MPPI 단계용)
"""
import argparse
import hashlib
import json
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
os.chdir(REPO)
from deep_dynamics.golf import data as D  # noqa: E402
from deep_dynamics.golf import mppi, train  # noqa: E402
from deep_dynamics.golf.configs import ALL, road_of  # noqa: E402

TEAM_SPLIT = dict(val=["2023-11_4", "2023-11_7", "2023-11_8"], test=["2023-10_1", "2023-11_1", "2023-11_5"])
# 11/19 에 만든 트랙 지도 (우리가 MPPI·P2 검증에 쓴 것). Colab 에서도 이 파일을 올려 쓴다 (OSM 을 다시 받으면 바뀔 수 있음)
BOUNDARY_MD5 = "0e3cabca84b489124312efed3490b770"

ok_all = True


def check(name, cond, detail=""):
    global ok_all
    ok_all &= bool(cond)
    print(f"[{'OK' if cond else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="GOLF_THUNDERHILL_G")
    a = ap.parse_args()

    # 1. 분할·데이터
    sp = D.get_split()
    check("분할 = 팀원 분할", sp["val"] == TEAM_SPLIT["val"] and sp["test"] == TEAM_SPLIT["test"] and len(sp["train"]) == 14,
          f"val {sp['val']} test {sp['test']}")
    names = D.available()
    check("원본 CSV 20개", len(names) == 20 and set(names) == set(sp["train"] + sp["val"] + sp["test"]), f"{len(names)}개 ({D.DATA_DIR})")

    # 2. 설정
    cfg = ALL[a.cfg]
    check("설정 GOLF_THUNDERHILL_G", cfg.name == "GOLF_THUNDERHILL_G" and cfg.use_grade, cfg.name)
    check("질량 1496.9 kg, 앞 제동 몫 0.75", abs(cfg.m - 1496.9) < 1e-6 and abs(cfg.brake_front_share - 0.75) < 1e-9,
          f"m {cfg.m}, brake_front {cfg.brake_front_share}")
    check("dt 0.04 s, hist 8, 입력 7열", abs(cfg.dt - 0.04) < 1e-12 and cfg.hist == 8 and cfg.nf == 7, f"dt {cfg.dt} hist {cfg.hist} nf {cfg.nf}")
    check("노면 = grade", road_of(cfg) == "grade")
    for f in ("run_train.py", "evaluate.py"):
        src = open(os.path.join(REPO, "deep_dynamics", "golf", f), encoding="utf-8").read()
        check(f"{f} 기본 --cfg GOLF_THUNDERHILL_G", '"--cfg", default="GOLF_THUNDERHILL_G"' in src)

    # 3. 힘 정의
    import pandas as pd
    n0 = sp["train"][0]
    raw = pd.read_csv(os.path.join(D.DATA_DIR, n0 + ".csv"), comment="#")
    F, _g = D.drive_force(raw)
    fxf = 0.699 * F - (raw["brake_fl_N"].to_numpy() + raw["brake_fr_N"].to_numpy())
    df = D.read_raw(n0)
    check("Fxf = 0.699·구동 − 앞 제동", D.DRIVE_EFF == 0.699 and np.allclose(df["Fxf"].to_numpy(), fxf, equal_nan=True), n0)

    # 4. 노면·경사 지도
    path = os.path.normpath(D.TRACK_MAP)
    check("트랙 지도 있음", os.path.exists(path), path)
    if os.path.exists(path):
        md5 = hashlib.md5(open(path, "rb").read()).hexdigest()
        check("트랙 지도 = 11/19 OSM 지도", md5 == BOUNDARY_MD5, md5)
        info = json.load(open(path.replace("_boundary.npz", "_info.json"), encoding="utf-8"))
        check("트랙 방식 osm", info.get("method") == "osm", info.get("method"))
    segs = [sg for n in sp["train"] for sg in D.load_run(n, road=road_of(cfg))]
    U = np.concatenate([sg["u"] for sg in segs])
    S = np.concatenate([sg["s"] for sg in segs])
    check("학습 입력 4열 [Fxf, δ, bank, grade]", U.shape[1] == 4, str(U.shape))
    check("bank = 0", np.all(U[:, 2] == 0.0))
    g = np.degrees(U[:, 3])
    check("경사 크기 정상 (|경사| < 6°, 0 아닌 값 대부분)", np.abs(g).max() < 6 and np.mean(g != 0) > 0.9,
          f"p1 {np.percentile(g, 1):.2f}° p99 {np.percentile(g, 99):.2f}°")
    check("정지 구간 제외 (vx ≥ 0.5)", S[:, 0].min() >= D.V_STOP - 1e-9, f"min vx {S[:, 0].min():.2f}")
    check("Fxf 범위 정상", -20000 < U[:, 0].min() and U[:, 0].max() < 8000, f"{U[:, 0].min():.0f} ~ {U[:, 0].max():.0f} N")

    # 5. 학습 규칙
    check("최대 300 에폭, 조기 종료 25, 학습률 감소 10",
          train.MAX_EPOCHS == 300 and train.STOP_PATIENCE == 25 and train.PLATEAU_PATIENCE == 10)
    src = open(os.path.join(REPO, "deep_dynamics", "golf", "run_train.py"), encoding="utf-8").read()
    check("기본 롤아웃 25 스텝 (1초 손실)", '"--roll", type=int, default=25' in src)
    W = train.windows(D.split_segs(sp["train"], road=road_of(cfg)), cfg.hist, horizon=25)
    check("학습 창 입력 폭 7", W[0].shape[-1] == 7, f"창 {len(W[0])}개, 모양 {tuple(W[0].shape)}")

    # 6. MPPI 용
    check("이탈 기준 0.9 m", mppi.OFF_TRACK_M == 0.9)

    print("\n모두 통과" if ok_all else "\n실패 항목 있음 — 학습하지 말 것")
    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
