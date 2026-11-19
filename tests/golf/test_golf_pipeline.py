"""Golf 로더·트랙 계산 점검 (docs/experiment_plan_golf.md 검증 항목). Golf CSV 가 있어야 돌아간다."""
import os

import numpy as np

from deep_dynamics.golf import data as D
from deep_dynamics.golf import track as T
from deep_dynamics.golf.configs import GOLF_THUNDERHILL as CFG
from deep_dynamics.golf.train import windows


def test_files_and_split():
    names = D.available()
    assert len(names) == 20 and not set(D.DUPLICATES) & set(names)
    sp = D.make_split(names, seed=0)
    assert sp == D.make_split(list(reversed(names)), seed=0)  # 순서와 무관하게 재현
    assert (len(sp["train"]), len(sp["val"]), len(sp["test"])) == (14, 3, 3)
    assert not (set(sp["train"]) & set(sp["val"]) or set(sp["train"]) & set(sp["test"]) or set(sp["val"]) & set(sp["test"]))
    assert D.get_split() == sp


def test_loader_shapes_and_dt():
    segs = D.load_run("2023-11_9")
    for sg in segs:
        assert sg["s"].shape[1] == 3 and sg["u"].shape[1] == 3
        assert np.isfinite(sg["s"]).all() and np.isfinite(sg["u"]).all()
        assert np.allclose(np.diff(sg["t"]), D.DT, atol=1e-3)
        assert (sg["s"][:, 0] >= D.V_STOP).all()
    X, Y, U = windows([(sg["s"], sg["u"]) for sg in segs], CFG.hist, horizon=25)
    assert X.shape[1:] == (CFG.hist, 6) and Y.shape[1:] == (25, 3) and U.shape[1:] == (24, 3)


def test_front_force_sign_and_size():
    df = D.read_raw("2023-11_9")
    assert (df[["brake_fl_N", "brake_fr_N", "brake_rl_N", "brake_rr_N"]] >= 0).all().all()  # 제동력은 양수로 기록
    mv = df["Vx_mps"] > 5
    p1 = np.percentile(df["Fxf"][mv], 1)
    assert -8000 < p1 < -4000  # 문서 5-5: 앞 제동력 하위 1% 약 −6 kN
    # 출발 중 구동력이 0 이 아님 (클러치 미끄러짐 처리)
    launch = (df["Vx_mps"] > 1) & (df["Vx_mps"] < 5) & (df["throttleCmd_percent"] > 20) & (df["F_brake_f"] < 1)
    assert np.median(df["F_drive"][launch]) > 1000


def test_speed_profile_simple_cases():
    n = 2000
    v = T.speed_profile(np.zeros(n), mu=1.2, ax_acc=4.3, ax_dec=-5.1, vmax=30.0)
    assert np.allclose(v, 30.0)  # 직선: 상한 속도
    k = 1 / 50.0
    v = T.speed_profile(np.full(n, k), mu=1.2)
    assert np.allclose(v, np.sqrt(1.2 * 9.81 / k))  # 원: √(μg/κ)
    # 열린 경로에서 가속 한계: 출발 0 근처 코너 뒤 v² 증가율 ≤ 2 a ds
    kap = np.zeros(n)
    kap[0] = 1.0
    v = T.speed_profile(kap, mu=1.2, ax_acc=4.3, closed=False)
    assert np.all(np.diff(v ** 2) <= 2 * 4.3 + 1e-6)


def test_track_outputs_inside_boundary():
    d = os.path.join(os.path.dirname(__file__), "..", "..", "experiments", "golf", "results", "tracks")
    for k in ["recorded", "center", "mincurv"]:
        t = np.load(os.path.join(d, f"thunderhill_{k}.npz"))
        assert np.minimum(t["width_l"], t["width_r"]).min() > 0.9
        assert 2500 < len(t["x"]) < 2700
