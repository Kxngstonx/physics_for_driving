"""학습 규칙(조기 종료 + 검증 손실 기반 학습률 감소 + 검증 최저 모델 반환)과 출발 제외 로더 점검."""
import numpy as np
import torch

from deep_dynamics.golf import data as D
from deep_dynamics.golf.configs import GOLF_THUNDERHILL as CFG
from deep_dynamics.golf.train import STOP_PATIENCE, fit, loss_fn, windows


def _toy(n, seed):
    """R5 가 금방 맞출 수 있는 작은 합성 데이터 (vx 감속 + 잡음)."""
    rng = np.random.default_rng(seed)
    segs = []
    for _ in range(4):

        s = np.zeros((n, 3))
        s[0] = [20.0, 0.0, 0.0]
        u = np.stack([rng.normal(0, 500, n), rng.normal(0, 0.02, n), np.zeros(n)], 1)
        for k in range(1, n):
            s[k] = s[k - 1] + [0.04 * u[k - 1, 0] / 1400.0, 0.0, 0.5 * u[k - 1, 1]] + rng.normal(0, 0.01, 3)
        segs.append((s, u))
    return windows(segs, CFG.hist, horizon=1)


def test_fit_stops_early_and_returns_best_val_model():
    tr, va = _toy(400, 0), _toy(200, 1)
    model, rec = fit("R5", CFG, tr, va, max_epochs=300, log_every=0)
    hist = np.array(rec["hist"])
    assert rec["stop_epoch"] < 299 and rec["stop_reason"] == "no_improvement"
    assert hist[:, 3].min() < hist[0, 3]  # 학습률이 줄어든 적이 있음
    # 반환된 모델의 검증 손실 = 기록된 최저 검증 손실
    w = 1.0 / model.norm.dsd
    with torch.no_grad():
        vl = loss_fn(model, *(torch.as_tensor(z) for z in va), w).item()
    assert abs(vl - rec["best_val"]) < 1e-4 * max(1.0, rec["best_val"])
    assert rec["best_val"] == hist[:, 2].min() and rec["best_epoch"] == int(hist[:, 2].argmin())


def test_stop_counter_agrees_with_lr_scheduler():
    """11/18 버그 재발 방지: 조기 종료 기준이 스케줄러와 어긋나 학습률이 줄기 전에 멈추던 문제."""
    model, rec = fit("R5", CFG, _toy(400, 0), _toy(200, 1), max_epochs=300, log_every=0)
    lrs = np.array(rec["hist"])[:, 3]
    assert len(set(lrs)) >= 3  # 멈추기 전에 학습률이 적어도 두 번 줄었음 (patience 10 < 멈춤 25)
    assert rec["stop_epoch"] - rec["ref_epoch"] == STOP_PATIENCE


def test_split_segs_launch_cut():
    sp = D.get_split()
    segs = D.split_segs(sp["test"], v_stop=D.V_LAUNCH)
    assert all((s[:, 0] >= D.V_LAUNCH).all() for s, _ in segs)
    full = sum(len(s) for s, _ in D.split_segs(sp["test"]))
    assert 0 < sum(len(s) for s, _ in segs) < full
