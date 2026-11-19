"""[11/19] 경사 채널 (configs.GOLF_THUNDERHILL_G, data.load_run(road='grade'), mppi 경사 입력) 검증."""
import os

import numpy as np
import torch

from deep_dynamics.golf import data as D
from deep_dynamics.golf import ladder, mppi
from deep_dynamics.golf.configs import GOLF_THUNDERHILL as CFG, GOLF_THUNDERHILL_G as CFG_G, road_of

TRACK = "experiments/golf/results/tracks/thunderhill_recorded.npz"


def _same_params(m_from, m_to):
    sd = m_from.state_dict()
    tgt = m_to.state_dict()
    for k, v in sd.items():
        if k in tgt and tgt[k].shape == v.shape:
            tgt[k].copy_(v)


def test_physics_rungs_grade_zero_equals_no_grade():
    """R1·R3: 경사 0 을 넣은 경사 모델 = 경사 없는 모델 (같은 계수). 경사 > 0(오르막)이면 vx 가 더 줄어든다."""
    torch.manual_seed(0)
    h6 = torch.randn(16, CFG.hist, 6)
    h6[..., 0] = 20 + h6[..., 0].abs()
    h6[..., 5] = 0.0
    for rung in ("R1", "R3"):
        m6, m7 = ladder.build(rung, CFG), ladder.build(rung, CFG_G)
        _same_params(m6, m7)
        h7 = torch.cat([h6, torch.zeros(16, CFG.hist, 1)], -1)
        assert torch.allclose(m6(h6), m7(h7), atol=1e-6)
        up = h7.clone()
        up[..., 6] = 0.05
        assert (m7(up)[:, 0] < m7(h7)[:, 0]).all()


def test_load_run_grade_columns():
    a = D.load_run("2023-11_9")
    g = D.load_run("2023-11_9", road="grade")
    assert len(a) == len(g)
    for sa, sg in zip(a, g):
        assert sg["u"].shape[1] == 4 and np.all(sg["u"][:, 2] == 0)
        assert np.allclose(sa["u"][:, :2], sg["u"][:, :2])
        assert np.abs(sg["u"][:, 3]).max() < np.radians(8)
    assert road_of(CFG_G) == "grade" and road_of(CFG) == "bank"


def test_mppi_episode_with_grade_model():
    """경사 모델을 MPPI 내부 모델로 써도 돌아간다 (세계 = 경사 없는 모델, 앞 6열만 받음). 이탈 기준 0.9 m."""
    assert mppi.OFF_TRACK_M == 0.9
    if not os.path.exists(TRACK):
        return
    tr = mppi.Track(TRACK)
    assert tr.grade.shape[0] == tr.n
    torch.manual_seed(0)
    ctrl = ladder.build("R1", CFG_G).to(mppi.DEV)
    world = ladder.build("R1", CFG).to(mppi.DEV)
    res, log = mppi.episode(ctrl, world, tr, K=64, N=10, max_time=2.0, use_graph=False)
    assert res["status"] in ("timeout", "off_track", "stalled", "finished") and len(log["vx"]) > 0
