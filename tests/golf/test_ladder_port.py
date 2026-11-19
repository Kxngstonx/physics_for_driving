"""golf/ladder.py 가 11/17 공유본(ver2 ladder_models/ladder.py)과 같은 출력을 내는지 확인.

공유본이 있을 때(정리 전)는 두 모듈을 직접 비교하고 기준 출력을 tests/golf/data/ladder_ref.npz 에 저장한다.
공유본을 지운 뒤에는 저장된 기준 출력과 비교한다.
"""
import importlib.util
import os

import numpy as np
import torch

from deep_dynamics.golf import ladder
from deep_dynamics.golf.configs import AV21_IAC, GTI_THUNDERHILL

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OLD = os.path.join(ROOT, "share_source_team_ver2_models_only", "ladder_models", "ladder.py")
REF = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "ladder_ref.npz")


def _inputs(cfg, g):
    B = 64
    s = torch.stack([10 + 25 * torch.rand(B, cfg.hist, generator=g), torch.randn(B, cfg.hist, generator=g),
                     0.4 * torch.randn(B, cfg.hist, generator=g)], -1)
    lon = (torch.rand(B, cfg.hist, generator=g) - 0.5) * (8000.0 if cfg.lon_input == "force" else 1.6)
    u = torch.stack([lon, 0.1 * torch.randn(B, cfg.hist, generator=g), 0.05 * torch.randn(B, cfg.hist, generator=g)], -1)
    hist = torch.cat([s, u], -1)
    U = torch.cat([lon[:, :1].repeat(1, 24).unsqueeze(-1), 0.1 * torch.randn(B, 24, 1, generator=g),
                   torch.zeros(B, 24, 1)], -1)
    return hist, U


def _outputs(mod, cfg):
    out = {}
    g = torch.Generator().manual_seed(0)
    hist, U = _inputs(cfg, g)
    norm = mod.Norm(hist.numpy(), hist[:, -1:, :3].numpy() + 0.1)
    for rung in ["R1", "R2", "R3", "R4", "R5", "R6"]:
        torch.manual_seed(1)
        m = mod.build(rung, cfg, norm).eval()
        # 신경망 마지막 층이 0으로 시작하는 단이 있어서, 비교가 의미 있도록 모든 가중치를 조금 흔든다
        torch.manual_seed(2)
        with torch.no_grad():
            for p in m.parameters():
                p.add_(0.01 * torch.randn_like(p))
        with torch.no_grad():
            out[f"{cfg.name}_{rung}_1"] = m(hist).numpy()
            out[f"{cfg.name}_{rung}_25"] = mod.rollout(m, hist, U).numpy()
    return out


def test_ladder_same_as_shared_version():
    new = {}
    for cfg in (AV21_IAC, GTI_THUNDERHILL):
        new.update(_outputs(ladder, cfg))
    if os.path.exists(OLD):
        spec = importlib.util.spec_from_file_location("ladder_ver2", OLD)
        old_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(old_mod)
        old = {}
        for cfg in (AV21_IAC, GTI_THUNDERHILL):
            # 설정 객체는 같은 값이어야 하므로 새 모듈 dataclass 를 그대로 넘긴다 (구조가 같아 그대로 읽힘)
            old.update(_outputs(old_mod, cfg))
        for k in new:
            assert np.array_equal(new[k], old[k]), k
        os.makedirs(os.path.dirname(REF), exist_ok=True)
        np.savez_compressed(REF, **new)
    ref = np.load(REF)
    for k in new:
        np.testing.assert_allclose(new[k], ref[k], rtol=1e-5, atol=1e-5, err_msg=k)
