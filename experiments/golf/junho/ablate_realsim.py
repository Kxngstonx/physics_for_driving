"""어떤 요소가 재생 오차를 키우는지: 요소를 하나씩 끄고 같은 계수(v1)로 검증 파일 재생."""
import sys, os, json, numpy as np
REPO = os.path.abspath(sys.argv[1]); [sys.path.insert(0, p) for p in (REPO, os.path.join(REPO, "experiments", "golf"), os.path.dirname(os.path.abspath(__file__)))]
import validate_sim as V, golf_realsim as RS
from deep_dynamics.golf import data as D
W = V.windows(D.get_split()["val"], 25, stride=50)
S_ = np.stack([w["s"] for w in W]); sd = np.array([7.09, 0.657, 0.27])
base = dict(mu=1.2, ca_f=82700.0, ca_r=72300.0, iz=1797.0, load_sens=0.0)
def run(params, **mod):
    old = {k: getattr(RS, k) for k in mod}
    for k, v in mod.items(): setattr(RS, k, v)
    w = RS.RealWorld(params=params, cap=False); P = np.full((len(W), 25, 3), np.nan)
    for i, win in enumerate(W):
        vx, vy, r = win["s0"]; w.reset(0, 0, 0, vx, vy, r, delta=win["d0"])
        for k in range(25):
            w.set_grade(win["g"][k]); P[i, k] = w.step(float(win["F"][k]), float(win["d"][k]))
    for k, v in old.items(): setattr(RS, k, v)
    ok = np.isfinite(P).all((1, 2)); e = P[ok, -1] - S_[ok, -1]; rm = np.sqrt((e ** 2).mean(0))
    return f"score {(rm/sd).sum():.3f} vx {rm[0]:.3f} vy {rm[1]:.3f} r {rm[2]:.4f} div {(~ok).sum()}"
print("창", len(W))
print("기본 (모든 요소)          ", run(base))
print("앞뒤 하중 이동 끔 (h=0)   ", run(base, H_CG=0.0))
print("조향 속도 제한 끔         ", run(dict(base, steer_rate=50.0)))
print("ABS 제한 끔 (1.0)         ", run(base, ABS_FRAC=1.0))
print("하중이동+조향제한 끔      ", run(dict(base, steer_rate=50.0), H_CG=0.0))
print("위 + Iz 2500              ", run(dict(base, steer_rate=50.0, iz=2500.0), H_CG=0.0))
