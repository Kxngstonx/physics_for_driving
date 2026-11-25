"""시뮬레이터가 실제 차보다 더 감속하는가: 시험 파일의 실제 조작(Fxf·조향·경사)을 1초 재생 → 1초 뒤 vx 편향 (시뮬레이터 − 실제).
fit_sim·fit_realsim 의 재생과 같은 방식 (지연 없음, 힘 상한 없음). 사용: python sim_bias.py REPO → sim_bias.json"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(sys.argv[1])
sys.path[:0] = [REPO, os.path.join(REPO, "experiments", "golf"), HERE]
import validate_sim as V  # noqa: E402
import golf_realsim as RS  # noqa: E402
from deep_dynamics.golf import data as D  # noqa: E402
from deep_dynamics.golf import sim  # noqa: E402

W = V.windows(D.get_split()["test"], 25, stride=25)
worlds = {
    "CommonRoad": lambda: sim.calibrated_world(delays=None, cap=False),
    "v3(데이터 맞춤)": lambda: RS.RealWorld(params=json.load(open(os.path.join(HERE, "realsim_out/fit_realsim.json")))["params"], cap=False),
    "횡가속도 고정판": lambda: RS.RealWorld(params=json.load(open(os.path.join(HERE, "realsim_phys3/fit_realsim.json")))["params"], cap=False),
}
Fm = np.array([np.mean(w["F"]) for w in W])
out = {}
for name, mk in worlds.items():
    w = mk()
    e = []
    for win in W:
        vx, vy, r = win["s0"]
        w.reset(win["X0"], win["Y0"], win["psi0"], vx, vy, r, delta=win["d0"])
        s = (np.nan,) * 3
        for k in range(len(win["F"])):
            w.set_grade(win["g"][k])
            s = w.step(float(win["F"][k]), float(win["d"][k]))
        e.append(np.array(s) - win["s"][-1])
    e = np.array(e)
    ok = np.isfinite(e).all(1)
    res = {}
    for tag, m in (("전체", ok), ("제동 중", ok & (Fm < -1000)), ("구동 중", ok & (Fm > 1000))):
        res[tag] = dict(n=int(m.sum()), bias_vx=round(float(e[m, 0].mean()), 3), rmse_vx=round(float(np.sqrt((e[m, 0] ** 2).mean())), 3))
    out[name] = res
    print(name, json.dumps(res, ensure_ascii=False), flush=True)
json.dump(out, open(os.path.join(HERE, "sim_bias.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
