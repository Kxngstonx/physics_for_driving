"""구동력 이득 k 를 넣은 CommonRoad 에서 모델 없는 운전자 재검증: 저장소 closed_free 가 고른 운전자 설정·조향 지연 그대로, 튜닝 랩과 검증 랩."""
import json, os, sys
K = float(sys.argv[2])
sys.argv = [sys.argv[0], sys.argv[1], "x", "commonroad", "9.0"]
os.environ["SEEDS"] = "0"
REPO = os.path.abspath(sys.argv[1]); HERE = os.getcwd()
sys.path[:0] = [REPO, os.path.join(REPO, "experiments", "golf"), HERE]
import validate_sim as V
from deep_dynamics.golf import sim
cf = json.load(open(V.JS))["closed_free"]
gains, sd = cf["gains"], cf["steer_delay"]
def gw(grade):
    w = sim.calibrated_world(delays=dict(steer=sd), grade_by_idx=grade)
    orig = w.step
    def step(F, d):
        F = min(max(F, sim.F_BRAKE_MIN), float(sim.force_cap(w.x[3])))
        return orig(F * K if F > 0 else F, d)
    w.cap = False; w.step = step
    return w
out = {}
for tag, info0 in (("tune", V.lap_info(V.TUNE_LAP_RUN)), ("val", V.lap_info())):
    tr, grade, info = V.recorded_track(info0)
    for kk in (1.0, K):
        globals()["K"] = kk
        r = V.drive_free(gw(grade), tr, gains=gains, i0=info["start_idx"])
        r["rel_err"] = (r["lap_time"] - info["lap_time"]) / info["lap_time"] if r["finished"] else None
        out[f"{tag}_k{kk}"] = {k: r[k] for k in ("status", "lap_time", "rel_err", "ey_mean", "ey_max")}
        print(tag, kk, json.dumps(out[f"{tag}_k{kk}"]), flush=True)
json.dump(out, open(os.path.join(HERE, "free_gain.json"), "w"), indent=1)
