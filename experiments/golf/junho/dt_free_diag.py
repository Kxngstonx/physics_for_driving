"""dt 에서 모델 없는 운전자가 왜 이탈하는지: 조향 속도 한계 / LSD / 언더스티어 보정(K_us) 을 하나씩 바꿔 봄. 운전자 설정은 prev_t 0.2, kp_v 1.0."""
import os, sys, json
sys.argv = [sys.argv[0], "physics_for_driving", "free"]
os.environ.setdefault("SIM_KIND", "dt")
sys.path.insert(0, os.getcwd())
import drive_compare as DC
import dt_world as DW
V = DC.V
tr, grade, info = V.recorded_track(V.lap_info(V.TUNE_LAP_RUN))
gains = dict(V.FREE, prev_t=0.2, kp_v=1.0)
def go(tag, **kw):
    rate = kw.pop("rate", None); lsd = kw.pop("lsd", True); bmin = kw.pop("bmin", None)
    old = DW.STEER_RATE_MAX
    if rate: DW.STEER_RATE_MAX = rate
    w = DW.DTWorld(DC.sim.Actuator, delays=dict(DC.sim.load_delays(DC.DELAYS) or {}, steer=0.0), grade_by_idx=grade, main_units=False, lsd=lsd)
    oldb = DW.F_BRAKE_MIN
    if bmin: DW.F_BRAKE_MIN = bmin
    r = V.drive_free(w, tr, gains=gains, i0=info["start_idx"])
    DW.STEER_RATE_MAX = old; DW.F_BRAKE_MIN = oldb
    print(tag, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items() if k in ("status", "progress_frac", "lap_time", "ey_mean", "ey_max")}, flush=True)
import json
spec = json.load(open(DC.sim.SPECS))
spec["GOLF"]["final"]["Cf_lin"], spec["GOLF"]["final"]["Cr_lin"] = DW._v("tire_stiffness_f"), DW._v("tire_stiffness_r")
json.dump(spec, open(os.path.join("dt_out", "specs_dt_kus.json"), "w"), indent=1)
go("kus_main")
DC.sim.SPECS = os.path.join("dt_out", "specs_dt_kus.json")
go("kus_branch")
