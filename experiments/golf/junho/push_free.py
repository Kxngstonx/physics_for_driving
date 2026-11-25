"""시뮬레이터 자체 한계: 모델 없는 운전자(저장소 drive_free, 검증에서 고른 설정)로 목표 속도 배율별 주행."""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.argv.insert(2, "run")
import drive_compare as DC  # noqa: E402
d0 = DC.load(); gains, steer = d0["free"]["gains"], d0["setup"]["steer_delay"]["realsim"]
out = []
for sc in [float(x) for x in sys.argv[3].split(",")]:
    tr, grade, info = DC.V.recorded_track(DC.V.lap_info("2023-10_5"))
    tr.vref = tr.vref * sc
    r = DC.V.drive_free(DC.real_world(grade, steer), tr, gains=gains, i0=info["start_idx"])
    row = dict(sim="realsim", driver="model_free", scale=sc, **{k: r[k] for k in ("status", "finished", "progress_frac", "lap_time", "ey_mean", "ey_max")},
               progress_m=round(r["progress_frac"] * 2620))
    print(json.dumps(row, ensure_ascii=False), flush=True); out.append(row)
json.dump(out, open(os.path.join(DC.OUT, "push_free.json"), "w"), indent=1)
