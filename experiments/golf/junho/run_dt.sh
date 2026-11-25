# 팀원 이중 궤도(dt_golf): 맞춤(별도 실행 중) 기다림 → 모델 없는 운전자 검증 → ×1.00 주행 → 한계 9.0 시드 3개
cd "$(dirname "$0")"
PY=${PY:-python}
export PYTHONUTF8=1 PYTHONIOENCODING=utf-8
st() { echo "$(date '+%H:%M:%S') $*" | tee -a dt_out/status.txt; }
st "ok fit $(cat pfd_dt/experiments/golf/results/sim/calibration_dt.json | tr -d '\n ')"
export REALSIM_OUT=dt_out SIM_KIND=dt OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
"$PY" -X utf8 -W ignore drive_compare.py "${REPO:-../../..}" free > dt_out/free.log 2>&1 || { st "FAILED free"; exit 1; }
st "ok free"
"$PY" -X utf8 -W ignore drive_compare.py "${REPO:-../../..}" run dt > dt_out/run.log 2>&1 || { st "FAILED run"; exit 1; }
st "ok run x1.00"
SEEDS=0,1,2 "$PY" -X utf8 -W ignore drive_profile.py "${REPO:-../../..}" dt 9.0 > dt_out/profile.log 2>&1 || { st "FAILED profile"; exit 1; }
st "ALL DONE"
