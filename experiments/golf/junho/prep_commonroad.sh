#!/bin/bash
# 저장소 CommonRoad 시뮬레이터 준비 단계 (run_phase2*.sh 순서): 제원 → 구동력표 → 보정 → 지연 → 계수 맞춤 → 검증
set -u
cd "$(dirname "$0")/"${REPO:-../../..}""
PY=${PY:-python}
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONUTF8=1 PYTHONIOENCODING=utf-8
R=experiments/golf/results; mkdir -p $R/sim
st() { echo "$(date '+%H:%M:%S') $*" | tee -a $R/prep_status.txt; }
step() { local name=$1; shift; st "start $name"; if "$PY" -W ignore "$@" > $R/prep_$name.log 2>&1; then st "ok $name"; else st "FAILED $name (see $R/prep_$name.log)"; exit 1; fi; }
step specs ../spec_golf_only.py .
step forcecap experiments/golf/drive_force_limit.py
step calibrate experiments/golf/calibrate_sim.py
step delays experiments/golf/fit_delays.py
step fit_sim experiments/golf/fit_sim.py
for s in replay mu closed_free report; do step validate_$s experiments/golf/validate_sim.py $s; done
st "PREP DONE"
