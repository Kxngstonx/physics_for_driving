#!/bin/bash
# fit_sim 이후 이어서: 모델 없는 운전자 검증(조향 지연 선택) → R1·R6 주행 ×1.00 → ×1.01, ×1.02
set -u
cd "$(dirname "$0")"
PY=${PY:-python}
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUTF8=1 PYTHONIOENCODING=utf-8
R="${REPO:-../../..}"/experiments/golf/results
st() { echo "$(date '+%H:%M:%S') $*" | tee -a $R/prep_status.txt; }
st "start validate_closed_free"
(cd "${REPO:-../../..}" && "$PY" -W ignore experiments/golf/validate_sim.py closed_free > results_tmp.log 2>&1); rc=$?
mv "${REPO:-../../..}"/results_tmp.log $R/prep_validate_closed_free.log
[ $rc -eq 0 ] && st "ok validate_closed_free" || { st "FAILED validate_closed_free"; exit 1; }
st "start drive commonroad"
"$PY" -X utf8 -W ignore drive_compare.py "${REPO:-../../..}" run commonroad > realsim_out/drive_commonroad.log 2>&1 && st "ok drive commonroad" || { st "FAILED drive"; exit 1; }
st "start push commonroad"
"$PY" -X utf8 -W ignore push_speed.py "${REPO:-../../..}" commonroad 1.01,1.02 > realsim_out/push_commonroad.log 2>&1 && st "ok push commonroad" || { st "FAILED push"; exit 1; }
st "ALL COMMONROAD DONE"
