# 물리 범위 제한판: 맞춤 → 모델 없는 운전자 검증 → ×1.00 주행 → 한계 9.0 시드 3개
cd "$(dirname "$0")"
PY=${PY:-python}
export PYTHONUTF8=1 PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
st() { echo "$(date '+%H:%M:%S') $*" | tee -a realsim_phys3/status.txt; }
st "start fit"
"$PY" -X utf8 -W ignore fit_realsim_phys3.py "${REPO:-../../..}" realsim_phys3/fit_realsim.json > realsim_phys3/fit.log 2>&1 || { st "FAILED fit"; exit 1; }
st "ok fit"
export REALSIM_OUT=realsim_phys3 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
"$PY" -X utf8 -W ignore drive_compare.py "${REPO:-../../..}" free > realsim_phys3/free.log 2>&1 || { st "FAILED free"; exit 1; }
st "ok free"
"$PY" -X utf8 -W ignore drive_compare.py "${REPO:-../../..}" run realsim > realsim_phys3/run.log 2>&1 || { st "FAILED run"; exit 1; }
st "ok run x1.00"
SEEDS=0,1,2 "$PY" -X utf8 -W ignore drive_profile.py "${REPO:-../../..}" realsim 9.0 > realsim_phys3/profile.log 2>&1 || { st "FAILED profile"; exit 1; }
st "ALL DONE"
