cd "$(dirname "$0")"
PY=${PY:-python}
export PYTHONUTF8=1 PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 SEEDS=0,1,2
(REALSIM_OUT=cr_mu095 MU_MULT=0.95 "$PY" -X utf8 -W ignore drive_profile.py "${REPO:-../../..}" commonroad 9.0 > cr_mu095/profile.log 2>&1; echo done >> cr_mu095/profile.log) &
(REALSIM_OUT=cr_mu105 MU_MULT=1.05 "$PY" -X utf8 -W ignore drive_profile.py "${REPO:-../../..}" commonroad 9.0 > cr_mu105/profile.log 2>&1; echo done >> cr_mu105/profile.log) &
wait
