cd "$(dirname "$0")"
PY=${PY:-python}
export PYTHONUTF8=1 OMP_NUM_THREADS=4
"$PY" -X utf8 -W ignore drive_profile.py "${REPO:-../../..}" commonroad 9.0,9.5,10.0 > realsim_out/profile3_commonroad.log 2>&1
"$PY" -X utf8 -W ignore drive_profile.py "${REPO:-../../..}" realsim 9.5,10.0 > realsim_out/profile3_realsim.log 2>&1
