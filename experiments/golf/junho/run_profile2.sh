cd "$(dirname "$0")"
PY=${PY:-python}
export PYTHONUTF8=1 OMP_NUM_THREADS=4
"$PY" -X utf8 -W ignore drive_profile.py "${REPO:-../../..}" commonroad 10.0,10.5,11.0 > realsim_out/profile_commonroad2.log 2>&1
"$PY" -X utf8 -W ignore drive_profile.py "${REPO:-../../..}" realsim 10.5,11.0,11.5 > realsim_out/profile_realsim2.log 2>&1
