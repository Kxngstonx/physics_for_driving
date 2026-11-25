# 1+2: 시드 0·1·2 × 한계 9.0/9.25/9.5/9.75 (CommonRoad), 시드 1·2 × 9.0 (제 시뮬레이터). 2개 프로세스 병렬, 결과 파일 분리
cd "$(dirname "$0")"
PY=${PY:-python}
export PYTHONUTF8=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
( SEEDS=1,2 PROFILE_JS=profile_a.json "$PY" -X utf8 -W ignore drive_profile.py "${REPO:-../../..}" commonroad 9.0,9.5
  SEEDS=1,2 PROFILE_JS=profile_a.json "$PY" -X utf8 -W ignore drive_profile.py "${REPO:-../../..}" realsim 9.0 ) > realsim_out/seeds_a.log 2>&1 &
( SEEDS=0,1,2 PROFILE_JS=profile_b.json "$PY" -X utf8 -W ignore drive_profile.py "${REPO:-../../..}" commonroad 9.25,9.75 ) > realsim_out/seeds_b.log 2>&1 &
wait
echo ALL DONE
