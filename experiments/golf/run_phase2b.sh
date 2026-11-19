#!/bin/bash
# 2단계 뒤쪽 (11/19 사용자 결정 "튜닝 먼저, 그 뒤 재판정"): MPPI 튜닝 A·B → 설정 선택 → 튜닝한 설정으로 시뮬레이터 폐루프 재판정.
#   setsid nohup bash experiments/golf/run_phase2b.sh > experiments/golf/results/run_phase2b.log 2>&1 < /dev/null &
# 진행 상황: experiments/golf/results/phase2_status.txt (run_phase2.sh 와 같은 파일에 이어 씀). 마지막 줄 "PHASE2B DONE" = 정상 종료
set -u
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO"
RES=experiments/golf/results
STATUS=$RES/phase2_status.txt
PY="${PY:-/home/okdol/miniconda3/envs/deep_dynamics/bin/python}"
say() { echo "$(date '+%m-%d %H:%M:%S') $*" | tee -a "$STATUS"; }
run() { bash experiments/limited.sh "$PY" -W ignore "$@"; }

say "phase2b start (pid $$): MPPI tuning → re-validation with tuned MPPI"
run experiments/golf/tune_mppi.py run A > $RES/tune_A.log 2>&1 &
pa=$!
run experiments/golf/tune_mppi.py run B > $RES/tune_B.log 2>&1 &
pb=$!
wait $pa; ra=$?
wait $pb; rb=$?
if [ $ra -ne 0 ] || [ $rb -ne 0 ]; then say "FAILED tune_mppi (A exit $ra, B exit $rb)"; exit 1; fi
run experiments/golf/tune_mppi.py select > $RES/tune_select.log 2>&1 || { say "FAILED tune_mppi select"; exit 1; }
say "mppi tuning chosen: $($PY -c "import json;print(json.load(open('$RES/mppi_tune.json'))['chosen'])")"

for st in closed_tuned report; do
  run experiments/golf/validate_sim.py $st >> $RES/sim/validate_$st.log 2>&1 || { say "FAILED validate_sim $st"; exit 1; }
done
say "re-validation (tuned): $($PY -c "import json;c=json.load(open('$RES/sim/validation.json'))['closed_tuned'];s=c['summary'][str(c['steer_delay'])];print(('PASS' if c['passed'] else 'FAIL'),'steer',c['steer_delay'],'finished',s['finished'],'median_finished',round(s['median_lap_finished'],2),'rel',round(100*s['rel_err_finished'],2))")"
say "PHASE2B DONE"
