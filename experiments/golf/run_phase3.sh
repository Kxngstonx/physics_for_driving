#!/bin/bash
# 13번 본 실험 1부 (11/19 사용자 결정): 측정 트랙 폭 10.9 m 반영 → 검증·A 튜닝 재실행 → B 방식 튜닝(1·2·4초) →
#   B 본 실험(54회) + 길이별 재학습(2·4초 손실, 시드 0, 12개) → ablation 주행(12회) → 보고서
#   setsid nohup bash experiments/golf/run_phase3.sh > experiments/golf/results/run_phase3.log 2>&1 < /dev/null &
# 진행 상황: experiments/golf/results/phase3_status.txt. 마지막 줄 "PHASE3 DONE" = 정상 종료
# 동시에 최대 2개 (레인 2개), 모두 limited.sh (5 GB). 이미 끝난 주행·모델은 각 스크립트가 건너뛴다 → 다시 실행하면 이어서 함
set -u
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO"
RES=experiments/golf/results
STATUS=$RES/phase3_status.txt
PY="${PY:-/home/okdol/miniconda3/envs/deep_dynamics/bin/python}"
say() { echo "$(date '+%m-%d %H:%M:%S') $*" | tee -a "$STATUS"; }
run() { bash experiments/limited.sh "$PY" -W ignore "$@"; }
two() {  # 두 명령을 나란히, 둘 다 성공해야 0
  bash -c "$1" & local a=$!
  bash -c "$2" & local b=$!
  wait $a; local ra=$?; wait $b; local rb=$?
  [ $ra -eq 0 ] && [ $rb -eq 0 ]
}
export -f run
export PY

say "phase3 start (pid $$)"
run -m deep_dynamics.golf.track > $RES/tracks/build.log 2>&1 || { say "FAILED track"; exit 1; }
for st in closed_free report; do
  run experiments/golf/validate_sim.py $st >> $RES/sim/validate_$st.log 2>&1 || { say "FAILED validate_sim $st"; exit 1; }
done
say "model-free validation (width 10.9): $($PY -c "import json;c=json.load(open('$RES/sim/validation.json'))['closed_free'];r=[x for x in c['runs'] if x['steer_delay']==c['steer_delay']];print(('PASS' if c['passed'] else 'FAIL'),'lap',r[0]['lap_time'] if r else None,'rel%',round(100*r[0]['rel_err'],2) if r and r[0]['rel_err'] is not None else None)")"

# 튜닝: 폭·B 비용이 바뀌어 처음부터 (이전 결과는 지움 — 재현 가능)
[ -f $RES/tune_phase3_started ] || { rm -f $RES/mppi_tune.json $RES/mppi_tune_before_laptime_fix.json; touch $RES/tune_phase3_started; }
two "run experiments/golf/tune_mppi.py run A 25 > $RES/tune_A25.log 2>&1" "run experiments/golf/tune_mppi.py run B 25 > $RES/tune_B25.log 2>&1" \
  || { say "FAILED tuning A25/B25"; exit 1; }
two "run experiments/golf/tune_mppi.py run B 50 > $RES/tune_B50.log 2>&1" "run experiments/golf/tune_mppi.py run B 100 > $RES/tune_B100.log 2>&1" \
  || { say "FAILED tuning B50/B100"; exit 1; }
run experiments/golf/tune_mppi.py select > $RES/tune_select.log 2>&1 || { say "FAILED tune select"; exit 1; }
say "tuning chosen: $($PY -c "import json;print(json.load(open('$RES/mppi_tune.json'))['chosen'])")"

# B 본 실험 (레인 1) + 길이별 재학습 (레인 2)
lane1="for n in 25 50 100; do run experiments/golf/run_b.py main \$n >> $RES/b_main.log 2>&1 || exit 1; done"
lane2="for roll in 50 100; do for r in R1 R2 R3 R4 R5 R6; do [ -f $RES/models/\${r}_s0_roll\${roll}.json ] && continue; \
  run -m deep_dynamics.golf.run_train \$r --seed 0 --roll \$roll > $RES/logs/\${r}_s0_roll\${roll}.log 2>&1 || exit 1; done; done"
mkdir -p $RES/logs
two "$lane1" "$lane2" || { say "FAILED B main or retraining (see b_main.log, logs/*roll50*, *roll100*)"; exit 1; }
say "B main 54 runs + retraining 12 models done"

two "run experiments/golf/run_b.py ablation 50 >> $RES/b_main.log 2>&1" "run experiments/golf/run_b.py ablation 100 >> $RES/b_main.log 2>&1" \
  || { say "FAILED ablation runs"; exit 1; }
run experiments/golf/run_b.py report > $RES/b_report.log 2>&1 || { say "FAILED report"; exit 1; }
say "PHASE3 DONE"
