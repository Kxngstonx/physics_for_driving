#!/bin/bash
# 2단계 자동 실행 (11/19): 재학습 54개 → 시뮬레이터 검증 → (통과 시) MPPI 튜닝.
#   setsid nohup bash experiments/golf/run_phase2.sh > experiments/golf/results/run_phase2.log 2>&1 < /dev/null &
# 진행 상황: experiments/golf/results/phase2_status.txt
#   마지막 줄 "PHASE2 DONE" = 정상 종료, "FAILED ..." = 실패, "STOP sim validation ..." = 검증 미달로 계획대로 멈춤
set -u
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO"
RES=experiments/golf/results
STATUS=$RES/phase2_status.txt
PY="${PY:-/home/okdol/miniconda3/envs/deep_dynamics/bin/python}"
LIM="bash experiments/limited.sh"
say() { echo "$(date '+%m-%d %H:%M:%S') $*" | tee -a "$STATUS"; }
run() { $LIM "$PY" -W ignore "$@"; }

say "start (pid $$)"

# 1. 이전 1단계 결과 보관 (클러치 신호 오류가 있던 입력으로 학습한 것). 이미 옮겼으면 건너뜀
ARC=$RES/archive_1118_clutchbug
if [ ! -d "$ARC" ]; then
  mkdir -p "$ARC"
  for f in models logs train_main.out train_loss.out train_launch.out run_status.txt run_until_done.log; do
    [ -e "$RES/$f" ] && mv "$RES/$f" "$ARC/"
  done
  mv $RES/eval_* $RES/learning_curves_* "$ARC/" 2>/dev/null
  [ -e "$RES/sim/validation.json" ] && mv "$RES/sim/validation.json" "$ARC/validation_before_calibration.json"
  say "archived phase-1 results → $ARC"
fi

# 2. 재학습 + 평가 (main, loss, launch)
RES=$RES NO_WAIT=1 bash experiments/golf/run_until_done.sh > $RES/run_until_done.log 2>&1
if ! tail -n 1 $RES/run_status.txt | grep -q "ALL MODES DONE"; then
  say "FAILED retraining — $(tail -n 1 $RES/run_status.txt)"
  exit 1
fi
say "retraining done (54 models + eval)"

# 3. 시뮬레이터 검증
for st in replay mu closed report; do
  if ! run experiments/golf/validate_sim.py $st >> $RES/sim/validate_$st.log 2>&1; then
    say "FAILED validate_sim $st (see $RES/sim/validate_$st.log)"
    exit 1
  fi
  say "validate_sim $st done"
done
verdict=$($PY -c "import json;c=json.load(open('$RES/sim/validation.json'))['closed'];s=c['summary'][str(c['steer_delay'])];print(('PASS' if c['passed'] else 'FAIL'), 'steer', c['steer_delay'], 'finished', s['finished'], 'median', round(s['median_lap'],2), 'rel', round(100*s['rel_err'],2))")
say "sim validation: $verdict"
case "$verdict" in
  PASS*) ;;
  *) say "STOP sim validation — MPPI 튜닝 안 함 (계획: 미달이면 멈추고 보고)"; exit 0 ;;
esac

# 4. MPPI 튜닝 (A, B 두 레인)
run experiments/golf/tune_mppi.py run A > $RES/tune_A.log 2>&1 &
pa=$!
run experiments/golf/tune_mppi.py run B > $RES/tune_B.log 2>&1 &
pb=$!
wait $pa; ra=$?
wait $pb; rb=$?
if [ $ra -ne 0 ] || [ $rb -ne 0 ]; then
  say "FAILED tune_mppi (A exit $ra, B exit $rb)"
  exit 1
fi
run experiments/golf/tune_mppi.py select > $RES/tune_select.log 2>&1 || { say "FAILED tune_mppi select"; exit 1; }
say "mppi tuning: $($PY -c "import json;print(json.load(open('$RES/mppi_tune.json'))['chosen'])")"
say "PHASE2 DONE"
