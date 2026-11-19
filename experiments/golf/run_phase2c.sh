#!/bin/bash
# 2단계 (11/19 사용자 결정 "실제 주행 데이터로 시뮬레이터 튜닝, 과적합 주의, 모델과 무관하게"):
#   시스템 식별(fit_sim) → 재생 검증 → 모델 없는 운전자 폐루프 검증 → (통과 시) MPPI 튜닝 (다른 실제 랩 2023-10_5 기준)
#   setsid nohup bash experiments/golf/run_phase2c.sh > experiments/golf/results/run_phase2c.log 2>&1 < /dev/null &
# 진행 상황: experiments/golf/results/phase2_status.txt. 마지막 줄 "PHASE2C DONE" = 정상 종료, "STOP ..." = 검증 미달로 멈춤
set -u
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO"
RES=experiments/golf/results
STATUS=$RES/phase2_status.txt
PY="${PY:-/home/okdol/miniconda3/envs/deep_dynamics/bin/python}"
say() { echo "$(date '+%m-%d %H:%M:%S') $*" | tee -a "$STATUS"; }
run() { bash experiments/limited.sh "$PY" -W ignore "$@"; }

say "phase2c start (pid $$): fit_sim → replay → closed_free → (pass) MPPI tuning"
run experiments/golf/fit_sim.py > $RES/sim/fit_sim.log 2>&1 || { say "FAILED fit_sim (see $RES/sim/fit_sim.log)"; exit 1; }
say "fit_sim done: $($PY -c "import json;d=json.load(open('$RES/sim/fit.json'));print({k:round(v,3) for k,v in d['after'].items()}, 'train score', round(d['train']['before']['all']['score'],3),'→',round(d['train']['after']['all']['score'],3), 'test', round(d['test']['before']['all']['score'],3),'→',round(d['test']['after']['all']['score'],3))")"

for st in replay mu closed_free report; do
  run experiments/golf/validate_sim.py $st >> $RES/sim/validate_$st.log 2>&1 || { say "FAILED validate_sim $st"; exit 1; }
done
verdict=$($PY -c "import json;c=json.load(open('$RES/sim/validation.json'))['closed_free'];r=[x for x in c['runs'] if x['steer_delay']==c['steer_delay']];print(('PASS' if c['passed'] else 'FAIL'),'steer',c['steer_delay'],'lap',r[0]['lap_time'] if r else None,'rel',round(100*r[0]['rel_err'],2) if r and r[0]['rel_err'] is not None else None, 'finished_delays',[x['steer_delay'] for x in c['runs'] if x['finished']])")
say "model-free validation: $verdict"
case "$verdict" in
  PASS*) ;;
  *) say "STOP sim validation (model-free) — MPPI 튜닝 안 함"; exit 0 ;;
esac

run experiments/golf/tune_mppi.py run A > $RES/tune_A.log 2>&1 &
pa=$!
run experiments/golf/tune_mppi.py run B > $RES/tune_B.log 2>&1 &
pb=$!
wait $pa; ra=$?
wait $pb; rb=$?
if [ $ra -ne 0 ] || [ $rb -ne 0 ]; then say "FAILED tune_mppi (A exit $ra, B exit $rb)"; exit 1; fi
run experiments/golf/tune_mppi.py select > $RES/tune_select.log 2>&1 || { say "FAILED tune_mppi select"; exit 1; }
say "mppi tuning chosen: $($PY -c "import json;print(json.load(open('$RES/mppi_tune.json'))['chosen'])")"
say "PHASE2C DONE"
