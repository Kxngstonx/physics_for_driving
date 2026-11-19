#!/bin/bash
# 계획된 실험 세 가지(main, 손실 ablation, 출발 구간 ablation)를 모두 정상 종료할 때까지 차례로 실행한다.
#   setsid nohup bash experiments/golf/run_until_done.sh > experiments/golf/results/run_until_done.log 2>&1 < /dev/null &
# - 실행 도구의 시간 제한을 받지 않도록 독립 프로세스로 띄우는 것을 전제로 함
# - 각 실험은 train_all.sh 가 "ALL DONE (모드)" 를 출력할 때(18개 학습 + 평가 완료)까지 최대 MAX_TRY 번 다시 실행.
#   train_all.sh 는 끝난 모델을 건너뛰므로, 다시 실행하면 실패·중단된 모델만 학습함
# - 진행 상황: experiments/golf/results/run_status.txt  (마지막 줄이 "ALL MODES DONE" 이면 정상 종료, "FAILED" 로 시작하면 실패)
set -u
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO"
RES="${RES:-experiments/golf/results}"
STATUS=$RES/run_status.txt
MAX_TRY="${MAX_TRY:-3}"
say() { echo "$(date '+%m-%d %H:%M:%S') $*" | tee -a "$STATUS"; }

say "start (pid $$)"
# 이미 돌고 있는 학습(이전에 시작한 묶음)이 끝날 때까지 기다림 → 같은 모델을 두 번 동시에 학습하지 않게
[ -n "${NO_WAIT:-}" ] || while pgrep -f "[d]eep_dynamics.golf.run_train|[e]xperiments/golf/train_all.sh" > /dev/null; do sleep 30; done  # [d] 처럼 감싸면 이 패턴 문자열을 담은 명령 자신은 걸리지 않음
say "no other training running — begin"

for m in main loss launch; do
  done_ok=0
  for try in $(seq 1 "$MAX_TRY"); do
    say "$m: try $try"
    tmp=$(mktemp)
    bash experiments/golf/train_all.sh "$m" > "$tmp" 2>&1
    cat "$tmp" >> "$RES/train_$m.out"
    grep -h "FAILED" "$tmp" | while read -r line; do say "$m: $line"; done
    if grep -q "ALL DONE ($m)" "$tmp"; then
      done_ok=1
      rm -f "$tmp"
      say "$m: ALL DONE"
      break
    fi
    say "$m: not finished — $(tail -n 1 "$tmp")"
    rm -f "$tmp"
  done
  if [ "$done_ok" -ne 1 ]; then
    say "FAILED $m after $MAX_TRY tries — stopping (see $RES/logs)"
    exit 1
  fi
done
say "ALL MODES DONE"
