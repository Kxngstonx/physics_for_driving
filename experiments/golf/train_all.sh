#!/bin/bash
# R1–R6 × 시드 0,1,2 학습 + 평가 (docs/experiment_plan_golf.md 6-3절). 작업 2개씩 병렬, 각 작업은 메모리 제한 안에서.
#   bash experiments/golf/train_all.sh main     # 1초 손실 (roll 25)
#   bash experiments/golf/train_all.sh loss     # ablation: 1스텝 손실 (roll 1)
#   bash experiments/golf/train_all.sh launch   # ablation: 출발 구간 제외 (6 m/s 미만 제외, roll 25)
# 모든 모드가 같은 학습 규칙 (train.fit: 조기 종료 + 검증 손실 기반 학습률 감소, 최대 300 에폭, 검증 최저 모델).
# 이미 끝난 모델(.json 있음)은 건너뜀 → 중단되면 다시 실행하면 됨. 모든 학습이 끝나면 평가까지 실행.
# RUNGS 로 일부 모델만 학습할 수 있음 (예: RUNGS="R2 R4" bash ... launch). 18개가 다 모였을 때만 평가함.
# 모델 이름 끝부분(SUF)은 run_train.py 의 태그 규칙과 같아야 함: roll{N}, 출발 제외는 _v6 추가.
set -uo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO"
export PY="${PY:-/home/okdol/miniconda3/envs/deep_dynamics/bin/python}"
export OUT="${OUT:-experiments/golf/results/models}"
# [변경 11/20, 사용자 결정] 모든 학습·평가에 종경사 입력. 경사 없는 이전 설정은 CFG=GOLF_THUNDERHILL 로만
export CFG="${CFG:-GOLF_THUNDERHILL_G}"
export LOG="${LOG:-experiments/golf/results/logs}"
mkdir -p "$OUT" "$LOG"
case "${1:-main}" in
  main)   export ROLL=25 EXTRA="" SUF="roll25" ;;
  loss)   export ROLL=1  EXTRA="" SUF="roll1" ;;
  launch) export ROLL=25 EXTRA="--vmin 6" SUF="roll25_v6" ;;
  *) echo "main | loss | launch"; exit 1 ;;
esac
job() {  # job R4 0
  local tag="$1_s$2_$SUF"
  if [ -f "$OUT/$tag.json" ]; then
    # 이미 있는 모델은 같은 설정(CFG)으로 학습한 것만 '끝남'으로 본다. 다른 설정이면 덮어쓰지도 건너뛰지도 않고 멈춤
    local have; have=$("$PY" -c "import json,sys; print(json.load(open(sys.argv[1])).get('cfg'))" "$OUT/$tag.json")
    if [ "$have" = "$CFG" ]; then echo "skip (done): $tag"; return; fi
    echo "FAILED $tag: $OUT/$tag.json 은 cfg=$have 로 학습됨 (지금 CFG=$CFG). 이전 모델을 다른 폴더로 옮긴 뒤 다시 실행"; return
  fi
  if bash experiments/limited.sh "$PY" -m deep_dynamics.golf.run_train "$1" --seed "$2" --roll $ROLL $EXTRA --cfg "$CFG" --out $OUT > "$LOG/train_$tag.log" 2>&1; then
    echo "ok $tag"
  else
    echo "FAILED $tag (see $LOG/train_$tag.log)"
  fi
}
export -f job
# 느린 물리 모델(R1–R4)이 앞에 오도록 줄 세운 뒤, 비는 쪽이 다음 작업을 가져가는 공유 대기열 (작업 2개씩)
JOBS=$(for r in ${RUNGS:-R3 R1 R2 R4 R5 R6}; do for s in 0 1 2; do echo "$r $s"; done; done)
echo "$JOBS" | xargs -P 2 -L 1 bash -c 'job "$0" "$1"'
want=18
n=$(ls $OUT/R?_s?_$SUF.json 2>/dev/null | wc -l)
echo "trained models ($SUF): $n / $want"
[ "$n" -eq "$want" ] || { echo "아직 $n / $want — 평가는 18개가 모두 끝난 뒤 실행"; exit 0; }
bash experiments/limited.sh "$PY" -m deep_dynamics.golf.evaluate --tag $SUF --cfg "$CFG" > "$LOG/eval_$SUF.log" 2>&1 \
  && echo "evaluated $SUF" || { echo "EVAL FAILED $SUF"; exit 1; }
echo "ALL DONE ($1)"
