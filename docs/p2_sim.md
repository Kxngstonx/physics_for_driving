# P2 자전거 시뮬레이터 (11/25–26, 우리 쪽 정답 세계)

> **main 판 안내 (11/26).** 이 파일과 `deep_dynamics/golf/p2.py`, `experiments/golf/fit_p2.py`, `tests/golf/test_p2.py` 는 shinyoung 에서 옮긴 것이다.
> - **차량 값은 11/20 제원으로 `p2.py` 에 고정**: 질량 1496.9 kg, lf/lr 1.026/1.604 m, 앞 제동 몫 0.75, 구동 효율 0.699, 바퀴 반지름 0.319 m,
>   감속비 [8.8, 5.65, 3.93, 3.0], 제동 하한 −6000 N, 구동력 상한 표, 브레이크 지연 35 ms + 20 ms. main 의 `specs.json` (m 1777, η 0.852) 을 읽지 않는다.
> - main 에서 shinyoung 과 같은 결과가 나오는 것을 확인함: 데이터 구동력·Fxf 합, 구동력 상한, 300 스텝 주행 궤적이 같고, 시험 1초 RMSE 도 같음 (아래 표).
> - 모델 없는 운전자 검증(3절)은 shinyoung 의 `validate_sim.py closed_free p2` 로 돌렸다 (main 의 validate_sim 에는 세계 선택이 없고, 3절의 출발 이음매 수정도 없음). 재현은 shinyoung 에서.
> - 이 제원으로 학습한 R1–R6·N 모델과 평가, MPPI 결과는 shinyoung `docs/exp_1125_summary.md`.
> - 시험: `python -c "import tests.golf.test_p2 as t; [getattr(t, n)() for n in dir(t) if n.startswith('test_')]"`

CommonRoad·GTI 두 바퀴 줄 시뮬레이터를 접은 뒤(`archive/sim-multibody-4wheel-1122`), 우리 쪽 실험의 정답 세계로 P2 를 쓴다.
팀원의 자전거 모델이 올라오면 두 번째 세계로 같은 검증을 돌려 비교할 예정.

## 1. 모델 (`deep_dynamics/golf/p2.py`)

R1(사다리 순수 물리)과 같은 식이면 R1 이 구조적으로 유리해지므로, R1 에 없는 것을 넣었다.

| | R1 | P2 |
|---|---|---|
| 타이어 최대 힘 | 축마다 고정 D | μ·수직 하중 |
| 앞뒤 하중 이동 | 없음 | 무게중심 높이 h |
| 결합 슬립 | 없음 | 마찰원 (앞뒤로 쓴 만큼 옆 힘 최대치 감소) |
| 적분 | 오일러 | RK4 (0.01 s, 제어 0.04 s 에 4번) |
| 공통 | | Pacejka 옆 힘, 굴림·공기 저항, 종경사 −g·sin, 좌우 기울기 0 |

입력은 팀 정의 그대로: 앞바퀴 힘 Fxf (구동 0.699·구동력, 제동은 앞 몫 0.75 → 전체 Fx = Fxf/0.75), 조향각.
`P2World` 는 `sim.World` 와 같은 형식(`reset`, `set_pos`, `set_grade`, `pose`, `state`, `step`, `world(hist)`, `g_slope`)이고,
지연은 `sim.Actuator` 를 그대로 쓰고, 구동력 상한·제동 하한은 `p2.force_cap`, `p2.F_BRAKE_MIN` (11/20 제원 값, shinyoung `sim` 과 같은 값) 을 쓴다.

## 2. 계수 맞추기 (`experiments/golf/fit_p2.py`)

팀 분할의 학습 파일, 기록 입력으로 1초(100 스텝) 굴린 [r, vy, vx] 오차 (표준편차로 나눔), Adam 400회, 검증으로 가장 좋은 상태.
고정: m 1496.9 kg, 축거 2.63 m, 앞 제동 몫 0.75 (`p2.MASS`, `p2.LF + p2.LR`, `p2.BRAKE_FRONT`).

| a | Iz | cf | cr | Cf | Cr | μ | h | c0 | c2 | ds |
|---|---|---|---|---|---|---|---|---|---|---|
| 1.420 m | 4099 kg m² | 12.73 /rad | 18.32 /rad | 1.248 | 1.856 | 1.144 | 0.273 m | 30 N | 0.585 N/(m/s)² | 0.349 m |

a 와 ds 는 서로 보완해 따로 정해지지 않는다 (a − ds 만 정해짐). a 를 무게 배분으로 읽지 말 것. h 는 하한(0.25) 근처.

시험 주행(10_1, 11_1, 11_5) 롤아웃 RMSE, 기록 입력:

| | 0.5 s | 1 s | 2 s |
|---|---|---|---|
| P2 r [rad/s] / vy / vx [m/s] | 0.042 / 0.277 / 0.138 | 0.043 / 0.285 / 0.245 | 0.042 / 0.279 / 0.427 |
| 현재값 유지 | 0.098 / 0.316 / 0.938 | 0.148 / 0.446 / 1.850 | 0.253 / 0.720 / 3.538 |
| P2, 0.8 g 이상 창 | 0.054 / 0.290 | 0.054 / 0.305 | 0.049 / 0.284 |

## 3. 모델 없는 운전자 검증

```
python experiments/golf/fit_p2.py                       # → results/sim/p2_fit.json (약 16분, CPU 8 스레드)
python experiments/golf/validate_sim.py closed_free p2  # → validation.json 의 closed_free_p2
```
튜닝 랩 2023-10_5 (기록 108.20 s): 9개 설정 모두 완주, 오차 0.0 ~ +0.7%. 고른 설정 prev_t 0.3, kp_v 2.0.
검증 랩 2023-10_8 (기록 107.28 s): 조향 지연 0 에서 108.16 s (**+0.8%, 통과**), 횡오차 평균 0.61 m·최대 3.45 m.
조향 지연 0.04·0.08·0.12 s 는 이탈 — 운전자(pure pursuit)가 조향 지연을 보상하지 않아서로 보이며, 세계의 조향 지연은 0 으로 둔다.

### 같이 고친 것 (`validate_sim.recorded_track`)
기록 속도를 위치별로 모을 때 랩 앞/뒤 절반을 트랙 번호 0 기준으로 나눠, 출발이 이음매 직전인 랩(10_5 는 2614/2620, 10_8 은 2602/2620)은
출발 지점 몇 m 의 목표 속도가 랩 끝 속도(32 m/s)가 됐다 → 정지 출발 대신 달리며 출발, 랩타임이 3.4% 빠르게 나옴.
이제 실제 출발 위치에서 잰 거리로 나눈다 (출발 목표 속도 0.8·1.2 m/s). CommonRoad 세계 검증에도 같은 영향이 있었을 수 있다.
info.json 을 읽을 때 `encoding="utf-8"` (윈도우 cp949 오류).

## 4. 시험
`tests/golf/test_p2.py` (shinyoung 에서는 run_all.py 에 포함, main 에서는 따로 실행): 오르막 감속 = g·sin, 제동하면 회전 응답이 바뀜, 마찰원, World 형식 + MPPI 한 판.
