# 경사 입력과 이탈 판정 (11/19 팀 결정, `shinyoung` 브랜치)

## 1. 경사 입력 (우리 방식)

세계(시뮬레이터)에는 이미 경사가 있었지만(`sim.calibrated_world(grade_by_idx=…)`), 모델과 MPPI는 경사를 몰랐다. 이제 모델 입력과 MPPI에도 넣을 수 있다. **경사를 끄면(기본 `GOLF_THUNDERHILL`) 이전과 출력이 같다** (`tests/golf/test_ladder_port.py` 통과).

| 곳 | 바꾼 것 |
|---|---|
| `ladder.py` | `VehicleConfig.use_grade`. 켜면 hist (B, H, 7) = [vx, vy, r, Fxf, δ, bank, grade], 물리 식(R1~R4) dvx 에 −g·sin(경사). R5·R6 은 입력 특징으로 배움 |
| `configs.py` | `GOLF_THUNDERHILL_G` (같은 차량 값 + `use_grade`), `road_of(cfg)`, `cfg_of_model(pt)` (모델 json 의 cfg 로 설정 고르기) |
| `data.py` | `load_run(..., road="grade")`: u = [Fxf, δ, **bank = 0**, **grade**]. 경사는 트랙 지도(`results/tracks/thunderhill_boundary.npz`, `track_osm.py`)에서 위치로 찾음. 좌우 기울기는 0 (데이터 `bank_rad` 열은 센서와 안 맞고, 시뮬레이터·MPPI 도 평평한 길) |
| `train.py` | `windows()` 가 입력 폭(6 또는 7)을 데이터에서 읽음 |
| `run_train.py`, `evaluate.py` | 설정에 맞는 데이터를 자동으로 (`road_of(cfg)`) |
| `mppi.py` | `Track.grade` (트랙 npz 의 `grade`, 없으면 0). 모델이 `use_grade` 면 굴리는 매 스텝의 **예측 위치 경사**를 입력에 붙임. 세계에는 앞 6열만 넘김 |
| `record_race.py`, `run_b.py`, `tune_mppi.py`, `validate_sim.py`, `viz2d.py` | 모델을 json 의 cfg 로 불러옴 (경사 모델도 그대로 씀) |

```
python -m deep_dynamics.golf.track                                               # 트랙 + 경사 지도 (먼저)
python -m deep_dynamics.golf.run_train R2 --seed 0 --roll 25 --cfg GOLF_THUNDERHILL_G --out experiments/golf/results/models_grade
python -m deep_dynamics.golf.evaluate --models experiments/golf/results/models_grade --cfg GOLF_THUNDERHILL_G ...
```

경사 지도 오차: 시험 주행 3개의 GPS 상하 속도와 RMS 0.09° ([`track_osm.md`](track_osm.md) 4절). 우리 저장소 실험에서는 경사를 빼면 1초 vx 오차가 약 50% 늘었다.

## 2. 이탈 판정

`mppi.episode`: 차 중심이 경계 밖으로 **0.9 m(`OFF_TRACK_M`, Golf 폭 1.79 m의 절반)** 를 넘으면 `off_track`. 차 전체가 경계 밖으로 나간 상태다(레이싱의 "네 바퀴 모두 밖"). 이전 2.0 m.
MPPI 비용의 이탈 벌점 여유(A 1.5 m, B 0.9 m, 경계 안쪽 기준)는 튜닝 값이라 그대로 두었다.

## 3. 시험

`tests/golf/test_grade.py`: 경사 0 을 넣은 경사 모델 = 경사 없는 모델(R1·R3, 같은 계수), 오르막이면 vx 가 더 줄어듦, 데이터 열 모양, MPPI 가 경사 모델로 돈다, 이탈 기준 0.9 m.
`run_all.py` 의 사다리·파이프라인·학습 규칙 시험도 통과 (시뮬레이터 시험은 `results/specs.json` 등 로컬 결과 파일이 있어야 돈다).
