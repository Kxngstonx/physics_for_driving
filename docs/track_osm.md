# 썬더힐 West 트랙: OSM 노면 중앙선 방식 (11/19 팀 결정)

코드 [`deep_dynamics/golf/track_osm.py`](../deep_dynamics/golf/track_osm.py) · 실측 폭 위치 [`deep_dynamics/golf/thunderhill_width_points.csv`](../deep_dynamics/golf/thunderhill_width_points.csv)

```
python -m deep_dynamics.golf.track                    # = track_osm (기본). 이전 방식은 --method recorded
python -m deep_dynamics.golf.track_osm --uniform-width  # 위치별 대신 평균 10.9 m 균일
python experiments/golf/plot_track.py                 # 그림
```

출력은 이전과 같은 `experiments/golf/results/tracks/thunderhill_{recorded,center,mincurv}.npz`, `thunderhill_boundary.npz`, `thunderhill_info.json`이다. `mppi.Track`, `validate_sim`, `tune_mppi`, `run_b`, `viz2d`, `plot_track`이 그대로 읽는다. 경로 npz에 `grade`(rad, 오르막 +)·`z`(고도 m) 키가 더 있다.

## 1. 무엇이 바뀌었나

| | 이전 (`--method recorded`) | 지금 (`track_osm`) |
|---|---|---|
| 경계 기준선 | 가장 빠른 기록 주행선 | **OSM 노면 중앙선** (D7 레이싱 라인으로 꺾임만 다듬음) |
| 폭 | ± 5.45 m 균일 | **실측 14곳 위치별** (10.1~11.6 m), 측정 없는 600~2,350 m는 평균 10.9 m |
| 경사 | 없음 (시뮬레이터는 랩마다 Vu/속력) | **학습 파일 GPS 고도 지도** → 경로 npz `grade` |
| recorded 경로, 목표 속도, 파일 형식 | | 같음 (`track.best_lap` 2023-10_8, `track.speed_profile`) |

**경계 기준선을 바꾼 이유:** 레이싱 라인은 트랙 가운데가 아니라 코너 안·바깥 끝을 지난다. OSM 중앙선에서 레이싱 라인까지 거리는 중앙값 2.1 m, 99분위 4.1 m, 최대 4.4 m다. 주행선 ± 5.45 m로 두면 코너 안쪽 경계가 실제 잔디·연석 쪽으로 최대 약 4 m 열리고, 바깥쪽 아스팔트는 잘린다. B 방식(경계 안 최대 전진)과 최소 곡률 경로에서 영향이 크다.

## 2. 만드는 법

1. **OSM 노면 선:** Overpass API로 썬더힐 반경 3.5 km의 `highway=raceway` 선(13개)을 받아 `results/tracks/thunderhill_osm.json`에 캐시한다. © OpenStreetMap contributors, ODbL.
2. **D7이 달린 구간만:** West 선 3개(`OSM_WEST_WAYS`). 위경도 → ECEF → D7 지역 좌표(D1 트랙 자료와 같은 기준점). OSM 선이 D7 주행 위에 2 m 안으로 겹친다.
3. **중앙선 점 모으기:** 학습 파일의 닫힌 한 바퀴(시작·끝 1 m 안)를 따라가며 가장 가까운 OSM 점(0.25 m 간격으로 채움)을 모은다. 갈림길에서는 D7이 간 쪽만 따라간다.
4. **꺾임 다듬기:** OSM 선은 점이 듬성한 곳에서 꺾인다(1 m당 최대 4.7°). 레이싱 라인에서 OSM 선까지 좌우 거리를 10 m로 평활해 레이싱 라인에 더한다. 모양은 RTK-GPS 레이싱 라인, 위치는 OSM이다. 다듬은 선은 OSM 원래 선과 중앙값 0.9 m, 최대 2.8 m 다르다.
5. **노면 지도:** 학습 파일 14개의 GPS 고도를 중앙선 점마다 중앙값(5 m 평활) → 거리로 미분해 경사. 고도차 23.8 m, 경사 −4.5°~+3.9° (1~99분위). 좌우 기울기는 "롤 = 기울기 + k·횡가속도"로 나눴지만 k가 0.026°/(m/s²)로 작아 분리가 깔끔하지 않다(참고용, 시뮬레이터는 평평한 길).
6. **폭:** 실측 14곳을 트랙 거리로 옮겼다(설명과 기준점으로 추정, ±20 m. 기준점: 피트 레인 2,479~110 m 나란히·2,404 m 갈라짐, East 코스 길 350 m, 북쪽 갈림길 527~557 m, 곡률 정점). 이웃 측정점 사이 150 m 이하는 선형 보간, 먼 곳은 30 m에 걸쳐 평균 10.9 m로.
7. **경로 3종:** recorded = `track.best_lap`의 실제 주행선(1 m 균일), center = 중앙선, mincurv = 경계에서 1 m 안쪽 안에서 곡률 제곱 합 최소(기록선에서 시작, L-BFGS-B).

## 3. 결과 (`thunderhill_info.json`)

| 항목 | 값 |
|---|---|
| 중앙선 길이 | 2,660 m |
| 폭 | 10.12~11.61 m (중앙값 10.9) |
| D7 주행 20개가 경계 안에 드는가 (바퀴 기준) | 3 m 구간에서 1 cm 걸침 (캐러셀 북쪽 10.1 m 지점 근처), 나머지 모두 안 |
| recorded (2023-10_8) | 2,620 m, 경계까지 최소 1.08 m |
| mincurv | 2,581 m, 최소 0.99 m. 반복 한계(5만)에서 멈춤, 목적값은 기록선보다 27% 낮음 |
| 프로파일 랩타임 (μ 1.2) | recorded 83.2 s, mincurv 83.7 s, center 93.3 s (실제 기록 107.3 s보다 약 22% 빠름) |

## 4. 오차

| 항목 | 크기 | 근거 |
|---|---|---|
| 경사 | RMS 0.09°, 95%가 0.2° 안 (가속도 0.016 m/s²) | 지도에 쓰지 않은 시험 주행 3개의 GPS 상하 속도와 비교 |
| 폭, 측정 지점 | ±0.2 m | 3회 평균 |
| 폭, 측정 없는 곳 | ±0.5~0.8 m (추정) | 실측 범위 안의 평균 사용 |
| 경계 좌우 위치 | 보통 코너 ±1 m, 헤어핀(약 1,850·2,400 m) 안쪽 최대 +2 m | 코너 정점 22곳에서 D7 바퀴와 안쪽 경계 사이 간격: 중앙값 1.15 m, 최소 −0.03 m(음수 없음 = 경계가 좁게 잡힌 곳 없음), 큰 곳은 헤어핀. 간격이 큰 곳은 "트랙이 넓게 잡힘"과 "늦은 정점"을 구분할 수 없어 상한으로 본다 |

A 방식(기록선 추종)에서는 영향이 작다. B 방식에서는 헤어핀 안쪽 경계가 가장 큰 불확실성이다. 측정자에게 14곳 측정선의 양 끝 좌표를 받으면 경계 위치와 측정 지점 위치를 함께 확인할 수 있다.
