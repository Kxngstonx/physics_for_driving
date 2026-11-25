# junho: R1 vs R6 썬더힐 주행 비교 (11/24–11/25)

정준호 작업 폴더. **저장소 코드는 한 줄도 고치지 않았고**, 모든 스크립트는 바깥에서 저장소 모듈을 불러 감싸 실행한다. main(d3a49a8) 기준.

전체 기록은 [`report/썬더힐_R1_R6_실험노트.html`](report/썬더힐_R1_R6_실험노트.html)(브라우저로 열기), 노션용 정리는 [`report/R1_vs_R6_주행비교.md`](report/R1_vs_R6_주행비교.md).

## 결론

| | R1 (물리, 계수 13개) | R6 (MLP, 계수 2.4만 개) |
|---|---|---|
| 실제 시험 파일 1초 vx 오차, 모델 학습 시드 0/1/2 | 0.265 / 0.267 / 0.274 | **0.158 / 0.175 / 0.154** |
| 제동 중 vx 편향 (예측 − 실제) | −0.34 ~ −0.35 | **−0.04 ~ −0.10** |
| CommonRoad 한계 9.0, MPPI 시드 3개 | **3/3 · 106.56–106.64 s** | 2/3 · 107.84–107.96 s |
| 위 + 출발 구간 수정 | **3/3 · 105.68 s** | 1/3 · 106.20 s |
| CommonRoad 감속 보정(구동력 ×1.11) | **3/3 · 105.96–106.04 s** | 0/3 |
| CommonRoad, 모델 학습 시드 1 · 2 | **106.56 · 106.52 s** | 107.92 · 107.20 s |

- 실제 데이터 1초 예측은 R6가 정확하고, 시뮬레이터 폐루프는 R1이 빠르고 견고하다. 둘이 다른 답을 낸다.
- 기각된 가설: (1) 헤어핀 데이터가 적어서 R6가 약하다, (2) 시뮬레이터가 실제보다 더 감속해서 R1이 유리하다(감속을 고쳐도 R1 우세).
- 발견한 설정 문제: 기록 랩 2023-10_5가 정지 출발이라 트랙 0–120 m의 기록 속도가 5 m/s부터 시작한다. `START_FIX=121`로 고치면 R6의 200 m 코너 이탈이 사라진다. 이 수정 전 결과는 서로 비교는 공정하지만 R6 이탈 위치·횟수에 이 문제가 섞여 있다.

## 조건 (저장소와 같음 / 다름)

- 같음: 차량 제원, 트랙(OSM + 실측 폭), 팀 데이터 분할, MPPI A 방식(K 1024, N 25, λ 0.3, 잡음 ×0.5), 이탈 0.9 m, 학습 명령(`run_train`, roll 25, GOLF_THUNDERHILL_G).
- 다름: R1·R6만 학습(시드 0·1·2). 목표 속도에 **곡률 속도 계획** 추가(`speed_profile.py`, 기록 속도로는 CommonRoad 완주 불가). 저장소 검증 중 마찰 탐색·보고서 단계는 건너뜀.

## 실행

이 폴더에서, 저장소 루트를 `REPO`로 넘긴다(기본 `../../..`). 학습된 모델 `.pt`와 결과 `.npz`는 저장소 규칙대로 커밋하지 않았다.

```bash
cd experiments/golf/junho
# 모델 학습 (Windows용 train_all.sh 대체, 전처리 캐시 사용)
python cached_run.py ../../.. build
SEEDS="0 1 2" RUNGS="R1 R6" python run_golf_all.py ../../.. 2 main

# CommonRoad: 곡률 속도 계획, 한계 9.0, MPPI 시드 3개
REALSIM_OUT=realsim_out SEEDS=0,1,2 python drive_profile.py ../../.. commonroad 9.0
# 변형: MU_MULT=0.95 | DRIVE_GAIN=1.11 | MODEL_SEED=1 | START_FIX=121  (REALSIM_OUT 폴더에 drive_compare.json 복사 후)
```

`drive_profile.py`가 읽는 환경 변수: `SIM_KIND`(realsim / dt), `MU_MULT`, `DRIVE_GAIN`, `MODEL_SEED`, `START_FIX`, `SEEDS`, `PROFILE_JS`. `run_*.sh`는 `PY`, `REPO` 환경 변수로 경로를 바꿀 수 있다.

## 파일

| 묶음 | 파일 |
|---|---|
| 직접 만든 시뮬레이터 | `golf_realsim.py`, `fit_realsim.py`(v3), `fit_realsim_phys.py`(물리 범위), `fit_realsim_phys2.py`(+하중 이동 지연), `fit_realsim_phys3.py`(최대 횡가속도 고정), `ablate_realsim.py` |
| 주행 | `drive_compare.py`(모델 없는 운전자 검증, ×1.00 주행), `drive_profile.py`(곡률 계획 주행), `speed_profile.py`, `accel_env.py`, `push_speed.py`, `push_free.py`, `steer_sens.py` |
| 팀원 dt_golf 연결 | `dt_world.py`(어댑터, `pfd_dt/`에 `origin/archive/sim-multibody-4wheel-1122` 워크트리 필요), `run_fit_dt.py`(Windows용 fit_sim --model dt), `dt_calib_mu.py`, `dump_lsd_table.py`, `dt_check_*.py`(팀원 World 와 출력 동일 확인), `dt_diag.py`, `dt_mu_curve.py`, `dt_step.py`, `dt_free_diag.py`, `dt_trace.py` |
| 분석 | `openloop_r1r6.py`, `seed_eval.py`, `data_coverage.py`(헤어핀 가설), `model_vs_world.py`, `sim_bias.py`, `sim_gain.py`, `free_gain.py`, `corner_analysis.py`, `corner_lat.py`, `start_check.py`, `make_video.py` |
| 실행 도구 | `cached_run.py`, `run_golf_all.py`, `spec_golf_only.py`, `*.sh` |
| 결과 (json/log만) | `realsim_out/`(CommonRoad·v3 주 결과), `realsim_phys*/`, `dt_out/`, `cr_mu095/`, `cr_mu105/`, `cr_gain111/`, `cr_mseed1/`, `cr_mseed2/`, `cr_startfix/`, `summary/`(분석 결과 모음) |

영상(GIF)은 `make_video.py`로 다시 만든다(저장소 규칙: 영상은 커밋하지 않음).
