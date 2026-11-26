"""차량별 설정. 모델 구조는 ladder.py 하나이고, 차가 바뀌면 여기 값만 바뀜.

AV21_IAC         : 11/12 IAC 실험에 쓴 값 그대로 (참조용)
GTI_THUNDERHILL  : 썬더힐 Stanford VW GTI 용 초안 (참조용. 제원 추정 검증에 알려진 값으로 씀)
GOLF_THUNDERHILL : 본 실험 차량. [변경 11/21] 차량 값은 deep_dynamics/golf/specs.json (experiments/golf/estimate_specs.py 총괄)
"""
from dataclasses import replace

from . import golf_specs
from .ladder import VehicleConfig

# ------------------------------------------------------------ IAC AV-21 (Deep Dynamics 공개 데이터)
# 제원: DD 설정 파일 deep_dynamics_iac.yaml. 범위: DD iac 설정을 바탕으로 넓히고 Cb·kb·ds 추가
AV21_IAC = VehicleConfig(
    name="AV21_IAC", m=790.0, lf=1.248, lr=1.7328, dt=0.04, hist=8, lon_input="pedal",
    axle_fmax0=(6000.0, 7000.0),
    phys=[  # 이름, 초기값, 하한, 상한
        ("Bf", 12.0, 5.0, 30.0), ("Cf", 1.3, 0.5, 2.0), ("Df", 5000.0, 100.0, 12000.0), ("Ef", -0.5, -2.0, 0.0),
        ("Br", 12.0, 5.0, 30.0), ("Cr", 1.3, 0.5, 2.0), ("Dr", 6000.0, 100.0, 12000.0), ("Er", -0.5, -2.0, 0.0),
        ("Cm1", 6000.0, 100.0, 15000.0), ("Cm2", 20.0, 0.0, 200.0), ("Cb", 8000.0, 100.0, 20000.0),
        ("Cr0", 200.0, 0.0, 2000.0), ("Cr2", 0.5, 0.0, 3.0), ("Iz", 1000.0, 500.0, 2000.0),
        ("kb", 0.0, -1.5, 1.5),   # 뱅크 중력 성분 이득: dvy += kb * g * sin(bank)
        ("ds", 0.0, -3.0, 3.0)],  # 측정 vy 위치 보정 (m): vy_측정 = vy_무게중심 + ds * r
    note="lon = 스로틀/100, 브레이크 눌리면 -브레이크압/2757.9. bank = LVMS roll 열(1초 평활), Putnam 0",
)

# ------------------------------------------------------------ Stanford VW GTI "Niki" (Thunderhill) — 초안
# 제원: Stanford learning_params.py / vehicles.py (m 1776.2, a 1.194, b 1.437, Izz 2763.49,
#       코너링 강성 Cf 150 kN/rad, Cr 170 kN/rad, 굴림저항 255 N, 공기저항 0.3638 N/(m/s)^2, 제동 앞 몫 0.66)
# 초기값: 정적 축하중 앞 9517 N, 뒤 7909 N (μ=1 이면 최대 횡력) → Df 9500, Dr 7900.
#         B 는 B·C·D 가 Stanford 코너링 강성과 맞게: 12.1·1.3·9500 ≈ 150 k, 16.5·1.3·7900 ≈ 170 k
# 범위: D 상한 25 kN (μ 약 2.6 까지 허용), Iz 1000~5000. 정해진 값 아님 — 학습해 보고 범위 끝에 붙으면 조정
# 데이터에 페달이 없고 앞바퀴 힘 fxf(N)가 있어서 lon_input="force"
GTI_THUNDERHILL = VehicleConfig(
    name="GTI_THUNDERHILL", m=1776.2, lf=1.194, lr=1.437, dt=0.04, hist=8, lon_input="force",
    brake_front_share=0.66, axle_fmax0=(9500.0, 7900.0),
    phys=[
        ("Bf", 12.1, 3.0, 30.0), ("Cf", 1.3, 0.5, 2.0), ("Df", 9500.0, 100.0, 25000.0), ("Ef", -0.5, -2.0, 0.0),
        ("Br", 16.5, 3.0, 30.0), ("Cr", 1.3, 0.5, 2.0), ("Dr", 7900.0, 100.0, 25000.0), ("Er", -0.5, -2.0, 0.0),
        ("Cr0", 255.0, 0.0, 2000.0), ("Cr2", 0.3638, 0.0, 3.0), ("Iz", 2763.49, 1000.0, 5000.0),
        ("kb", 0.0, -1.5, 1.5),   # 썬더힐 데이터에는 경사 열이 없어 bank=0 → 효과 없음 (형식 유지)
        ("ds", 0.0, -1.0, 1.0)],
    note="초안. lon = fxf (N). dt 0.04 는 100 Hz 를 4개마다 뽑는다는 가정 (thunderhill_data.py)",
)

# ------------------------------------------------------------ VW Golf (Stanford hh613qz0317, Thunderhill West) — 본 실험
# [변경 11/21] 차량 값은 손으로 적지 않고 deep_dynamics/golf/specs.json 에서 읽는다 (항목별 출처·근거는 그 파일과
#   docs/experiment_plan_golf.md [결정 11/21]). 질량·무게중심·요 관성 = 공개값(데이터 차량 = Niki 가정), 앞 제동 몫·타이어 = 데이터.
# 물리 계수 초기값 (학습 시작점, Physics Guard 범위는 그대로):
#   Df, Dr = 정적 축하중 × Fiala 축별 최대 마찰.  Bf, Br = Fiala 코너링 강성 ÷ (C 1.3 × D)  (B·C·D = 0 근처 기울기)
#   Iz, Cr0, Cr2 = specs 값.  이전 값(11/20): m 1496.9, lf 1.026 / lr 1.604, Iz 1797, Df 10750 / Dr 6870, Bf 6.1 / Br 8.4
_v = golf_specs.value
_m, _a, _b = _v("mass"), _v("cg_a"), _v("cg_b")
_Fzf, _Fzr = _m * 9.81 * _b / (_a + _b), _m * 9.81 * _a / (_a + _b)
_Df, _Dr = _Fzf * _v("tire_mu_f"), _Fzr * _v("tire_mu_r")
_Bf, _Br = _v("tire_stiffness_f") / (1.3 * _Df), _v("tire_stiffness_r") / (1.3 * _Dr)
GOLF_THUNDERHILL = VehicleConfig(
    name="GOLF_THUNDERHILL", m=_m, lf=_a, lr=_b, dt=0.04, hist=8, lon_input="force",
    brake_front_share=_v("brake_front_share"), axle_fmax0=(_Df, _Dr),
    phys=[
        ("Bf", _Bf, 2.0, 30.0), ("Cf", 1.3, 0.5, 2.0), ("Df", _Df, 100.0, 25000.0), ("Ef", -0.5, -2.0, 0.0),
        ("Br", _Br, 2.0, 30.0), ("Cr", 1.3, 0.5, 2.0), ("Dr", _Dr, 100.0, 25000.0), ("Er", -0.5, -2.0, 0.0),
        ("Cr0", _v("resist_cr0"), 0.0, 2000.0), ("Cr2", _v("resist_cr2"), 0.0, 3.0), ("Iz", _v("yaw_inertia"), 800.0, 4000.0),
        ("kb", 0.0, -1.5, 1.5),   # 뱅크 중력 성분 이득 (Golf 데이터 bank_rad 사용)
        ("ds", 0.0, -1.0, 1.0)],
    note="lon = Fxf (N). 200 Hz → 10 Hz 저역 통과 → 8개마다 1개 (dt 0.04). 차량 값은 deep_dynamics/golf/specs.json",
)

# [11/19 팀 결정] 경사 입력 (우리 방식): 같은 차량 값 + use_grade. 학습 데이터 u = [Fxf, delta, bank=0, grade]
#   (data.load_run(road="grade"): 경사는 트랙 지도 experiments/golf/results/tracks/thunderhill_boundary.npz 의 grade,
#    좌우 기울기는 0 — 데이터 bank_rad 열은 센서와 안 맞고 시뮬레이터·MPPI 도 평평한 길이므로). 물리 식 dvx 에 −g·sin(경사)
GOLF_THUNDERHILL_G = replace(GOLF_THUNDERHILL, name="GOLF_THUNDERHILL_G", use_grade=True,
                             note=GOLF_THUNDERHILL.note + " + 경사 채널 (use_grade, bank=0)")

ALL = {c.name: c for c in (AV21_IAC, GTI_THUNDERHILL, GOLF_THUNDERHILL, GOLF_THUNDERHILL_G)}


def road_of(cfg):
    """설정에 맞는 데이터 노면 열: 'grade' ([bank=0, grade]) 또는 'bank' (이전, 데이터 bank_rad)."""
    return "grade" if getattr(cfg, "use_grade", False) else "bank"


def cfg_of_model(pt_path, default=GOLF_THUNDERHILL):
    """학습한 모델(.pt) 옆 .json 의 cfg 이름으로 설정을 고른다 (run_train 이 저장). 없으면 default."""
    import json
    import os
    js = os.path.splitext(pt_path)[0] + ".json"
    if os.path.exists(js):
        return ALL.get(json.load(open(js)).get("cfg"), default)
    return default
