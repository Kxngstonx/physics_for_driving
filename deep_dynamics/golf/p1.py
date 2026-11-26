"""P1 자전거 시뮬레이터 (11/26 결정: 정답 세계로 P2 대신 P1 을 쓴다).

  from deep_dynamics.golf import p1
  w = p1.P1World.from_fit()        # experiments/golf/results/sim/p1_fit.json (python experiments/golf/fit_p2.py --model p1)

P2 (p2.py) 와 같은 코드에서 두 가지만 끈다:
  - 앞뒤 하중 이동 없음: 축 하중 = 정지 상태 값 (m·g·b/L, m·g·a/L). 무게중심 높이 h 를 쓰지 않음
  - 마찰원 없음: 앞뒤로 힘을 써도 옆 힘 최대치 = μ·정지 하중 그대로
같은 점: 자전거 모델, Pacejka 옆 힘 (최대 = μ·Fz), 굴림·공기 저항, 종경사, RK4 0.01 s, 지연·구동력 상한·제동 하한 (sim.Actuator 등).
차량 값: m 1496.9 kg, 축거 2.63 m, 앞 제동 몫 0.75 고정 (configs.GOLF_THUNDERHILL), 힘 정의 Fx = 0.699·구동 − 브레이크 (data.DRIVE_EFF).
         나머지(a, Iz, 강성, 형상, μ, 저항, ds)는 D7 학습 파일로 맞춤.
팀원 sim-recheck 의 st 모델(4바퀴 식에서 무게 쏠림을 끈 것, 데이터 Fiala 축 타이어)과 같은 계열이다. 제원은 다름 (팀원 m 1777, η 0.852).
"""
import os
from dataclasses import dataclass

from . import p2

FIT = os.path.join(os.path.dirname(p2.FIT), "p1_fit.json")


@dataclass
class P1Params(p2.P2Params):
    p1: bool = True


def load_fit(path=FIT):
    assert os.path.exists(path), f"{path} 없음: python experiments/golf/fit_p2.py --model p1 를 먼저 실행"
    import json
    d = json.load(open(path))
    return P1Params(**{k: v for k, v in d["params"].items() if k in P1Params.__dataclass_fields__ and k != "p1"})


class P1World(p2.P2World):
    def __init__(self, params=None, **kw):
        super().__init__(params=params or P1Params(), **kw)

    @classmethod
    def from_fit(cls, path=FIT, **kw):
        if "delays" not in kw:
            from . import sim
            kw["delays"] = sim.load_delays() or p2.DEFAULT_DELAYS
        return cls(params=load_fit(path), **kw)
