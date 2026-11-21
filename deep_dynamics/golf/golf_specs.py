"""Golf 차량 계수 한 곳: deep_dynamics/golf/specs.json (11/21 사용자 결정).

만드는 곳: experiments/golf/estimate_specs.py (총괄). 모델(configs)·데이터(data)·시뮬레이터(sim)가 모두 여기서 읽는다.
항목마다 {"value", "source": published|data|assumed, "basis": 근거, ...}. 출처 규칙은 docs/experiment_plan_golf.md [결정 11/21].
입력 설정이므로 results/ 와 달리 git 에 커밋한다.
"""
import functools
import json
import os

SPECS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "specs.json")


@functools.lru_cache(maxsize=None)
def load(path=SPECS_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} 없음: python experiments/golf/estimate_specs.py 로 만든다")
    return json.load(open(path, encoding="utf-8"))


def value(name):
    return load()["items"][name]["value"]
