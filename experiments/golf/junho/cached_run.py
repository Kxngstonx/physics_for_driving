"""저장소 코드를 고치지 않고 data._filtered(CSV 읽기 + 필터, 가장 무거운 부분)만 디스크 캐시로 바꿔 끼운 뒤 모듈을 실행.

  python cached_run.py REPO build                                   # 캐시 만들기 (CSV 를 한 번만 읽음) + 동일성 확인
  python cached_run.py REPO deep_dynamics.golf.run_train R4 --seed 0 --roll 25 --out ...
  python cached_run.py REPO deep_dynamics.golf.evaluate --tag roll25
"""
import os
import pickle
import runpy
import sys

REPO = os.path.abspath(sys.argv[1])
sys.path.insert(0, REPO)
os.chdir(REPO)
import numpy as np  # noqa: E402

import deep_dynamics.golf.data as D  # noqa: E402

CACHE = os.path.join(REPO, "deep_dynamics", "data", "golf", "_filtered_cache.pkl")

if sys.argv[2] == "build":
    names = D.available()
    cache = {n: D._filtered(n) for n in names}
    with open(CACHE, "wb") as f:
        pickle.dump(cache, f, protocol=pickle.HIGHEST_PROTOCOL)
    # 동일성 확인: 캐시를 다시 읽어 split_segs 결과가 원래와 같은지
    direct = {v: D.split_segs(names, v_stop=v) for v in (D.V_STOP, D.V_LAUNCH)}
    with open(CACHE, "rb") as f:
        c2 = pickle.load(f)
    D._filtered = lambda name, data_dir=D.DATA_DIR: c2[name]
    D.load_run.cache_clear()
    for v, segs in direct.items():
        again = D.split_segs(names, v_stop=v)
        assert len(again) == len(segs) and all(np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1]) for a, b in zip(again, segs))
        print(f"v_stop={v}: 구간 {len(segs)}개, 샘플 {sum(len(s) for s, _ in segs)} — 캐시와 동일")
    print("cache", CACHE, f"{os.path.getsize(CACHE) / 1e6:.1f} MB, 파일 {len(names)}개")
    sys.exit(0)

with open(CACHE, "rb") as f:
    _C = pickle.load(f)
D._filtered = lambda name, data_dir=D.DATA_DIR: _C[name]
mod = sys.argv[2]
sys.argv = [mod] + sys.argv[3:]
runpy.run_module(mod, run_name="__main__", alter_sys=True)
