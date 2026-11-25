"""estimate_specs.py 의 golf() 만 실행 (참조 차량 GTI·AV-21 데이터가 없어서 방법 검증 부분은 건너뜀). 결과 형식은 같다."""
import json, os, sys
REPO = os.path.abspath(sys.argv[1])
sys.path.insert(0, os.path.join(REPO, "experiments", "golf")); sys.path.insert(0, REPO)
import estimate_specs as E
g = E.golf()
os.makedirs(E.OUT, exist_ok=True)
json.dump(dict(GOLF=g, note="GOLF only: reference vehicles (GTI, AV-21) not available here"), open(os.path.join(E.OUT, "specs.json"), "w"), indent=1, default=float)
print(json.dumps(g["final"], ensure_ascii=False))
