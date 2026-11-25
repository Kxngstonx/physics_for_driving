"""팀원 compare_lsd.lsd_table_from_train() 의 LSD 작동 비율 표를 json 으로 (dt 시뮬레이터 폐루프용). pfd_dt 에서 실행."""
import json, os, sys
REPO = os.getcwd()
sys.path[:0] = [os.path.join(REPO, "experiments", "golf"), REPO]
import compare_lsd as CL
T, _ = CL.lsd_table_from_train()
json.dump(dict(F_BINS=CL.F_BINS.tolist(), AY_BINS=CL.AY_BINS.tolist(), T=T.tolist()), open(sys.argv[1], "w"), indent=1)
print(T)
