"""팀원 fit_sim.py --model dt 를 Windows(spawn)에서 실행: 풀 작업자에도 MODEL='dt' 를 넣는다 (저장소 코드는 그대로).
사용: (pfd_dt 에서) python ../run_fit_dt.py"""
import os
import sys

REPO = os.getcwd()
sys.path[:0] = [os.path.join(REPO, "experiments", "golf"), REPO]
import fit_sim as FS  # noqa: E402

_orig_init = FS._init


def init_dt(W):
    import fit_sim
    fit_sim.MODEL = "dt"
    _orig_init(W)


if __name__ == "__main__":
    FS.MODEL = "dt"
    FS._init = init_dt
    FS.main_dt()
