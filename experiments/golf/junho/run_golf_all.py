"""physics_for_driving 저장소의 train_all.sh(리눅스 전용)를 Windows 용으로 옮긴 런처.
같은 명령 `python -m deep_dynamics.golf.run_train RUNG --seed S --roll N [--vmin 6] --out OUT` 을 병렬 실행하고,
모드별 18개가 다 모이면 `python -m deep_dynamics.golf.evaluate --tag SUF` 를 실행한다. 끝난 모델(.json)은 건너뜀.
사용: python run_golf_all.py REPO [WORKERS] [MODES...]   (MODES 기본: main loss launch)
"""
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

REPO = os.path.abspath(sys.argv[1])
WORKERS = int(sys.argv[2]) if len(sys.argv) > 2 else 18
MODES = sys.argv[3:] or ["main", "loss", "launch"]
OUT = os.path.join(REPO, "experiments", "golf", "results", "models")
LOG = os.path.join(REPO, "experiments", "golf", "results", "logs")
os.makedirs(OUT, exist_ok=True)
os.makedirs(LOG, exist_ok=True)
SPEC = {"main": (25, [], "roll25"), "loss": (1, [], "roll1"), "launch": (25, ["--vmin", "6"], "roll25_v6")}
ENV = dict(os.environ, OMP_NUM_THREADS=os.environ.get("TRAIN_THREADS", "1"), MKL_NUM_THREADS=os.environ.get("TRAIN_THREADS", "1"), PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
WRAP = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cached_run.py")  # CSV 대신 전처리 캐시를 읽게 함
STATUS = os.path.join(REPO, "experiments", "golf", "results", "run_status.txt")


def say(msg):
    line = time.strftime("%m-%d %H:%M:%S ") + msg
    print(line, flush=True)
    with open(STATUS, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def job(args):
    mode, rung, seed = args
    roll, extra, suf = SPEC[mode]
    tag = f"{rung}_s{seed}_{suf}"
    if os.path.exists(os.path.join(OUT, tag + ".json")):
        return say(f"skip (done) {tag}")
    t0 = time.time()
    with open(os.path.join(LOG, f"train_{tag}.log"), "w", encoding="utf-8") as f:
        p = subprocess.run([sys.executable, WRAP, REPO, "deep_dynamics.golf.run_train", rung, "--seed", str(seed), "--roll", str(roll),
                            *extra, "--out", OUT], cwd=REPO, env=ENV, stdout=f, stderr=subprocess.STDOUT)
    say(f"{'ok' if p.returncode == 0 else 'FAILED'} {tag} ({(time.time() - t0) / 60:.0f} min)")


# 느린 물리 모델을 앞에, 모드 순서대로 하나의 대기열
SEEDS = [int(x) for x in os.environ.get("SEEDS", "0 1 2").split()]  # 예: SEEDS="0" 이면 시드 0만
RUNGS = os.environ.get("RUNGS", "R3 R1 R2 R4 R5 R6").split()  # 예: RUNGS="R1 R6"
jobs = [(m, r, s) for m in MODES for r in RUNGS for s in SEEDS]
say(f"start {len(jobs)} jobs, {WORKERS} workers, modes {MODES}")
evaluated = set()


def maybe_eval():
    for m in MODES:
        suf = SPEC[m][2]
        n = sum(os.path.exists(os.path.join(OUT, f"{r}_s{s}_{suf}.json")) for r in ["R1", "R2", "R3", "R4", "R5", "R6"] for s in (0, 1, 2))
        if n == 18 and m not in evaluated:
            evaluated.add(m)
            with open(os.path.join(LOG, f"eval_{suf}.log"), "w", encoding="utf-8") as f:
                p = subprocess.run([sys.executable, WRAP, REPO, "deep_dynamics.golf.evaluate", "--tag", suf], cwd=REPO, env=ENV,
                                   stdout=f, stderr=subprocess.STDOUT)
            say(f"{'evaluated' if p.returncode == 0 else 'EVAL FAILED'} {suf}")


with ThreadPoolExecutor(WORKERS) as ex:
    futs = [ex.submit(job, j) for j in jobs]
    for fu in futs:
        fu.result()
        maybe_eval()
maybe_eval()
say("ALL DONE")
