"""학습 파일(팀 분할 train)에서 속도별 최대 가속·최대 감속 포락선 → realsim_out/accel_env.json
ax = dvx/dt - vy·r (차체 앞방향 가속). 속도 2 m/s 구간마다 가속은 상위 0.5%, 감속은 하위 0.5% 값.  검증·시험 파일은 쓰지 않음."""
import json, os, pickle, sys
import numpy as np
REPO = os.path.abspath(sys.argv[1]); sys.path.insert(0, REPO); os.chdir(REPO)
import deep_dynamics.golf.data as D
C = pickle.load(open(os.path.join(REPO, "deep_dynamics/data/golf/_filtered_cache.pkl"), "rb"))
D._filtered = lambda name, data_dir=D.DATA_DIR: C[name]
V, A = [], []
for n in D.get_split()["train"]:
    for sg in D.load_run(n):
        s, t = sg["s"], sg["t"]
        ax = np.gradient(s[:, 0], t) - s[:, 1] * s[:, 2]
        V.append(s[:, 0]); A.append(ax)
V, A = np.concatenate(V), np.concatenate(A)
edges = np.arange(4, 42, 2.0)
rows = []
for lo, hi in zip(edges[:-1], edges[1:]):
    m = (V >= lo) & (V < hi)
    if m.sum() < 200:
        continue
    rows.append(dict(v=(lo + hi) / 2, n=int(m.sum()), acc=float(np.percentile(A[m], 99.5)), brk=float(np.percentile(A[m], 0.5))))
    print(f"v {lo:4.0f}-{hi:4.0f}  n {m.sum():6d}  가속 {rows[-1]['acc']:5.2f}  감속 {rows[-1]['brk']:6.2f}")
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "realsim_out", "accel_env.json")
json.dump(rows, open(out, "w"), indent=1); print("saved", out)
