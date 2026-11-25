import os, sys, json
REPO=os.getcwd(); sys.path[:0]=[os.path.join(REPO,"experiments","golf"),REPO]
from deep_dynamics.golf import sim
import compare_lsd as CL
_, look = CL.lsd_table_from_train() if len(sys.argv)>1 else (None,None)
w = sim.World(model="dt", delays=dict(sim.load_delays() or {}, steer=0.08), lsd=dict(mode="active") if look else None, lsd_table=look)
w.reset(0,0,0,25.0)
out=[]
for k in range(100):
    F = 3000 if k<40 else (-4000 if k<60 else 1500); d = 0.05*min(k,30)/30
    out.append(w.step(F, d))
print(json.dumps([list(map(float,o)) for o in out[::10]]))
