import os, sys, json
sys.path[:0]=[os.path.join(os.getcwd(),"physics_for_driving"), os.getcwd()]
from deep_dynamics.golf import sim
import dt_world
w = dt_world.DTWorld(sim.Actuator, delays=dict(sim.load_delays(os.path.join("physics_for_driving","experiments/golf/results/sim/delays.json")) or {}, steer=0.08), lsd=len(sys.argv)>1, main_units=False)
w.reset(0,0,0,25.0)
out=[]
for k in range(100):
    F = 3000 if k<40 else (-4000 if k<60 else 1500); d = 0.05*min(k,30)/30
    out.append(w.step(F, d))
print(json.dumps([list(map(float,o)) for o in out[::10]]))
