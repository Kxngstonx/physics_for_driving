"""pytest 없이 tests/golf 의 test_* 함수를 모두 실행:  python tests/golf/run_all.py"""
import importlib, os, sys, traceback
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
fails = 0
for mod in ["tests.golf.test_ladder_port", "tests.golf.test_golf_pipeline", "tests.golf.test_training_rules", "tests.golf.test_sim_mppi", "tests.golf.test_sim_dt"]:
    m = importlib.import_module(mod)
    for name in sorted(n for n in dir(m) if n.startswith("test_")):
        try:
            getattr(m, name)()
            print("PASS", mod, name)
        except Exception:
            fails += 1
            print("FAIL", mod, name)
            traceback.print_exc()
print("failures:", fails)
sys.exit(1 if fails else 0)
