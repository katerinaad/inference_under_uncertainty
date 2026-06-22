import os, subprocess, sys

env = {**os.environ, "FINE_OBS": "1"}
result = subprocess.run(
    [sys.executable, "inf_layered_vap_exp.py"],
    env=env,
    cwd=os.path.dirname(os.path.abspath(__file__)),
)
sys.exit(result.returncode)
