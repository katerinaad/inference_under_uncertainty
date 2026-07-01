"""
Run Stage 2 MC ablation: n_obs in {10, 50, 100}, 2 seeds each, 50 iterations.
Logs are written to stage2_log_mc_n{N}_s{S}.csv as each run progresses.

Usage:
    python3 run_mc_ablation.py
"""
import subprocess
import sys
import itertools
import os

N_OBS_LIST = [20, 50, 100]
SEEDS      = [0, 1]
VAR_ITERS  = 20

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MAIN     = os.path.join(BASE_DIR, "inf_layered_vap_exp.py")

runs = list(itertools.product(N_OBS_LIST, SEEDS))
print(f"Running {len(runs)} Stage 2 ablation jobs "
      f"(n_obs={N_OBS_LIST}, seeds={SEEDS}, iters={VAR_ITERS})\n")

for n_obs, seed in runs:
    tag  = f"mc_n{n_obs}_s{seed}"
    log  = os.path.join(BASE_DIR, f"stage2_log_{tag}.csv")
    cmd  = [
        sys.executable, MAIN,
        f"--mc_obs={n_obs}",
        f"--seed={seed}",
        f"--var_iters={VAR_ITERS}",
        f"--tag={tag}",
    ]
    print(f"── n_obs={n_obs:>3d}  seed={seed}  → stage2_log_{tag}.csv")
    result = subprocess.run(cmd, cwd=BASE_DIR)
    if result.returncode != 0:
        print(f"   WARNING: run exited with code {result.returncode}")

print("\nAll runs complete.")
print("Logs written to stage2_log_mc_n*.csv")
