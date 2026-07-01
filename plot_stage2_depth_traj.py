"""
Plot mean and variance depth trajectories from stage2_depth_log.csv.
Shows: true (h_obs / sigma2_obs), starting (iter=1), and final (last iter).
"""
import numpy as np
import matplotlib.pyplot as plt
from datetime import date

# ── load ──────────────────────────────────────────────────────────────────────
with open("stage2_depth_log.csv") as f:
    lines = f.readlines()

h_obs      = np.array([float(v) for v in lines[0].lstrip("# h_obs,").split(",")])
sigma2_obs = np.array([float(v) for v in lines[1].lstrip("# sigma2_obs,").split(",")])

# parse data rows (skip 2 comment lines + header)
header = lines[2].strip().split(",")
rows   = [list(map(float, l.strip().split(","))) for l in lines[3:] if l.strip()]
data   = np.array(rows)

T   = len(h_obs)
t   = np.arange(T)

n_params = 6  # iter, rho_melt1_surf, rho_melt1_deep, kappa_surf, kappa_deep, y_trans, width → 7 cols before h_pred
offset   = 7  # iter + 6 params

h_start   = data[0,  offset          : offset + T]
var_start = data[0,  offset + T      : offset + 2*T]
h_final   = data[-1, offset          : offset + T]
var_final = data[-1, offset + T      : offset + 2*T]

n_iters = int(data[-1, 0])

# ── plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
fig.suptitle(
    f"Stage 2  |  depth mean & variance trajectories  |  {date.today()}",
    fontsize=11)

# --- mean ---
ax = axes[0]
ax.plot(t, h_obs,     "k-",  lw=2.0, label="true  $h_{obs}$")
ax.plot(t, h_start,   "b--", lw=1.5, label="starting (iter 1)")
ax.plot(t, h_final,   "r-",  lw=1.5, label=f"final  (iter {n_iters})")
ax.set_ylabel("Mean ablation depth  (m)")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# --- variance ---
ax = axes[1]
ax.semilogy(t, sigma2_obs + 1e-30, "k-",  lw=2.0, label=r"true  $\sigma^2_{obs}$")
ax.semilogy(t, var_start  + 1e-30, "b--", lw=1.5, label="starting (iter 1)")
ax.semilogy(t, var_final  + 1e-30, "r-",  lw=1.5, label=f"final  (iter {n_iters})")
ax.set_xlabel("Timestep")
ax.set_ylabel("Variance  (m²)")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3, which="both")

plt.tight_layout()
out = "stage2_depth_traj.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved {out}")
