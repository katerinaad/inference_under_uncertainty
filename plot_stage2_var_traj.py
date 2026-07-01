"""
Focused variance trajectory plot for Stage 2.
Panel 1: all iterations' var_pred vs true sigma2_obs (log scale)
Panel 2: relative error  (var_pred - sigma2_obs) / sigma2_obs  per iteration
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from datetime import date

# ── load ──────────────────────────────────────────────────────────────────────
with open("stage2_depth_log.csv") as f:
    lines = f.readlines()

sigma2_obs = np.array([float(v) for v in lines[1].lstrip("# sigma2_obs,").split(",")])
rows = [list(map(float, l.strip().split(","))) for l in lines[3:] if l.strip()]
data = np.array(rows)

T      = len(sigma2_obs)
t      = np.arange(T)
offset = 7  # iter + 6 params (rho_surf, rho_deep, kappa_surf, kappa_deep, y_trans, width)

iters    = data[:, 0].astype(int)
var_pred = data[:, offset + T : offset + 2*T]   # (N_iters, T)

N = len(iters)
cmap  = cm.plasma
norm  = plt.Normalize(iters.min(), iters.max())
colors = [cmap(norm(i)) for i in iters]

# ── figure ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
fig.suptitle(
    f"Variance trajectory  |  {date.today()}",
    fontsize=11)

# --- panel 1: log-scale trajectories ---
ax = axes[0]
for k in range(N):
    if k%5==0:
        ax.semilogy(t, var_pred[k] + 1e-35, color=colors[k],
                    lw=1.4, alpha=0.85, label=f"iter {iters[k]}")
ax.semilogy(t, sigma2_obs + 1e-35, "k-", lw=2.5, zorder=10, label=r"true $\sigma^2_{obs}$")

sm = cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, pad=0.01)
cbar.set_label("iteration", fontsize=10.5)

ax.set_ylabel(r"Variance  (m²)")
ax.legend(fontsize=10.5, ncol=2, loc="upper left")
ax.grid(True, alpha=0.3, which="both")

# --- panel 2: relative error ---
ax = axes[1]
for k in range(N):
    if k %2==0:
        rel = (var_pred[k] - sigma2_obs) / (sigma2_obs + 1e-35)
        ax.plot(t, rel, color=colors[k], lw=1.4, alpha=0.85, label=f"iter {iters[k]}")

ax.axhline(0, color="k", lw=1.5, ls="--")
ax.set_xlabel("Timestep")
ax.set_ylabel(r"Relative error")
ax.legend(fontsize=10.5, ncol=2, loc="upper left")
ax.grid(True, alpha=0.3)

plt.tight_layout()
out = "stage2_var_traj.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved {out}")
