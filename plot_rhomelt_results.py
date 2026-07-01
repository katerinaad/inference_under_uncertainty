import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm

log       = pd.read_csv("stage2_log_rhomelt_s.csv",       comment="#")
depth_log = pd.read_csv("stage2_depth_log_rhomelt_s.csv", comment="#")

n_iter  = len(log)
n_times = sum(1 for c in depth_log.columns if c.startswith("h_pred_t"))

h_cols   = [f"h_pred_t{i}" for i in range(1, n_times)]
var_cols = [f"var_pred_t{i}" for i in range(1, n_times)]
h_matrix   = depth_log[h_cols].values
var_matrix = depth_log[var_cols].values
t_plot = np.arange(1, n_times) * 0.075

with open("stage2_depth_log_rhomelt_s.csv") as _f:
    for line in _f:
        if line.startswith("# h_obs"):
            h_obs = np.array([float(x) for x in line.strip().split(",")[1:]])
        if line.startswith("# sigma2_obs"):
            sigma2_obs = np.array([float(x) for x in line.strip().split(",")[1:]])

cmap = cm.plasma

def smooth(arr, w=7):
    return pd.Series(arr).rolling(w, center=True, min_periods=1).mean().values

def smooth_var(arr, w=40):
    return pd.Series(arr).rolling(w, center=True, min_periods=1).mean().values

early     = np.arange(0, min(10, n_iter))
later     = np.arange(10, n_iter, 3)
sel_iters = np.unique(np.concatenate([early, later]))
colors    = cmap(sel_iters / (n_iter - 1))

iters = log["iter"].values

# ── figure 1: variance trajectory convergence ─────────────────────────────
fig, ax = plt.subplots(figsize=(6, 4.5))

for color, idx in zip(colors, sel_iters):
    ax.plot(t_plot, smooth_var(var_matrix[idx]), color=color, lw=1.0, alpha=0.3)

ax.plot(t_plot, smooth_var(var_matrix[0]),    color="black", lw=1.5, ls="--", zorder=5, label="Initial")
ax.plot(t_plot, smooth_var(var_matrix[-1]),   color="blue",  lw=1.5, ls="--", zorder=5, label="Final")
ax.plot(t_plot, smooth_var(sigma2_obs[1:]),   color="red",   lw=1.5, ls="--", zorder=5, label="True")

sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=1, vmax=n_iter))
sm.set_array([])
fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04, label="Iteration")

ax.legend(fontsize=9, loc="upper left")
ax.set_xlabel("Time (s)", fontsize=11)
ax.set_ylabel(r"Variance (m$^2$)", fontsize=11)
ax.set_title(r"Variance trajectory convergence ($L_{f,1}$ inference)", fontsize=11)
ax.tick_params(labelsize=9)

plt.tight_layout()
plt.savefig("rhomelt_var_convergence.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved rhomelt_var_convergence.png")

# ── figure 2: variance residuals ──────────────────────────────────────────
t_act = 37
t_res = t_plot[t_act - 1:]

fig, ax = plt.subplots(figsize=(6, 4.5))

for color, idx in zip(colors, sel_iters):
    resid = smooth_var(var_matrix[idx][t_act - 1:]) - sigma2_obs[t_act:]
    ax.plot(t_res, resid, color=color, lw=1.0, alpha=0.3)

ax.plot(t_res, smooth_var(var_matrix[0][t_act - 1:])  - sigma2_obs[t_act:],
        color="black", lw=1.5, ls="--", zorder=5, label="Initial")
ax.plot(t_res, smooth_var(var_matrix[-1][t_act - 1:]) - sigma2_obs[t_act:],
        color="blue",  lw=1.5, ls="--", zorder=5, label="Final")
ax.axhline(0, color="k", lw=1.0, ls=":")

sm2 = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=1, vmax=n_iter))
sm2.set_array([])
fig.colorbar(sm2, ax=ax, fraction=0.046, pad=0.04, label="Iteration")

ax.legend(fontsize=9, loc="best")
ax.set_xlabel("Time (s)", fontsize=11)
ax.set_ylabel(r"$\hat{\sigma}^2_\mathrm{pred} - \sigma^2_\mathrm{obs}$ (m$^2$)", fontsize=11)
ax.set_title(r"Variance residuals ($L_{f,1}$ inference)", fontsize=11)
ax.tick_params(labelsize=9)

plt.tight_layout()
plt.savefig("rhomelt_var_residuals.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved rhomelt_var_residuals.png")
