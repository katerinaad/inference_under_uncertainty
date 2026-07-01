import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# ── load data ──────────────────────────────────────────────────────────────
log       = pd.read_csv("stage2_log_3kappa_50KL_new.csv",       comment="#")
depth_log = pd.read_csv("stage2_depth_log_3kappa_50kl_new.csv", comment="#")

n_iter  = len(log)
n_times = sum(1 for c in depth_log.columns if c.startswith("h_pred_t"))

h_cols   = [f"h_pred_t{i}" for i in range(1, n_times)]
h_matrix = depth_log[h_cols].values
t_plot   = np.arange(1, n_times) * 0.075

with open("stage2_depth_log_3kappa_50kl_new.csv") as _f:
    for line in _f:
        if line.startswith("# h_obs"):
            h_obs = np.array([float(x) for x in line.strip().split(",")[1:]])
        if line.startswith("# sigma2_obs"):
            sigma2_obs = np.array([float(x) for x in line.strip().split(",")[1:]])

var_cols   = [f"var_pred_t{i}" for i in range(1, n_times)]
var_matrix = depth_log[var_cols].values

# truth values
kappa_surf_truth = 200.0
kappa_deep_truth = 100.0
y_trans_truth    = 0.189
width_truth      = 0.0021

cmap = cm.plasma

def smooth(arr, w=7):
    return pd.Series(arr).rolling(w, center=True, min_periods=1).mean().values

# iteration sampling: dense early, spread later
early     = np.arange(0, min(10, len(depth_log)))
later     = np.arange(10, len(depth_log), 5)
sel_iters = np.unique(np.concatenate([early, later]))
colors    = cmap(sel_iters / (len(depth_log) - 1))

iters = log["iter"].values

# ── figure 1: depth trajectory convergence ────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 4.5))

for color, idx in zip(colors, sel_iters):
    ax.plot(t_plot, 0.21 - smooth(h_matrix[idx]), color=color, lw=1.0, alpha=0.8)

ax.plot(t_plot, 0.21 - smooth(h_matrix[0]),  color="black", lw=1.5, ls="--", zorder=5, label="Initial")
ax.plot(t_plot, 0.21 - smooth(h_matrix[-1]), color="blue",  lw=1.5, ls="--", zorder=5, label="Final")
ax.plot(t_plot, 0.21 - smooth(h_obs[1:]),    color="red",   lw=1.5, ls="--", zorder=5, label="True")

sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=1, vmax=n_iter))
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label("Iteration", fontsize=10)

ax.legend(fontsize=9, loc="upper left")
ax.set_xlabel("Time (s)", fontsize=11)
ax.set_ylabel("Depth (m)", fontsize=11)
ax.set_title("Depth trajectory convergence", fontsize=12)
ax.tick_params(labelsize=9)

plt.tight_layout()
plt.savefig("stage2_depth_convergence.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved stage2_depth_convergence.png")

# ── figure 2: parameter convergence ──────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), constrained_layout=True)
fig.suptitle("Matérn correlation hyperparameter convergence", fontsize=16)

params = [
    ("kappa_surf",  kappa_surf_truth, r"$\kappa_\mathrm{surf}$"),
    ("kappa_deep",  kappa_deep_truth, r"$\kappa_\mathrm{deep}$"),
    ("y_trans",     y_trans_truth,    r"$y_\mathrm{trans}$ (m)"),
]

sc_norm = plt.Normalize(vmin=iters[0], vmax=iters[-1])

for ax, (col, truth, label) in zip(axes.flat, params):
    vals = log[col].values
    sc = ax.scatter(iters, vals, c=iters, cmap="plasma", s=25, norm=sc_norm, zorder=3)
    ax.plot(iters, vals, color="grey", lw=0.8, alpha=0.5, zorder=2)
    ax.axhline(truth, color="red", lw=1.5, ls="--", zorder=4, label="Truth")
    ax.scatter(iters[0],  vals[0],  marker="o", s=60, color="royalblue", zorder=5, label="Initial")
    ax.scatter(iters[-1], vals[-1], marker="o", s=60, color=cmap(1.0),   zorder=5, label="Final")
    ax.set_ylabel(label, fontsize=11, rotation=0, ha="right", labelpad=10)
    ax.set_xlabel("Iteration", fontsize=10)
    ax.legend(fontsize=8, loc="upper right")
    ax.tick_params(labelsize=9)

fig.colorbar(plt.cm.ScalarMappable(cmap="plasma", norm=sc_norm),
             ax=axes, fraction=0.02, pad=0.04, label="Iteration")
plt.savefig("stage2_param_convergence.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved stage2_param_convergence.png")

# ── figure 3: depth residuals ─────────────────────────────────────────────
t_act = 37
t_res = t_plot[t_act - 1:]

fig, ax = plt.subplots(figsize=(6, 4.5))

for color, idx in zip(colors, sel_iters):
    resid = smooth(h_matrix[idx][t_act - 1:]) - h_obs[t_act:]
    ax.plot(t_res, resid, color=color, lw=1.0, alpha=0.8)

ax.plot(t_res, smooth(h_matrix[0][t_act - 1:])  - h_obs[t_act:], color="black", lw=1.5, ls="--", zorder=5, label="Initial")
ax.plot(t_res, smooth(h_matrix[-1][t_act - 1:]) - h_obs[t_act:], color="blue",  lw=1.5, ls="--", zorder=5, label="Final")
ax.axhline(0, color="k", lw=1.0, ls=":")

sm3 = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=1, vmax=n_iter))
sm3.set_array([])
fig.colorbar(sm3, ax=ax, fraction=0.046, pad=0.04, label="Iteration")

ax.legend(fontsize=9, loc="best")
ax.set_xlabel("Time (s)", fontsize=11)
ax.set_ylabel(r"$h_\mathrm{pred} - h_\mathrm{obs}$ (m)", fontsize=11)
ax.set_title("Depth residuals", fontsize=12)
ax.tick_params(labelsize=9)

plt.tight_layout()
plt.savefig("stage2_residuals.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved stage2_residuals.png")

# ── figure 4: variance trajectory convergence ─────────────────────────────
fig, ax = plt.subplots(figsize=(6, 4.5))

def smooth_var(arr, w=40):
    return pd.Series(arr).rolling(w, center=True, min_periods=1).mean().values

for color, idx in zip(colors, sel_iters):
    ax.plot(t_plot, smooth_var(var_matrix[idx]), color=color, lw=1.0, alpha=0.3)

ax.plot(t_plot, smooth_var(var_matrix[0]),  color="black", lw=1.5, ls="--", zorder=5, label="Initial")
ax.plot(t_plot, smooth_var(var_matrix[-1]), color="blue",  lw=1.5, ls="--", zorder=5, label="Final")
ax.plot(t_plot, smooth_var(sigma2_obs[1:]), color="red",   lw=1.5, ls="--", zorder=5, label="True")

sm4 = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=1, vmax=n_iter))
sm4.set_array([])
fig.colorbar(sm4, ax=ax, fraction=0.046, pad=0.04, label="Iteration")

ax.legend(fontsize=9, loc="upper left")
ax.set_xlabel("Time (s)", fontsize=11)
ax.set_ylabel(r"Variance (m$^2$)", fontsize=11)
ax.set_title("Variance trajectory convergence", fontsize=12)
ax.tick_params(labelsize=9)

plt.tight_layout()
plt.savefig("stage2_var_convergence.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved stage2_var_convergence.png")
