import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# ── load data ──────────────────────────────────────────────────────────────
log = pd.read_csv("stage1_log.csv", comment="#")
depth_log = pd.read_csv("stage1_depth_log.csv", comment="#")

n_iter  = len(log)
n_times = sum(1 for c in depth_log.columns if c.startswith("h_pred_t"))

h_cols   = [f"h_pred_t{i}" for i in range(1, n_times)]   # skip t0 (always 0)
h_matrix = depth_log[h_cols].values                       # shape (n_iter, n_times-1)
t_plot   = np.arange(1, n_times) * 0.075

cmap = cm.plasma

# parse h_obs from comment line
with open("stage1_depth_log.csv") as _f:
    h_obs = np.array([float(x) for x in _f.readline().strip().split(",")[1:]])

# ── figure 1: depth mean trajectory convergence ───────────────────────────
fig, ax = plt.subplots(figsize=(6, 4.5))

# dense sampling in early iterations (where most change happens) + spread later
early = np.arange(0, min(10, len(depth_log)))
later = np.arange(10, len(depth_log), 5)
sel_iters = np.unique(np.concatenate([early, later]))
colors = cmap(sel_iters / (len(depth_log) - 1))

def smooth(arr, w=7):
    return pd.Series(arr).rolling(w, center=True, min_periods=1).mean().values

for color, idx in zip(colors, sel_iters):
    ax.plot(t_plot, 0.21 - smooth(h_matrix[idx]), color=color, lw=1.0, alpha=0.8)

# reference lines (drawn on top)
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
plt.savefig("stage1_depth_convergence.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved stage1_depth_convergence.png")

# ── figure 2: 2-D parameter convergence ──────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 4.5))

rho   = log["rho_vap0"].values
f0    = log["f0_m"].values
iters = log["iter"].values

sc = ax.scatter(rho / 1e9,1- f0, c=iters, cmap="plasma", s=30, zorder=3,
                norm=plt.Normalize(vmin=iters[0], vmax=iters[-1]))
ax.plot(rho / 1e9, 1-f0, color="grey", lw=0.8, alpha=0.5, zorder=2)

ax.scatter(rho[0] / 1e9,  1-f0[0],  marker="o", s=80, color="royalblue", zorder=4, label="Initial")
ax.scatter(rho[-1] / 1e9, 1-f0[-1], marker="o", s=80, color=cmap(1.0),   zorder=4, label="Final")
ax.scatter(1.3664, 1-0.8, marker="o", s=120, facecolors="none", edgecolors="black", linewidths=1.5, zorder=6, label="Truth")

cbar2 = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
cbar2.set_label("Iteration", fontsize=10)

ax.set_xlabel(r"Mean latent heat of vaporisation $L_v$ ($\times 10^5$ kJ kg$^{-1}$)", fontsize=11)
ax.set_ylabel(r"Mean reflectivity of melted material $\rho$", fontsize=11)
ax.set_title(r"Recovery of mean $L_v$ and $\rho$", fontsize=12)
ax.legend(fontsize=9, loc="best")
ax.tick_params(labelsize=9)

plt.tight_layout()
plt.savefig("stage1_param_convergence.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved stage1_param_convergence.png")

# ── figure 3: depth residuals ─────────────────────────────────────────────
# skip early flat region (ablation not yet active)
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
cbar3 = fig.colorbar(sm3, ax=ax, fraction=0.046, pad=0.04)
cbar3.set_label("Iteration", fontsize=10)

ax.legend(fontsize=9, loc="best")
ax.set_xlabel("Time (s)", fontsize=11)
ax.set_ylabel("$h_{\\mathrm{pred}} - h_{\\mathrm{obs}}$ (m)", fontsize=11)
ax.set_title("Depth residuals", fontsize=12)
ax.tick_params(labelsize=9)

plt.tight_layout()
plt.savefig("stage1_residuals.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved stage1_residuals.png")
