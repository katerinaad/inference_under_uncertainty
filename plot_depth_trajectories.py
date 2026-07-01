"""
Plot depth trajectories from stage1_log.npz.
Each iteration's mean depth trajectory is shown as a line, coloured by log10(J).
The observed depth h_obs_hist is overlaid in black.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from datetime import date

# ── load ─────────────────────────────────────────────────────────────────────
data = np.load("stage1_log.npz")

iters      = data["iters"]          # (N,)
J          = data["J"]              # (N,)
rho        = data["rho_vap0"]       # (N,)
f0v        = data["f0_v"]           # (N,)
depth_traj = data["depth_traj"]     # (N, T)
h_obs      = data["h_obs_hist"]     # (T,)
rho_truth  = float(data["rho_truth"])
f0_truth   = float(data["f0_truth"])

N, T = depth_traj.shape
t    = np.arange(T)

# ── colour map: log10(J) low → bright, high → grey ───────────────────────────
logJ  = np.log10(np.clip(J, 1e-6, None))
norm  = plt.Normalize(logJ.min(), logJ.max())
cmap  = cm.plasma_r

# drop outlier bounds-probe iterations (keep only J ≤ 10× best J) for focused plots
J_threshold = J.min() * 10
keep = J <= J_threshold
norm_focused = plt.Normalize(logJ[keep].min(), logJ[keep].max())

# ── Figure 1: all trajectories + obs ─────────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 5))
fig.suptitle(
    f"Mean depth trajectories per function evaluation\n"
    f"Stage 1  |  truth: rho_vap0={rho_truth:.3e}  f0={f0_truth:.3f}  |  {date.today()}",
    fontsize=10)

for i in range(N):
    ax.plot(t, depth_traj[i], color=cmap(norm(logJ[i])),
            lw=0.8, alpha=0.7)

ax.plot(t, h_obs, 'k-', lw=2.0, label="h_obs (truth)")

sm   = cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, pad=0.01)
cbar.set_label("log₁₀(J)", fontsize=9)

ax.set_xlabel("Timestep")
ax.set_ylabel("Mean ablation depth  (m)")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("depth_trajectories_all.png", dpi=150, bbox_inches="tight")
print("Saved: depth_trajectories_all.png")
plt.close()

# ── Figure 1b: converging iterations only (outliers removed) ─────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle(
    f"Converging iterations (outliers removed, J ≤ {J_threshold:.2e})\n"
    f"Stage 1  |  truth: rho_vap0={rho_truth:.3e}  f0={f0_truth:.3f}  |  {date.today()}",
    fontsize=10)

ax = axes[0]
for i in np.where(keep)[0]:
    ax.plot(t, depth_traj[i], color=cmap(norm_focused(logJ[i])), lw=1.0, alpha=0.8)
ax.plot(t, h_obs, 'k-', lw=2.0, label="h_obs (truth)")
sm2 = cm.ScalarMappable(cmap=cmap, norm=norm_focused)
sm2.set_array([])
fig.colorbar(sm2, ax=ax, pad=0.01).set_label("log₁₀(J)", fontsize=9)
ax.set_xlabel("Timestep"); ax.set_ylabel("Mean ablation depth  (m)")
ax.set_title("Depth trajectories"); ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

ax = axes[1]
for i in np.where(keep)[0]:
    ax.plot(t, depth_traj[i] - h_obs, color=cmap(norm_focused(logJ[i])), lw=1.0, alpha=0.8)
ax.axhline(0, color='k', lw=1.5, ls='--')
sm3 = cm.ScalarMappable(cmap=cmap, norm=norm_focused)
sm3.set_array([])
fig.colorbar(sm3, ax=ax, pad=0.01).set_label("log₁₀(J)", fontsize=9)
ax.set_xlabel("Timestep"); ax.set_ylabel("depth − h_obs  (m)")
ax.set_title("Residuals"); ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("depth_trajectories_focused.png", dpi=150, bbox_inches="tight")
print("Saved: depth_trajectories_focused.png")
plt.close()

# ── Figure 2: first, last and obs only — cleaner view ─────────────────────────
fig, ax = plt.subplots(figsize=(11, 5))
fig.suptitle(
    f"Depth trajectory: x0, final, obs  |  {date.today()}", fontsize=10)

ax.plot(t, depth_traj[0],  color='#d62728', lw=1.2,
        label=f"iter 1   J={J[0]:.2e}  rho={rho[0]:.3e}  f0={f0v[0]:.3f}")
ax.plot(t, depth_traj[-1], color='#1f77b4', lw=1.2,
        label=f"iter {int(iters[-1])}  J={J[-1]:.2e}  rho={rho[-1]:.3e}  f0={f0v[-1]:.3f}")
ax.plot(t, h_obs,          'k-', lw=2.0, label="h_obs (truth)")

ax.set_xlabel("Timestep")
ax.set_ylabel("Mean ablation depth  (m)")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("depth_trajectories_first_last.png", dpi=150, bbox_inches="tight")
print("Saved: depth_trajectories_first_last.png")
plt.close()

# ── Figure 3: residual (depth - h_obs) per iteration ─────────────────────────
fig, ax = plt.subplots(figsize=(11, 5))
fig.suptitle(f"Depth residual (model − obs)  |  {date.today()}", fontsize=10)

for i in range(N):
    residual = depth_traj[i] - h_obs
    ax.plot(t, residual, color=cmap(norm(logJ[i])), lw=0.8, alpha=0.7)

ax.axhline(0, color='k', lw=1.5, ls='--')

sm = cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, pad=0.01)
cbar.set_label("log₁₀(J)", fontsize=9)

ax.set_xlabel("Timestep")
ax.set_ylabel("depth − h_obs  (m)")
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("depth_residuals.png", dpi=150, bbox_inches="tight")
print("Saved: depth_residuals.png")
plt.close()

# ── print summary ─────────────────────────────────────────────────────────────
print(f"\n{N} function evaluations loaded.")
print(f"  J:       {J[0]:.3e}  →  {J[-1]:.3e}")
print(f"  rho_vap0: {rho[0]:.4e}  →  {rho[-1]:.4e}  (truth {rho_truth:.4e})")
print(f"  f0_v:    {f0v[0]:.4f}  →  {f0v[-1]:.4f}  (truth {f0_truth:.4f})")
