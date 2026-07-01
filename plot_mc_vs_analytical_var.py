"""
Compare analytical sigma2_obs_hist against MC estimates obtained by sampling
the PC depth expansion.

Usage:
    python3 plot_mc_vs_analytical_var.py          # default N_obs=50
    python3 plot_mc_vs_analytical_var.py 20 50 100
"""
import sys
import math
import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial.hermite_e import hermeval

# ── load saved PC coefficients ────────────────────────────────────────────────
data = np.load("depth_pc_coeffs.npz")
depth_pc      = data["depth_pc"]        # (time_steps, P)  d_k[t] = U_k @ g
multi_idx     = data["multi_idx"]       # (P, N_KL)
h_obs_hist    = data["h_obs_hist"]      # (time_steps,)
sigma2_anal   = data["sigma2_obs_hist"] # (time_steps,)
N_KL          = int(data["N_KL"])
P             = int(data["P"])
time_steps    = int(data["time_steps"])

# ── Hermite basis (same as inf_layered_vap_exp.py) ───────────────────────────
def eval_psi(xi, alpha):
    val = 1.0
    for m, p in enumerate(alpha):
        coeff = np.zeros(p + 1); coeff[-1] = 1.0
        val *= hermeval(xi[m], coeff) / math.sqrt(math.factorial(p))
    return val

def sample_depth_trajectory(xi):
    """Evaluate depth at all timesteps for one KL realisation xi."""
    psi = np.array([eval_psi(xi, tuple(alpha)) for alpha in multi_idx])  # (P,)
    return depth_pc @ psi  # (time_steps,)

# ── MC sampling ───────────────────────────────────────────────────────────────
def mc_sigma2(n_obs, seed=0):
    rng = np.random.default_rng(seed)
    samples = np.zeros((time_steps, n_obs))
    for i in range(n_obs):
        xi = rng.standard_normal(N_KL)
        samples[:, i] = sample_depth_trajectory(xi)
    return samples.var(axis=1, ddof=1)

# ── parse args ────────────────────────────────────────────────────────────────
n_obs_list = [int(x) for x in sys.argv[1:]] if len(sys.argv) > 1 else [50]

# ── plot ──────────────────────────────────────────────────────────────────────
t_axis = np.arange(time_steps)

fig, axes = plt.subplots(1, 2, figsize=(11, 4))

colors = plt.cm.tab10.colors

# left: full trajectory
axes[0].plot(t_axis, sigma2_anal * 1e6, 'k-', lw=2, label="Analytical", zorder=10)
for idx, n in enumerate(n_obs_list):
    s2 = mc_sigma2(n)
    axes[0].plot(t_axis, s2 * 1e6, '--', color=colors[idx],
                 lw=1.2, alpha=0.8, label=f"MC $N={n}$")

axes[0].set_xlabel("Timestep")
axes[0].set_ylabel(r"$\sigma^2_{\mathrm{depth}}$ [$\times 10^{-6}$ m$^2$]")
axes[0].set_title("Depth variance: analytical vs MC")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# right: relative error |MC - analytical| / analytical  (skip t=0 where sigma2=0)
mask = sigma2_anal > 0
for idx, n in enumerate(n_obs_list):
    s2 = mc_sigma2(n)
    rel_err = np.abs(s2[mask] - sigma2_anal[mask]) / sigma2_anal[mask]
    axes[1].plot(t_axis[mask], rel_err, color=colors[idx],
                 lw=1.2, alpha=0.8, label=f"$N={n}$")

axes[1].set_xlabel("Timestep")
axes[1].set_ylabel("Relative error")
axes[1].set_title("MC relative error vs analytical")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
out = "mc_vs_analytical_var.png"
plt.savefig(out, dpi=150, bbox_inches='tight')
print(f"Saved {out}")
