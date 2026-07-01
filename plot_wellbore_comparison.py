"""
Side-by-side wellbore plots: truth (VAP_obs/MELT_obs) vs starting point (VAP/MELT).

Runs both forward solves and calls _fig8_wellbore_contours for each, then
stitches the two figures into one side-by-side PNG.
"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from datetime import date

# ── import everything from the main script ────────────────────────────────────
# inf_layered_vap_exp.py is a script, not a module, but all the names we need
# (run_forward, VAP_obs, VAP, MELT_obs, MELT, SOLID_obs, U0, multi_idx,
#  eval_psi, Nx, Ny, Lx, Ly, num_nodes, P, N_KL, T_abl, dt, ell,
#  theta_kappa_obs, kappa_param_obs) are defined at module level once
# imported.
sys.path.insert(0, "/Users/a.adamopoulou/code")
import inf_layered_vap_exp as sim

from plot_sg_capabilities import _fig8_wellbore_contours, _RC

plt.rcParams.update(_RC)

# ── common kwargs for _fig8_wellbore_contours ─────────────────────────────────
common = dict(
    Nx        = sim.Nx,
    Ny        = sim.Ny,
    num_nodes = sim.num_nodes,
    P         = sim.P,
    N_KL      = sim.N_KL,
    multi_idx = sim.multi_idx,
    eval_psi_fn = sim.eval_psi,
    Lx        = sim.Lx,
    Ly        = sim.Ly,
    T_abl     = sim.T_abl,
    dt        = sim.dt,
    n_show    = 4,
    n_samples = 100,
    seed      = 0,
)

# ── truth forward solve (already computed in sim, reuse U_obs) ────────────────
print("Truth forward solve already available as U_obs.")
_fig8_wellbore_contours(
    sim.U_obs,
    **common,
    out="wellbore_truth.png",
)

# ── starting-point forward solve ──────────────────────────────────────────────
VAP_start  = {**sim.VAP_obs,
              'rho_vap0': sim.VAP['rho_vap0'],
              'rho_vap1': sim.VAP['rho_vap1']}
MELT_start = {**sim.MELT_obs, 'f0': sim.MELT['f0']}

print(f"\nStarting-point params:")
print(f"  rho_vap0 = {VAP_start['rho_vap0']:.4e}  (truth {sim.VAP_obs['rho_vap0']:.4e})")
print(f"  f0_m     = {MELT_start['f0']:.4f}        (truth {sim.MELT_obs['f0']:.4f})")

print("Running starting-point forward solve...")
U_start, *_ = sim.run_forward(
    sim.U0, sim.SOLID_obs, MELT_start, VAP_start,
    sim.ell, theta_kappa=sim.theta_kappa_obs,
    kappa_param=sim.kappa_param_obs,
)

_fig8_wellbore_contours(
    U_start,
    **common,
    out="wellbore_start.png",
)

# ── stitch side by side ───────────────────────────────────────────────────────
img_truth = mpimg.imread("wellbore_truth.png")
img_start = mpimg.imread("wellbore_start.png")

fig, axes = plt.subplots(1, 2, figsize=(22, 7))
fig.suptitle(
    f"Wellbore comparison  |  {date.today()}\n"
    f"Left: truth  (rho_vap0={sim.VAP_obs['rho_vap0']:.3e}, f0={sim.MELT_obs['f0']:.2f})   "
    f"Right: x0  (rho_vap0={VAP_start['rho_vap0']:.3e}, f0={MELT_start['f0']:.2f})",
    fontsize=11,
)
axes[0].imshow(img_truth); axes[0].axis("off"); axes[0].set_title("Truth (VAP_obs / MELT_obs)", fontsize=10)
axes[1].imshow(img_start); axes[1].axis("off"); axes[1].set_title("Starting point (VAP / MELT prior)", fontsize=10)

plt.tight_layout()
plt.savefig("wellbore_comparison.png", dpi=150, bbox_inches="tight")
print("\nSaved: wellbore_comparison.png")
plt.close()
