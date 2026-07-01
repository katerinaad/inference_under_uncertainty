"""
ridge_analysis_free.py
======================
Ridge analysis using ONLY the already-computed stage1_log.npz data.
No extra forward solves needed.

Three things:
  1. Gradient-based ridge direction (local, linear approx)
  2. Quadratic approximation of J around the stalled point
     (using the gradient evolution along the path)
  3. Plot: optimisation path, gradient arrows, and estimated ridge
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

# ── Load data ─────────────────────────────────────────────────────────────────
d = np.load("stage1_log.npz", allow_pickle=True)
rho_all    = d['rho_vap0']           # (18,)
f0_all     = d['f0_v']               # (18,)
J_all      = d['J']                  # (18,)
g_all      = d['g']                  # (18, 2) — already in log-space: dJ/d(log θ)
rho_truth  = float(d['rho_truth'])
f0_truth   = float(d['f0_truth'])

# Stalled point (last non-duplicate iter with non-zero step)
# Iter 8 is where J first flattens; use that as the "stall" point.
stall_idx  = np.argmin(J_all)        # index of best J found
rho_stall  = rho_all[stall_idx]
f0_stall   = f0_all[stall_idx]
g_stall    = g_all[stall_idx]        # (g_log_rho, g_log_f0) in log-space

print("=" * 60)
print("  GRADIENT-BASED RIDGE ANALYSIS (free, no new forward solves)")
print("=" * 60)
print(f"\nStall point:  rho = {rho_stall:.4e}  f0 = {f0_stall:.6f}  J = {J_all[stall_idx]:.4e}")
print(f"Truth:        rho = {rho_truth:.4e}  f0 = {f0_truth:.6f}")
print(f"\nLog-space gradient at stall: g_s = {g_stall[0]:.4e}  g_t = {g_stall[1]:.4e}")

# ── 1. Ridge direction from gradient ─────────────────────────────────────────
# The ridge (flat direction) is perpendicular to the gradient in log-space.
# In log-space coordinates (s=log ρ, t=log f0):
g_norm     = np.linalg.norm(g_stall)
g_hat      = g_stall / g_norm         # unit gradient (steepest descent direction)
ridge_hat  = np.array([-g_hat[1],  g_hat[0]])   # 90° rotation = ridge direction
print(f"\nGradient direction (log-space, unit): {g_hat}")
print(f"Ridge direction    (log-space, unit): {ridge_hat}")
print(f"  → in log(rho), log(f0) coords: d(log ρ)/d(log f0) along ridge = {ridge_hat[0]/ridge_hat[1]:.3f}")

# ── 2. Secant-based curvature along gradient vs ridge direction ───────────────
# Use consecutive gradient differences to estimate local curvature.
# H ≈ Δg / Δθ  between converged iterates (all near the stall).
late_mask  = J_all < 1e4             # only the converged tail
rho_late   = rho_all[late_mask]
f0_late    = f0_all[late_mask]
g_late     = g_all[late_mask]
J_late     = J_all[late_mask]

if len(rho_late) >= 3:
    # Use pairs of consecutive iterates to estimate the 2x2 Hessian
    s_late = np.log(rho_late)
    t_late = np.log(f0_late)
    H_est  = np.zeros((2, 2))
    n_pairs = 0
    for i in range(len(s_late) - 1):
        ds = np.array([s_late[i+1] - s_late[i], t_late[i+1] - t_late[i]])
        dg = g_late[i+1] - g_late[i]
        if np.linalg.norm(ds) > 1e-10:
            H_est += np.outer(dg, ds) / (np.dot(ds, ds))
            n_pairs += 1
    if n_pairs > 0:
        H_est /= n_pairs
        H_sym = 0.5 * (H_est + H_est.T)   # symmetrise
        eigvals, eigvecs = np.linalg.eigh(H_sym)
        cond = eigvals[1] / max(eigvals[0], 1e-30)
        print(f"\nSecant Hessian (log-space, from {n_pairs} late-iterate pairs):")
        print(f"  H ≈ {H_sym}")
        print(f"  Eigenvalues:  λ_min = {eigvals[0]:.3e}  λ_max = {eigvals[1]:.3e}")
        print(f"  Condition number κ ≈ {cond:.1f}")
        print(f"  Ridge direction (eigvec of λ_min): {eigvecs[:,0]}")
        hess_ridge_hat = eigvecs[:, 0]
    else:
        hess_ridge_hat = ridge_hat
else:
    hess_ridge_hat = ridge_hat

# ── 3. Quadratic ridge approximation ─────────────────────────────────────────
# Build a quadratic approx J(θ) ≈ J* + ½ (θ-θ*)^T H (θ-θ*)
# Level sets of this quadratic are ellipses → the long axis IS the ridge.

# Estimate H from the gradient drop from iter 3→8  (the descent phase)
# Using 4 key iters: first good iterate (3), midway (5), stall (7,8)
key_idx  = [2, 4, 6, stall_idx]   # 0-indexed
s_key = np.log(rho_all[key_idx])
t_key = np.log(f0_all[key_idx])
g_key = g_all[key_idx]

# Least-squares fit: g_i ≈ H (θ_i - θ_stall)
A = np.column_stack([s_key - np.log(rho_stall),
                     t_key - np.log(f0_stall)])  # (4, 2)
G = g_key                                         # (4, 2)
# Vec form: G.ravel() ≈ (A ⊗ I₂) vec(H)  → but H is 2x2 symmetric (3 unknowns)
# Simple: H ≈ A^+ G  (pseudoinverse, each column independently)
H_ls, _, _, _ = np.linalg.lstsq(A, G, rcond=None)
H_ls_sym = 0.5 * (H_ls + H_ls.T)

print(f"\nLeast-squares Hessian estimate (from {len(key_idx)} key iterates):")
print(f"  H_ls = {H_ls_sym}")
eigvals_ls, eigvecs_ls = np.linalg.eigh(H_ls_sym)
print(f"  Eigenvalues: {eigvals_ls}")
cond_ls = eigvals_ls[1] / max(eigvals_ls[0], 1e-30)
print(f"  Condition number κ ≈ {cond_ls:.1f}")

# ── 4. Plot ───────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# --- Left: full path + gradient arrows in log-space ---
ax = axes[0]
s_path = np.log10(rho_all)
t_path = np.log10(f0_all)

ax.plot(s_path, t_path, 'o-', color='royalblue', lw=1.5, ms=5, zorder=5, label='optim path')
ax.plot(np.log10(rho_stall), np.log10(f0_stall), '*', color='red', ms=14, zorder=10, label=f'stall  J={J_all[stall_idx]:.1e}')
ax.plot(np.log10(rho_truth),  np.log10(f0_truth),  'D', color='green', ms=10, zorder=10, label=f'truth')

# Gradient arrows (scaled, pointing towards lower J = -g direction)
for i in range(len(rho_all)):
    gs  = g_all[i, 0]; gt = g_all[i, 1]
    gn  = max(np.sqrt(gs**2 + gt**2), 1e-15)
    scale = 0.08
    ax.annotate("", xy=(s_path[i] - scale*gs/gn, t_path[i] - scale*gt/gn),
                xytext=(s_path[i], t_path[i]),
                arrowprops=dict(arrowstyle="-|>", color='orange', lw=1.0, mutation_scale=8))

# Ridge direction arrow from stall point
s0, t0 = np.log10(rho_stall), np.log10(f0_stall)
arrow_len = 0.25
rh = hess_ridge_hat / np.log(10)   # convert from ln to log10 scale
ax.annotate("", xy=(s0 + arrow_len*rh[0], t0 + arrow_len*rh[1]),
            xytext=(s0 - arrow_len*rh[0], t0 - arrow_len*rh[1]),
            arrowprops=dict(arrowstyle="<->", color='purple', lw=2.0, mutation_scale=12))
ax.text(s0 + arrow_len*rh[0] + 0.02, t0 + arrow_len*rh[1],
        'ridge\ndirection', color='purple', fontsize=9)

ax.set_xlabel('log₁₀(ρ_vap0)')
ax.set_ylabel('log₁₀(f0)')
ax.set_title('Optimisation path + gradient arrows\n(orange: -∇J direction, purple: estimated ridge)')
ax.legend(fontsize=9)
ax.grid(True, ls=':')

# --- Right: J along path with index labels ---
ax2 = axes[1]
ax2.semilogy(J_all, 'o-', color='royalblue', lw=2, ms=5)
for i in range(len(J_all)):
    ax2.text(i + 0.1, J_all[i] * 1.1, str(i+1), fontsize=7, color='gray')
ax2.axhline(J_all[stall_idx], color='red', ls='--', lw=1.5, label=f'stall J={J_all[stall_idx]:.1e}')
ax2.set_xlabel('Iteration')
ax2.set_ylabel('J (log scale)')
ax2.set_title('Cost function convergence')
ax2.legend()
ax2.grid(True, which='both', ls=':')

plt.tight_layout()
plt.savefig('ridge_analysis_free.png', dpi=150)
print("\nSaved ridge_analysis_free.png")

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  SUMMARY")
print("=" * 60)
print(f"\nStall location (log-space): s* = {np.log(rho_stall):.3f}  t* = {np.log(f0_stall):.3f}")
print(f"Truth location  (log-space): s_T = {np.log(rho_truth):.3f}  t_T = {np.log(f0_truth):.3f}")
delta_s = np.log(rho_truth) - np.log(rho_stall)
delta_t = np.log(f0_truth)  - np.log(f0_stall)
print(f"Vector stall→truth (log):    Δs = {delta_s:.3f}  Δt = {delta_t:.3f}")
print(f"  angle with gradient: {np.degrees(np.arccos(np.clip(np.dot([delta_s,delta_t], g_hat)/np.linalg.norm([delta_s,delta_t]),  -1,1))):.1f}°")
print(f"  angle with ridge:    {np.degrees(np.arccos(np.clip(np.dot([delta_s,delta_t], hess_ridge_hat)/np.linalg.norm([delta_s,delta_t]),  -1,1))):.1f}°")

print(f"\nIf the ridge direction ({hess_ridge_hat}) aligns closely with the stall→truth")
print(f"vector ({delta_s:.3f}, {delta_t:.3f}), J is nearly constant between stall and truth.")
print(f"The gradient there just isn't large enough to drive the optimizer to truth.\n")
