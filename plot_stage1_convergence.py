"""
Stage 1 convergence diagnostics — data extracted from run on 2026-05-19.
Truth: rho_vap0 = 1.3664e9, f0_v = 0.2
"""
import numpy as np
import matplotlib.pyplot as plt
from datetime import date

# ── Stage 1a data (rho_vap0 + f0_v only, phase params fixed) ─────────────────
s1a = [
    # (iter, J, rho_vap0, f0_v)
    ( 1, 2.1830e+02, 1.2664e+09, 0.2000),
    ( 2, 3.8256e+03, 4.0000e+09, 0.0800),  # bounds corner — rejected
    ( 3, 8.5834e+02, 1.7632e+09, 0.1536),
    ( 4, 9.2271e+01, 1.4015e+09, 0.1845),
    ( 5, 3.0433e+01, 1.3683e+09, 0.1876),
    ( 6, 3.4296e+00, 1.3321e+09, 0.1919),
    ( 7, 3.0534e+00, 1.3353e+09, 0.1922),
    ( 8, 1.3471e+00, 1.3488e+09, 0.1948),
    ( 9, 2.1151e+00, 1.3613e+09, 0.2019),
    (10, 4.1275e-01, 1.3538e+09, 0.1976),
    (11, 1.5111e-01, 1.3653e+09, 0.2006),
    (12, 4.4648e+01, 1.4233e+09, 0.1967),  # line-search overstep
    (13, 7.9435e-02, 1.3700e+09, 0.2003),
    (14, 7.1094e-03, 1.3680e+09, 0.2004),
    (15, 6.5063e-03, 1.3678e+09, 0.2004),
    (16, 2.4087e-03, 1.3672e+09, 0.2002),
    (17, 1.9725e-02, 1.3669e+09, 0.1998),
    (18, 1.2370e-03, 1.3671e+09, 0.2001),
    (19, 6.7922e-04, 1.3669e+09, 0.2001),
    (20, 3.3561e-02, 1.3639e+09, 0.1998),
    (21, 6.3748e-04, 1.3662e+09, 0.2000),
    (22, 1.2249e-04, 1.3666e+09, 0.2000),
    (23, 6.9384e-05, 1.3665e+09, 0.2000),
    (24, 8.6043e-06, 1.3663e+09, 0.2000),
    (25, 6.4617e-04, 1.3666e+09, 0.2000),
    (26, 1.0071e-05, 1.3664e+09, 0.2000),
    (27, 7.9057e-06, 1.3664e+09, 0.2000),
    (28, 6.2498e-06, 1.3664e+09, 0.2000),
    (29, 3.4219e-06, 1.3664e+09, 0.2000),
    (30, 4.7735e-05, 1.3663e+09, 0.2000),
    (31, 2.6594e-06, 1.3664e+09, 0.2000),
    (32, 2.4556e-06, 1.3664e+09, 0.2000),  # converged
]

# ── Stage 1b data (all 10 params, interrupted) ────────────────────────────────
# phase params all stayed at truth throughout — only showing J and key params
s1b = [
    # (iter, J, rho_vap0, f0_v, k0_v, m0_v, k0_m, m0_m, f0_m, k0_s, m0_s, f0_s)
    (1, 2.4556e-06, 1.3664e+09, 0.2000, 0.2600, 4.1605e+06, 3.0000, 2.7825e+06, 0.9000, 2.0000, 2.7825e+06, 1.0000),
    (2, 1.4348e+05, 4.0000e+09, 0.5000, 0.1000, 1.0000e+06, 8.0000, 8.0000e+06, 0.5000, 6.0000, 8.0000e+06, 0.7000),
    (3, 7.0255e-06, 1.3664e+09, 0.2000, 0.2600, 4.1605e+06, 3.0000, 2.7825e+06, 0.9000, 2.0000, 2.7825e+06, 1.0000),
    (4, 2.2352e-06, 1.3664e+09, 0.2000, 0.2600, 4.1605e+06, 3.0000, 2.7825e+06, 0.9000, 2.0000, 2.7825e+06, 1.0000),
    (5, 2.1817e-06, 1.3664e+09, 0.2000, 0.2600, 4.1605e+06, 3.0000, 2.7825e+06, 0.9000, 2.0000, 2.7825e+06, 1.0000),
    (6, 1.7021e-06, 1.3664e+09, 0.2000, 0.2600, 4.1605e+06, 3.0000, 2.7825e+06, 0.9000, 2.0000, 2.7825e+06, 1.0000),
    (7, 9.0849e-07, 1.3664e+09, 0.2000, 0.2600, 4.1605e+06, 3.0000, 2.7825e+06, 0.9000, 2.0000, 2.7825e+06, 1.0000),
]

RHO_TRUTH = 1.3664e9
F0_TRUTH  = 0.2

# truth values for phase params (obs values, which are also Stage 1b starting point)
TRUTH = {
    "rho_vap0": 1.3664e9,
    "f0_v":     0.2,
    "k0_v":     0.26,
    "m0_v":     4.1605e6,
    "k0_m":     3.0,
    "m0_m":     2.7825e6,
    "f0_m":     0.9,
    "k0_s":     2.0,
    "m0_s":     2.7825e6,
    "f0_s":     1.0,
}

s1a = np.array(s1a)   # cols: iter, J, rho_vap0, f0_v
s1b = np.array(s1b)   # cols: iter, J, rho_vap0, f0_v, k0_v, m0_v, k0_m, m0_m, f0_m, k0_s, m0_s, f0_s

colors = {"1a": "#1f77b4", "1b": "#d62728"}

# ── Figure 1: Stage 1a — 2 opt vars ──────────────────────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(9, 9))
fig.suptitle("Stage 1a convergence  (opt vars: rho_vap0, f0_v)\n"
             f"Phase params fixed at obs values  |  {date.today()}", fontsize=11)

ax = axes[0]
ax.semilogy(s1a[:, 0], s1a[:, 1], 'o-', color=colors["1a"], ms=4)
ax.set_ylabel("J"); ax.set_title("Objective"); ax.grid(True, which="both", alpha=0.3)

ax = axes[1]
ax.plot(s1a[:, 0], s1a[:, 2] / 1e9, 'o-', color=colors["1a"], ms=4)
ax.axhline(TRUTH["rho_vap0"] / 1e9, color='k', ls='--', lw=1.2,
           label=f"truth = {TRUTH['rho_vap0']:.4e}")
ax.set_ylabel("rho_vap0  (×10⁹)"); ax.legend(fontsize=9)
ax.set_title("rho_vap0"); ax.grid(True, alpha=0.3)

ax = axes[2]
ax.plot(s1a[:, 0], s1a[:, 3], 'o-', color=colors["1a"], ms=4)
ax.axhline(TRUTH["f0_v"], color='k', ls='--', lw=1.2,
           label=f"truth = {TRUTH['f0_v']}")
ax.set_ylabel("f0_v"); ax.legend(fontsize=9); ax.set_xlabel("Function evaluation")
ax.set_title("f0_v"); ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("stage1a_convergence.png", dpi=150, bbox_inches="tight")
print("Saved: stage1a_convergence.png")
plt.close()

# ── Figure 2: Stage 1b — all 10 opt vars ─────────────────────────────────────
# cols in s1b: 0=iter,1=J,2=rho,3=f0_v,4=k0_v,5=m0_v,6=k0_m,7=m0_m,8=f0_m,9=k0_s,10=m0_s,11=f0_s
params_1b = [
    ("rho_vap0", 2,  1e9,  "×10⁹ J/m³"),
    ("f0_v",     3,  1.0,  "—"),
    ("k0_v",     4,  1.0,  "W/(m·K)"),
    ("m0_v",     5,  1e6,  "×10⁶ J/(m³·K)"),
    ("k0_m",     6,  1.0,  "W/(m·K)"),
    ("m0_m",     7,  1e6,  "×10⁶ J/(m³·K)"),
    ("f0_m",     8,  1.0,  "—"),
    ("k0_s",     9,  1.0,  "W/(m·K)"),
    ("m0_s",     10, 1e6,  "×10⁶ J/(m³·K)"),
    ("f0_s",     11, 1.0,  "— (at upper bound)"),
]

fig, axes = plt.subplots(6, 2, figsize=(13, 18))
fig.suptitle("Stage 1b convergence  (all 10 opt vars, INTERRUPTED at iter 7)\n"
             f"Warm-started from Stage 1a result  |  {date.today()}", fontsize=11)
axes_flat = axes.ravel()

# J in first panel
ax = axes_flat[0]
ax.semilogy(s1b[:, 0], s1b[:, 1], 's-', color=colors["1b"], ms=5)
ax.set_title("Objective J"); ax.set_ylabel("J")
ax.grid(True, which="both", alpha=0.3)

for i, (name, col, scale, unit) in enumerate(params_1b):
    ax = axes_flat[i + 1]
    vals = s1b[:, col] / scale
    ax.plot(s1b[:, 0], vals, 's-', color=colors["1b"], ms=5)
    tv = TRUTH[name] / scale
    ax.axhline(tv, color='k', ls='--', lw=1.2, label=f"truth={tv:.4g}")
    ax.set_title(name); ax.set_ylabel(unit if unit != "—" else "")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    ax.set_xlabel("Function eval")

# hide unused panel (11 params + J = 12 panels exactly, nothing unused)
plt.tight_layout()
plt.savefig("stage1b_convergence.png", dpi=150, bbox_inches="tight")
print("Saved: stage1b_convergence.png")
plt.close()

# ── Summary document ──────────────────────────────────────────────────────────
summary = f"""Stage 1 Inference Summary
=========================
Run date : {date.today()}

Truth values
------------
  rho_vap0 = {RHO_TRUTH:.6e}  J/m³
  f0_v     = {F0_TRUTH:.4f}

Stage 1a  (warm-start: rho_vap0 + f0_v, phase params fixed at obs)
-------------------------------------------------------------------
  Starting point : rho_vap0 = 1.2664e+09  f0_v = 0.2000  (f0_v fixed to truth)
  Bounds         : rho_vap0 ∈ [4e8, 4e9]   f0_v ∈ [0.08, 0.5]
  Iterations     : {len(s1a)}  function evaluations
  J start        : {s1a[0,1]:.4e}
  J final        : {s1a[-1,1]:.4e}
  Converged      : yes  (gtol = 2e-4)

  Final estimates vs truth:
    rho_vap0 = {s1a[-1,2]:.6e}   truth {RHO_TRUTH:.6e}   err = {abs(s1a[-1,2]-RHO_TRUTH)/RHO_TRUTH*100:.4f}%
    f0_v     = {s1a[-1,3]:.6f}         truth {F0_TRUTH:.6f}         err = {abs(s1a[-1,3]-F0_TRUTH)/F0_TRUTH*100:.4f}%

Stage 1b  (all 10 params: + k0_v, m0_v, k0_m, m0_m, f0_m, k0_s, m0_s, f0_s)
------------------------------------------------------------------------------
  Warm start     : from Stage 1a result
  Status         : INTERRUPTED after {len(s1b)} evals
  J start        : {s1b[0,1]:.4e}
  J at interrupt : {s1b[-1,1]:.4e}
  Phase params   : all stayed at truth (no drift observed before interrupt)

  Iter 2 bounds probe (J = {s1b[1,1]:.4e})  — correctly rejected, returned to truth
    rho_vap0 = 4.0e+09  f0_v = 0.50  k0_v = 0.10  m0_v = 1e6  ...  (all at bounds)

Notes
-----
- sigma2_obs_hist at early timesteps is O(1e-16), creating large J weights;
  residual J ~ 1e-6 is numerical noise at this scale, not a real mismatch.
- f0_s = 1.0 is at the upper bound; gradient info lost there. Consider fixing
  f0_s = 1.0 in Stage 1 if it becomes an issue in future runs.
"""

with open("stage1_summary.txt", "w") as f:
    f.write(summary)
print("Saved: stage1_summary.txt")
print(summary)
