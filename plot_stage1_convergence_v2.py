"""
Stage 1 convergence — rho_vap0 + f0_v only (phase params fixed at obs).
Truth: rho_vap0 = 1.37e9, f0_v = 0.10
Data extracted from run on 2026-05-19.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from datetime import date

# ── data: (iter, J, rho_vap0, f0_v, g0, g1) ──────────────────────────────────
data = [
    ( 1, 9.8866e+03, 1.2664e+09, 0.2500,  -97245.83,   32586.54),
    ( 2, 6.6801e+03, 1.0000e+10, 0.0500,    2230.89,      -4.96),
    ( 3, 6.6254e+03, 9.5820e+09, 0.0506,    2244.65,      -5.33),
    ( 4, 6.4043e+03, 8.0777e+09, 0.0529,    2307.75,      -7.21),
    ( 5, 5.4377e+03, 4.0795e+09, 0.0635,    2851.81,     -32.26),
    ( 6, 4.9680e+04, 7.0000e+08, 0.1015, -458590.42,  111208.97),  # bounds probe
    ( 7, 3.7494e+03, 2.2435e+09, 0.0745,   12761.85,   -1210.37),
    ( 8, 7.3049e+02, 1.1196e+09, 0.0896,  -19169.72,    3333.42),
    ( 9, 1.1662e+03, 1.7051e+09, 0.0801,   14551.27,   -2217.82),
    (10, 1.1092e+01, 1.3384e+09, 0.0854,    1109.46,    -236.87),
    (11, 1.4497e+05, 7.0000e+08, 0.1707, -949404.36,  329852.06),  # bounds probe
    (12, 9.4083e+02, 1.1401e+09, 0.1014,  -22602.74,    4436.03),
    (13, 1.9423e+01, 1.2851e+09, 0.0892,   -1855.66,     254.91),
    (14, 4.1415e+00, 1.3141e+09, 0.0871,    -140.07,     -32.60),
    (15, 4.0094e+00, 1.3162e+09, 0.0871,     -54.65,     -46.13),
    (16, 3.9103e+00, 1.3177e+09, 0.0872,      -4.20,     -53.75),
    (17, 3.6917e+00, 1.3200e+09, 0.0875,      38.50,     -59.12),
    (18, 2.4025e+00, 1.3288e+09, 0.0898,      38.12,     -49.00),
    (19, 1.9440e+02, 1.2353e+09, 0.0989,   -7781.44,    1414.16),  # line-search spike
    (20, 4.9716e+00, 1.3149e+09, 0.0911,    -758.03,      86.72),
    (21, 2.4083e+00, 1.3258e+09, 0.0901,    -128.13,     -20.85),
    (22, 2.3695e+00, 1.3274e+09, 0.0899,     -41.01,     -35.62),
    (23, 2.3195e+00, 1.3275e+09, 0.0901,     -52.42,     -33.23),
    (24, 2.1366e+00, 1.3283e+09, 0.0905,     -98.24,     -23.56),
    # ── plateau: J~1.89, large gradient, Hessian collapsed ──────────────────
    (25, 1.8960e+00, 1.3481e+09, 0.0990,    -588.57,      98.04),
    (26, 4.5268e+00, 1.3822e+09, 0.1095,    -729.44,     172.45),
    (27, 1.9795e+00, 1.3532e+09, 0.1005,    -605.49,     108.20),
    (28, 1.9038e+00, 1.3491e+09, 0.0993,    -591.72,      99.94),
    (29, 1.8962e+00, 1.3483e+09, 0.0991,    -589.05,      98.39),
    (30, 1.8954e+00, 1.3481e+09, 0.0990,    -588.53,      98.08),
    (31, 1.8958e+00, 1.3482e+09, 0.0990,    -588.85,      98.28),
    (32, 1.8959e+00, 1.3481e+09, 0.0990,    -588.66,      98.13),
    (33, 1.8961e+00, 1.3481e+09, 0.0990,    -588.66,      98.11),
    (34, 1.8954e+00, 1.3481e+09, 0.0990,    -588.53,      98.08),
    # ── breakthrough via bounds probe ────────────────────────────────────────
    (35, 5.1218e+03, 3.3634e+09, 0.0500,    3228.80,     -43.82),  # bounds probe
    (36, 3.1430e+00, 1.3764e+09, 0.0975,     724.78,    -134.07),
    (37, 6.7136e-02, 1.3611e+09, 0.0983,      39.73,     -13.65),
    (38, 5.5925e-02, 1.3602e+09, 0.0984,      -3.71,      -5.93),
    (39, 5.4953e-02, 1.3602e+09, 0.0984,      -6.23,      -5.42),
    (40, 5.1682e-02, 1.3603e+09, 0.0985,     -16.52,      -3.34),
    (41, 4.0396e-02, 1.3609e+09, 0.0988,     -43.32,       2.78),
    (42, 5.4705e-02, 1.3649e+09, 0.1003,     -97.99,      18.17),
    (43, 3.3612e-02, 1.3618e+09, 0.0992,     -55.67,       6.22),
    (44, 3.5274e-01, 1.3777e+09, 0.1039,    -108.16,      34.51),
    (45, 2.2946e-02, 1.3642e+09, 0.0999,     -62.96,      10.35),
    (46, 5.4941e+00, 1.4190e+09, 0.1080,     816.37,    -118.31),  # line-search spike
    (47, 2.0910e-01, 1.3769e+09, 0.1017,     157.65,     -20.94),
    (48, 9.3558e-03, 1.3690e+09, 0.1006,      21.96,      -1.62),
    (49, 1.7839e-03, 1.3675e+09, 0.1003,      -0.93,       1.34),
]

RHO_TRUTH = 1.37e9
F0_TRUTH  = 0.10

arr = np.array(data)
iters   = arr[:, 0]
J       = arr[:, 1]
rho     = arr[:, 2]
f0v     = arr[:, 3]
gnorm   = np.sqrt(arr[:, 4]**2 + arr[:, 5]**2)

# mark bounds probes / large line-search spikes
probes = np.array([2, 6, 11, 35, 46]) - 1   # 0-indexed
plateau = np.arange(24, 34)                  # iters 25–34, 0-indexed

color_main   = "#1f77b4"
color_probe  = "#d62728"
color_plateau = "#ff7f0e"

fig, axes = plt.subplots(4, 1, figsize=(10, 12), sharex=True)
fig.suptitle(
    f"Stage 1 convergence  (opt vars: rho_vap0, f0_v)\n"
    f"Phase params fixed at obs values  |  {date.today()}",
    fontsize=11)

# ── panel 1: J ────────────────────────────────────────────────────────────────
ax = axes[0]
mask_normal = np.ones(len(iters), bool)
mask_normal[probes] = False
ax.semilogy(iters[mask_normal], J[mask_normal], 'o-',
            color=color_main, ms=4, label="J")
ax.semilogy(iters[probes], J[probes], 'x',
            color=color_probe, ms=8, mew=2, label="bounds probe / spike")
ax.axvspan(iters[plateau[0]], iters[plateau[-1]], alpha=0.12,
           color=color_plateau, label="plateau (large grad, flat J)")
ax.set_ylabel("J"); ax.grid(True, which="both", alpha=0.3)
ax.legend(fontsize=8, loc="upper right")
ax.set_title("Objective J")

# ── panel 2: rho_vap0 ─────────────────────────────────────────────────────────
ax = axes[1]
ax.plot(iters[mask_normal], rho[mask_normal] / 1e9, 'o-',
        color=color_main, ms=4)
ax.plot(iters[probes], rho[probes] / 1e9, 'x',
        color=color_probe, ms=8, mew=2)
ax.axhline(RHO_TRUTH / 1e9, color='k', ls='--', lw=1.2,
           label=f"truth = {RHO_TRUTH:.2e}")
ax.axvspan(iters[plateau[0]], iters[plateau[-1]], alpha=0.12, color=color_plateau)
ax.set_ylabel("rho_vap0  (×10⁹  J/m³)")
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
ax.set_title("rho_vap0")

# ── panel 3: f0_v ─────────────────────────────────────────────────────────────
ax = axes[2]
ax.plot(iters[mask_normal], f0v[mask_normal], 'o-',
        color=color_main, ms=4)
ax.plot(iters[probes], f0v[probes], 'x',
        color=color_probe, ms=8, mew=2)
ax.axhline(F0_TRUTH, color='k', ls='--', lw=1.2,
           label=f"truth = {F0_TRUTH}")
ax.axvspan(iters[plateau[0]], iters[plateau[-1]], alpha=0.12, color=color_plateau)
ax.set_ylabel("f0_v"); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
ax.set_title("f0_v")

# ── panel 4: gradient norm ────────────────────────────────────────────────────
ax = axes[3]
ax.semilogy(iters[mask_normal], gnorm[mask_normal], 'o-',
            color=color_main, ms=4)
ax.semilogy(iters[probes], gnorm[probes], 'x',
            color=color_probe, ms=8, mew=2)
ax.axvspan(iters[plateau[0]], iters[plateau[-1]], alpha=0.12, color=color_plateau)
ax.set_ylabel("|g|"); ax.set_xlabel("Function evaluation")
ax.grid(True, which="both", alpha=0.3)
ax.set_title("Gradient norm")

plt.tight_layout()
plt.savefig("stage1_convergence_v2.png", dpi=150, bbox_inches="tight")
print("Saved: stage1_convergence_v2.png")
plt.show()

# ── print summary ─────────────────────────────────────────────────────────────
final = data[-1]
print(f"\nFinal (iter {int(final[0])}):")
print(f"  J        = {final[1]:.4e}")
print(f"  rho_vap0 = {final[2]:.6e}  truth {RHO_TRUTH:.6e}  err = {abs(final[2]-RHO_TRUTH)/RHO_TRUTH*100:.3f}%")
print(f"  f0_v     = {final[3]:.6f}        truth {F0_TRUTH:.6f}        err = {abs(final[3]-F0_TRUTH)/F0_TRUTH*100:.3f}%")
print(f"  |g|      = {gnorm[-1]:.4f}")
