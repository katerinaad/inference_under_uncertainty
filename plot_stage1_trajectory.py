"""
Stage 1 optimiser trajectory in (rho_vap0, f0_v) space.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from datetime import date

data = [
    ( 1, 9.8866e+03, 1.2664e+09, 0.2500),
    ( 2, 6.6801e+03, 1.0000e+10, 0.0500),
    ( 3, 6.6254e+03, 9.5820e+09, 0.0506),
    ( 4, 6.4043e+03, 8.0777e+09, 0.0529),
    ( 5, 5.4377e+03, 4.0795e+09, 0.0635),
    ( 6, 4.9680e+04, 7.0000e+08, 0.1015),
    ( 7, 3.7494e+03, 2.2435e+09, 0.0745),
    ( 8, 7.3049e+02, 1.1196e+09, 0.0896),
    ( 9, 1.1662e+03, 1.7051e+09, 0.0801),
    (10, 1.1092e+01, 1.3384e+09, 0.0854),
    (11, 1.4497e+05, 7.0000e+08, 0.1707),
    (12, 9.4083e+02, 1.1401e+09, 0.1014),
    (13, 1.9423e+01, 1.2851e+09, 0.0892),
    (14, 4.1415e+00, 1.3141e+09, 0.0871),
    (15, 4.0094e+00, 1.3162e+09, 0.0871),
    (16, 3.9103e+00, 1.3177e+09, 0.0872),
    (17, 3.6917e+00, 1.3200e+09, 0.0875),
    (18, 2.4025e+00, 1.3288e+09, 0.0898),
    (19, 1.9440e+02, 1.2353e+09, 0.0989),
    (20, 4.9716e+00, 1.3149e+09, 0.0911),
    (21, 2.4083e+00, 1.3258e+09, 0.0901),
    (22, 2.3695e+00, 1.3274e+09, 0.0899),
    (23, 2.3195e+00, 1.3275e+09, 0.0901),
    (24, 2.1366e+00, 1.3283e+09, 0.0905),
    (25, 1.8960e+00, 1.3481e+09, 0.0990),
    (26, 4.5268e+00, 1.3822e+09, 0.1095),
    (27, 1.9795e+00, 1.3532e+09, 0.1005),
    (28, 1.9038e+00, 1.3491e+09, 0.0993),
    (29, 1.8962e+00, 1.3483e+09, 0.0991),
    (30, 1.8954e+00, 1.3481e+09, 0.0990),
    (31, 1.8958e+00, 1.3482e+09, 0.0990),
    (32, 1.8959e+00, 1.3481e+09, 0.0990),
    (33, 1.8961e+00, 1.3481e+09, 0.0990),
    (34, 1.8954e+00, 1.3481e+09, 0.0990),
    (35, 5.1218e+03, 3.3634e+09, 0.0500),
    (36, 3.1430e+00, 1.3764e+09, 0.0975),
    (37, 6.7136e-02, 1.3611e+09, 0.0983),
    (38, 5.5925e-02, 1.3602e+09, 0.0984),
    (39, 5.4953e-02, 1.3602e+09, 0.0984),
    (40, 5.1682e-02, 1.3603e+09, 0.0985),
    (41, 4.0396e-02, 1.3609e+09, 0.0988),
    (42, 5.4705e-02, 1.3649e+09, 0.1003),
    (43, 3.3612e-02, 1.3618e+09, 0.0992),
    (44, 3.5274e-01, 1.3777e+09, 0.1039),
    (45, 2.2946e-02, 1.3642e+09, 0.0999),
    (46, 5.4941e+00, 1.4190e+09, 0.1080),
    (47, 2.0910e-01, 1.3769e+09, 0.1017),
    (48, 9.3558e-03, 1.3690e+09, 0.1006),
    (49, 1.7839e-03, 1.3675e+09, 0.1003),
]

RHO_TRUTH = 1.37e9
F0_TRUTH  = 0.10

arr    = np.array(data)
iters  = arr[:, 0].astype(int)
J      = arr[:, 1]
rho    = arr[:, 2] / 1e9   # plot in units of 1e9
f0v    = arr[:, 3]

probes  = {2, 6, 11, 35, 46}    # 1-indexed iters that are bounds probes / spikes
plateau = set(range(25, 35))    # iters 25–34

fig, ax = plt.subplots(figsize=(9, 7))
fig.suptitle(
    f"Optimiser trajectory in (rho_vap0, f0_v) space\n"
    f"Stage 1  |  {date.today()}",
    fontsize=11)

# ── draw connecting line for non-probe steps ──────────────────────────────────
normal_idx = [i for i, it in enumerate(iters) if it not in probes]
ax.plot(rho[normal_idx], f0v[normal_idx],
        '-', color='#aec7e8', lw=0.8, zorder=1)

# ── colour scatter by log10(J) ────────────────────────────────────────────────
logJ   = np.log10(np.clip(J, 1e-4, None))
norm   = plt.Normalize(logJ.min(), logJ.max())
cmap   = cm.plasma_r

for i, (it, r, f, lj) in enumerate(zip(iters, rho, f0v, logJ)):
    if it in probes:
        ax.scatter(r, f, marker='x', s=80, color='#d62728',
                   linewidths=2, zorder=4)
    elif it in plateau:
        ax.scatter(r, f, marker='s', s=30,
                   color='#ff7f0e', zorder=3)
    else:
        ax.scatter(r, f, marker='o', s=30,
                   color=cmap(norm(lj)), zorder=3)

# ── label selected iterations ─────────────────────────────────────────────────
label_iters = {1, 10, 18, 25, 34, 36, 45, 49}
for i, it in enumerate(iters):
    if it in label_iters:
        ax.annotate(str(it), (rho[i], f0v[i]),
                    textcoords="offset points", xytext=(5, 4),
                    fontsize=7, color='#333333')

# ── truth and start markers ───────────────────────────────────────────────────
ax.scatter(RHO_TRUTH / 1e9, F0_TRUTH, marker='*', s=250,
           color='green', zorder=5, label=f"truth ({RHO_TRUTH:.2e}, {F0_TRUTH})")
ax.scatter(rho[0], f0v[0], marker='D', s=60,
           color='black', zorder=5, label=f"x0 ({rho[0]:.3f}×10⁹, {f0v[0]:.2f})")

# ── colourbar ─────────────────────────────────────────────────────────────────
sm = cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, pad=0.02)
cbar.set_label("log₁₀(J)", fontsize=9)

# ── legend for special markers ────────────────────────────────────────────────
from matplotlib.lines import Line2D
legend_elements = [
    ax.get_legend_handles_labels()[0][0],   # truth star
    ax.get_legend_handles_labels()[0][1],   # x0 diamond
    Line2D([0], [0], marker='x', color='#d62728', lw=0,
           markersize=8, markeredgewidth=2, label='bounds probe / spike'),
    Line2D([0], [0], marker='s', color='#ff7f0e', lw=0,
           markersize=6, label='plateau (iters 25–34)'),
]
ax.legend(handles=legend_elements, fontsize=8, loc='upper right')

ax.set_xlabel("rho_vap0  (×10⁹  J/m³)", fontsize=10)
ax.set_ylabel("f0_v", fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("stage1_trajectory.png", dpi=150, bbox_inches="tight")
print("Saved: stage1_trajectory.png")
plt.show()
