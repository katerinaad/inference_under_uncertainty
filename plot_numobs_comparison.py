import numpy as np
import matplotlib.pyplot as plt

import os
BASE = os.path.dirname(os.path.abspath(__file__))

logs = [
    (os.path.join(BASE, "stage2_log_rhomelt_s.csv"),  100),
    (os.path.join(BASE, "stage2_log_rho_melt20.csv"),  20),
    (os.path.join(BASE, "stage2_log.csv"),              10),
]

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

colors = plt.cm.tab10.colors

for idx, (fname, n_obs) in enumerate(logs):
    rows = []
    with open(fname) as f:
        for line in f:
            line = line.strip()
            if line.startswith('#') or line.startswith('iter'):
                continue
            rows.append([float(x) for x in line.split(',')])
    arr = np.array(rows)
    iters    = arr[:, 0].astype(int)
    J        = arr[:, 1]
    rho_surf = arr[:, 2]
    rho_deep = arr[:, 3]
    c = colors[idx]
    label = f"$N_{{obs}}={n_obs}$"

    axes[0].semilogy(iters, J, 'o-', color=c, label=label, markersize=4, lw=1.5)
    axes[1].plot(iters, rho_surf * 1e-8, 'o-', color=c, label=f"{label} surf", markersize=4, lw=1.5)
    axes[1].plot(iters, rho_deep * 1e-8, 's--', color=c, label=f"{label} deep", markersize=4, lw=1.5)

# truth lines
rho_surf_truth = 3.0e8
rho_deep_truth = 5.0e7
axes[1].axhline(rho_surf_truth * 1e-8, color='k', lw=1, ls=':', label='truth surf')
axes[1].axhline(rho_deep_truth * 1e-8, color='k', lw=1, ls='--', label='truth deep')

axes[0].set_xlabel("Iteration")
axes[0].set_ylabel("$J_{var}$")
axes[0].set_title("Objective convergence")
axes[0].legend()
axes[0].grid(True, which='both', alpha=0.3)

axes[1].set_xlabel("Iteration")
axes[1].set_ylabel(r"$\rho_{melt}$ [$\times 10^8$ kg m$^{-3}$]")
axes[1].set_title("Parameter convergence")
axes[1].legend(fontsize=8)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("numobs_comparison.png", dpi=150, bbox_inches='tight')
print("saved numobs_comparison.png")
