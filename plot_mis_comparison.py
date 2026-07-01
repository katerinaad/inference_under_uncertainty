import numpy as np
import matplotlib.pyplot as plt

import os
_base = os.path.dirname(os.path.abspath(__file__))
def p(f): return os.path.join(_base, f)

truth = {"kappa_surf": 200.0, "kappa_deep": 100.0, "y_trans": 0.9 * 0.21}

files = {
    r"Correct ($L_{f,1}=3\times10^8$)":    p("stage2_log_nonmis.csv"),
    r"Mis. ($L_{f,1}=2.5\times10^8$)":     p("stage2_log_mis_2.5e8.csv"),
    r"Mis. ($L_{f,1}=2\times10^8$)":       p("stage2_log_mis_2e8.csv"),
}
colors = {
    r"Correct ($L_{f,1}=3\times10^8$)":    "#4878cf",
    r"Mis. ($L_{f,1}=2.5\times10^8$)":     "#e07b39",
    r"Mis. ($L_{f,1}=2\times10^8$)":       "#d43f3a",
}

def load_csv(fname):
    with open(fname) as f:
        lines = [l for l in f if not l.startswith("#")]
    header = lines[0].strip().split(",")
    rows   = [list(map(float, l.strip().split(","))) for l in lines[1:] if l.strip()]
    return {h: np.array([r[i] for r in rows]) for i, h in enumerate(header)}

def param_rms_err(d):
    e_ks = (d["kappa_surf"] - truth["kappa_surf"]) / truth["kappa_surf"]
    e_kd = (d["kappa_deep"] - truth["kappa_deep"]) / truth["kappa_deep"]
    e_yt = (d["y_trans"]    - truth["y_trans"])    / truth["y_trans"]
    return np.sqrt((e_ks**2 + e_kd**2 + e_yt**2) / 3.0)

data = {label: load_csv(fname) for label, fname in files.items()}

fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))

for label, d in data.items():
    kw = dict(color=colors[label], marker="o", markersize=3, lw=1.5, label=label)
    rms = param_rms_err(d)
    axes[0].semilogy(d["iter"], rms * 100, **kw)
    axes[1].semilogy(d["iter"], d["J"], **kw)

axes[0].set_xlabel("Iteration", fontsize=11)
axes[0].set_ylabel(r"Param.\ RMS error (\%)", fontsize=11)
axes[0].set_title(r"(a) Parameter recovery", fontsize=11)
axes[0].legend(fontsize=9, framealpha=0.8)
axes[0].grid(True, which="both", ls="--", alpha=0.4)

axes[1].set_xlabel("Iteration", fontsize=11)
axes[1].set_ylabel(r"Objective $J$", fontsize=11)
axes[1].set_title(r"(b) Objective convergence", fontsize=11)
axes[1].legend(fontsize=9, framealpha=0.8)
axes[1].grid(True, which="both", ls="--", alpha=0.4)

fig.tight_layout()
fig.savefig("mis_comparison_params.png", dpi=150, bbox_inches="tight")
print("Saved mis_comparison_params.png")
plt.show()
