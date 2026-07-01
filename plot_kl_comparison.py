import numpy as np
import matplotlib.pyplot as plt

import os
_base = os.path.dirname(os.path.abspath(__file__))
def p(f): return os.path.join(_base, f)

files  = {"40 KL": p("stage2_log_3kappa_40KL_new.csv"),
          "45 KL": p("stage2_log_3kappa_45kl_new.csv"),
          "50 KL": p("stage2_log_3kappa_50KL_new.csv")}
colors = {"40 KL": "#e07b39", "45 KL": "#4878cf", "50 KL": "#6acc65"}

truth = {"kappa_surf": 200.0, "kappa_deep": 100.0, "y_trans": 0.9 * 0.21}

def load_csv(fname):
    with open(fname) as f:
        lines = [l for l in f if not l.startswith("#")]
    header = lines[0].strip().split(",")
    rows   = [list(map(float, l.strip().split(","))) for l in lines[1:] if l.strip()]
    return {h: np.array([r[i] for r in rows]) for i, h in enumerate(header)}

def trim(d):
    """Keep only up to the first iteration where J reaches its final minimum."""
    J = d["J"]
    j_min = J.min()
    cutoff = np.where(J == j_min)[0][0] + 1   # include first occurrence
    return {k: v[:cutoff] for k, v in d.items()}

data = {label: trim(load_csv(fname)) for label, fname in files.items()}

def param_rms_err(d):
    e_ks = (d["kappa_surf"] - truth["kappa_surf"]) / truth["kappa_surf"]
    e_kd = (d["kappa_deep"] - truth["kappa_deep"]) / truth["kappa_deep"]
    e_yt = (d["y_trans"]   - truth["y_trans"])    / truth["y_trans"]
    return np.sqrt((e_ks**2 + e_kd**2 + e_yt**2) / 3.0)

fig, axes = plt.subplots(1, 2, figsize=(8, 3.8))

for label, d in data.items():
    kw = dict(color=colors[label], marker="o", markersize=3, lw=1.5, label=label)
    axes[0].semilogy(d["iter"], d["J"], **kw)
    axes[1].semilogy(d["iter"], param_rms_err(d), **kw)

for ax, title, ylabel in zip(
    axes,
    ["(a) Objective $J$", r"(b) Parameter RMS relative error"],
    ["$J$",               r"$\frac{1}{\sqrt{3}}\left(\sum_i\left(\frac{\theta_i - \theta_i^*}{\theta_i^*}\right)^2\right)^{1/2}$"],
):
    ax.set_xlabel("Iteration", fontsize=10)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=9, framealpha=0.7)
    ax.grid(True, which="both", ls="--", alpha=0.4)

fig.tight_layout()
fig.savefig("kl_comparison.png", dpi=150, bbox_inches="tight")
print("Saved kl_comparison.png")
plt.show()
