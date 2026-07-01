import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

files = {
    r"$L_{f,1}=4\times10^ J/kg5$ (+33%)" : (
        "/Users/a.adamopoulou/code/stage2_log_4e8.csv",
        {"kappa_surf": 200.0, "kappa_deep": 100.0, "y_trans": 0.189},
    ),
    r"$L_{f,1}=3\times10^5 J/kg$ (0%)" : (
        "/Users/a.adamopoulou/code/stage2_log_nonmis.csv",
        {"kappa_surf": 200.0, "kappa_deep": 100.0, "y_trans": 0.189},
    ),
    r"$L_{f,1}=2.5\times10^5 J/kg$ ($-17$%)" : (
        "/Users/a.adamopoulou/code/stage2_log_mis_2.5e8.csv",
        {"kappa_surf": 200.0, "kappa_deep": 100.0, "y_trans": 0.189},
    ),
    r"$L_{f,1}=2\times10^5 J/kg$ ($-33$%)" : (
        "/Users/a.adamopoulou/code/stage2_log_mis_2e8.csv",
        {"kappa_surf": 200.0, "kappa_deep": 100.0, "y_trans": 0.189},
    ),
}

colors = ["#d62728", "#1f77b4", "#ff7f0e", "#2ca02c"]
params = ["kappa_surf", "kappa_deep", "y_trans"]
MAX_ITER = 40

def pad_to(iters, vals):
    if iters[-1] < MAX_ITER:
        iters = np.append(iters, MAX_ITER)
        vals  = np.append(vals,  vals[-1])
    return iters, vals

fig, (ax_J, ax_rmse) = plt.subplots(1, 2, figsize=(11, 4.5))

for (label, (fpath, truth)), color in zip(files.items(), colors):
    df = pd.read_csv(fpath, comment="#")
    iters = df["iter"].values

    J_runmin = np.minimum.accumulate(df["J"].values)
    iters_J, J_runmin = pad_to(iters.copy(), J_runmin)
    ax_J.semilogy(iters_J, J_runmin, color=color, label=label, lw=1.8)

    sq_rel = np.stack([
        ((df[p].values - truth[p]) / truth[p]) ** 2
        for p in params
    ], axis=1)
    rmse_runmin = np.minimum.accumulate(np.sqrt(sq_rel.mean(axis=1)))
    iters_r, rmse_runmin = pad_to(iters.copy(), rmse_runmin)
    ax_rmse.semilogy(iters_r, rmse_runmin, color=color, label=label, lw=1.8)

ax_J.set_title("Objective $J$ (running minimum)", fontsize=11)
ax_J.set_xlabel("Iteration", fontsize=10)
ax_J.set_ylabel("$J$", fontsize=10)
ax_J.grid(True, which="both", ls="--", alpha=0.35)
ax_J.legend(fontsize=8.5, framealpha=0.85)

ax_rmse.set_title("Normalised RMS parameter error (running minimum)", fontsize=11)
ax_rmse.set_xlabel("Iteration", fontsize=10)
ax_rmse.set_ylabel("RMS error", fontsize=10)
ax_rmse.yaxis.set_major_formatter(plt.FuncFormatter(
    lambda y, _: f"{y*100:.1f}%" if y < 0.1 else f"{y*100:.0f}%"
))
ax_rmse.grid(True, which="both", ls="--", alpha=0.35)
ax_rmse.legend(fontsize=8.5, framealpha=0.85)

fig.suptitle(r"Matern hyperparameter convergence under $L_{f,1}$ misspecification", fontsize=12)
fig.tight_layout()
plt.savefig("/Users/a.adamopoulou/code/stage2_convergence_comparison.png",
            dpi=150, bbox_inches="tight")
plt.close()
print("Saved stage2_convergence_comparison.png")
