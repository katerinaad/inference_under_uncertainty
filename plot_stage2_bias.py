import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

truth = {"kappa_surf": 200.0, "kappa_deep": 100.0, "y_trans": 0.189}

# Final inferred values per run (last CSV row; 3e8 read from formatted table)
final = {
    r"$4\times10^8$": dict(pd.read_csv("/Users/a.adamopoulou/code/stage2_log_4e8.csv", comment="#").iloc[-1]),
    r"$3\times10^8$": {"kappa_surf": 200.0006, "kappa_deep": 99.9996, "y_trans": 0.189000},
    r"$2\times10^8$": dict(pd.read_csv("/Users/a.adamopoulou/code/stage2_log_2e8_new.csv", comment="#").iloc[-1]),
    r"$1.5\times10^8$": dict(pd.read_csv("/Users/a.adamopoulou/code/stage2_log1.5e8.csv", comment="#").iloc[-1]),
}

params = ["kappa_surf", "kappa_deep", "y_trans"]
param_labels = [r"$\kappa_\mathrm{surf}$", r"$\kappa_\mathrm{deep}$", r"$y_\mathrm{trans}$"]
run_labels = list(final.keys())
colors = ["#d62728", "#1f77b4", "#ff7f0e", "#2ca02c"]

pct_bias = {
    run: [100 * (final[run][p] - truth[p]) / truth[p] for p in params]
    for run in run_labels
}

x = np.arange(len(params))
width = 0.18
fig, ax = plt.subplots(figsize=(8, 4))

offsets = np.linspace(-1.5, 1.5, len(run_labels)) * width
for offset, (run, color) in zip(offsets, zip(run_labels, colors)):
    bars = ax.bar(x + offset, pct_bias[run], width, label=r"$\rho_\mathrm{melt}=$" + run,
                  color=color, alpha=0.85, edgecolor="k", linewidth=0.5)

ax.axhline(0, color="k", linewidth=0.8, linestyle="--")
ax.set_xticks(x)
ax.set_xticklabels(param_labels, fontsize=12)
ax.set_ylabel("Relative bias (%)")
ax.set_title(r"Parameter bias under $\rho_\mathrm{melt}$ misspecification")
ax.legend(fontsize=9, title=r"True $\rho_\mathrm{melt}$", title_fontsize=9)
ax.grid(True, axis="y", alpha=0.3)

fig.tight_layout()
plt.savefig("/Users/a.adamopoulou/code/stage2_bias.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved stage2_bias.png")
