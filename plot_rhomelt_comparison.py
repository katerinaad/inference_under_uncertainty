import numpy as np
import matplotlib.pyplot as plt
import os

_base = os.path.dirname(os.path.abspath(__file__))
def p(f): return os.path.join(_base, f)

files = {
    r"$\rho_{m,1}^*=1.5\times10^8$": p("stage2_log1.5e8.csv"),
    r"$\rho_{m,1}^*=2.0\times10^8$": p("stage2_log_2e8_new.csv"),
    r"$\rho_{m,1}^*=3.0\times10^8$": p("stage2_log3e8_new.csv"),
}
colors = {
    r"$\rho_{m,1}^*=1.5\times10^8$": "#e07b39",
    r"$\rho_{m,1}^*=2.0\times10^8$": "#4878cf",
    r"$\rho_{m,1}^*=3.0\times10^8$": "#6acc65",
}

# GMRF parameter truth values (same across all runs)
truth = {"kappa_surf": 200.0, "kappa_deep": 100.0, "y_trans": 0.9 * 0.21}

def load_csv(fname):
    csv_rows = []
    table_rows = []  # parsed from │-delimited pretty-print lines
    header = None

    with open(fname, encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            if line.isascii():
                if header is None:
                    header = line.strip().split(",")
                else:
                    csv_rows.append(list(map(float, line.strip().split(","))))
            elif "│" in line:
                # pretty-print row: │ iter │ J │ kappa_surf │ kappa_deep │ y_trans │
                parts = [p.strip() for p in line.split("│") if p.strip()]
                if len(parts) == 5:
                    try:
                        table_rows.append(list(map(float, parts)))
                    except ValueError:
                        pass  # skip separator lines

    data = {h: np.array([r[i] for r in csv_rows]) for i, h in enumerate(header)}

    if table_rows:
        table_arr = np.array(table_rows)   # shape (n, 5): iter, J, kappa_surf, kappa_deep, y_trans
        table_cols = ["iter", "J", "kappa_surf", "kappa_deep", "y_trans"]
        # fill missing columns with NaN so shapes stay consistent
        for h in header:
            if h not in table_cols:
                data[h] = np.concatenate([data[h], np.full(len(table_arr), np.nan)])
        for i, h in enumerate(table_cols):
            data[h] = np.concatenate([data[h], table_arr[:, i]])
        # re-sort by iteration number
        order = np.argsort(data["iter"])
        data = {h: v[order] for h, v in data.items()}

    return data

data = {label: load_csv(fname) for label, fname in files.items()}

def param_rms_err(d):
    e_ks = (d["kappa_surf"] - truth["kappa_surf"]) / truth["kappa_surf"]
    e_kd = (d["kappa_deep"] - truth["kappa_deep"]) / truth["kappa_deep"]
    e_yt = (d["y_trans"]   - truth["y_trans"])    / truth["y_trans"]
    return np.sqrt((e_ks**2 + e_kd**2 + e_yt**2) / 3.0)

fig, ax = plt.subplots(figsize=(6, 4))

for label, d in data.items():
    ax.semilogy(d["iter"], param_rms_err(d),
                color=colors[label], marker="o", markersize=3, lw=1.5, label=label)

ax.set_xlabel("Iteration", fontsize=10)
ax.set_ylabel(r"$\frac{1}{\sqrt{3}}\left(\sum_i\left(\frac{\theta_i-\theta_i^*}{\theta_i^*}\right)^2\right)^{1/2}$", fontsize=9)
ax.set_title(r"Weighted parameter RMS error under $\rho_{m,1}$ misspecification", fontsize=10)
ax.legend(fontsize=9, framealpha=0.7)
ax.grid(True, which="both", ls="--", alpha=0.4)

fig.tight_layout()
fig.savefig("rhomelt_comparison.png", dpi=150, bbox_inches="tight")
print("Saved rhomelt_comparison.png")
plt.show()
