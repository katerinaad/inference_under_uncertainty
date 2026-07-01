"""
Two-panel figure for the non-inverse-crime / finite-sample ablation section.
  Plot 1: sigma2_obs(t) for N=50, 100, 500 + analytical SG
  Plot 2: RMS parameter error vs iteration for N=50, 100, 500
"""
import numpy as np
import matplotlib.pyplot as plt

# ── analytical SG variance (linearised, same as inference model) ──────────────
analytical = np.array([
    0.00000000e+00, 2.77511307e-21, 1.24078119e-20, 3.13515012e-20,
    6.29097687e-20, 1.11565392e-19, 1.83447292e-19, 2.86935272e-19,
    4.32983041e-19, 6.32790077e-19, 8.91959875e-19, 1.21062671e-18,
    1.58788272e-18, 2.02444010e-18, 2.52509383e-18, 3.10060111e-18,
    3.77240590e-18, 4.57631705e-18, 5.56615307e-18, 6.81853823e-18,
    8.43963865e-18, 1.05859920e-17, 1.34965568e-17, 1.75459361e-17,
    2.33481592e-17, 3.19604388e-17, 4.53154541e-17, 6.72167101e-17,
    1.05920866e-16, 1.81616023e-16, 3.47861509e-16, 7.21237876e-16,
    1.48067255e-15, 2.93093031e-15, 5.64800409e-15, 1.06989652e-14,
    2.00878688e-14, 3.77808368e-14, 7.22302972e-14, 1.43383832e-13,
    3.05156296e-13, 7.33842880e-13, 2.17173223e-12, 8.46572438e-12,
    3.36169517e-11, 9.76086987e-11, 2.33017574e-10, 5.67712619e-10,
    9.11487654e-10, 8.56202000e-10, 7.52593906e-10, 9.43947661e-10,
    1.32873116e-09, 1.37747324e-09, 1.30539037e-09, 1.37342559e-09,
    1.29147888e-09, 1.08716145e-09, 1.00736203e-09, 1.11860841e-09,
    1.14160502e-09, 1.00699114e-09, 9.21513703e-10, 9.74060577e-10,
    9.93534623e-10, 8.81959223e-10, 7.74759150e-10, 7.71032650e-10,
    8.37587901e-10, 8.16795571e-10, 7.20759823e-10, 6.51029606e-10,
    6.62415467e-10, 7.03988314e-10, 6.71627240e-10, 5.91227103e-10,
    5.30420826e-10, 5.28010932e-10, 5.61457760e-10, 5.54097048e-10,
    4.97240019e-10, 4.38143816e-10, 4.10807933e-10, 4.23263959e-10,
    4.37946210e-10, 4.14226828e-10, 3.67168087e-10, 3.25437359e-10,
    3.07535836e-10, 3.15237884e-10, 3.23951387e-10, 3.08284034e-10,
    2.75729931e-10, 2.44114937e-10, 2.25563570e-10, 2.24850188e-10,
    2.32481161e-10, 2.29975439e-10, 2.12320086e-10, 1.89433154e-10,
    1.70306840e-10, 1.60991322e-10, 1.62537260e-10, 1.67776502e-10,
    1.65890283e-10, 1.54524291e-10, 1.39598025e-10, 1.26548517e-10,
    1.19232112e-10, 1.19219914e-10, 1.23651873e-10, 1.25875820e-10,
    1.21512353e-10, 1.12529778e-10, 1.02869318e-10, 9.54425259e-11,
    9.24026167e-11, 9.42274545e-11, 9.85622288e-11, 1.00941147e-10,
    9.85813660e-11, 9.27309395e-11, 8.60566244e-11, 8.06044199e-11,
    7.79943434e-11, 7.90536319e-11, 8.29536153e-11, 8.69702612e-11,
    8.80444659e-11, 8.54448933e-11, 8.07907366e-11, 7.59345915e-11,
    7.22833205e-11, 7.10044415e-11, 7.26649795e-11, 7.66832199e-11,
    8.09912249e-11, 8.31109952e-11, 8.20258234e-11, 7.86976865e-11,
    7.46693105e-11, 7.11609015e-11, 6.91968684e-11, 6.95626847e-11,
    7.23987220e-11, 7.68195840e-11, 8.09401854e-11, 8.28252898e-11,
    8.18736126e-11, 7.89691864e-11, 7.53907091e-11, 7.21581027e-11,
    7.01331462e-11, 7.00398556e-11, 7.21920670e-11, 7.61821136e-11,
    8.07339631e-11, 8.40767493e-11, 8.49627827e-11, 8.34425065e-11,
    8.04900694e-11, 7.71481820e-11, 7.42332470e-11, 7.24642277e-11,
    7.24501764e-11, 7.44777754e-11, 7.82741073e-11, 8.28664014e-11,
    8.67637639e-11, 8.86467032e-11, 8.81336831e-11, 8.58256131e-11,
    8.26699025e-11, 7.94705961e-11, 7.68773306e-11, 7.54889867e-11,
    7.57872429e-11, 7.79604340e-11, 8.17296419e-11, 8.62492323e-11,
    9.02394035e-11, 9.24921945e-11, 9.24959305e-11, 9.06040338e-11,
    8.76148738e-11, 8.42920553e-11, 8.12342304e-11, 7.89676754e-11,
    7.79755506e-11, 7.86024576e-11, 8.09186135e-11, 8.46011276e-11,
    8.88228373e-11, 9.24781339e-11, 9.45882437e-11, 9.46620830e-11,
    9.29144250e-11, 8.99905279e-11, 8.65700467e-11, 8.31929606e-11,
])

# ── load MC sigma2_obs from depth logs ───────────────────────────────────────
def load_sigma2(path):
    with open(path) as f:
        for line in f:
            if line.startswith("# sigma2_obs,"):
                return np.array(line.strip().split(",")[1:], dtype=float)

sigma2 = {
    50:  load_sigma2("/Users/a.adamopoulou/code/stage2_depth_log_mc50.csv"),
    100: load_sigma2("/Users/a.adamopoulou/code/stage2_depth_log_mc100.csv"),
    500: load_sigma2("/Users/a.adamopoulou/code/stage2_depth_log_nic500.csv"),
}

# ── load parameter logs ───────────────────────────────────────────────────────
truth = {"kappa_surf": 200.0, "kappa_deep": 100.0, "y_trans": 0.189}

def load_params(path):
    iters, ks, kd, yt = [], [], [], []
    with open(path) as f:
        for line in f:
            if line.startswith("#") or line.startswith("iter") or not line.strip():
                continue
            parts = line.strip().split(",")
            iters.append(int(parts[0]))
            ks.append(float(parts[4]))
            kd.append(float(parts[5]))
            yt.append(float(parts[6]))
    return np.array(iters), np.array(ks), np.array(kd), np.array(yt)

logs = {
    50:  load_params("/Users/a.adamopoulou/code/stage2_log_mc50.csv"),
    100: load_params("/Users/a.adamopoulou/code/stage2_log_mc100.csv"),
    500: load_params("/Users/a.adamopoulou/code/stage2_log_nic_500.csv"),
}

def rms_err(ks, kd, yt):
    e_ks = (ks - truth["kappa_surf"]) / truth["kappa_surf"]
    e_kd = (kd - truth["kappa_deep"]) / truth["kappa_deep"]
    e_yt = (yt - truth["y_trans"])    / truth["y_trans"]
    return np.sqrt((e_ks**2 + e_kd**2 + e_yt**2) / 3)

colors = {50: "#e07b39", 100: "#4878cf", 500: "#6acc65"}
t = np.arange(len(analytical))
eps = 1e-35

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# ── Plot 1: sigma2_obs (N=500 MC vs analytical) — model discrepancy ───────────
ax = axes[0]
ax.semilogy(t, analytical + eps, color="k", lw=2, ls="--",
            label="Analytical SG (inference model)", zorder=5)
ax.semilogy(t, sigma2[500] + eps, color=colors[500], lw=1.8, alpha=0.9,
            label=r"MC $N=500$ (observations)")
# annotate the systematic gap in the smooth region
t_ann = 20
ax.annotate("", xy=(t_ann, analytical[t_ann]),
            xytext=(t_ann, sigma2[500][t_ann]),
            arrowprops=dict(arrowstyle="<->", color="#c0392b", lw=1.5))
ax.text(t_ann + 2, np.sqrt(analytical[t_ann] * sigma2[500][t_ann]),
        r"${\sim}11.5\%$", color="#c0392b", fontsize=9, va="center")
ax.set_xlim(0, 100)
ax.set_xlabel("Time step", fontsize=11)
ax.set_ylabel(r"$\sigma^2(t)$ (m$^2$)", fontsize=11)
ax.set_title("Model discrepancy: MC observations vs analytical SG", fontsize=11)
ax.legend(fontsize=9, framealpha=0.8)
ax.grid(True, which="both", ls="--", alpha=0.35)

# ── Plot 2: per-parameter error bars (signed) across N values ─────────────────
ax = axes[1]

Ns     = [20, 50, 100, 500]
params = [r"$\kappa_\mathrm{surf}$", r"$\kappa_\mathrm{deep}$", r"$y_\mathrm{trans}$"]
errs = {
    20:  [-1.16, +15.97, -0.70],
    50:  [-3.03,  -1.49, -0.25],
    100: [+4.55,  -1.25, +0.22],
    500: [-0.15,  -1.75, +0.28],
}
bar_colors = {20: "#9b59b6", 50: "#e07b39", 100: "#4878cf", 500: "#6acc65"}

n_params = 3
n_Ns     = len(Ns)
width    = 0.18
x        = np.arange(n_params)

for i, N in enumerate(Ns):
    offset = (i - (n_Ns - 1) / 2) * width
    bars = ax.bar(x + offset, errs[N], width=width,
                  color=bar_colors[N], label=f"$N={N}$", alpha=0.85)

ax.axhline(0, color="k", lw=1.2)
ax.set_xticks(x)
ax.set_xticklabels(params, fontsize=11)
ax.set_ylabel("Parameter error (\\%)", fontsize=11)
ax.set_title("Converged parameter errors by sample size", fontsize=11)
ax.legend(fontsize=9, framealpha=0.8)
ax.grid(True, axis="y", ls="--", alpha=0.35)

fig.suptitle("Non-inverse-crime ablation: varying MC sample size $N$", fontsize=12, y=1.01)
fig.tight_layout()
fig.savefig("/Users/a.adamopoulou/code/nic_ablation.png", dpi=150, bbox_inches="tight")
print("Saved nic_ablation.png")
plt.show()
