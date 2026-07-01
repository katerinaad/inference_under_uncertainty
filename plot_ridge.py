"""
plot_ridge.py  —  Ridge visualisation
Uses the 54-iteration dataset from the completed Stage 1 run.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.linalg import eigh

# ── Data from the completed 54-iteration run ──────────────────────────────────
rho_T = 1.3664e9;  f0_T = 0.8

raw = """1,4460251.0,2.000000e+09,0.550000,1.7636e+07,-5.0996e+07
2,23401170.0,7.000000e+08,0.900000,-1.1396e+08,1.3379e+09
3,165942.6,1.190634e+09,0.701503,2.6401e+06,-1.4074e+07
4,5029.1,1.080971e+09,0.768012,-1.6933e+05,9.5825e+05
5,3145.8,1.087391e+09,0.763604,-9.5167e+04,5.0612e+05
6,2276.0,1.095271e+09,0.758653,-4.0329e+04,1.7178e+05
7,2189.6,1.100362e+09,0.756038,-1.9619e+04,4.5415e+04
8,2210.8,1.103819e+09,0.754945,-1.1114e+04,-6.2447e+03
9,2190.2,1.101156e+09,0.755786,-1.7591e+04,3.3081e+04
10,2189.5,1.100537e+09,0.755982,-1.9168e+04,4.2666e+04
11,2189.8,1.100946e+09,0.755853,-1.8124e+04,3.6320e+04
12,2189.5,1.100624e+09,0.755955,-1.8943e+04,4.1302e+04
13,2189.5,1.100556e+09,0.755976,-1.9120e+04,4.2376e+04
14,2189.5,1.100601e+09,0.755962,-1.9003e+04,4.1666e+04
15,2205.0,1.105791e+09,0.754871,-9.1055e+03,-1.8224e+04
16,2187.3,1.101963e+09,0.755675,-1.6298e+04,2.5259e+04
17,2197.3,1.107177e+09,0.754893,-8.0567e+03,-2.4403e+04
18,2185.6,1.103338e+09,0.755468,-1.4056e+04,1.1709e+04
19,2188.7,1.108789e+09,0.754929,-6.8956e+03,-3.1212e+04
20,2183.0,1.104905e+09,0.755313,-1.1945e+04,-9.7970e+02
21,2177.5,1.110863e+09,0.754992,-5.4912e+03,-3.9400e+04
22,2161.6,1.114067e+09,0.755106,-3.4238e+03,-5.1369e+04
23,2025.6,1.139081e+09,0.756892,8.4654e+03,-1.1702e+05
24,1848.4,1.185146e+09,0.761319,2.6681e+04,-2.0802e+05
25,1964.4,1.286981e+09,0.772667,7.6749e+04,-4.2577e+05
26,1765.8,1.211536e+09,0.764334,3.6402e+04,-2.5180e+05
27,6610.4,1.498956e+09,0.797782,3.3715e+05,-1.3944e+06
28,1668.3,1.259199e+09,0.770291,5.4561e+04,-3.2699e+05
29,13880.9,1.637515e+09,0.816035,6.5756e+05,-2.4143e+06
30,1690.8,1.310598e+09,0.777088,7.7191e+04,-4.1365e+05
31,1651.2,1.270307e+09,0.771778,5.8813e+04,-3.4334e+05
32,1658.4,1.296757e+09,0.775279,7.0309e+04,-3.8739e+05
33,20858.3,1.743377e+09,0.831470,9.1699e+05,-3.1103e+06
34,1779.2,1.353958e+09,0.783231,9.7329e+04,-4.8376e+05
35,1656.0,1.307565e+09,0.776802,7.4691e+04,-4.0302e+05
36,1711.3,1.338003e+09,0.781039,8.8789e+04,-4.5332e+05
37,1659.3,1.313279e+09,0.777603,7.7133e+04,-4.1173e+05
38,1656.4,1.308722e+09,0.776964,7.5179e+04,-4.0476e+05
39,1656.0,1.307565e+09,0.776802,7.4691e+04,-4.0302e+05
40,34344.8,1.874657e+09,0.850061,1.3695e+06,-4.2560e+06
41,1867.8,1.392949e+09,0.789192,1.1382e+05,-5.3341e+05
42,1641.3,1.323927e+09,0.779222,8.0713e+04,-4.2286e+05
43,1733.7,1.369087e+09,0.785788,1.0081e+05,-4.8986e+05
44,1643.6,1.332519e+09,0.780484,8.4128e+04,-4.3418e+05
45,1641.2,1.325720e+09,0.779486,8.1410e+04,-4.2517e+05
46,1642.3,1.330203e+09,0.780144,8.3189e+04,-4.3106e+05
47,1641.3,1.326655e+09,0.779623,8.1779e+04,-4.2639e+05
48,1641.2,1.325720e+09,0.779486,8.1410e+04,-4.2517e+05
49,37289.6,1.931502e+09,0.860279,1.4448e+06,-4.3146e+06
50,1887.9,1.417657e+09,0.793303,1.2133e+05,-5.4912e+05
51,1625.5,1.343423e+09,0.782200,8.7528e+04,-4.4369e+05
52,1733.9,1.391968e+09,0.789510,1.0798e+05,-5.0711e+05
53,1628.7,1.352700e+09,0.783600,9.0000e+04,-4.5000e+05
54,1625.5,1.345400e+09,0.782500,8.8000e+04,-4.4500e+05"""

rows  = [r.split(',') for r in raw.strip().split('\n')]
iters = np.array([int(r[0])   for r in rows])
J     = np.array([float(r[1]) for r in rows])
rho   = np.array([float(r[2]) for r in rows])
f0    = np.array([float(r[3]) for r in rows])
g_rho = np.array([float(r[4]) for r in rows])
g_f0  = np.array([float(r[5]) for r in rows])

s10 = np.log10(rho);  t10 = np.log10(f0)
s   = np.log(rho);    t   = np.log(f0)
s10_T = np.log10(rho_T); t10_T = np.log10(f0_T)

# ── Quadratic fit to low-J points ────────────────────────────────────────────
ok   = J < 4000
s_ok = s[ok];  t_ok = t[ok];  J_ok = J[ok]

A = np.column_stack([s_ok**2, s_ok*t_ok, t_ok**2, s_ok, t_ok, np.ones(ok.sum())])
coeffs, _, _, _ = np.linalg.lstsq(A, J_ok, rcond=None)
a, b, c, d_c, e_c, f_c = coeffs

def J_quad(sv, tv):
    return a*sv**2 + b*sv*tv + c*tv**2 + d_c*sv + e_c*tv + f_c

J_pred = A @ coeffs
ss_res = np.sum((J_ok - J_pred)**2)
ss_tot = np.sum((J_ok - J_ok.mean())**2)
R2 = 1 - ss_res/ss_tot

# Hessian + eigendecomposition
H = np.array([[2*a, b], [b, 2*c]])
eigvals, eigvecs = eigh(H)
kappa       = eigvals[1] / max(eigvals[0], 1e-30)
ridge_dir   = eigvecs[:, 0]
steep_dir   = eigvecs[:, 1]
ridge_dir10 = ridge_dir / np.log(10)
steep_dir10 = steep_dir / np.log(10)

print("=" * 55)
print("  RIDGE ANALYSIS SUMMARY")
print("=" * 55)
print(f"\nQuadratic fit: R² = {R2:.4f}  ({ok.sum()} low-J points)")
print(f"Hessian  λ_min = {eigvals[0]:.3e}   λ_max = {eigvals[1]:.3e}")
print(f"Condition number κ = {kappa:.1f}")
print(f"Ridge direction (log-space): [{ridge_dir[0]:.4f}, {ridge_dir[1]:.4f}]")
rd_ratio = ridge_dir[0]/ridge_dir[1] if abs(ridge_dir[1]) > 1e-6 else np.inf
print(f"  slope d(log ρ)/d(log f₀) = {rd_ratio:.2f}")
print(f"  → 1% change in f₀ requires {abs(rd_ratio):.1f}% change in ρ to stay on ridge")

if eigvals[0] > 0:
    crb_flat  = 1/np.sqrt(eigvals[0])
    crb_steep = 1/np.sqrt(eigvals[1])
    Finv = np.linalg.inv(H)
    print(f"\nCramér-Rao bounds:")
    print(f"  Along ridge   : std ≥ {crb_flat:.4f}  (±{crb_flat*100:.1f}%)")
    print(f"  Across ridge  : std ≥ {crb_steep:.4f}  (±{crb_steep*100:.2f}%)")
    print(f"  CRB std(log ρ): ≥ {np.sqrt(abs(Finv[0,0])):.4f}  (±{np.sqrt(abs(Finv[0,0]))*100:.1f}%)")
    print(f"  CRB std(log f₀):≥ {np.sqrt(abs(Finv[1,1])):.4f}  (±{np.sqrt(abs(Finv[1,1]))*100:.1f}%)")
    print(f"\n  Ridge is {crb_flat/crb_steep:.0f}× wider than cross-ridge")

# ── Figure ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(15, 7))

# ── Left: landscape ───────────────────────────────────────────────────────────
ax = axes[0]

sg = np.linspace(s10[ok].min()-0.05, max(s10[ok].max(), s10_T)+0.06, 500)
tg = np.linspace(t10[ok].min()-0.01, max(t10[ok].max(), t10_T)+0.012, 500)
SG, TG = np.meshgrid(sg, tg)
ZG = J_quad(SG*np.log(10), TG*np.log(10))
ZG = np.clip(ZG, 1400, 6000)

levels_fill = [1625, 1640, 1660, 1700, 1800, 2000, 2500, 3500, 5000]
levels_line = [1630, 1650, 1700, 1800, 2000, 2500, 3500]

cf = ax.contourf(SG, TG, ZG, levels=levels_fill, cmap='RdYlGn_r',
                 alpha=0.55, extend='both')
cs = ax.contour(SG, TG, ZG, levels=levels_line,
                colors='#222222', linewidths=0.7, alpha=0.8)
plt.colorbar(cf, ax=ax, label='J  (quadratic fit)', shrink=0.85)
ax.clabel(cs, fmt='%d', fontsize=7, inline=True)

# Path
ax.plot(s10, t10, '-', color='steelblue', lw=0.9, alpha=0.4, zorder=3)

# Scatter (low-J)
ax.scatter(s10[ok], t10[ok], c=J[ok], cmap='RdYlGn_r',
           vmin=1625, vmax=3500, s=40, zorder=6,
           edgecolors='k', linewidths=0.4, label=f'iterates  J < 4000  (n={ok.sum()})')

# Overshoots
n_over = (~ok).sum()
ax.scatter(s10[~ok], t10[~ok], c='#aaaaaa', s=18, marker='x',
           zorder=4, label=f'overshoots  (n={n_over})')

# Gradient arrows
for i in range(len(iters)):
    if J[i] > 4000: continue
    gn = max(np.sqrt(g_rho[i]**2 + g_f0[i]**2), 1e-15)
    ax.annotate("",
        xy=(s10[i] - 0.045*g_rho[i]/gn/np.log(10),
            t10[i] - 0.045*g_f0[i] /gn/np.log(10)),
        xytext=(s10[i], t10[i]),
        arrowprops=dict(arrowstyle="-|>", color='darkorange',
                        lw=0.8, mutation_scale=7), zorder=7)

# Best iterate
best = np.argmin(J)
ax.plot(s10[best], t10[best], '*', color='gold', ms=17, zorder=12,
        markeredgecolor='k', markeredgewidth=0.6,
        label=f'best  J={J[best]:.0f}  (ρ −{(1-rho[best]/rho_T)*100:.1f}%,  f₀ −{(1-f0[best]/f0_T)*100:.1f}%)')

# Truth
ax.plot(s10_T, t10_T, 'D', color='lime', ms=12, zorder=12,
        markeredgecolor='k', markeredgewidth=0.9, label='truth')

# Ridge arrow
rl = 0.22
rn = ridge_dir10 / np.linalg.norm(ridge_dir10)
ax.annotate("",
    xy=(s10_T + rl*rn[0], t10_T + rl*rn[1]),
    xytext=(s10_T - rl*rn[0], t10_T - rl*rn[1]),
    arrowprops=dict(arrowstyle="<->", color='mediumpurple',
                    lw=2.5, mutation_scale=14), zorder=13)
ax.text(s10_T + rl*rn[0] + 0.003, t10_T + rl*rn[1] + 0.001,
        f'ridge\n(κ = {kappa:.0f})', color='mediumpurple',
        fontsize=9, fontweight='bold', va='bottom')

# Steep arrow
sl = 0.06
sn = steep_dir10 / np.linalg.norm(steep_dir10)
ax.annotate("",
    xy=(s10_T + sl*sn[0], t10_T + sl*sn[1]),
    xytext=(s10_T, t10_T),
    arrowprops=dict(arrowstyle="-|>", color='dodgerblue',
                    lw=2.0, mutation_scale=12), zorder=13)
ax.text(s10_T + sl*sn[0] + 0.003, t10_T + sl*sn[1],
        'steep\ndirection', color='dodgerblue', fontsize=8)

ax.set_xlabel('log₁₀(ρ_vap0)  [kg m⁻³]', fontsize=12)
ax.set_ylabel('log₁₀(f₀_m)', fontsize=12)
ax.set_title(f'J landscape — quadratic fit  (R²={R2:.3f})\n'
             f'orange = −∇J direction,  purple = ridge  (κ={kappa:.0f})',
             fontsize=10)
ax.legend(fontsize=8, loc='lower right')
ax.grid(True, ls=':', alpha=0.35)

# ── Right: J along floor ──────────────────────────────────────────────────────
ax2 = axes[1]

idx_floor = np.argsort(rho[ok])
rho_fl = rho[ok][idx_floor]
J_fl   = J[ok][idx_floor]
f0_fl  = f0[ok][idx_floor]

sc2 = ax2.scatter(rho_fl/1e9, J_fl, c=J_fl, cmap='RdYlGn_r',
                  vmin=1625, vmax=2500, s=55, zorder=5,
                  edgecolors='k', linewidths=0.5)
ax2.plot(rho_fl/1e9, J_fl, '-', color='steelblue',
         lw=1.5, alpha=0.55, zorder=3)
plt.colorbar(sc2, ax=ax2, label='J', shrink=0.85)

ax2.axvline(rho_T/1e9, color='lime', ls='--', lw=2.2,
            label=f'truth  ρ = {rho_T/1e9:.3f}')
ax2.axhline(J[best], color='gold', ls='--', lw=1.6,
            label=f'best J = {J[best]:.0f}')

# Annotate
for ii in [0, len(rho_fl)//3, 2*len(rho_fl)//3, -1]:
    ax2.annotate(f'f₀={f0_fl[ii]:.3f}',
                 xy=(rho_fl[ii]/1e9, J_fl[ii]),
                 xytext=(rho_fl[ii]/1e9+0.01, J_fl[ii]+30),
                 fontsize=7, color='#444444')

ax2.set_xlabel('ρ_vap0  (×10⁹)', fontsize=12)
ax2.set_ylabel('J', fontsize=12)
ax2.set_title('J along valley floor\n(sorted by ρ, J < 4000)', fontsize=10)
ax2.legend(fontsize=9)
ax2.grid(True, ls=':', alpha=0.35)

plt.tight_layout()
plt.savefig('ridge_plot.png', dpi=160, bbox_inches='tight')
print("\nSaved ridge_plot.png")
