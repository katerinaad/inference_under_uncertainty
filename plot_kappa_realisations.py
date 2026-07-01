import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from stable_eigh_test import SigmoidLayeredKappa, precompute_fem_coo, assemble_K_M

# ── Parameters (match inf_layered_vap_exp.py) ─────────────────────────────
Nx, Ny   = 95, 95
Lx, Ly   = 0.21, 0.21
N_KL     = 95
sigma_field = 10.0
m0 = 2650 * 1050    # mean rho*Cp  [J/(m³·K)]
m1 = 2650 * 2000     # stochastic amplitude

theta_init  = np.array([180.0, 150.0, 0.8 * Ly, 0.01 * Ly])
theta_truth = np.array([380.0, 200.0, 0.9 * Ly, 0.01 * Ly])

kappa_param = SigmoidLayeredKappa(
    Ny=Ny, Ly=Ly,
    kappa_surface=200.0, kappa_deep=100.0,
    y_transition=0.9 * Ly, width=0.01 * Ly,
)

# ── KL modes via SPDE ─────────────────────────────────────────────────────
coo = precompute_fem_coo(Nx, Ny, Lx, Ly)
n   = coo['n']    # (Nx+1)*(Ny+1) nodes

def get_kl_modes(theta):
    K, M = assemble_K_M(coo, kappa_param.kappa_y(theta))
    Kf   = spla.factorized((K + 1e-10 * sp.eye(n, format='csc')).tocsc())
    def Cv(z): v = Kf(z); return Kf(M @ v)
    C_op = spla.LinearOperator((n, n), matvec=Cv, dtype=float)
    ev, phi = spla.eigsh(C_op, k=N_KL, which='LM', tol=1e-10)
    idx = np.argsort(ev)[::-1]
    return ev[idx], phi[:, idx]

print("Computing KL modes for initial kappa...")
ev_init,  phi_init  = get_kl_modes(theta_init)
print("Computing KL modes for truth kappa...")
ev_truth, phi_truth = get_kl_modes(theta_truth)

# ── rho*Cp realisations ───────────────────────────────────────────────────
np.random.seed(42)
n_samples = 2

def sample_rho_cp(ev, phi, xi):
    grf = sigma_field * (phi * np.sqrt(np.maximum(ev, 0.0))[None, :]) @ xi
    return (m0 + m1 * grf).reshape(Nx + 1, Ny + 1)

xis = np.random.randn(N_KL, n_samples)
samples_init  = [sample_rho_cp(ev_init,  phi_init,  xis[:, i]) for i in range(n_samples)]
samples_truth = [sample_rho_cp(ev_truth, phi_truth, xis[:, i]) for i in range(n_samples)]

# ── Load log files for final kappa estimates ───────────────────────────────
def load_csv(fname):
    with open(fname) as f:
        lines = [l for l in f if not l.startswith("#")]
    header = lines[0].strip().split(",")
    rows   = [list(map(float, l.strip().split(","))) for l in lines[1:] if l.strip()]
    return {h: np.array([r[i] for r in rows]) for i, h in enumerate(header)}

files  = {
          "50 KL": "stage2_log_3kappa_50KL_new.csv"}
colors = {"50 KL": "#6acc65"}

finals = {}
for label, fname in files.items():
    d = load_csv(fname)
    finals[label] = np.array([d["kappa_surf"][-1], d["kappa_deep"][-1],
                               d["y_trans"][-1],   d["width"][-1]])

# ── Plot helpers ──────────────────────────────────────────────────────────
x_nodes = np.linspace(0, Lx, Nx + 1) * 100    # cm
y_nodes = np.linspace(0, Ly, Ny + 1) * 100    # cm
y_elem  = (np.arange(Ny) * (Ly / Ny) + 0.5 * (Ly / Ny)) * 100   # cm

# shared colourbar range across all rho_Cp panels
all_vals = np.concatenate([s.ravel() for s in samples_init + samples_truth])
vmin, vmax = np.percentile(all_vals, 2), np.percentile(all_vals, 98)

def save(fig, fname):
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    print(f"Saved {fname}")
    plt.close(fig)

# (a) kappa(y) profile ─────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(4, 6))
ax.plot(kappa_param.kappa_y(theta_init),  y_elem, color="grey", ls=":",  lw=1.8, label="Initial")
ax.plot(kappa_param.kappa_y(theta_truth), y_elem, color="k",    ls="--", lw=2.0, label="Truth")
for label, theta in finals.items():
    ax.plot(kappa_param.kappa_y(theta), y_elem,
            color=colors[label], lw=1.5, label=f"Final ({label})")
ax.set_xlabel(r"$\kappa(y)$", fontsize=11)
ax.set_ylabel("Depth (cm)", fontsize=11)
ax.set_title(r"Correlation length profile $\kappa(y)$", fontsize=11)
ax.invert_yaxis()
ax.legend(fontsize=9, framealpha=0.7, loc="lower right")
ax.grid(True, ls="--", alpha=0.4)
fig.tight_layout()
save(fig, "kappa_profile.png")

# (b,c) rho_Cp — initial kappa ────────────────────────────────────────────
for j in range(n_samples):
    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.pcolormesh(x_nodes, y_nodes, samples_init[j].T,
                       cmap="RdYlBu_r", vmin=vmin, vmax=vmax, shading="auto")
    ax.set_title(fr"$\rho C_p$ — initial $\kappa$, sample {j + 1}", fontsize=11)
    ax.set_xlabel("$x$ (cm)", fontsize=10)
    ax.set_ylabel("$y$ (cm)", fontsize=10)
    plt.colorbar(im, ax=ax, label=r"$\rho C_p$ [J m$^{-3}$ K$^{-1}$]")
    fig.tight_layout()
    save(fig, f"rhocp_init_sample{j + 1}.png")

# (d,e) rho_Cp — truth kappa ──────────────────────────────────────────────
for j in range(n_samples):
    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.pcolormesh(x_nodes, y_nodes, samples_truth[j].T,
                       cmap="RdYlBu_r", vmin=vmin, vmax=vmax, shading="auto")
    ax.set_title(fr"$\rho C_p$ — truth $\kappa$, sample {j + 1}", fontsize=11)
    ax.set_xlabel("$x$ (cm)", fontsize=10)
    ax.set_ylabel("$y$ (cm)", fontsize=10)
    plt.colorbar(im, ax=ax, label=r"$\rho C_p$ [J m$^{-3}$ K$^{-1}$]")
    fig.tight_layout()
    save(fig, f"rhocp_truth_sample{j + 1}.png")
