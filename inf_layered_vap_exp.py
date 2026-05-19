import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.fft import fft2, ifft2, fftfreq
from scipy.sparse import lil_matrix, csr_matrix, kron, identity
from visualize_spatial import visualize_spatial_weights_history, visualize_variance_field
# Import fast assembly module (place fast_assembly.py in same directory or on PYTHONPATH)
try:
    from fast_assembly import FastAssembler, FastAssemblerNumba, HAS_NUMBA
    USE_FAST_ASSEMBLY = False
    print(f"Fast assembly enabled (Numba: {HAS_NUMBA})")
except ImportError:
    USE_FAST_ASSEMBLY = False
    print("Fast assembly not available, using original implementation")
from scipy.sparse.linalg import spsolve
from scipy.sparse.linalg import cg, gmres, LinearOperator, spilu
from scipy.sparse import coo_matrix
from scipy.sparse import diags
from numpy.polynomial.hermite_e import hermeval   # <- probabilists’ Hermite
import time
from scipy.sparse import diags, isspmatrix
from scipy.sparse.linalg import cg, LinearOperator, factorized
import jax
from scipy.sparse import coo_matrix
import functools
import os, tempfile
import gc; gc.collect()
import os
import jax
import jax.numpy as jnp
from stable_eigh_test import SPDEKLDifferentiator, LayeredKappa, SmoothKappa, SigmoidLayeredKappa
import scipy.sparse as sp
from depth_objective import softmin_depth, _trap_weights

def rss_mb():
    with open(f"/proc/{os.getpid()}/status") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                return float(line.split()[1]) / 1024.0
    return float("nan")

# -------------------------------
# Domain & FE Mesh Parameters (2D physical space)
# ------------------------------
Lx, Ly = 0.21, 0.21
Nx, Ny = 40,40
k0= 1.0       # baseline conductivity
k1 = 0.2 # scaling for fluctuations
m0 = 1e6
m1 = 1e5
f0= 1.0
f1 = 0.0
rhoL = 0
#rho_vap0 = 1e9    # latent heat of vaporisation (J/m^3); set >0 to activate vapour transition
#rho_vap1 = 1e7
T_final = 3.0
dt = 0.1
N_KL = 40 # number of KL terms
P = N_KL + 1     # total chaos modes (0th = mean, 1...N_KL = KL terms)
dx, dy = Lx/Nx, Ly/Ny
num_obs =100
alpha_k = alpha_m=0
t_off = 50000
sigma_d = 1e-10
#KAPPA OBS
#kappa_param = LayeredKappa(
#    Ny=Ny, Ly=Ly,
#    layer_boundaries=[0.0,0.06,0.1],  # from geology
#    kappa_values=[100,10.0]             # initial guess
#)
#kappa_param = SmoothKappa(Ny, Ly, kappa0=30.0, strength=0.0)
kappa_param_obs = SigmoidLayeredKappa(
    Ny=Ny,
    Ly=Ly,
    kappa_surface=80.0,    # short correlation near surface (weathered/fractured)
    kappa_deep=60.0,       # longer correlation at depth (competent granite)
    y_transition=0.75*Ly,   # transition at 60% depth
    width=0.1*Ly,          # transition zone ~10% of domain width
    )
kappa_param_init = SigmoidLayeredKappa(
    Ny=Ny,
    Ly=Ly,
    kappa_surface=100.0,    # short correlation near surface (weathered/fractured)
    kappa_deep=40.0,       # longer correlation at depth (competent granite)
    y_transition=0.85*Ly,   # transition at 60% depth
    width=0.1*Ly,          # transition zone ~10% of domain width
    )
theta_kappa_init = np.array([100.0,40.0, 0.85*Ly,0.1*Ly])
theta_kappa_obs = np.array([80.0,60.0, 0.75*Ly,0.1*Ly])
np.random.seed(0)
#Generate GMRF  and assemble SG matrices
# Generate the GMRF
sigma = 1.0  # standard deviation
ell = 0.5 # correlation length
xi_sample = np.random.normal(0, 1, N_KL)
steps = int(T_final/dt)

fname = "U_hist.dat"   # or tempfile.NamedTemporaryFile(delete=False).name

names = ["k0","k1","m0","m1", "f0", "f1", "ell"]
# ---- phase-change controls (choose) ----
T_melt_lo =173.0
T_melt_hi =273.0
Delta_melt = 50.0   # smoothing width

# ---- solid / melt parameter sets ----
SOLID =dict(k0=2.0,  k1=0.0, m0=2650 * 1050, m1= 2650 * 1050, f0=1.0, f1=0.0)
MELT  =  dict(k0=2.0,  k1=0.0, m0=2650 * 1050, m1= 2650 * 1050, f0=1.0, f1=0.0)
#SOLID_obs = dict(k0=2.0,  k1=0.0, m0=2.3e6, m1=2e5, f0=1.0, f1=0.0)
#MELT_obs  = dict(k0=2.0,  k1=0.0, m0=2.0e6, m1=2e5, f0=1.0, f1=0.0)
# ---- vapour parameter set (properties above the vaporisation front) ----
#VAP_obs = dict(k0=1.0, k1=0.0, m0=2.0e6, m1=5e5, f0=0.1, f1=0.0, rho_vap0 = 5e8, rho_vap1 = 1.5e7)

SOLID_obs = dict(
    k0  = 2.0,              # W/(m·K) — Gokhale Table 1
    k1  = 0.0,
    m0  = 2650 * 1050,      # 2.7825e6 J/(m³·K) — rho*Cp solid
    m1  = 2650 * 1050,
    f0  = 1.0,             # 0.75 * 0.83 — transmissivity * absorptivity correction
    f1  = 0.0,
)

MELT_obs = dict(
    k0  = 2.0,              # W/(m·K) — same as solid in Gokhale Table 1
    k1  = 0.0,
    m0  = 2650 * 1050,      # 4.1605e6 J/(m³·K) — rho*Cp melt
    m1  = 2650 * 1050,
    f0  = 1.0,             # same absorptivity correction as solid
    f1  = 0.0
)

# ---- vapour parameter set ----
VAP = dict(k0=0.26, k1=0.0, m0=2650  * 1570, m1=2650  * 1570, f0=0.25, f1=0.0,
           rho_vap0=1266400000, rho_vap1=15664000*0.2)

VAP_obs = dict(
    k0      = 0.26,            # W/(m·K) — vapour thermal conductivity
    k1      = 0.0,
    m0      =  2650  * 1570,      # 1010.6 J/(m³·K) — rho_v * Cp_v
    m1      =  2650  * 1570,
    f0      = 0.1,
    f1      = 0.0,
   # rho_vap0 = 2650 * 1, # 3.621e10 J/m³ — rho_melt * L_v (Gokhale Table 1)414.65414574953286
    rho_vap0 =  1366400000 ,
    rho_vap1 = 13664000*0.2 ,
)
theta_lab = np.array([VAP['rho_vap0'], VAP['rho_vap1'], 100, 40, 0.85*Ly, 0.1*Ly])

# ---- vaporisation phase controls ----
T_vap_lo    = 354.0          # onset  of vaporisation window
T_vap_hi    = 554.0          # end    of vaporisation window
Delta_vap   = 50.0            # smoothing half-width (same role as Delta_melt)
T_abl = 450.0

# Mesh
x = np.linspace(0, Lx, Nx + 1)
y = np.linspace(0, Ly, Ny + 1)
hx, hy = x[1] - x[0], y[1] - y[0]
num_nodes = (Nx + 1) * (Ny + 1)
num_elements = (Nx+1) * (Ny+1)
p_order = 1                    # ← raise this 1,2,3,...

# Initialize fast assembler (if available)
_fast_assembler = None
if USE_FAST_ASSEMBLY:
    # Use regular FastAssembler (NumPy-based) - avoids Numba JIT compilation overhead
    # FastAssemblerNumba is faster after JIT, but the compilation adds ~300ms on first call
    # Since assembly is only ~1% of total runtime, the JIT overhead isn't worth it
    _fast_assembler = FastAssembler(Nx, Ny, Lx, Ly, x, y)



from itertools import product

def elem_to_node_weights(coeff_elem, Nx, Ny):
    """Convert element coefficients to node weights (averaging)."""
    num_nodes = (Nx + 1) * (Ny + 1)
    Ny1 = Ny + 1
   
    node_weights = np.zeros(num_nodes)
    node_counts = np.zeros(num_nodes)
   
    ii, jj = np.meshgrid(np.arange(Nx), np.arange(Ny), indexing='ij')
    ii, jj = ii.ravel(), jj.ravel()
    c = coeff_elem.ravel()
   
    n0 = ii * Ny1 + jj
    n1 = ii * Ny1 + jj + 1
    n2 = (ii + 1) * Ny1 + jj
    n3 = (ii + 1) * Ny1 + jj + 1
   
    np.add.at(node_weights, n0, c)
    np.add.at(node_weights, n1, c)
    np.add.at(node_weights, n2, c)
    np.add.at(node_weights, n3, c)
    np.add.at(node_counts, n0, 1)
    np.add.at(node_counts, n1, 1)
    np.add.at(node_counts, n2, 1)
    np.add.at(node_counts, n3, 1)
   
    return node_weights / np.maximum(node_counts, 1)


def apply_K1_spatial(v, k1_elem, K_kl_global, G_list, sqrt_lam, P, num_nodes, Nx, Ny):
    """
    Apply K_SG_K1 with spatially-varying k1.
   
    Replaces: k1_eff * (K_SG_K1 @ v)
   
    Uses symmetric diagonal scaling for accuracy:
        K_m(k1) ≈ D^{1/2} @ K_m @ D^{1/2}
   
    Parameters
    ----------
    v : ndarray (P * num_nodes,)
        Input vector
    k1_elem : ndarray (Nx, Ny)
        Element-wise k1 coefficient
    K_kl_global : list of sparse matrices (N_KL,)
        Per-mode stiffness matrices (WITH sqrt_lam baked in)
    G_list : list of sparse matrices (N_KL,)
        Polynomial chaos coupling matrices
    sqrt_lam : ndarray (N_KL,)
        Square root of KL eigenvalues (for reference, already in K_kl_global)
    """
    N_KL = len(K_kl_global)
   
    # Convert element weights to node weights
    k1_nodes = elem_to_node_weights(k1_elem, Nx, Ny)
    k1_sqrt = np.sqrt(np.maximum(k1_nodes, 0))
   
    V = v.reshape(P, num_nodes)
    Y = np.zeros_like(V)
   
    # Scale input by sqrt(k1)
    V_scaled = V * k1_sqrt[None, :]
   
    for m in range(N_KL):
        G_m = G_list[m].tocoo()
        K_m = K_kl_global[m]  # Already has sqrt_lam[m] in it
       
        # For each chaos mode coupling
        for idx in range(len(G_m.data)):
            p, q = G_m.row[idx], G_m.col[idx]
            g_pq = G_m.data[idx]
           
            if abs(g_pq) < 1e-15:
                continue
           
            # K_m(k1) @ V[q] ≈ k1_sqrt * (K_m @ (k1_sqrt * V[q]))
            Kv = K_m @ V_scaled[q]
            Y[p] += g_pq * k1_sqrt * Kv
   
    return Y.ravel()


def apply_M1_spatial(v, m1_elem, M_kl_global, G_list, sqrt_lam, P, num_nodes, Nx, Ny):
    """
    Apply M_SG_M1 with spatially-varying m1.
   
    Replaces: m1_eff * (M_SG_M1 @ v)
    """
    N_KL = len(M_kl_global)
   
    m1_nodes = elem_to_node_weights(m1_elem, Nx, Ny)
    m1_sqrt = np.sqrt(np.maximum(m1_nodes, 0))
   
    V = v.reshape(P, num_nodes)
    Y = np.zeros_like(V)
   
    V_scaled = V * m1_sqrt[None, :]
   
    for m in range(N_KL):
        G_m = G_list[m].tocoo()
        M_m = M_kl_global[m]
       
        for idx in range(len(G_m.data)):
            p, q = G_m.row[idx], G_m.col[idx]
            g_pq = G_m.data[idx]
           
            if abs(g_pq) < 1e-15:
                continue
           
            Mv = M_m @ V_scaled[q]
            Y[p] += g_pq * m1_sqrt * Mv
   
    return Y.ravel()


# ============================================================================
# OPTIMIZED VERSION: Cache G structure and reuse
# ============================================================================

class SpatialK1M1Operator:
    """
    Optimized operator for repeated applications with changing k1/m1.
   
    Usage:
        op = SpatialK1M1Operator(K_kl_global, M_kl_global, G_list, P, num_nodes, Nx, Ny)
       
        # In Picard loop:
        op.update_weights(k1_elem, m1_elem)
       
        # In matvec:
        y_K1 = op.apply_K1(v)
        y_M1 = op.apply_M1(v)
    """
   
    def __init__(self, K_kl_global, M_kl_global, G_list, P, num_nodes, Nx, Ny):
        self.K_kl = [K.tocsr() for K in K_kl_global]
        self.M_kl = [M.tocsr() for M in M_kl_global]
        self.P = P
        self.num_nodes = num_nodes
        self.Nx = Nx
        self.Ny = Ny
        self.N_KL = len(K_kl_global)
       
        # Precompute G structure
        self.G_data = []
        for m in range(self.N_KL):
            G_coo = G_list[m].tocoo()
            mask = np.abs(G_coo.data) > 1e-15
            self.G_data.append({
                'p': G_coo.row[mask].astype(np.int32),
                'q': G_coo.col[mask].astype(np.int32),
                'g': G_coo.data[mask]
            })
       
        self.k1_sqrt = None
        self.m1_sqrt = None
   
    def update_weights(self, k1_elem, m1_elem):
        """Update spatial weights. Call once per Picard iteration."""
        k1_nodes = elem_to_node_weights(k1_elem, self.Nx, self.Ny)
        m1_nodes = elem_to_node_weights(m1_elem, self.Nx, self.Ny)
        self.k1_sqrt = np.sqrt(np.maximum(k1_nodes, 0))
        self.m1_sqrt = np.sqrt(np.maximum(m1_nodes, 0))
   
    def apply_K1(self, v):
        """Apply spatially-weighted K1."""
        V = v.reshape(self.P, self.num_nodes)
        Y = np.zeros_like(V)
       
        V_scaled = V * self.k1_sqrt[None, :]
       
        for m in range(self.N_KL):
            gd = self.G_data[m]
            if len(gd['p']) == 0:
                continue
           
            K_m = self.K_kl[m]
           
            # Cache K_m @ V_scaled[q] for unique q values
            unique_q, inv_idx = np.unique(gd['q'], return_inverse=True)
            Kv_cache = [self.k1_sqrt * (K_m @ V_scaled[q]) for q in unique_q]
           
            # Accumulate
            for i, (p, g) in enumerate(zip(gd['p'], gd['g'])):
                Y[p] += g * Kv_cache[inv_idx[i]]
       
        return Y.ravel()
   
    def apply_M1(self, v):
        """Apply spatially-weighted M1."""
        V = v.reshape(self.P, self.num_nodes)
        Y = np.zeros_like(V)
       
        V_scaled = V * self.m1_sqrt[None, :]
       
        for m in range(self.N_KL):
            gd = self.G_data[m]
            if len(gd['p']) == 0:
                continue
           
            M_m = self.M_kl[m]
           
            unique_q, inv_idx = np.unique(gd['q'], return_inverse=True)
            Mv_cache = [self.m1_sqrt * (M_m @ V_scaled[q]) for q in unique_q]
           
            for i, (p, g) in enumerate(zip(gd['p'], gd['g'])):
                Y[p] += g * Mv_cache[inv_idx[i]]
       
        return Y.ravel()
    def apply_K1_jacobian_k1(self, v, delta_k1):
        """
        Compute d(apply_K1)/d(k1) @ delta_k1
       
        This gives the directional derivative of apply_K1(v)
        in the direction delta_k1.
        """
        V = v.reshape(self.P, self.num_nodes)
        Y = np.zeros_like(V)
       
        # Derivative of sqrt(k1) w.r.t. k1
        # d(sqrt(k1))/d(k1) = 1/(2*sqrt(k1))
        w = self.k1_sqrt
        # Avoid division by zero
        dw_dk1 = np.where(w > 1e-15, 0.5 / w, 0.0)
        delta_w = dw_dk1 * delta_k1  # Chain rule: delta_w = dw/dk1 * delta_k1
       
        V_scaled = V * w[None, :]           # w * v
        V_delta_scaled = V * delta_w[None, :]  # delta_w * v
       
        for m in range(self.N_KL):
            gd = self.G_data[m]
            if len(gd['p']) == 0:
                continue
           
            K_m = self.K_kl[m]
            unique_q, inv_idx = np.unique(gd['q'], return_inverse=True)
           
            # Two terms from product rule:
            # Term 1: delta_w * (K_m @ (w * v_q))
            # Term 2: w * (K_m @ (delta_w * v_q))
           
            Kv_cache_term1 = [delta_w * (K_m @ V_scaled[q]) for q in unique_q]
            Kv_cache_term2 = [w * (K_m @ V_delta_scaled[q]) for q in unique_q]
           
            for i, (p, g) in enumerate(zip(gd['p'], gd['g'])):
                Y[p] += g * (Kv_cache_term1[inv_idx[i]] + Kv_cache_term2[inv_idx[i]])
       
        return Y.ravel()


    def apply_K1_vjp_k1(self, v, grad_output):
        """
        Compute the vector-Jacobian product: grad_output @ d(apply_K1)/d(k1)
       
        This is useful for backpropagation. Returns gradient w.r.t. k1.
        """
        V = v.reshape(self.P, self.num_nodes)
        G = grad_output.reshape(self.P, self.num_nodes)
       
        w = self.k1_sqrt
        dw_dk1 = np.where(w > 1e-15, 0.5 / w, 0.0)
       
        V_scaled = V * w[None, :]
        grad_w = np.zeros(self.num_nodes)
       
        for m in range(self.N_KL):
            gd = self.G_data[m]
            if len(gd['p']) == 0:
                continue
           
            K_m = self.K_kl[m]
           
            # For the gradient, we need to accumulate contributions
            # from both terms of the product rule
           
            unique_q, inv_idx = np.unique(gd['q'], return_inverse=True)
            unique_p, inv_idx_p = np.unique(gd['p'], return_inverse=True)
           
            # Precompute K_m @ V_scaled[q] and K_m.T @ G[p]
            Kv_cache = {q: K_m @ V_scaled[q] for q in unique_q}
            KtG_cache = {p: K_m.T @ G[p] for p in unique_p}
           
            for i, (p, q, g) in enumerate(zip(gd['p'], gd['q'], gd['g'])):
                # Term 1: delta_w * (K_m @ (w * v_q))
                # Gradient w.r.t. w: g * G[p] * (K_m @ (w * v_q))
                grad_w += g * G[p] * Kv_cache[q]
               
                # Term 2: w * (K_m @ (delta_w * v_q))
                # Gradient w.r.t. w at position of v_q: g * (K_m.T @ (w * G[p])) * v_q
                grad_w += g * (K_m.T @ (w * G[p])) * V[q]
       
        # Chain rule: grad_k1 = grad_w * dw/dk1
        grad_k1 = grad_w * dw_dk1
       
        return grad_k1


    def apply_M1_jacobian_m1(self, v, delta_m1):
        """
        Compute d(apply_M1)/d(m1) @ delta_m1
        """
        V = v.reshape(self.P, self.num_nodes)
        Y = np.zeros_like(V)
       
        w = self.m1_sqrt
        dw_dm1 = np.where(w > 1e-15, 0.5 / w, 0.0)
        delta_w = dw_dm1 * delta_m1
       
        V_scaled = V * w[None, :]
        V_delta_scaled = V * delta_w[None, :]
       
        for m in range(self.N_KL):
            gd = self.G_data[m]
            if len(gd['p']) == 0:
                continue
           
            M_m = self.M_kl[m]
            unique_q, inv_idx = np.unique(gd['q'], return_inverse=True)
           
            Mv_cache_term1 = [delta_w * (M_m @ V_scaled[q]) for q in unique_q]
            Mv_cache_term2 = [w * (M_m @ V_delta_scaled[q]) for q in unique_q]
           
            for i, (p, g) in enumerate(zip(gd['p'], gd['g'])):
                Y[p] += g * (Mv_cache_term1[inv_idx[i]] + Mv_cache_term2[inv_idx[i]])
       
        return Y.ravel()


    def apply_M1_vjp_m1(self, v, grad_output):
        """
        Compute the vector-Jacobian product for M1 w.r.t. m1.
        """
        V = v.reshape(self.P, self.num_nodes)
        G = grad_output.reshape(self.P, self.num_nodes)
       
        w = self.m1_sqrt
        dw_dm1 = np.where(w > 1e-15, 0.5 / w, 0.0)
       
        V_scaled = V * w[None, :]
        grad_w = np.zeros(self.num_nodes)
       
        for m in range(self.N_KL):
            gd = self.G_data[m]
            if len(gd['p']) == 0:
                continue
           
            M_m = self.M_kl[m]
           
            unique_q = np.unique(gd['q'])
            unique_p = np.unique(gd['p'])
           
            Mv_cache = {q: M_m @ V_scaled[q] for q in unique_q}
            MtG_cache = {p: M_m.T @ G[p] for p in unique_p}
           
            for i, (p, q, g) in enumerate(zip(gd['p'], gd['q'], gd['g'])):
                grad_w += g * G[p] * Mv_cache[q]
                grad_w += g * (M_m.T @ (w * G[p])) * V[q]
       
        grad_m1 = grad_w * dw_dm1
       
        return grad_m1


def enumerate_multi_indices(d, p):
    out, a = [], [0]*d
    def rec(i, remaining):
        if i == d:
            out.append(tuple(a)); return
        for v in range(remaining+1):
            a[i] = v
            rec(i+1, remaining-v)
    rec(0, p)
    return np.array(out, dtype=int)

multi_idx = enumerate_multi_indices(N_KL, p_order)   # shape (P, N_KL)
P          = multi_idx.shape[0]    
# -------------------------------
# Elements for Stochastic Galerkin Assembly
# -------------------------------

# Quadrature points and weights for 2x2 Gauss quadrature
quad_points = np.array([[-1 / np.sqrt(3), -1 / np.sqrt(3)],
                        [1 / np.sqrt(3), -1 / np.sqrt(3)],
                        [1 / np.sqrt(3), 1 / np.sqrt(3)],
                        [-1 / np.sqrt(3), 1 / np.sqrt(3)]])
quad_weights = np.array([1, 1, 1, 1])
X, Y = np.meshgrid(np.linspace(0, Lx, Nx, endpoint=False),
                   np.linspace(0, Ly, Ny, endpoint=False))
nodes = np.column_stack((X.flatten(), Y.flatten()))


# build a map from multi-index tuple -> row index
idx_of = {tuple(a): i for i,a in enumerate(multi_idx)}
rows = [[] for _ in range(N_KL)]
cols = [[] for _ in range(N_KL)]
data = [[] for _ in range(N_KL)]

for a_idx, alpha in enumerate(multi_idx):
    for m in range(N_KL):
        a = alpha[m]
        # up
        alpha_up = list(alpha); alpha_up[m] = a+1
        tup_up = tuple(alpha_up)
        if tup_up in idx_of:
            rows[m].append(idx_of[tup_up]); cols[m].append(a_idx); data[m].append(np.sqrt(a+1.0))
        # down
        if a > 0:
            alpha_dn = list(alpha); alpha_dn[m] = a-1
            tup_dn = tuple(alpha_dn)
            j = idx_of[tup_dn]
            rows[m].append(j); cols[m].append(a_idx); data[m].append(np.sqrt(a))
G_list = [coo_matrix((data[m], (rows[m], cols[m])), shape=(P,P)).tocsr() for m in range(N_KL)]


params = dict(
    Nx=Nx,
    Ny=Ny,
    num_nodes=num_nodes,
    x=jnp.asarray(x),
    y=jnp.asarray(y),
    Xg=jnp.asarray(np.meshgrid(x, y, indexing="ij")[0]),
    Yg=jnp.asarray(np.meshgrid(x, y, indexing="ij")[1]),
    Lx=Lx,
    Ly=Ly,
    quad_points=jnp.asarray(quad_points),
    quad_weights=jnp.asarray(quad_weights),

    # keep these as plain Python / NumPy objects, not JAX arrays
    multi_idx=multi_idx,   # list of tuples (0,1,0,...) etc.
    idx_of=idx_of,         # dict from tuple -> mode index

    w0=2.079e-2,
    P0 = 42500,
    lam=3e-3,
    d=1e-3,
    x_c=0.105,
    h_conv=200.0,
    sigma_SB = 5.67e-8,
    T_inf=0.0,
    eps=0.85,
    delta=None,
)

def melt_fraction_from_Tmean(Tmean_nodes):
    Tm = T_melt_lo
    Delta = max(Delta_melt, 1e-12)

    z = (Tmean_nodes - Tm) / Delta
    S_nodes = 0.5 * (1.0 + np.tanh(z))

    sech2 = 1.0 - np.tanh(z)**2
    dS_dT_nodes = 0.5 * sech2 / Delta

    # element-average from the 4 nodes
    S_elem = 0.25 * (
        S_nodes[:-1, :-1] + S_nodes[1:, :-1] + S_nodes[:-1, 1:] + S_nodes[1:, 1:]
    )
    dS_dT_elem = 0.25 * (
        dS_dT_nodes[:-1, :-1] + dS_dT_nodes[1:, :-1] +
        dS_dT_nodes[:-1, 1:] + dS_dT_nodes[1:, 1:]
    )

    return S_nodes, S_elem, dS_dT_elem


def vap_fraction_from_Tmean(Tmean_nodes):
    """
    Smooth vaporisation fraction V ∈ [0, 1] and its nodal / elemental derivatives.

    Mirrors ``melt_fraction_from_Tmean`` but uses the vaporisation temperature
    window (T_vap_lo, T_vap_hi, Delta_vap).

      V = 0.5 * (1 + tanh((T - T_vap_mid) / Delta_vap))

    Returns
    -------
    V_nodes   : (Nx+1, Ny+1)  nodal vaporisation fraction
    V_elem    : (Nx,   Ny  )  element-averaged fraction
    dV_dT_elem: (Nx,   Ny  )  element-averaged dV/dT  (for apparent heat capacity)
    """
    Tv_mid  = T_vap_lo
    Delta   = max(Delta_vap, 1e-12)

    z       = (Tmean_nodes - Tv_mid) / Delta
    V_nodes = 0.5 * (1.0 + np.tanh(z))

    sech2          = 1.0 - np.tanh(z)**2
    dV_dT_nodes    = 0.5 * sech2 / Delta

    V_elem = 0.25 * (
        V_nodes[:-1, :-1] + V_nodes[1:, :-1] + V_nodes[:-1, 1:] + V_nodes[1:, 1:]
    )
    dV_dT_elem = 0.25 * (
        dV_dT_nodes[:-1, :-1] + dV_dT_nodes[1:, :-1] +
        dV_dT_nodes[:-1, 1:]  + dV_dT_nodes[1:, 1:]
    )

    return V_nodes, V_elem, dV_dT_elem


def _top_boundary_nodes_and_xy():
    """Return node indices + (x,y) for the top boundary (evap surface)."""
    ix = np.arange(Nx+1)
    node_idx = ix*(Ny+1) + Ny
    x_nodes = x[ix]                      # uses global x from your mesh
    y_nodes = np.full_like(x_nodes, Ly)  # all at y = Ly
    return node_idx.astype(int), x_nodes.astype(float), y_nodes.astype(float)

def radiometer_fov(angle_deg, center=(None, None), aggregate=False,
                   weights='uniform'):
    """
    angle_deg: half-angle of FOV around the +y direction.
    center: (xc, yc). Default is domain center.
    aggregate: if True, return a single weighted average measurement.
               if False, return one measurement per visible top node.
    weights: 'uniform' | 'cosine' | 'inverse_square'
    Returns: (H, visible_node_indices)
      H is (m x num_nodes) CSR with either pick-rows (aggregate=False)
      or one row that averages with weights (aggregate=True).
    """
    if center[0] is None: center = (Lx/2.0, Ly/2.0)
    xc, yc = center

    top_idx, xs, ys = _top_boundary_nodes_and_xy()
    dxs = xs - xc
    dys = ys - yc

    # angle to +y axis: tan(theta) = |dx| / dy
    # (guard dy>0; yc below top makes sense for “middle of domain”)
    thetas = np.arctan2(np.abs(dxs), np.maximum(dys, 1e-16))
    mask = thetas <= np.deg2rad(angle_deg)

    sel = top_idx[mask]
    dxs = dxs[mask]
    dys = dys[mask]

    if sel.size == 0:
        # empty FOV -> empty operator
        return csr_matrix((0, num_nodes)), sel

    # ----- pick weights -----
    if weights == 'uniform':
        w = np.ones(sel.size, float)
    elif weights == 'cosine':
        # Lambert-like weighting ~ cos(view angle)
        r = np.sqrt(dxs**2 + dys**2)
        w = np.maximum(dys / np.maximum(r, 1e-16), 0.0)
    elif weights == 'inverse_square':
        r2 = dxs**2 + dys**2
        w = 1.0 / np.maximum(r2, 1e-16)
    else:
        w = np.ones(sel.size, float)

    if aggregate:
        w = w / np.maximum(w.sum(), 1e-16)
        row = np.zeros(num_nodes)
        row[sel] = w
        H = csr_matrix(row.reshape(1, -1))
    else:
        # one measurement per visible node (simple selection)
        rows = np.arange(sel.size)
        H = csr_matrix((np.ones(sel.size), (rows, sel)),
                       shape=(sel.size, num_nodes))
    return H, sel
import numpy as np
import numpy as np
import matplotlib.pyplot as plt

def _plot_on_grid_from_vis(vis_idx, values_vis, x_grid, y_grid, title):
    Nxg, Nyg = x_grid.size, y_grid.size
    field = np.full(Nxg * Nyg, np.nan, dtype=float)
    field[vis_idx] = values_vis
    field = field.reshape(Nxg, Nyg)

    plt.figure()
    plt.imshow(field.T, origin="lower", aspect="auto")
    plt.colorbar()
    plt.title(title)
    plt.xlabel("x index")
    plt.ylabel("y index")
    plt.show()

def visualise_likelihood_terms(
    T_obs, U_final,
    x_vis, y_vis, x_grid, y_grid,
    sigma_obs, num_obs,
    jitter=1e-12,
):
    Nxg, Nyg   = x_grid.size, y_grid.size
    num_nodes  = Nxg * Nyg
    P          = U_final.size // num_nodes

    U_modes = U_final.reshape(P, num_nodes)
    mu_T    = U_modes[0]

    vis_mask = _visible_mask_from_xy(x_vis, y_vis, x_grid, y_grid)
    vis_idx  = np.where(vis_mask.ravel())[0]
    M        = vis_idx.size
    if M == 0:
        raise ValueError("No visible nodes selected.")

    mu_y = mu_T[vis_idx]                 # (M,)
    Y    = T_obs[vis_idx, :]             # (M, N)
    R    = (mu_y[:, None] - Y)           # (M, N)
    N    = num_obs

    s2 = float(sigma_obs**2)
    s2 = max(s2, 1e-30)
    s2j = s2 + jitter

    if P <= 1:
        # Diagonal Sigma only
        Sigma_diag = np.full(M, s2j)
        Sinv_diag  = np.full(M, 1.0 / s2j)

        r2_per_i = np.sum(R * R, axis=1)              # (M,)
        quad = np.sum(r2_per_i * Sinv_diag)           # scalar
        logdet_Sigma = M * np.log(s2j)

        quad_term   = 0.5 * quad
        logdet_term = 0.5 * N * logdet_Sigma

        # Per-node "quad contributions"
        quad_i = 0.5 * (Sinv_diag * r2_per_i)

        _plot_on_grid_from_vis(vis_idx, quad_i, x_grid, y_grid,
                               "Per-visible-node quad contribution (diag approx)")
        _plot_on_grid_from_vis(vis_idx, Sigma_diag, x_grid, y_grid,
                               "Per-visible-node marginal variance Σ_ii")

        print("Scalar summands:")
        print("  0.5 * quad         =", quad_term)
        print("  0.5 * N * logdetΣ   =", logdet_term)
        print("  0.5 * N*M*log(2π)   =", 0.5 * N * M * np.log(2*np.pi))
        return

    # Low-rank case
    U = U_modes[1:, vis_idx].T           # (M, K)
    K = U.shape[1]

    # Build A and its Cholesky for logdet and Woodbury pieces
    UtU = U.T @ U
    UtU = 0.5 * (UtU + UtU.T)
    A = np.eye(K) + (1.0 / s2j) * UtU
    A = 0.5 * (A + A.T)

    L_A = np.linalg.cholesky(A)

    logdet_A = 2.0 * np.sum(np.log(np.diag(L_A)))
    logdet_Sigma = M * np.log(s2j) + logdet_A

    # Compute Sigma^{-1} R
    UtR = U.T @ R                         # (K, N)
    X = np.linalg.solve(L_A, UtR)
    X = np.linalg.solve(L_A.T, X)         # X = A^{-1} UtR
    SinvR = (1.0 / s2j) * R - (1.0 / (s2j*s2j)) * (U @ X)

    quad = np.sum(R * SinvR)

    quad_term   = 0.5 * quad
    logdet_term = 0.5 * N * logdet_Sigma

    # ---- Per-node visualisations (diagnostic, not exact decomposition) ----

    # 1) Per-node residual energy
    r2_per_i = np.sum(R * R, axis=1)   # (M,)

    # 2) Diagonal of Sigma (exact): Σ_ii = s2 + ||U_i||^2
    Sigma_diag = s2j + np.sum(U * U, axis=1)

    # 3) Diagonal of Sigma^{-1} (exact, cheap via Woodbury):
    # Σ^{-1} = (1/s2)I - (1/s2^2) U A^{-1} U^T
    # diag(Σ^{-1}) = 1/s2 - 1/s2^2 * rowwise( U * (A^{-1} U^T)^T )
    # Compute B = A^{-1} U^T  (K, M)
    Ut = U.T                              # (K, M)
    B = np.linalg.solve(L_A, Ut)
    B = np.linalg.solve(L_A.T, B)         # (K, M) = A^{-1} U^T
    UAinvUt_diag = np.sum(U * B.T, axis=1)  # (M,)
    Sinv_diag = (1.0 / s2j) - (1.0 / (s2j*s2j)) * UAinvUt_diag

    # 4) Per-node quad diagnostic (diag approximation):
    quad_i = 0.5 * (Sinv_diag * r2_per_i)

    _plot_on_grid_from_vis(vis_idx, quad_i, x_grid, y_grid,
                           "Per-visible-node quad contribution (using diag(Σ^{-1}))")

    _plot_on_grid_from_vis(vis_idx, Sigma_diag, x_grid, y_grid,
                           "Per-visible-node marginal variance Σ_ii (s2 + ||U_i||^2)")

    _plot_on_grid_from_vis(vis_idx, Sinv_diag, x_grid, y_grid,
                           "Per-visible-node precision diag(Σ^{-1})")

    print("Scalar summands:")
    print("  0.5 * quad         =", quad_term)
    print("  0.5 * N * logdetΣ   =", logdet_term)
    print("  0.5 * N*M*log(2π)   =", 0.5 * N * M * np.log(2*np.pi))
    print("  Total (no dxdy)     =", quad_term + logdet_term + 0.5 * N * M * np.log(2*np.pi))
def _visible_mask_from_xy(x_vis, y_vis, x_grid, y_grid):
    # If someone accidentally passed a time-history list, catch it early
    if isinstance(x_vis, (list, tuple)) and len(x_vis) > 0 and isinstance(x_vis[0], (list, tuple, np.ndarray)):
        raise TypeError(
            "x_vis looks like a time-history (list of arrays). "
            "Pass a single timestep: x_vis_hist[n], y_vis_hist[n]."
        )

    x_vis  = np.asarray(x_vis, dtype=float).ravel()
    y_vis  = np.asarray(y_vis, dtype=float).ravel()
    x_grid = np.asarray(x_grid, dtype=float).ravel()
    y_grid = np.asarray(y_grid, dtype=float).ravel()

    ix = np.abs(x_grid[None, :] - x_vis[:, None]).argmin(axis=1)
    iy = np.abs(y_grid[None, :] - y_vis[:, None]).argmin(axis=1)

    vis_mask = np.zeros((x_grid.size, y_grid.size), dtype=bool)
    vis_mask[ix, iy] = True
    return vis_mask


def full_gaussian_likelihood_visible_grad(
    U_final, T_obs,
    x_vis, y_vis, x_grid, y_grid,
    sigma_obs, num_obs,
    var_floor=1e-3,
):
    """
    Gradient of the diagonal Gaussian likelihood wrt U_final.

    Uses Sigma_i = sigma_obs^2 + sum_{k>=1} u_{k,i}^2 + var_floor
    """
    Nxg, Nyg   = x_grid.size, y_grid.size
    num_nodes  = Nxg * Nyg
    P          = U_final.size // num_nodes

    U_modes = U_final.reshape(P, num_nodes)
    mu_T    = U_modes[0]

    vis_mask = _visible_mask_from_xy(x_vis, y_vis, x_grid, y_grid)
    vis_idx  = np.where(vis_mask.ravel())[0]
    M        = vis_idx.size
    if M == 0:
        raise ValueError("No visible nodes selected: vis_mask is empty.")

    mu_y = mu_T[vis_idx]                 # (M,)
    Y    = T_obs[vis_idx, :]             # (M, N)
    R    = (mu_y[:, None] - Y)           # (M, N)
    N    = int(num_obs)

    s2 = max(float(sigma_obs**2), 1e-30)

    dL_dU = np.zeros_like(U_final)
    dL_dU_modes = dL_dU.reshape(P, num_nodes)

    if P <= 1:
        Sigma = s2 + float(var_floor)              # scalar
        # dL/dmu_i = sum_n r_{i,n} / Sigma
        dL_dmu = np.sum(R, axis=1) / Sigma         # (M,)
        dL_dU_modes[0, vis_idx] += dL_dmu
        return dL_dU_modes.ravel() * dx * dy

    # P > 1
    U = U_modes[1:, vis_idx]             # (K, M)
    var_model = np.sum(U*U, axis=0)      # (M,)
    Sigma = s2 + var_model + float(var_floor)  # (M,)

    invS = 1.0 / Sigma                   # (M,)
    invS2 = invS * invS

    # dL/dmu_i
    dL_dmu = np.sum(R, axis=1) * invS    # (M,)
    dL_dU_modes[0, vis_idx] += dL_dmu

    # dL/dSigma_i = 0.5 * ( N/Sigma_i - sum_n r^2 / Sigma_i^2 )
    r2_sum = np.sum(R*R, axis=1)         # (M,)
    dL_dSigma = 0.5 * (N * invS - r2_sum * invS2)   # (M,)

    # For each mode k>=1: Sigma_i depends on u_{k,i} via u_{k,i}^2
    # dL/du_{k,i} = dL/dSigma_i * 2 u_{k,i}
    dL_dU_nonmean = (2.0 * U) * dL_dSigma[None, :]  # (K, M)

    dL_dU_modes[1:, vis_idx] += dL_dU_nonmean
   

    return dL_dU_modes.ravel() * dx * dy

def full_gaussian_likelihood_visible(
    T_obs, U_final,
    x_vis, y_vis, x_grid, y_grid,
    sigma_obs, num_obs,
    var_floor=1e-3,     # <<< variance floor (in variance units)
):
    """
    Diagonal Gaussian likelihood using per-node marginal variance.

    Sigma_i = sigma_obs^2 + sum_{k>=1} u_{k,i}^2 + var_floor

    Returns: scalar L * dx * dy (to match your existing scaling).
    """
    Nxg, Nyg   = x_grid.size, y_grid.size
    num_nodes  = Nxg * Nyg
    P          = U_final.size // num_nodes

    U_modes = U_final.reshape(P, num_nodes)
    mu_T    = U_modes[0]

    vis_mask = _visible_mask_from_xy(x_vis, y_vis, x_grid, y_grid)
    vis_idx  = np.where(vis_mask.ravel())[0]
    M        = vis_idx.size
    if M == 0:
        raise ValueError("No visible nodes selected: vis_mask is empty.")

    mu_y = mu_T[vis_idx]                 # (M,)
    Y    = T_obs[vis_idx, :]             # (M, N)
    R    = (mu_y[:, None] - Y)           # (M, N)
    N    = int(num_obs)

    s2 = max(float(sigma_obs**2), 1e-30)

    if P <= 1:
        # no model variance, just measurement noise + floor
        Sigma = s2 + float(var_floor)
    else:
        U = U_modes[1:, vis_idx]         # (K, M) with K=P-1
        # per-node marginal variance from modes
        var_model = np.sum(U*U, axis=0)  # (M,)
        Sigma = s2 + var_model + float(var_floor)  # (M,)

    # Quadratic term: sum_{i,n} r^2 / Sigma_i
    quad = np.sum((R*R) / Sigma[:, None])

    # Logdet term: N * sum_i log Sigma_i  (diagonal logdet)
    logdet = np.sum(np.log(Sigma))

    L = 0.5 * quad + 0.5 * N * (M * np.log(2*np.pi) + logdet)
    return L * dx * dy

def _visible_mask_from_xy(x_vis, y_vis, x_grid, y_grid):
    """
    Build a boolean mask (Nxg, Nyg) that is True at grid nodes
    closest to the (x_vis, y_vis) visible points.
    """
    Nxg, Nyg = x_grid.size, y_grid.size
    mask = np.zeros((Nxg, Nyg), dtype=bool)
   
    for xv, yv in zip(x_vis, y_vis):
        ix = np.argmin(np.abs(x_grid - xv))
        iy = np.argmin(np.abs(y_grid - yv))
        mask[ix, iy] = True
   
    return mask


def get_visible_weights_from_xy(x_vis, y_vis, w_vis, x_grid, y_grid):
    """
    Map the visible weights back to grid node indices.
   
    Returns:
        vis_idx: array of visible node indices (flattened)
        weights: corresponding normalized weights
    """
    Nxg, Nyg = x_grid.size, y_grid.size
    num_nodes = Nxg * Nyg
   
    vis_idx_list = []
    weights_list = []
   
    for xv, yv, wv in zip(x_vis, y_vis, w_vis):
        ix = np.argmin(np.abs(x_grid - xv))
        iy = np.argmin(np.abs(y_grid - yv))
        node_idx = ix * Nyg + iy
        vis_idx_list.append(node_idx)
        weights_list.append(float(wv))
   
    vis_idx = np.array(vis_idx_list, dtype=int)
    weights = np.array(weights_list, dtype=float)
   
    # Normalize weights (should already be normalized, but ensure)
    weights = weights / (np.sum(weights) + 1e-12)
   
    return vis_idx, weights


# =============================================================================
# Synthetic observation generation with weighted average
# =============================================================================
def make_T_obs_history_weighted(U_hist_true, x_vis_hist, y_vis_hist, w_vis_hist,
                                x_grid, y_grid, sigma_obs, num_obs,
                                multi_idx, eval_psi_func, N_KL):

    time_steps = U_hist_true.shape[0]
    num_nodes  = x_grid.size * y_grid.size
    P          = U_hist_true.shape[1] // num_nodes

    T_obs_hist = np.zeros((time_steps, num_obs))

    for t in range(1, time_steps):
        if x_vis_hist[t] is None or len(x_vis_hist[t]) == 0:
            continue

        vis_idx, weights = get_visible_weights_from_xy(
            x_vis_hist[t], y_vis_hist[t], w_vis_hist[t],
            x_grid, y_grid
        )

        # U_modes: (P, num_nodes) — reshape once
        U_modes = U_hist_true[t].reshape(P, num_nodes)

        # Weighted spatial response for each PC mode: (P,)
        # g[k] = weights · U_modes[k, vis_idx]
        g = U_modes[:, vis_idx] @ weights          # (P,)

        # Draw all xi samples at once: (num_obs, N_KL)
        xi_samples = np.random.randn(num_obs, N_KL)

        # Evaluate all psi basis values: Psi[i, k] = psi_k(xi_i)
        # shape (num_obs, P)
        Psi = np.array([[eval_psi_func(xi_samples[i], alpha)
                         for alpha in multi_idx]
                        for i in range(num_obs)])   # (num_obs, P)

        # Weighted radiometer readings: (num_obs,)
        T_obs_hist[t] = Psi @ g + sigma_obs * np.random.randn(num_obs)

    T_obs_mean = np.mean(T_obs_hist, axis=1)
    T_obs_var  = np.var(T_obs_hist,  axis=1, ddof=1)

    return T_obs_hist
def make_T_obs_history_weighted_old(U_hist_true, x_vis_hist, y_vis_hist, w_vis_hist,
                                 x_grid, y_grid, sigma_obs, num_obs,
                                 multi_idx, eval_psi_func, N_KL):
    """
    Generate synthetic radiometer observations as weighted averages.
   
    Each observation is a scalar: y_n = sum_i w_i * T_i(xi_n) + noise
   
    Args:
        U_hist_true: (time_steps, n_dofs) forward solution history
        x_vis_hist, y_vis_hist: lists of visible point coordinates per timestep
        w_vis_hist: list of radiometer weights per timestep
        x_grid, y_grid: mesh coordinates
        sigma_obs: observation noise std
        num_obs: number of observation samples per timestep
        multi_idx: polynomial chaos multi-indices
        eval_psi_func: function to evaluate PC basis
        N_KL: number of KL modes
   
    Returns:
        T_obs_hist: (time_steps, num_obs) array of scalar observations
    """
    time_steps = U_hist_true.shape[0]
    num_nodes = (x_grid.size) * (y_grid.size)
    P = U_hist_true.shape[1] // num_nodes
   
    T_obs_hist = np.zeros((time_steps, num_obs))
   
    def sample_solution(Ucoeff, xi_sample):
        u = np.zeros(num_nodes)
        for k, alpha in enumerate(multi_idx):
            psi = eval_psi_func(xi_sample, alpha)
            u += psi * Ucoeff[k*num_nodes:(k+1)*num_nodes]
        return u
   
    for t in range(1, time_steps):
        if x_vis_hist[t] is None or len(x_vis_hist[t]) == 0:
            continue
           
        # Get visible indices and weights
        vis_idx, weights = get_visible_weights_from_xy(
            x_vis_hist[t], y_vis_hist[t], w_vis_hist[t],
            x_grid, y_grid
        )
        #weights = np.ones_like(weights)
       
        for i in range(num_obs):
            xi_sample = np.random.randn(N_KL)
            T_full = sample_solution(U_hist_true[t, :], xi_sample)
           
            # Weighted average observation
            T_weighted_avg = np.sum(weights * T_full[vis_idx])
           
            # Add measurement noise
            T_obs_hist[t, i] = T_weighted_avg + sigma_obs * np.random.randn()
            #print(T_obs_hist[t,i])
    #print(T_obs_hist.shape)
    # Mean radiometer reading at each timestep
    T_obs_mean = np.mean(T_obs_hist, axis=1)      # shape (time_steps,)

    # Sample variance (unbiased)
    T_obs_var = np.var(T_obs_hist, axis=1, ddof=1)  # shape (time_steps,)
    t = np.arange(time_steps)

   # plt.figure()
   # plt.plot(t, T_obs_mean, label="mean radiometer signal")
   # plt.fill_between(
   #     t,
   #     T_obs_mean - np.sqrt(T_obs_var),
   #     T_obs_mean + np.sqrt(T_obs_var),
   #     alpha=0.3,
   #     label="±1 std"
   # )
   # plt.xlabel("timestep")
   # plt.ylabel("radiometer reading")
   # plt.legend()
   # plt.show()
    return T_obs_hist
import numpy as np

def make_depth_obs_history(
    U_hist_true,
    x_grid, y_grid,
    sigma_obs, num_obs,
    multi_idx, eval_psi_func, N_KL,
    T_abl,
    restrict_to_visible=False,
    y_increases_with_depth=True,
    # new knobs:
    arctan_beta=1.0,        # scaling inside atan(beta*(T-Tabl))
    use_soft_depth=False,   # False = min-y hard depth, True = weighted soft depth
    no_hit_value=0.0,       # what to record if nothing exceeds threshold
):
    time_steps = U_hist_true.shape[0]
    Nx1, Ny1 = x_grid.size, y_grid.size
    num_nodes = Nx1 * Ny1
    P = U_hist_true.shape[1] // num_nodes

    depth_obs_hist = np.zeros((time_steps, num_obs), dtype=float)

    def sample_solution(Ucoeff, xi_sample):
        u = np.zeros(num_nodes, dtype=float)
        for k, alpha in enumerate(multi_idx):
            psi = eval_psi_func(xi_sample, alpha)
            u += psi * Ucoeff[k*num_nodes:(k+1)*num_nodes]
        return u

    y_mat = np.tile(y_grid[None, :], (Nx1, 1))  # (Nx1, Ny1)

    # If depth corresponds to "more negative y", convert to a "depth axis" where deeper = larger
    if y_increases_with_depth:
        y_depth = y_mat
        def depth_to_y(d):  # identity
            return float(d)
    else:
        y_depth = -y_mat
        def depth_to_y(d):  # invert back
            return float(-d)

    for t in range(1, time_steps):


        for i in range(num_obs):
            xi_sample = np.random.randn(N_KL)
            T_full = sample_solution(U_hist_true[t, :], xi_sample)
            T2 = T_full.reshape(Nx1, Ny1)

            # Exceedance above threshold
            dT = T2 - T_abl

            if restrict_to_visible:
                hit = (dT > 0.0) & vis_mask2
            else:
                hit = (dT > 0.0)

            if not np.any(hit):
                depth_y = no_hit_value
            else:
                if not use_soft_depth:
                    # ---- HARD DEPTH: minimum y where T > T_abl ----
                    # minimum in depth-axis => shallowest in depth-axis
                    # If you literally want "minimum y", do np.min(y_mat[hit]) instead.
                    # Since you said "minimum y where T>T_abl", we use y_mat directly:
                    y_min = np.min(y_mat[hit])
                    depth_y = float(y_min)
                else:
                    # ---- SOFT DEPTH (optional): arctan-weighted depth ----
                    # weights are 0 below threshold, atan(beta*dT) above threshold
                    w = np.zeros_like(dT, dtype=float)
                    w[hit] = np.arctan(arctan_beta * dT[hit])

                    # If you still want the "minimum y" but smoothed, you can
                    # take the minimum y among points with weight above a tiny cutoff:
                    # hit2 = w > 1e-12; y_min = np.min(y_mat[hit2])
                    # depth_y = float(y_min)

                    # Or a smoother depth estimate (less jumpy):
                    wsum = np.sum(w)
                    if wsum <= 0.0:
                        depth_y = no_hit_value
                    else:
                        depth_y = float(np.sum(w * y_mat) / wsum)

            depth_obs_hist[t, i] = depth_y + sigma_obs * np.random.randn()

    return depth_obs_hist



# =============================================================================
# Likelihood for weighted-average radiometer
# =============================================================================
def radiometer_likelihood_weighted(
    T_obs,           # (num_obs,) scalar observations at this timestep
    U_current,         # (P * num_nodes,) SG coefficients
    x_vis, y_vis,    # visible point coordinates
    w_vis,           # radiometer weights (normalized)
    x_grid, y_grid,
    sigma_obs,
    num_obs,
    var_floor=1e-3,
):
    """
    Gaussian likelihood for weighted-average radiometer observation.
   
    Model:
        y_n = w^T T + epsilon_n,   epsilon_n ~ N(0, sigma_obs^2)
       
    where T is the temperature field and w are the radiometer weights.
   
    The predicted mean is:   mu_y = w^T mu_T
    The predicted variance:  sigma_y^2 = sigma_obs^2 + w^T Sigma_T w + var_floor
   
    For diagonal PC variance: Sigma_T = diag(sum_k u_k^2), so
        w^T Sigma_T w = sum_i w_i^2 * var_T_i
   
    Returns: scalar negative log-likelihood (times dx*dy for consistency)
    """
    Nxg, Nyg = x_grid.size, y_grid.size
    num_nodes_local = Nxg * Nyg
    dx = x_grid[1] - x_grid[0]
    dy = y_grid[1] - y_grid[0]
   
    P = U_current.size // num_nodes_local
    U_modes = U_current.reshape(P, num_nodes_local)
    mu_T = U_modes[0]  # mean temperature field
   
    # Get visible indices and weights
    vis_idx, weights = get_visible_weights_from_xy(x_vis, y_vis, w_vis, x_grid, y_grid)
    M = vis_idx.size
    #weights=np.ones_like(weights)

    if M == 0:
        return 0.0
   
    # Predicted mean: weighted average of mean temperatures
    mu_y = np.sum(weights * mu_T[vis_idx])  # scalar
   
    # Predicted variance
    s2 = max(float(sigma_obs**2), 1e-30)
   
    if P <= 1:
        # No model variance
        sigma_y2 = s2 + float(var_floor)
    else:
        # Model variance at visible nodes
        U_vis = U_modes[:, vis_idx]        # (P, M)
        proj  = U_vis @ weights            # (P,)
        mu_y  = proj[0]
        var_y = np.sum(proj[1:]**2)        # (+ gamma if needed)
        sigma_y2 = sigma_obs**2 + var_y + var_floor
   
    # Residuals
    Y = np.asarray(T_obs)  # (num_obs,)
    R = mu_y - Y           # (num_obs,)
    N = len(T_obs)
   
    # Negative log-likelihood (up to constant)
    # L = 0.5 * sum_n (r_n^2 / sigma_y^2) + 0.5 * N * log(sigma_y^2)
    quad = np.sum(R * R) / sigma_y2
    logdet = N * np.log(sigma_y2)
   
    L = 0.5 * quad + 0.5 * (N * np.log(2 * np.pi) + logdet)
    S = float(np.sum((Y - mu_y)**2))
   # print("mu_y", mu_y, "sigma_y2", sigma_y2, "S", S, "S/N", S/N)
    #print("quad", S/sigma_y2, "log", N*np.log(2*np.pi*sigma_y2))

    return L *dx*dy


def radiometer_likelihood_weighted_grad(
    U_current,         # (P * num_nodes,)
    T_obs,           # (num_obs,) scalar observations
    x_vis, y_vis,    # visible point coordinates  
    w_vis,           # radiometer weights
    x_grid, y_grid,
    sigma_obs,
    num_obs,
    var_floor=1e-3,
):
    """
    Gradient of radiometer likelihood w.r.t. U_final.
   
    dL/dU has contributions from:
    1. dL/dmu_y * dmu_y/dU  (mean mode only)
    2. dL/dsigma_y^2 * dsigma_y^2/dU  (higher modes)
    """
    Nxg, Nyg = x_grid.size, y_grid.size
    num_nodes_local = Nxg * Nyg
    dx = x_grid[1] - x_grid[0]
    dy = y_grid[1] - y_grid[0]
   
    P = U_current.size // num_nodes_local
    U_modes = U_current.reshape(P, num_nodes_local)
    mu_T = U_modes[0]
   
    vis_idx, weights = get_visible_weights_from_xy(x_vis, y_vis, w_vis, x_grid, y_grid)
    M = vis_idx.size
   # weights=np.ones_like(weights)
    dL_dU = np.zeros_like(U_current)
    dL_dU_modes = dL_dU.reshape(P, num_nodes_local)
   
    if M == 0:
        return dL_dU
   
    # Predicted mean
    mu_y = np.sum(weights * mu_T[vis_idx])
   
    # Predicted variance
    s2 = max(float(sigma_obs**2), 1e-30)
   
    if P <= 1:
        sigma_y2 = s2 + float(var_floor)
        var_T_vis = None
    else:
        U_vis = U_modes[:, vis_idx]        # (P, M)
        proj  = U_vis @ weights            # (P,)
        mu_y  = proj[0]
        var_y = np.sum(proj[1:]**2)        # (+ gamma if needed)
        sigma_y2 = sigma_obs**2 + var_y + var_floor
           
    # Residuals
    Y = np.asarray(T_obs)
    R = mu_y - Y  # (N,)
    N = len(T_obs)
   
    inv_sigma2 = 1.0 / sigma_y2
   
    # --- Gradient w.r.t. mean mode ---
    # dL/dmu_y = sum_n r_n / sigma_y^2
    dL_dmu_y = np.sum(R) * inv_sigma2
   
    # dmu_y/d(mu_T[i]) = w_i for visible nodes
    # So dL/d(mu_T[vis_idx]) = dL_dmu_y * weights
    dL_dU_modes[0, vis_idx] += dL_dmu_y * weights
   
    # --- Gradient w.r.t. higher modes (variance contribution) ---
    if P > 1:
        # dL/dsigma_y^2 = 0.5 * (N / sigma_y^2 - sum_n r_n^2 / sigma_y^4)
        r2_sum = np.sum(R * R)
        dL_dsigma2 = 0.5 * (N * inv_sigma2 - r2_sum * inv_sigma2**2)
       
        # dsigma_y^2 / d(u_{k,i}) = 2 * w_i^2 * u_{k,i}  for visible nodes
        # dL/d(u_{k,i}) = dL_dsigma2 * 2 * w_i^2 * u_{k,i}
        for k in range(1, P):
            u_k_vis = U_modes[k, vis_idx]  # (M,)
            dL_dU_modes[k, vis_idx] += dL_dsigma2 * 2 * (weights**2) * u_k_vis
   
    return dL_dU_modes.ravel()*dx*dy

# ------------------------------------------------------------------------
def eval_psi(xi, alpha):
    val = 1.0
    for m, p in enumerate(alpha):
        coeff = np.zeros(p+1); coeff[-1] = 1.0
        He_p = hermeval(xi[m], coeff)            # He_p
        val *= He_p / np.sqrt(math.factorial(p))  # makes basis orthonormal under N(0,1)
    return val

def eval_psi_probabilists(xi, alpha):
    # uses probabilists' Hermite He_n with N(0,1) weight
    from numpy.polynomial.hermite_e import hermeval
    val = 1.0
    for m, p in enumerate(alpha):
        coeff = np.zeros(p+1); coeff[-1] = 1.0
        He_p = hermeval(xi[m], coeff)          # He_p
        val *= He_p / np.sqrt(math.factorial(p))   # orthonormalize
    return val


# -------------------------------
# Generate GMRF using Fourier-based method
# -------------------------------

def get_boundary_nodes(Nx, Ny):
    boundary_nodes = set()
    for ix in range(Nx+1):
        # y=0
        iy = 0
        boundary_nodes.add(ix*(Ny+1) + iy)
        # y=Ny (top boundary)
       # iy = Ny
       # boundary_nodes.add(ix*(Ny+1) + iy)

    for iy in range(Ny+1):
        # x=0
        ix = 0
        boundary_nodes.add(ix*(Ny+1) + iy)
        # x=Nx (right boundary)
        ix = Nx
        boundary_nodes.add(ix*(Ny+1) + iy)

    return sorted(boundary_nodes)

def apply_dirichlet_bc(K_SG, F_SG, boundary_nodes, T_bc, P, num_nodes):
    K_lil = K_SG.tolil()

    for bnode in boundary_nodes:
        for p in range(P):
            row = p * num_nodes + bnode
            # zero out entire row
            K_lil.rows[row] = []
            K_lil.data[row] = []
            # set diagonal entry = 1
            K_lil.rows[row].append(row)
            K_lil.data[row].append(1.0)

            # Adjust the RHS
            if p == 0:
                # Mean mode: T = T_bc
                F_SG[row] = T_bc
            else:
                # Higher modes: T = 0
                F_SG[row] = 0.0

    return K_lil.tocsr(), F_SG

def compute_local_mass_derivative_ell(x_coords, y_coords, phi_func, m, eigvals_trunc, dlambda_dell):
    M_local = np.zeros((4, 4))
    for qp, w in zip(quad_points, quad_weights):
        xi_qp, eta_qp = qp

        N = np.array([0.25*(1 - xi_qp)*(1 - eta_qp),
                      0.25*(1 + xi_qp)*(1 - eta_qp),
                      0.25*(1 + xi_qp)*(1 + eta_qp),
                      0.25*(1 - xi_qp)*(1 + eta_qp)])

        J = np.array([[np.dot([-0.25*(1 - eta_qp), 0.25*(1 - eta_qp), 0.25*(1 + eta_qp), -0.25*(1 + eta_qp)], x_coords),
                       np.dot([-0.25*(1 - eta_qp), 0.25*(1 - eta_qp), 0.25*(1 + eta_qp), -0.25*(1 + eta_qp)], y_coords)],
                      [np.dot([-0.25*(1 - xi_qp), -0.25*(1 + xi_qp), 0.25*(1 + xi_qp), 0.25*(1 - xi_qp)], x_coords),
                       np.dot([-0.25*(1 - xi_qp), -0.25*(1 + xi_qp), 0.25*(1 + xi_qp), 0.25*(1 - xi_qp)], y_coords)]])
        detJ = np.linalg.det(J)

        # Physical coordinates
        x_qp, y_qp = np.dot(N, x_coords), np.dot(N, y_coords)
       
        phi = phi_func(x_qp, y_qp)

        # Derivative w.r.t. ell appears via eigenvalue derivative
        dphi_term = 0.5 * dlambda_dell[m] / np.sqrt(eigvals_trunc[m]) * phi

        M_local += dphi_term * np.outer(N, N) * detJ * w

    return M_local



def form_dM_SG_dell(eigvals_trunc, dlambda_dell, eigvecs_reshaped, phi_gradients):
    M_kl_global_dell = []
    for m in range(N_KL):
        M_kl_dell = lil_matrix((num_nodes, num_nodes))
        phi, phi_x, phi_y  = phi_gradients[m]
       
        def phi_func(x,y):
            i = int(np.floor(x/Lx*Nx))
            j = int(np.floor(y/Ly*Ny))
            return phi[i,j]
           
        for ix in range(Nx):
            for iy in range(Ny):
                node1 = ix * (Ny + 1) + iy
                node2 = node1 + 1
                node3 = (ix + 1) * (Ny + 1) + iy
                node4 = node3 + 1
                nodes = [node1, node2, node3, node4]
                x_coords = np.array([x[ix], x[ix + 1], x[ix + 1], x[ix]])
                y_coords = np.array([y[iy], y[iy], y[iy + 1], y[iy + 1]])
               
                M_kl_local = compute_local_mass_derivative_ell(x_coords, y_coords, phi_func, m, eigvals_trunc, dlambda_dell)
                for a in range(4):
                    for b in range(4):
                        M_kl_dell[nodes[a], nodes[b]] += M_kl_local[a, b]
                       
        M_kl_global_dell.append(M_kl_dell.tocsr())  # MOVED outside element loop

    dM_dell = csr_matrix((P * num_nodes, P * num_nodes))

    for m in range(N_KL):
        # replace the hand-made 2-entry G_m with the Hermite coupling:
        G_m = G_list[m]
        dM_dell += kron(G_m, M_kl_global_dell[m].tocsr(), format='csr')


    return dM_dell



def generate_gmrf_periodic(Nx, Ny, Lx, Ly, sigma, ell):
    """
    Generate a GMRF with periodic boundary conditions using Fourier methods.
    """
    # Create a grid of frequencies
    kx = 2 * np.pi * fftfreq(Nx, d=Lx / Nx)
    ky = 2 * np.pi * fftfreq(Ny, d=Ly / Ny)
    KX, KY = np.meshgrid(kx, ky, indexing='ij')

    # Compute the spectral density of the periodic covariance function
    spectral_density = sigma**2 * np.exp(-(KX**2 + KY**2) * ell**2 / 2)

    # Generate random Fourier coefficients
    random_phase = np.random.normal(0, 1, (Nx, Ny)) + 1j * np.random.normal(0, 1, (Nx, Ny))
    random_field_fourier = np.sqrt(spectral_density) * random_phase

    # Inverse Fourier transform to get the spatial field
    random_field = np.real(ifft2(random_field_fourier))

    return random_field


def eigenvalue_derivatives_via_feynman(K, dK_dell, X, sigma,ell):
    # Compute eigenvalues and eigenvectors
    lambdas, v = np.linalg.eigh(K)  # K is symmetric, so eigh is preferred
    idx = np.argsort(lambdas)[::-1]
    lambdas = lambdas[idx]
    v = v[:, idx]
    # Truncate to N_KL terms
    lambdas = lambdas[:N_KL]
    v = v[:, :N_KL]
    # Apply Hellman-Feynman theorem
    dlambda_dell = np.array([v[:, i].T @ dK_dell @ v[:, i] for i in range(len(lambdas))])
    return dlambda_dell, lambdas

# -------------------------------
# Compute the Covariance Matrix
# -------------------------------
from scipy.sparse.linalg import LinearOperator, eigsh

def _periodic_cov_first_col(Nx, Ny, Lx, Ly, sigma, ell):
    """
    First column of the 2D periodic covariance (distance from (0,0)).
    BCCB structure => this fully defines the matrix.
    """
    # grid index distances (0..Nx-1, 0..Ny-1), periodic metric via sin^2
    ix = np.arange(Nx)[:, None]
    jy = np.arange(Ny)[None, :]

    sin2x = np.sin(np.pi * ix / Nx)**2   # shape (Nx,1)
    sin2y = np.sin(np.pi * jy / Ny)**2   # shape (1,Ny)
    A = sin2x + sin2y                    # shape (Nx,Ny)

    col = sigma**2 * np.exp(-2.0 * A / (ell**2))      # k((i,j),(0,0))
    dcol_dell = col * (4.0 * A / (ell**3))            # ∂k/∂ell at (i,j)
    return col, dcol_dell

def make_cov_fft_ops(Nx, Ny, Lx, Ly, sigma, ell):
    """
    Returns:
      C_op         : LinearOperator for y = C x  using FFTs (SPD)
      dC_dell_op   : LinearOperator for y = (∂C/∂ell) x using FFTs
      Lam2D        : eigenvalues on the Fourier grid (= fft2 of first column)
    """
    col, dcol = _periodic_cov_first_col(Nx, Ny, Lx, Ly, sigma, ell)

    # Precompute FFTs of first columns (define the convolution symbols)
    F_col  = np.fft.fft2(col)
    F_dcol = np.fft.fft2(dcol)

    def matvec(x):
        X = x.reshape(Nx, Ny)
        Y = np.fft.ifft2(F_col * np.fft.fft2(X)).real
        return Y.ravel()

    def dmatvec(x):
        X = x.reshape(Nx, Ny)
        Y = np.fft.ifft2(F_dcol * np.fft.fft2(X)).real
        return Y.ravel()

    n = Nx * Ny
    C_op       = LinearOperator((n, n), matvec=matvec, rmatvec=matvec, dtype=float)
    dC_dell_op = LinearOperator((n, n), matvec=dmatvec, rmatvec=dmatvec, dtype=float)
    Lam2D = np.real(np.fft.fft2(col))  # spectrum on the FFT grid (not sorted)

    return C_op, dC_dell_op, Lam2D
def set_rhs_from_f_field(f_field):
    # f_field is (Nx+1, Ny+1) on nodes
    set_source_grid(f_field)  # then f(x,y,*) will sample it
def assemble_F_mean_from_qgrid():
    F_det = np.zeros(num_nodes)
    for ix in range(Nx):
        for iy in range(Ny):
            node1 = ix*(Ny+1)+iy; node2 = node1+1
            node3 = (ix+1)*(Ny+1)+iy; node4 = node3+1
            nodes = [node1, node2, node3, node4]
            x_coords = np.array([x[ix], x[ix+1], x[ix+1], x[ix]])
            y_coords = np.array([y[iy], y[iy], y[iy+1], y[iy+1]])
            for (xi_qp,eta_qp), wq in zip(quad_points, quad_weights):
                N = np.array([
                    0.25*(1 - xi_qp)*(1 - eta_qp),
                    0.25*(1 + xi_qp)*(1 - eta_qp),
                    0.25*(1 + xi_qp)*(1 + eta_qp),
                    0.25*(1 - xi_qp)*(1 + eta_qp),
                ])
                x_qp = np.dot(N, x_coords)
                y_qp = np.dot(N, y_coords)
                # detJ is constant for Q1 rectangles: (hx*hy)/4
                detJ = (hx*hy)/4.0
                fq = f(x_qp, y_qp, _q_grid)   # samples the global q-grid
                scale = detJ * wq * fq
                for a in range(4):
                    F_det[nodes[a]] += scale * N[a]
    return F_det
def assemble_periodic_covariance_matrix_vec(nodes, sigma, ell, Lx, Ly):

    diff = nodes[:, np.newaxis, :] - nodes[np.newaxis, :, :]  
    # Extract differences in x and y
    diff_x = diff[:, :, 0]
    diff_y = diff[:, :, 1]
   
    # Compute the sine of the scaled differences
    sin_term_x = np.sin(np.pi * diff_x / Lx)
    sin_term_y = np.sin(np.pi * diff_y / Ly)
   
    # Square the sine terms
    sin_sq_x = sin_term_x**2
    sin_sq_y = sin_term_y**2
   
    # Combine the contributions from x and y
    exponent = -2 * (sin_sq_x + sin_sq_y) / (ell**2)
    A = sin_sq_x + sin_sq_y
    # Assemble the covariance matrix
    C = sigma**2 * np.exp(exponent)
    dC_dell = (4 * A / (ell**3)) * C
    return C, dC_dell
def assemble_sq_covariance_matrix_vec(nodes, sigma, ell, Lx, Ly):
    """
    Vectorized assembly of the covariance matrix using the squared exponential kernel.
    """
    N = nodes.shape[0]
    # Compute pairwise squared distances
    dist_sq = np.sum((nodes[:, np.newaxis, :] - nodes[np.newaxis, :, :]) ** 2, axis=-1)
    # Compute the covariance matrix
    C = sigma**2 * np.exp(-dist_sq / (2 * ell**2))
        # Assemble the periodic covariance matrix
    #C = sigma**2 * np.exp(-dist_sq / (2 * ell**2))
    dC_dell = sigma**2 * (dist_sq / (ell**3))*C
    return C, dC_dell
# Use Eigenvalues and Eigenvectors in KL Expansion
# -------------------------------
def k_kl(x, y, eigvals, eigvecs_reshaped, k0, k1, xi_sample):
    """
    Evaluate the KL expansion of the conductivity at a point (x,y) using a fixed sample xi_sample.
    """
    # Determine grid indices (assuming nodes aare on a grid of size (Nx,Ny))
    i = int(np.floor(x / Lx * Nx))
    j = int(np.floor(y / Ly * Ny))
    i = np.clip(i, 0, Nx-1)
    j = np.clip(j, 0, Ny-1)
   
    k_val = k0
    for m in range(N_KL):
        k_val += k1 * np.sqrt(eigvals[m]) * eigvecs_reshaped[i, j, m] * xi_sample[m]
   
    return k_val
def form_dK_SG_dell(eigvals_trunc, dlambda_dell, eigvecs_reshaped, phi_gradients):
    K_kl_global_dell = []
    for m in range(N_KL):
        K_kl_dell = lil_matrix((num_nodes, num_nodes))
        phi, phi_x, phi_y = phi_gradients[m]
       
        def phi_grad_func(x,y):
            i = int(np.floor(x/Lx*Nx))
            j = int(np.floor(y/Ly*Ny))
            return phi[i,j], phi_x[i,j], phi_y[i,j]
           
        for ix in range(Nx):
            for iy in range(Ny):
                node1 = ix * (Ny + 1) + iy
                node2 = node1 + 1
                node3 = (ix + 1) * (Ny + 1) + iy
                node4 = node3 + 1
                nodes = [node1, node2, node3, node4]
                x_coords = np.array([x[ix], x[ix + 1], x[ix + 1], x[ix]])
                y_coords = np.array([y[iy], y[iy], y[iy + 1], y[iy + 1]])
               
                K_kl_local_dell = compute_local_stiffness(
                    x_coords, y_coords,
                    lambda x,y: 0,  # Only the KL mode contributes
                    1.0,
                    phi_grad_func=phi_grad_func,
                    m=m,
                    eigvals_trunc=eigvals_trunc,
                    deigvals_trunc_dell = dlambda_dell,
                    grad_ell = True
                )
                for a in range(4):
                    for b in range(4):
                        K_kl_dell[nodes[a], nodes[b]] += K_kl_local_dell[a, b]
                       
        K_kl_global_dell.append(K_kl_dell.tocsr())  # MOVED outside element loop
    """Assemble the stochastic coupling terms"""
    # Initialize empty sparse matrix (CSR format directly)
    dK_SG_dell = csr_matrix((P * num_nodes, P * num_nodes))

    for m in range(N_KL):
        # Create coupling matrix directly in CSR format
        # in form_dK_SG_dell  (and same idea in form_dM_SG_dell)
# replace the hand-made 2-entry G_m with the Hermite coupling:
        G_m = G_list[m]
        dK_SG_dell += kron(G_m, K_kl_global_dell[m].tocsr(), format='csr')



    return dK_SG_dell


def compute_local_stiffness(x_coords, y_coords, k_func, k1, phi_grad_func=None, m=None, eigvals_trunc=None, deigvals_trunc_dell=None, grad_ell = False):
    K_local = np.zeros((4, 4))
    for qp, w in zip(quad_points, quad_weights):
        xi_qp, eta_qp = qp
       
        # Shape functions and derivatives (existing code)
        N = np.array([0.25 * (1 - xi_qp) * (1 - eta_qp),
                      0.25 * (1 + xi_qp) * (1 - eta_qp),
                      0.25 * (1 + xi_qp) * (1 + eta_qp),
                      0.25 * (1 - xi_qp) * (1 + eta_qp)])
        dN_dxi = np.array([-0.25 * (1 - eta_qp),
                           0.25 * (1 - eta_qp),
                           0.25 * (1 + eta_qp),
                           -0.25 * (1 + eta_qp)])
        dN_deta = np.array([-0.25 * (1 - xi_qp),
                            -0.25 * (1 + xi_qp),
                            0.25 * (1 + xi_qp),
                            0.25 * (1 - xi_qp)])
        J = np.array([[np.dot(dN_dxi, x_coords), np.dot(dN_dxi, y_coords)],
                      [np.dot(dN_deta, x_coords), np.dot(dN_deta, y_coords)]])
        detJ = np.linalg.det(J)
        invJ = np.linalg.inv(J)
        dN_dx = invJ[0, 0] * dN_dxi + invJ[0, 1] * dN_deta
        dN_dy = invJ[1, 0] * dN_dxi + invJ[1, 1] * dN_deta
       
        # Physical coordinates of quadrature point
        x_qp = np.dot(N, x_coords)
        y_qp = np.dot(N, y_coords)
       
        # Base conductivity
        k = k_func(x_qp, y_qp)
        if phi_grad_func and m is not None and eigvals_trunc is not None:
            if grad_ell == False:
                phi, phi_x, phi_y = phi_grad_func(x_qp, y_qp)
                k += k1 * np.sqrt(eigvals_trunc[m]) * phi
                #grad_enhancement = k1 * np.sqrt(eigvals_trunc[m]) * (phi_x * dN_dx + phi_y * dN_dy)
                grad_enhancement = 0
                grad_dot = np.outer(dN_dx, dN_dx + grad_enhancement) + \
                        np.outer(dN_dy, dN_dy + grad_enhancement)

            else:
                phi, phi_x, phi_y = phi_grad_func(x_qp, y_qp)
                k += 0.5*k1 *(1/ np.sqrt(eigvals_trunc[m]))*deigvals_trunc_dell[m] * phi
                #grad_enhancement = k1 * 0.5*(1/ np.sqrt(eigvals_trunc[m]))*deigvals_trunc_dell[m] * (phi_x * dN_dx + phi_y * dN_dy)
                grad_enhancement = 0
                grad_dot = np.outer(dN_dx, dN_dx + grad_enhancement) + \
                        np.outer(dN_dy, dN_dy + grad_enhancement)
        else:
            grad_dot = np.outer(dN_dx, dN_dx) + np.outer(dN_dy, dN_dy)

        K_local += k * grad_dot * detJ * w
    return K_local
def compute_local_base_mass(x_coords, y_coords, m_func, m1, m=None, phi_func = None, eigenvals=None):
    M_local = np.zeros((4, 4))  # Changed from K_local to M_local since we're computing mass matrix
    for qp, w in zip(quad_points, quad_weights):
        xi_qp, eta_qp = qp
       
        # Shape functions and derivatives
        N = np.array([0.25 * (1 - xi_qp) * (1 - eta_qp),
                      0.25 * (1 + xi_qp) * (1 - eta_qp),
                      0.25 * (1 + xi_qp) * (1 + eta_qp),
                      0.25 * (1 - xi_qp) * (1 + eta_qp)])
        dN_dxi = np.array([-0.25 * (1 - eta_qp),
                           0.25 * (1 - eta_qp),
                           0.25 * (1 + eta_qp),
                           -0.25 * (1 + eta_qp)])
        dN_deta = np.array([-0.25 * (1 - xi_qp),
                            -0.25 * (1 + xi_qp),
                            0.25 * (1 + xi_qp),
                            0.25 * (1 - xi_qp)])
        J = np.array([[np.dot(dN_dxi, x_coords), np.dot(dN_dxi, y_coords)],
                      [np.dot(dN_deta, x_coords), np.dot(dN_deta, y_coords)]])
        detJ = np.linalg.det(J)
       
        # Physical coordinates of quadrature point
        x_qp = np.dot(N, x_coords)
        y_qp = np.dot(N, y_coords)
       
        sum = m_func(x_qp, y_qp) * np.outer(N, N)  # m_func should return a scalar
       
        if eigenvals is not None and m is not None:
            phi = phi_func(x_qp, y_qp)  # Make sure this returns a scalar for mode m at (x_qp, y_qp)

            sum += m1*np.sqrt(eigenvals[m]) * phi * np.outer(N, N)
           
        M_local += sum * detJ * w
    return M_local

def form_k0_global_p(k0,k1):
    # Assemble global mass and stiffness matrices
    M_global = lil_matrix((num_nodes, num_nodes))
    K0_global = lil_matrix((num_nodes, num_nodes))  # Mean stiffness matrix

    for ix in range(Nx):
        for iy in range(Ny):
            node1 = ix * (Ny + 1) + iy
            node2 = node1 + 1
            node3 = (ix + 1) * (Ny + 1) + iy
            node4 = node3 + 1
            nodes = [node1, node2, node3, node4]
            x_coords = np.array([x[ix], x[ix + 1], x[ix + 1], x[ix]])
            y_coords = np.array([y[iy], y[iy], y[iy + 1], y[iy + 1]])
            K0_local = compute_local_stiffness(x_coords, y_coords, lambda x, y: k0,k1)
            for a in range(4):
                for b in range(4):
                    K0_global[nodes[a], nodes[b]] += K0_local[a, b]

    #print(M_SG)
    return K0_global
import numpy as np
from scipy.sparse import coo_matrix

def form_k0_global(k0, k1):
    """
    Assemble global stiffness matrix K0_global without Python loops
    for a structured Nx-by-Ny grid of Q1 elements.
    """
    global _fast_assembler
   
    # Use fast assembly if available (for uniform coefficient, ignores k1)
    if USE_FAST_ASSEMBLY and _fast_assembler is not None:
        return _fast_assembler.assemble_stiffness_uniform(k0)
   
    # Original implementation
    # 1) One representative element to get the local stiffness
    x_coords = np.array([x[0], x[1], x[1], x[0]])
    y_coords = np.array([y[0], y[0], y[1], y[1]])

    K0_local = compute_local_stiffness(
        x_coords,
        y_coords,
        lambda xx, yy: k0,  # constant coefficient
        k1
    )  # shape (4,4)

    # 2) Build element connectivity (Ne x 4) without loops
    Ne = Nx * Ny
    ix = np.arange(Nx)
    iy = np.arange(Ny)
    IX, IY = np.meshgrid(ix, iy, indexing="ij")   # (Nx, Ny)

    node1 = IX * (Ny + 1) + IY
    node2 = node1 + 1
    node3 = (IX + 1) * (Ny + 1) + IY
    node4 = node3 + 1

    elem_nodes = np.stack([node1, node2, node3, node4], axis=-1)  # (Nx, Ny, 4)
    elem_nodes = elem_nodes.reshape(-1, 4)                         # (Ne, 4)

    # 3) Row/column indices for all contributions
    rows = np.repeat(elem_nodes[:, :, None], 4, axis=2)  # (Ne, 4, 4)
    cols = np.repeat(elem_nodes[:, None, :], 4, axis=1)  # (Ne, 4, 4)

    rows = rows.ravel()
    cols = cols.ravel()

    # 4) Local matrix values repeated for each element
    data = np.tile(K0_local.ravel(), Ne)

    # 5) Assemble global sparse matrix
    K0_global = coo_matrix(
        (data, (rows, cols)),
        shape=(num_nodes, num_nodes)
    ).tocsr()   # or .tolil() if you really want LIL

    return K0_global

def form_M0_global_p(m0,m1):
    # Assemble global mass and stiffness matrices

    M0_global = lil_matrix((num_nodes, num_nodes))  # Mean stiffness matrix

    for ix in range(Nx):
        for iy in range(Ny):
            node1 = ix * (Ny + 1) + iy
            node2 = node1 + 1
            node3 = (ix + 1) * (Ny + 1) + iy
            node4 = node3 + 1
            nodes = [node1, node2, node3, node4]
            x_coords = np.array([x[ix], x[ix + 1], x[ix + 1], x[ix]])
            y_coords = np.array([y[iy], y[iy], y[iy + 1], y[iy + 1]])
            M0_local = compute_local_base_mass(x_coords, y_coords, lambda x, y: m0,m1)
            for a in range(4):
                for b in range(4):
                    M0_global[nodes[a], nodes[b]] += M0_local[a, b]

    return M0_global

def form_M0_global(m0, m1):
    """Assemble global mass matrix M0_global."""
    global _fast_assembler
   
    # Use fast assembly if available (for uniform coefficient, ignores m1)
    if USE_FAST_ASSEMBLY and _fast_assembler is not None:
        return _fast_assembler.assemble_mass_uniform(m0)
   
    # Original implementation
    x_coords = np.array([x[0], x[1], x[1], x[0]])
    y_coords = np.array([y[0], y[0], y[1], y[1]])

    M0_local = compute_local_base_mass(
        x_coords,
        y_coords,
        lambda xx, yy: m0,  # coeff function (constant here)
        m1
    )  # shape (4,4)

    # 2) Build element connectivity array (Ne x 4) without loops
    Ne = Nx * Ny
    ix = np.arange(Nx)
    iy = np.arange(Ny)
    IX, IY = np.meshgrid(ix, iy, indexing="ij")   # shape (Nx, Ny)

    node1 = IX * (Ny + 1) + IY
    node2 = node1 + 1
    node3 = (IX + 1) * (Ny + 1) + IY
    node4 = node3 + 1

    elem_nodes = np.stack([node1, node2, node3, node4], axis=-1)  # (Nx, Ny, 4)
    elem_nodes = elem_nodes.reshape(-1, 4)                         # (Ne, 4)

    # 3) Build row/col index arrays for all element contributions
    # rows[e, a, b] = global row index for local (a,b) of element e
    # cols[e, a, b] = global col index for local (a,b) of element e
    rows = np.repeat(elem_nodes[:, :, None], 4, axis=2)  # (Ne, 4, 4)
    cols = np.repeat(elem_nodes[:, None, :], 4, axis=1)  # (Ne, 4, 4)

    rows = rows.ravel()
    cols = cols.ravel()

    # 4) Repeat local matrix values for all elements
    data = np.tile(M0_local.ravel(), Ne)

    # 5) Assemble sparse global matrix
    M0_global = coo_matrix(
        (data, (rows, cols)),
        shape=(num_nodes, num_nodes)
    ).tocsr()  # or .tolil() if you prefer

    return M0_global

def form_K_SG_K0(K0_global):
    """Assemble the mean component of stochastic stiffness matrix"""
    # Use csr format throughout for efficiency
    return kron(identity(P, format='csr'), K0_global.tocsr(), format='csr')
def form_M_SG_M0(M0_global):
    """Assemble the mean component of stochastic mass matrix"""
    # Use csr format throughout for efficiency
    return kron(identity(P, format='csr'), M0_global.tocsr(), format='csr')
from scipy.sparse import kron, coo_matrix

def form_KM_SG_K1(K_kl_global, M_kl_global):
    """
    Build:
      K_SG_K1 = sum_m kron(G_m, K_kl_global[m])
      M_SG_K1 = sum_m kron(G_m, M_kl_global[m])
    where G_list[m] is a (P x P) matrix and K_kl_global[m] is (num_nodes x num_nodes).
    """
    # Build lists of kronecker products
    K_terms = [
        kron(G_list[m], K_kl_global[m], format="coo")
        for m in range(N_KL)
    ]
    M_terms = [
        kron(G_list[m], M_kl_global[m], format="coo")
        for m in range(N_KL)
    ]

    # Sum them; sum(coo_mats) is fine, SciPy combines them
    K_SG_K1 = sum(K_terms).tocsr()
    M_SG_K1 = sum(M_terms).tocsr()

    return K_SG_K1, M_SG_K1
def form_KM_SG_K1_p(K_kl_global, M_kl_global):
    K_SG_K1 = csr_matrix((P*num_nodes, P*num_nodes))
    M_SG_K1 = csr_matrix((P*num_nodes, P*num_nodes))
    for m in range(N_KL):
        G_m = G_list[m]                       # ← NEW
        K_SG_K1 += kron(G_m, K_kl_global[m], format='csr')
        M_SG_K1 += kron(G_m, M_kl_global[m], format='csr')
    return K_SG_K1, M_SG_K1
def form_KM_SG_K1_correct(K_kl_global, M_kl_global):
    """
    Correct coupling for KL expansion terms
    Each KL mode m couples only with the mean mode (0)
    """
    K_SG_K1 = csr_matrix((P*num_nodes, P*num_nodes))
    M_SG_K1 = csr_matrix((P*num_nodes, P*num_nodes))
   
    for m in range(N_KL):
        # Simple symmetric coupling: mean <-> m-th KL mode
        G_m = lil_matrix((P, P))
        G_m[0, m+1] = 1.0
        G_m[m+1, 0] = 1.0
        G_m = G_m.tocsr()
       
        K_SG_K1 += kron(G_m, K_kl_global[m], format='csr')
        M_SG_K1 += kron(G_m, M_kl_global[m], format='csr')
   
    return K_SG_K1, M_SG_K1


def dot_test_sym(A, trials=3):
    import numpy as np
    n = A.shape[0]
    rels = []
    for t in range(trials):
        v = np.random.randn(n)
        w = np.random.randn(n)
        lhs = v @ (A @ w)
        rhs = w @ (A @ v)
        rel = abs(lhs - rhs) / max(1.0, abs(lhs), abs(rhs))
        rels.append(rel)
    return rels

def skew_report(A):
    import numpy as np
    from scipy.sparse import csr_matrix
    S = (A - A.T).tocsr()
    s_inf = abs(S).max()
    # relative to A’s magnitude
    a_inf = abs(A).max()
    rel_inf = s_inf / max(1.0, a_inf)
    return s_inf, rel_inf, S
def enforce_dirichlet_sym(A, b, idx, values):
    """
    Efficient symmetric Dirichlet enforcement.
    A: CSR matrix
    b: numpy array
    idx: array of constrained dofs
    values: prescribed values (same length as idx)
    """
    import numpy as np
    from scipy.sparse import csr_matrix, diags

    # ensure arrays
    idx = np.asarray(idx, dtype=int)
    values = np.asarray(values, dtype=float)

    # Copy inputs (avoid in-place modification unless you want it)
    A = A.tocsr(copy=True)
    b = b.copy()

    # ---- RHS shift ----
    # Build a restriction operator once (if idx set doesn’t change).
    # Here just do one matvec to subtract the constrained cols.
    mask = np.zeros(A.shape[1])
    mask[idx] = values
    b -= A @ mask

    # ---- zero rows and set diag ----
    for ii, val in zip(idx, values):
        row_start, row_end = A.indptr[ii], A.indptr[ii+1]
        if row_end > row_start:
            A.data[row_start:row_end] = 0.0
        A[ii, ii] = 1.0
        b[ii] = val

    # ---- zero columns ----
    # Easiest fast trick: just zero the column entries by multiplying with identity mask
    keep = np.ones(A.shape[0])
    keep[idx] = 0.0
    D = diags(keep)
    A = D @ A @ D
    # then restore the diagonal ones
    A[idx, idx] = 1.0

    return A, b


def get_dirichlet_nodes_LRB(Nx, Ny):
    nodes = set()
    # bottom (y=0)
    for ix in range(Nx+1):
        nodes.add(ix*(Ny+1) + 0)
    # left (x=0) and right (x=Nx)
    for iy in range(Ny+1):
        nodes.add(0*(Ny+1)     + iy)
        nodes.add(Nx*(Ny+1)    + iy)
    return sorted(nodes)

bphys = get_dirichlet_nodes_LRB(Nx, Ny)  # ← exclude top
bc_idx, bc_val, abc_val = [], [], []
T_bc = 300.0
for p in range(P):
    for i in bphys:
        bc_idx.append(p*num_nodes + i)
        bc_val.append(T_bc if p == 0 else 0.0)
        abc_val.append(0.0)
bc_idx = np.array(bc_idx, int)
bc_val = np.array(bc_val, float)
abc_val = np.array(abc_val,float)

# quick sanity check
print("Top constrained? ->", any((i % (Ny+1)) == Ny for i in bphys))  # should be False

# --- terminal cost: linear functional setup ---
rng = np.random.default_rng(seed=12345)  # fixed seed for reproducibility

N = P*num_nodes
bc_mask = np.zeros(N, dtype=bool)
bc_mask[bc_idx] = True
interior_idx = np.where(~bc_mask)[0]

w = np.zeros(N)
w[interior_idx] = rng.standard_normal(interior_idx.size)

# Optional: scale for numerics (not mandatory)
# e.g., normalize by sqrt of interior size so magnitudes are stable:
w[interior_idx] /= np.sqrt(interior_idx.size)

# Convenience helpers
def interior(v):
    vv = v.copy()
    vv[bc_idx] = 0.0
    return vv

def interior_weight():
    # full-length weight already has zeros on bc, but this keeps intent clear
    return w.copy()

def kl_via_fft_direct_with_dlambda(Nx, Ny, Lx, Ly, sigma, ell, n_kl):
    # first column and its derivative w.r.t ell
    col, dcol = _periodic_cov_first_col(Nx, Ny, Lx, Ly, sigma, ell)
    Lam2D  = np.real(np.fft.fft2(col))
    dLam2D = np.real(np.fft.fft2(dcol))

    flat_idx = np.argsort(Lam2D.ravel())[::-1]

    eigvals = []
    deigs  = []
    modes   = []
    taken   = 0
    used = set()

    def real_modes_for(kx, ky):
        x = np.arange(Nx)[:, None]; y = np.arange(Ny)[None, :]
        phase = 2*np.pi*(x*kx/Nx + y*ky/Ny)
        cosm = np.cos(phase) / np.sqrt(Nx*Ny)
        sinm = np.sin(phase) / np.sqrt(Nx*Ny)
        if (kx == 0 or kx == Nx//2) and (ky == 0 or ky == Ny//2):
            return [cosm]
        return [cosm, sinm]

    for idx in flat_idx:
        if taken >= n_kl: break
        kx, ky = divmod(idx, Ny)
        key = tuple(sorted([(kx,ky), ((-kx)%Nx, (-ky)%Ny)]))
        if key in used: continue
        used.add(key)

        lam  = Lam2D[kx, ky]
        dlam = dLam2D[kx, ky]

        for v in real_modes_for(kx, ky):
            eigvals.append(lam)
            deigs.append(dlam)
            modes.append(v)
            taken += 1
            if taken >= n_kl: break

    #eigvals_trunc = np.array(eigvals[:n_kl])
    #dlambda_trunc = np.array(deigs[:n_kl])
    #eigvecs_reshaped = np.stack(modes[:n_kl], axis=2)  # (Nx, Ny, n_kl)

    return eigvals_trunc, dlambda_trunc, eigvecs_reshaped
# Caches for per-mode base matrices (independent of ℓ)
_K_mode_base = None   # list[csr_matrix] of shape (num_nodes, num_nodes)
_M_mode_base = None   # list[csr_matrix] of shape (num_nodes, num_nodes)
# --- KL mode bases for RHS (forcing) ---
_F_mode_base = None     # list(len = N_KL) of nodal vectors: ∫ f(x) φ_m(x) N dΩ   (NO √λ, NO f1)
_F_mean_base = None     # single nodal vector: ∫ f(x) N dΩ                       (NO f0)
_F_cache_t   = None     # if your f(x,y,t) depends on t, we reassemble when t changes

def _assemble_mode_bases(eigvals_trunc, eigvecs_reshaped, phi_gradients):
    """
    Faster build of per-mode base matrices K_m^base, M_m^base (no √λ).
    Uses fast vectorized assembly if available.
    """
    global _fast_assembler
   
    # Use fast assembly if available
   # if USE_FAST_ASSEMBLY and _fast_assembler is not None:
        # Fast assembler expects (Nx, Ny, N_KL) shape
    #    K_mode_base, M_mode_base = _fast_assembler.assemble_KL_modes_vectorized(
    #        eigvecs_reshaped, sqrt_lam=None  # no sqrt_lam scaling here
    #    )
    #    return K_mode_base, M_mode_base
   
    # Fallback to original implementation
    import numpy as np
    import scipy.sparse as sp

    # --- constants for the whole mesh ---
    # 2x2 Gauss on reference square
    gps = np.array([-1/np.sqrt(3), +1/np.sqrt(3)])
    Nq = 4  # 2x2
    # shape functions at 4 qps (order: (xi,eta) = (-,+) × (-,+))
    Nvals = []
    for xi in gps:
        for eta in gps:
            N = np.array([
                0.25*(1-xi)*(1-eta),
                0.25*(1-xi)*(1+eta),
                0.25*(1+xi)*(1-eta),
                0.25*(1+xi)*(1+eta),
            ])
            Nvals.append(N)
    Nvals = np.stack(Nvals, axis=0)              # (4,4)

    # derivatives on reference; affine map to physical ⇒ constant grads
    dN_dxi  = np.array([
        [-0.25*(1-eta) for eta in gps for _ in (0,)],  # not used directly; see below
    ])
    # We’ll compute dN_dx, dN_dy directly for Q1/affine on rectangle:
    invJ = np.array([[ 2.0/hx, 0.0 ],
                     [ 0.0,    2.0/hy ]])               # constant for all elements
    # Reference derivatives (per node)
    #dN_dxi  = np.array([-0.25,  0.25,  0.25, -0.25])
    #dN_deta = np.array([-0.25, -0.25,  0.25,  0.25])
    #dN_dx = invJ[0,0]*dN_dxi + invJ[0,1]*dN_deta        # (4,)
    #dN_dy = invJ[1,0]*dN_dxi + invJ[1,1]*dN_deta        # (4,)
    # Constant grad-dot-grad block for the element
    #GRAD = np.outer(dN_dx, dN_dx) + np.outer(dN_dy, dN_dy)   # (4,4)
    # --- replace your GRAD construction with this ---
    invJ = np.array([[2.0/hx, 0.0],
                    [0.0,    2.0/hy]], dtype=float)
    detJ = (hx*hy)/4.0
    wq = np.ones(4)

    # Precompute GRAD blocks at each qp, consistent with node order [LB, LT, RB, RT]
    GRAD_q = []
    for xi in gps:
        for eta in gps:
            dN_dxi = np.array([
                -0.25*(1.0-eta),  # LB
                -0.25*(1.0+eta),  # LT
                +0.25*(1.0-eta),  # RB
                +0.25*(1.0+eta),  # RT
            ], dtype=float)
            dN_deta = np.array([
                -0.25*(1.0-xi),   # LB
                +0.25*(1.0-xi),   # LT
                -0.25*(1.0+xi),   # RB
                +0.25*(1.0+xi),   # RT
            ], dtype=float)

            dN_dx = invJ[0,0]*dN_dxi + invJ[0,1]*dN_deta
            dN_dy = invJ[1,0]*dN_dxi + invJ[1,1]*dN_deta
            GRAD_q.append(np.outer(dN_dx, dN_dx) + np.outer(dN_dy, dN_dy))

    GRAD_q = np.stack(GRAD_q, axis=0)  # (4,4,4)
    # Mass “shape” blocks at each qp (these differ by qp, but are 4×4 constants)
    M_SHAPES = np.array([np.outer(N, N) for N in Nvals])     # (4,4,4)

    # detJ (constant for all elems) and Gauss weights
    detJ = (hx*hy)/4.0
    wq = np.ones(4)                                          # 2x2 Gauss

    # --- precompute: dof mapping per element (node indices) ---
    elem_nodes = []
    for ix in range(Nx):
        for iy in range(Ny):
            n1 = ix*(Ny+1) + iy
            n2 = n1 + 1
            n3 = (ix+1)*(Ny+1) + iy
            n4 = n3 + 1
            elem_nodes.append([n1, n2, n3, n4])
    elem_nodes = np.asarray(elem_nodes, dtype=np.int64)      # (Ne,4)
    Ne = elem_nodes.shape[0]

    # --- precompute: integer indices in the φ grid for the 4 Gauss points of each element ---
    # Physical qp coords: x = x[ix] + 0.5*(xi+1)*hx ; y = y[iy] + 0.5*(eta+1)*hy
    # Our sampling rule in the rest of the code is nearest-cell via floor(x/Lx*Nx).
    gp_shifts_x = 0.5*(gps+1.0) * hx
    gp_shifts_y = 0.5*(gps+1.0) * hy

    gp_offsets = []
    for xi in gp_shifts_x:
        for eta in gp_shifts_y:
            gp_offsets.append((xi, eta))
    gp_offsets = np.array(gp_offsets)                         # (4,2)

    elem_ix = np.repeat(np.arange(Nx), Ny)
    elem_iy = np.tile(np.arange(Ny), Nx)

    # physical lower-left corner of each element
    x_ll = x[elem_ix]
    y_ll = y[elem_iy]

    # (Ne,4,2) physical qp coords
    X_qp = x_ll[:,None] + gp_offsets[None,:,0]
    Y_qp = y_ll[:,None] + gp_offsets[None,:,1]

    # corresponding φ grid indices (clipped)
    I_qp = np.clip((X_qp / Lx * Nx).astype(int), 0, Nx-1)    # (Ne,4)
    J_qp = np.clip((Y_qp / Ly * Ny).astype(int), 0, Ny-1)    # (Ne,4)

    # --- triplet storage per mode ---
    rowsK = [[] for _ in range(N_KL)]
    colsK = [[] for _ in range(N_KL)]
    valsK = [[] for _ in range(N_KL)]
    rowsM = [[] for _ in range(N_KL)]
    colsM = [[] for _ in range(N_KL)]
    valsM = [[] for _ in range(N_KL)]

    # --- main loop over modes (outer) and elements (inner) ---
    # (This is already much cheaper; numba can be added later if needed.)
    for m in range(N_KL):
        phi = eigvecs_reshaped[:, :, m]     # (Nx,Ny)

        # φ at all 4 gp for all elements: (Ne,4)
        phi_qp = phi[I_qp, J_qp]

        # stiffness: sum_q w_q * φ(x_q)  (scalar per element)
        sphi = (phi_qp * wq[None, :]).sum(axis=1)            # (Ne,)

        # mass: linear combo of 4 constant shapes with weights φ_q
        # M_loc = sum_q (w_q * φ_q) * M_SHAPES[q] * detJ
        # We'll compute the 4 coeffs per element, then accumulate 4 small 4×4 adds.
        coeffs = phi_qp * wq[None, :]                        # (Ne,4)

        for e in range(Ne):
            nodes = elem_nodes[e]

            # K_loc = (sum_q w φ_q) * GRAD * detJ
           # K_loc = (sphi[e] * detJ) * GRAD
            # NEW (consistent quadrature)
            K_loc = detJ * (
                coeffs[e,0] * GRAD_q[0] +
                coeffs[e,1] * GRAD_q[1] +
                coeffs[e,2] * GRAD_q[2] +
                coeffs[e,3] * GRAD_q[3]
            )
            # M_loc = detJ * sum_q coeff[e,q] * M_SHAPES[q]
            # do it explicitly to stay in NumPy (4 only)
            M_loc = detJ * (
                coeffs[e,0] * M_SHAPES[0] +
                coeffs[e,1] * M_SHAPES[1] +
                coeffs[e,2] * M_SHAPES[2] +
                coeffs[e,3] * M_SHAPES[3]
            )

            # append 16 triplets (4×4) once
            for a in range(4):
                ra = nodes[a]
                for b in range(4):
                    cb = nodes[b]
                    vK = K_loc[a, b]
                    vM = M_loc[a, b]
                    if vK != 0.0:
                        rowsK[m].append(ra); colsK[m].append(cb); valsK[m].append(vK)
                    if vM != 0.0:
                        rowsM[m].append(ra); colsM[m].append(cb); valsM[m].append(vM)

    # build CSR once per mode
    K_mode_base = []
    M_mode_base = []
    for m in range(N_KL):
        K_mode_base.append(sp.coo_matrix((valsK[m], (rowsK[m], colsK[m])), shape=(num_nodes, num_nodes)).tocsr())
        M_mode_base.append(sp.coo_matrix((valsM[m], (rowsM[m], colsM[m])), shape=(num_nodes, num_nodes)).tocsr())
    return K_mode_base, M_mode_base
def _assemble_F_bases(eigvecs_reshaped, t=0.0):
    """
    Build time-slice forcing bases that DO NOT depend on √λ, f0, or f1:
      - F_mean_base: ∫ f(x,y,t) N dΩ
      - F_mode_base[m]: ∫ f(x,y,t) φ_m(x,y) N dΩ
    Uses the same quadrature, shapes and Jacobian as stiffness/mass assembly.
    """
    # mean base
    F_mean_base = np.zeros(num_nodes)
    # per-mode bases
    F_mode_base = [np.zeros(num_nodes) for _ in range(N_KL)]

    for ix in range(Nx):
        for iy in range(Ny):
            node1 = ix*(Ny+1)+iy
            node2 = node1 + 1
            node3 = (ix+1)*(Ny+1)+iy
            node4 = node3 + 1
            nodes = [node1, node2, node3, node4]

            x_coords = np.array([x[ix], x[ix+1], x[ix+1], x[ix]])
            y_coords = np.array([y[iy], y[iy], y[iy+1], y[iy+1]])

            for (xi_qp, eta_qp), wq in zip(quad_points, quad_weights):
                # bilinear shape funcs
                N = np.array([
                    0.25*(1 - xi_qp)*(1 - eta_qp),
                    0.25*(1 + xi_qp)*(1 - eta_qp),
                    0.25*(1 + xi_qp)*(1 + eta_qp),
                    0.25*(1 - xi_qp)*(1 + eta_qp),
                ])
                dN_dxi  = np.array([-0.25*(1-eta_qp),  0.25*(1-eta_qp),  0.25*(1+eta_qp), -0.25*(1+eta_qp)])
                dN_deta = np.array([-0.25*(1-xi_qp), -0.25*(1+xi_qp),  0.25*(1+xi_qp),  0.25*(1-xi_qp)])

                J = np.array([
                    [np.dot(dN_dxi,  x_coords), np.dot(dN_dxi,  y_coords)],
                    [np.dot(dN_deta, x_coords), np.dot(dN_deta, y_coords)],
                ])
                detJ = np.linalg.det(J)

                # physical qp and forcing value
                x_qp = np.dot(N, x_coords)
                y_qp = np.dot(N, y_coords)
                fq   = f(x_qp, y_qp, t)  # your forcing

                scale = detJ * wq * fq

                # add to mean base
                for a in range(4):
                    F_mean_base[nodes[a]] += scale * N[a]

                # map qp -> φ grid (same rule you use elsewhere)
                i = int(np.floor(x_qp / Lx * Nx)); i = 0 if i < 0 else (Nx-1 if i >= Nx else i)
                j = int(np.floor(y_qp / Ly * Ny)); j = 0 if j < 0 else (Ny-1 if j >= Ny else j)

                # per-mode bases (just φ_m, NO √λ, NO f1)
                for m in range(N_KL):
                    phi_qp = eigvecs_reshaped[i, j, m]
                    for a in range(4):
                        F_mode_base[m][nodes[a]] += scale * phi_qp * N[a]

    return F_mean_base, F_mode_base
def get_F_bases(eigvecs_reshaped, t=0.0):
    """
    Cached access to F_mean_base and F_mode_base at time t.
    Bases do NOT depend on √λ, f0, f1, or ℓ.
    """
    global _F_mean_base, _F_mode_base, _F_cache_t

    if (_F_mean_base is None) or (_F_mode_base is None) or (_F_cache_t != t):
        _F_mean_base, _F_mode_base = _assemble_F_bases(eigvecs_reshaped, t)
        _F_cache_t = t

    return _F_mean_base, _F_mode_base



def form_K_kl_global(eigvals_trunc, eigvecs_reshaped, phi_gradients):
    """
    Return list of per-mode stiffness matrices with √λ weighting baked in:
    K_kl_global[m] = √λ_m * K_m^base
    Always rebuilds _K_mode_base from the supplied eigvecs so that
    different kappa values (e.g. obs vs init) don't share stale bases.
    """
    global _K_mode_base, _M_mode_base
    _K_mode_base, _M_mode_base = _assemble_mode_bases(eigvals_trunc, eigvecs_reshaped, phi_gradients)

    w = np.sqrt(eigvals_trunc)                    # weights for K1/M1
    return [w[m] * _K_mode_base[m] for m in range(N_KL)]


def form_M_kl_global(eigvals_trunc, eigvecs_reshaped, phi_gradients):
    """
    Return list of per-mode mass matrices with √λ weighting baked in:
    M_kl_global[m] = √λ_m * M_m^base
    Relies on _K_mode_base/_M_mode_base set by form_K_kl_global (always called first
    in build_SG_operators), so no rebuild needed here.
    """
    global _K_mode_base, _M_mode_base
    if _M_mode_base is None:
        # Fallback: should not happen if build_SG_operators calls form_K_kl_global first
        _K_mode_base, _M_mode_base = _assemble_mode_bases(eigvals_trunc, eigvecs_reshaped, phi_gradients)

    w = np.sqrt(eigvals_trunc)
    return [w[m] * _M_mode_base[m] for m in range(N_KL)]


def form_dK_SG_dell(eigvals_trunc, dlambda_dell, eigvecs_reshaped, phi_gradients):
    """
    dK1/dℓ as a Kron-sum without reassembly:
        dK1/dℓ = Σ_m (0.5 * dlambda/√λ)_m * K_m^base
        ⇒ dK_SG_dell = Σ_m G_m ⊗ [ (0.5*dlambda/√λ)_m * K_m^base ]
    """
    global _K_mode_base, _M_mode_base
    if _K_mode_base is None:
        _K_mode_base, _M_mode_base = _assemble_mode_bases(eigvals_trunc, eigvecs_reshaped, phi_gradients)

    # safe weights (handles tiny λ)
    w = np.sqrt(eigvals_trunc)
    eps = 1e-30
    dw = 0.5 * (dlambda_dell / np.maximum(w, eps))
    #print(dw)
    dK_SG = csr_matrix((P * num_nodes, P * num_nodes))
    for m in range(N_KL):
        if dw[m] != 0.0:
            dK_SG += kron(G_list[m], dw[m] * _K_mode_base[m], format='csr')
    return dK_SG


def form_dM_SG_dell(eigvals_trunc, dlambda_dell, eigvecs_reshaped, phi_gradients):
    """
    dM1/dℓ analogous to stiffness, no reassembly.
    """
    global _K_mode_base, _M_mode_base
    if _M_mode_base is None:
        _K_mode_base, _M_mode_base = _assemble_mode_bases(eigvals_trunc, eigvecs_reshaped, phi_gradients)

    w = np.sqrt(eigvals_trunc)
    eps = 1e-30
    dw = 0.5 * (dlambda_dell / np.maximum(w, eps))
    #print(dw)
    dM_SG = csr_matrix((P * num_nodes, P * num_nodes))
    for m in range(N_KL):
        if dw[m] != 0.0:
            dM_SG += kron(G_list[m], dw[m] * _M_mode_base[m], format='csr')
    return dM_SG
_SRC_VERSION = 0  # bumps whenever q_grid is refreshed

def extract_interface_h_from_SG_jax(Tm, Tv, T_abl, x_nodes, y_nodes, eps_T=0.5):
    eps_T = float(eps_T)
    y_top = y_nodes[-1]

    def process_column(colT, colV):
        # Smooth indicator in T
        s = 0.5 * (1.0 + jnp.tanh((colT - T_abl) / (eps_T + 1e-16)))
        w = s
        w_sum = jnp.sum(w)

        def no_hot_region(_):
            # exactly like NumPy: put interface at top
            h = y_top
            return h, h, h

        def with_hot_region(_):
            h_col = jnp.sum(y_nodes * w) / (w_sum + 1e-16)
            # no clipping; h_lo = h_hi = h_col
            return h_col, h_col, h_col

        h_mean_col, h_lo_col, h_hi_col = jax.lax.cond(
            w_sum < 1e-16,          # same threshold as NumPy
            no_hot_region,
            with_hot_region,
            operand=None
        )

        return h_mean_col, h_lo_col, h_hi_col

    h_mean, h_lo, h_hi = jax.vmap(process_column, in_axes=(0, 0))(Tm, Tv)
    return x_nodes, h_mean, h_lo, h_hi


# ============================================================================
# Helper functions (matching 3D version style)
# ============================================================================

def _smooth_ramp_jax(T, T_low, T_high, Delta):
    """
    Smooth ramp from 0 to 1 as T goes from T_low to T_high.
    Delta controls the smoothing width.
    """
    return 0.5 * (1.0 + jnp.tanh((T - 0.5*(T_low + T_high)) / Delta))


def _smooth_heaviside_jax(x, epsilon):
    """
    Smooth Heaviside: ~0 for x<0, ~1 for x>0, with transition width epsilon.
    """
    return 0.5 * (1.0 + jnp.tanh(x / jnp.maximum(epsilon, 1e-16)))

@jax.jit
def gaussian_beam_source_2d_jax(P0, T, x, y, y_edges, props):
    """
    Volumetric beam heat source for 2D (x=radial, y=depth) FE model.

    Follows Gokhale et al. (2026) Sec 2.2:
      - Beer-Lambert absorption with temperature-dependent α(T) from Table 2
      - Absorptivity correction factor η = 0.83 (Table 2)
      - Surface transmissivity ||T||² = 0.75 (measured, Sec 4.1.1)
      - Beam divergence neglected inside rock (absorption length << domain)
      - 2D peak intensity: P0 / (sqrt(π/2) * ω0)  [W/m]
    """
    omega0            = props["beam_rad"]           # beam waist at rock surface [m] — use 20.79e-3
    x_center          = props["x_center"]           # beam centre x [m]
    T_vap             = props["T_vap"]              # vaporisation onset temperature [K]
    evap_range        = props["evap_range"]         # temperature range for H transition [K]
    Delta_vap         = props["Delta_vap"]          # smoothing width for H(T) [K]
    epsilon_smoothing = props["epsilon_smoothing"]  # spatial smoothing length [m]

    # ── Gokhale Table 2 absorption coefficients ───────────────────────────────
    a_abs = props.get("alpha_a", 32.71)    # constant term         [1/m]
    b_abs = props.get("alpha_b", 62455.03)  # pre-exponential term  [1/m]
    c_abs = props.get("alpha_c", 9998.36)   # exponent temperature  [K]

    # absorptivity correction (Table 2) and surface transmissivity (Sec 4.1.1)
    eta   = props.get("eta",           0.83)   # absorptivity correction factor [-]
    T_fac = props.get("transmissivity", 0.75)  # ||T||^2 at rock-vapour interface [-]

    # ── 2D peak intensity ─────────────────────────────────────────────────────
    # In 3D: I0 = 2P0 / (π ω0²)  [W/m²]
    # In 2D (one transverse dimension x):
    #   ∫ I0_2d * exp(-2x²/ω0²) dx = P0  =>  I0_2d = P0 / (sqrt(π/2) * ω0)  [W/m]
    I0 = 2.0 * P0 / (jnp.pi * omega0**2)
    # ── Surface tracking ──────────────────────────────────────────────────────
    H       = _smooth_ramp_jax(T, T_vap_lo, T_vap_hi, Delta_vap)  # (Nx+1, Ny+1)
    H_avg   = 0.5 * (H[:, :-1] + H[:, 1:])                               # (Nx+1, Ny)

    dy_vec  = y[1:] - y[:-1]                                              # (Ny,)
    thickness = jnp.sum(H_avg * dy_vec[None, :], axis=-1)                 # (Nx+1,)
    y_top   = y[-1]
    y0      = y_top - thickness                                            # (Nx+1,)

    # signed distance: > 0 inside rock, < 0 in vapour/air
    dist_s  =(Ly - y[None, :])                                    # (Nx+1, Ny+1)

    # ── Transverse Gaussian profile (no divergence inside rock) ──────────────
    dx2             = jnp.square(x - x_center)                            # (Nx+1,)
    gaussian_profile = jnp.exp(-2.0 * dx2[:, None] / (omega0**2 + 1e-30))# (Nx+1, Ny+1)
    I_surface       = I0 * gaussian_profile                               # (Nx+1, Ny+1)

    # ── Temperature-dependent absorption coefficient (Gokhale Table 2) ───────
    # α(T) = a + b * exp(-c/T)
    # Clamp T to avoid exp overflow at very low T
    T_safe  = jnp.maximum(T, 1.0)
    alpha_T = a_abs           # (Nx+1, Ny+1)

    # ── Beer-Lambert depth attenuation ────────────────────────────────────────
    depth       = jnp.maximum(dist_s, 0.0)                                # (Nx+1, Ny+1)
    absorption  = jnp.exp(-alpha_T * depth)                               # (Nx+1, Ny+1)

    # ── Interior mask ─────────────────────────────────────────────────────────
    interior_mask = _smooth_heaviside_jax(dist_s,5 )      # (Nx+1, Ny+1)

    # ── Volumetric heat source: Q = η * ||T||² * α * I * exp(-α*depth) ───────
    # Matches Gokhale Eq (44): S_beam = 2η * α_λ(T(ξ)) * I(x,y,T) * (1 - H(φ))
    Q = eta*T_fac* 4*alpha_T * I_surface *interior_mask * absorption

    return Q
@jax.jit
def gaussian_beam_source_plus_loss_term_jax(P0, T, x, y, y_edges, props):
    # --- Absorption source (volumetric) ---
    Q_abs = gaussian_beam_source_2d_jax(P0, T, x, y, y_edges, props)

    # --- Loss parameters ---
    h_conv     = props.get("h_conv")          # [W/(m^2 K)]
    T_inf      = props.get("T_inf")         # [K]
    emissivity = props.get("emissivity")      # [-]
    sigma_SB   = props.get("sigma_SB")  # [W/(m^2 K^4)]
    eps_s      = props.get("epsilon_smoothing")   # [m] thickness scale for spreading

    # --- Recompute tracked surface distance dist_s (same logic as gaussian_beam_source_2d_jax) ---
    T_vap      = props["T_vap"]
    evap_range = props["evap_range"]
    Delta_vap  = props["Delta_vap"]

    # smooth indicator for "ablated/vapor" region
    H = _smooth_ramp_jax(T, T_vap, T_vap + evap_range, Delta_vap)  # (Nx+1, Ny+1)

    # infer ablated thickness per x-column
    H_avg   = 0.5 * (H[:, :-1] + H[:, 1:])         # (Nx+1, Ny)
    dy_vec  = y[1:] - y[:-1]                        # (Ny,)
    thickness = jnp.sum(H_avg * dy_vec[None, :], axis=-1)  # (Nx+1,)

    y_top = y[-1]
    y0    = y_top - thickness                       # tracked surface y0(x)
    dist_s = y0[:, None] - y[None, :]               # >0 inside material, <0 outside

    # --- Smooth "inside" mask and a one-sided surface delta (integrates ~1 on inside) ---
    H_in = _smooth_heaviside_jax(dist_s, eps_s)     # ~1 inside, ~0 outside

    # d/dy of H_in gives a smooth delta; make it one-sided (inside) and renormalize
    sech2 = 1.0 / jnp.cosh(dist_s / jnp.maximum(eps_s, 1e-16))**2
    dH_dy = 0.5 * (1.0 / jnp.maximum(eps_s, 1e-16)) * sech2      # integrates to 1 over (-inf, inf)
    delta_surf_inside = 2.0 * H_in * dH_dy                        # inside-only, integrates ~1

    # --- Surface heat flux losses (W/m^2) ---
    q_conv = h_conv * (T - T_inf)
    q_rad  = emissivity * sigma_SB * (T**4 - T_inf**4)

    q_loss = q_conv + q_rad                                       # W/m^2

    # --- Spread surface flux into a volumetric sink (W/m^3) ---
    Q_loss = q_loss * delta_surf_inside

    # --- Ensure absorption only heats the material (optional but recommended) ---
    Q_abs_in = Q_abs

    # --- Net source term ---
    Q = Q_abs_in - Q_loss
    return Q_abs_in, Q_loss


def make_h_func_jax(x_nodes, h_mean):
    """
    Return a JAX-compatible function h_func(xq) that linearly interpolates
    h_mean on x_nodes. xq can be scalar or array.
    """
    x_nodes = jnp.asarray(x_nodes)
    h_mean  = jnp.asarray(h_mean)
    dx = x_nodes[1] - x_nodes[0]
    x0 = x_nodes[0]
    n  = x_nodes.shape[0]

    def h_func(xq):
        xq = jnp.asarray(xq)
        # scaled coordinate
        s = (xq - x0) / jnp.maximum(dx, 1e-16)
        i0 = jnp.clip(jnp.floor(s).astype(int), 0, n-2)
        t  = s - i0
        h0 = h_mean[i0]
        h1 = h_mean[i0 + 1]
        return (1.0 - t) * h0 + t * h1

    return h_func
import jax
import jax.numpy as jnp
from jax import lax

def build_beam_q_grid_jax(
    T_m,
    Xg,
    Yg,
    alpha_grid,
    x_nodes,
    y_nodes,
    T_abl,
    params,
    t,
):
    """
    Build q(x,y) on the nodal grid using a precomputed reflectivity field alpha_grid
    with shape (Nx+1, Ny+1) or (num_nodes,). JAX-jittable.

    Beam model:
        Q_abs, Q_loss = gaussian_beam_source_plus_loss_term_jax(P0, T, x, y, y_edges, props)

    Heating/cooling schedule:
        if t < 5:   q = Q_abs * alpha_grid - Q_loss
        else:       q = -Q_loss

    Uses jax.lax.cond to avoid Python control flow under jit.
    """
    # For nodal grids, y_edges can be y_nodes
    y_edges = y_nodes

    # Build props dict (must contain only JAX-friendly values when jitted)
    props = {
        "beam_rad": params.get("w0", 2e-2),
        "wavelength": params.get("lam", 10e-6),
        "x_center": params.get("x_c", 0.105),
        "T_vap": T_abl,
        "evap_range": params.get("evap_range", 0.0),
        "Delta_vap": params.get("Delta_vap", 5.0),
        "epsilon_smoothing": params.get("epsilon_smoothing", 2e3),
        "alpha_base": params.get("alpha_base", 50.0),
        "alpha_vap": params.get("alpha_vap", 10.0),
        "h_conv": params.get("h_conv", 0.0),
        "T_inf": params.get("T_inf", 0.0),
        "emissivity": params.get("emissivity", 0.0),
        "sigma_SB": params.get("sigma_SB", 5.670374419e-8),
    }
    P0 = params.get("P0", 1e4)

    # Beam model (should be JAX-friendly / jittable)
    Q_abs, Q_loss = gaussian_beam_source_plus_loss_term_jax(
        P0, T_m, x_nodes, y_nodes, y_edges, props
    )

    # Ensure t is a JAX scalar (works whether t is Python float/int or JAX array)
    t = jnp.asarray(t)

    def beam_on(_):
        # Heating phase: absorbed beam minus losses
        return Q_abs * alpha_grid

    def beam_off(_):
        # Cooling phase: only losses
        return -Q_loss

    return lax.cond(t < t_off, beam_on, beam_off, operand=None)


def He_n_jax(n, x):
    """Evaluate physicists' Hermite polynomial He_n(x)."""
    x = jnp.asarray(x)
    n = int(n)  # n must be a Python int

    if n == 0:
        return jnp.ones_like(x)
    if n == 1:
        return 2.0 * x

    def body(k, state):
        Hkm1, Hk = state
        Hkp1 = 2.0 * x * Hk - 2.0 * (k - 1) * Hkm1
        return (Hk, Hkp1)

    _, Hn = jax.lax.fori_loop(
        2, n + 1,
        body,
        (jnp.ones_like(x), 2.0 * x)
    )
    return Hn


def eval_psi_jax(xi_vec, alpha):
    """
    xi_vec: (N_KL,) JAX array
    alpha : tuple/list of ints (multi-index)
    """
    val = 1.0
    for m in range(len(alpha)):
        p = int(alpha[m])  # alpha is Python tuple, so this is fine
        if p == 0:
            He_p = jnp.ones_like(xi_vec[m])
        else:
            He_p = He_n_jax(p, xi_vec[m])

        val = val * He_p / jnp.sqrt(float(math.factorial(p)))
    return val

eval_psi_vec = jax.vmap(eval_psi_jax, in_axes=(None, 0))  # over alpha
def reconstruct_u_from_SG_jax_p1(Ucoeff, xi_vec, params):
    """
    Specialised reconstructor for total-degree p_order = 1.
    Ucoeff: (P*num_nodes,)
    xi_vec: (N_KL,)
    """
    idx_of    = params["idx_of"]

    U_blocks = Ucoeff.reshape(P, num_nodes)  # (P, num_nodes)

    # mean block is always multi-index (0,0,...)
    u = U_blocks[0, :]                       # (num_nodes,)

    # add first-order contributions ξ_m * U_m
    for m in range(N_KL):
        e = [0]*N_KL; e[m] = 1
        p = idx_of[tuple(e)]
        u = u + xi_vec[m] * U_blocks[p, :]

    return u

def reconstruct_u_from_SG_jax(Ucoeff, xi_vec, params):
    """
    Ucoeff: (P*num_nodes,) JAX array
    xi_vec: (N_KL,) JAX array
    params["multi_idx"]: list of length-P tuples
    """
    multi_idx = params["multi_idx"]  # Python list, not jnp array

    U_blocks = Ucoeff.reshape(P, num_nodes)  # (P, num_nodes)

    psi_vals = []
    for alpha in multi_idx:
        psi_vals.append(eval_psi_jax(xi_vec, alpha))
    psi_vec = jnp.stack(psi_vals)            # (P,)

    u = psi_vec @ U_blocks                  # (num_nodes,)
    return u

import jax
import jax.numpy as jnp
from dataclasses import dataclass
import jax
import jax.numpy as jnp



def assemble_F_from_qgrid_jax(q_grid, params):
    """
    JAX-compatible FE assembly of the load vector from a volumetric source q_grid.

    q_grid : (Nx+1, Ny+1) array (JAX)
        Source sampled on the same nodal grid as x,y.
    params : dict with at least
        Nx, Ny      : ints
        x, y        : 1D arrays of nodes
        Lx, Ly      : domain sizes (for sampling)
        quad_points : (Nq, 2) array of (xi, eta)
        quad_weights: (Nq,) array
        num_nodes   : total number of FE nodes

    Returns
    -------
    F_det : (num_nodes,) JAX array
        Deterministic load vector ∫ q N dΩ.
    """
   # Nx = int(params["Nx"])
   # Ny = int(params["Ny"])
    x  = params["x"]          # shape (Nx+1,)
    y  = params["y"]          # shape (Ny+1,)
    Lx = params["Lx"]
    Ly = params["Ly"]
    quad_points  = params["quad_points"]   # (Nq, 2)
    quad_weights = params["quad_weights"]  # (Nq,)

    hx = x[1] - x[0]
    hy = y[1] - y[0]

    # bilinear sampling of q_grid at arbitrary (xq, yq)
    def bilinear_sample(q_grid, xq, yq):
        # map to index space
        sx = xq / Lx * Nx
        sy = yq / Ly * Ny

        i0 = jnp.clip(jnp.floor(sx).astype(int), 0, Nx-1)
        j0 = jnp.clip(jnp.floor(sy).astype(int), 0, Ny-1)

        tx = sx - i0
        ty = sy - j0

        q00 = q_grid[i0,   j0  ]
        q10 = q_grid[i0+1, j0  ]
        q01 = q_grid[i0,   j0+1]
        q11 = q_grid[i0+1, j0+1]

        return ((1-tx)*(1-ty)*q00 +
                tx*(1-ty)*q10 +
                (1-tx)*ty*q01 +
                tx*ty*q11)

    # number of elements
    Ne = Nx * Ny
    F0 = jnp.zeros(num_nodes)

    # loop over elements
    def elem_body(e, F_det):
        ix = e // Ny
        iy = e % Ny

        node1 = ix*(Ny+1) + iy
        node2 = node1 + 1
        node3 = (ix+1)*(Ny+1) + iy
        node4 = node3 + 1
        nodes = jnp.array([node1, node2, node3, node4], dtype=jnp.int32)

        x_coords = jnp.array([x[ix], x[ix+1], x[ix+1], x[ix]])
        y_coords = jnp.array([y[iy], y[iy], y[iy+1], y[iy+1]])

        def qp_body(qi, F_det_inner):
            xi_qp, eta_qp = quad_points[qi]
            wq = quad_weights[qi]

            # bilinear shape functions on reference element
            N = jnp.array([
                0.25*(1 - xi_qp)*(1 - eta_qp),
                0.25*(1 + xi_qp)*(1 - eta_qp),
                0.25*(1 + xi_qp)*(1 + eta_qp),
                0.25*(1 - xi_qp)*(1 + eta_qp),
            ])

            x_qp = jnp.dot(N, x_coords)
            y_qp = jnp.dot(N, y_coords)

            # affine Q1 rectangle ⇒ detJ constant
            detJ = (hx * hy) / 4.0

            fq = bilinear_sample(q_grid, x_qp, y_qp)
            scale = detJ * wq * fq

            # add contributions to the 4 local nodes
            F_det_inner = F_det_inner.at[nodes].add(scale * N)
            return F_det_inner

        F_det = jax.lax.fori_loop(0, quad_points.shape[0], qp_body, F_det)
        return F_det

    F_det = jax.lax.fori_loop(0, Ne, elem_body, F0)
    return F_det


def make_gaussian_samples_antithetic(N_KL, K=64, seed=12345):
    assert K % 2 == 0
    rng = np.random.default_rng(seed)
    half = K // 2
    z = rng.standard_normal((half, N_KL))
    xi_samples = np.vstack([z, -z])
    xi_weights = np.full(K, 1.0 / K)
    return xi_samples, xi_weights


# e.g. in your setup:
xi_samples, xi_weights = make_gaussian_samples_antithetic(N_KL, K=512)
mu = np.sum(xi_weights[:, None] * xi_samples, axis=0)
xi_samples = xi_samples - mu[None, :]

params["xi_samples"] = xi_samples
params["xi_weights"] = xi_weights




@jax.jit
def _F_SG_jax_baked(Ucoeff_jax,f0,f1):
    return F_SG_jax(Ucoeff_jax,f0,f1)

@jax.jit
def F_SG_f_sens_jax(Ucoeff_jax):
    """
    Returns (F_f0, F_f1) such that:
      F(U,f0,f1) = f0 * F_f0 + f1 * F_f1
    assuming linearity of F in (f0,f1).
    """
    F_f0 = F_SG_jax(Ucoeff_jax, 1.0, 0.0)  # derivative wrt f0
    F_f1 = F_SG_jax(Ucoeff_jax, 0.0, 1.0)  # derivative wrt f1
    return F_f0, F_f1

def F_SG_df0_numpy(Ucoeff, params, f0, f1):
    U_jax = jnp.asarray(Ucoeff)
    F_f0, _ = F_SG_f_sens_jax(U_jax)
    return np.asarray(F_f0, dtype=float)

def F_SG_df1_numpy(Ucoeff, params, f0, f1):
    U_jax = jnp.asarray(Ucoeff)
    _, F_f1 = F_SG_f_sens_jax(U_jax)
    return np.asarray(F_f1, dtype=float)



def cell_to_nodal(q_cell):
    # q_cell: (Nx, Ny)  ->  returns: (Nx+1, Ny+1)
    q_pad = jnp.pad(q_cell, ((1, 1), (1, 1)), mode="edge")  # (Nx+2, Ny+2)
    q00 = q_pad[0:-1, 0:-1]
    q10 = q_pad[1:  , 0:-1]
    q01 = q_pad[0:-1, 1:  ]
    q11 = q_pad[1:  , 1:  ]
    return 0.25 * (q00 + q10 + q01 + q11)
# ---- one-time setup (module scope) ----
Nx = int(params["Nx"]); Ny = int(params["Ny"])
num_nodes = (Nx+1)*(Ny+1)

Xg = jnp.asarray(params["Xg"])
Yg = jnp.asarray(params["Yg"])
xi_samples = jnp.asarray(params["xi_samples"])       # (K, N_KL)
xi_weights = jnp.asarray(params["xi_weights"])       # (K,)

# p_idx computed once using idx_of (python dict)
idx_of = params["idx_of"]
p_idx_np = np.empty(N_KL, dtype=int)
for m in range(N_KL):
    e = [0]*N_KL; e[m] = 1
    p_idx_np[m] = idx_of[tuple(e)]
p_idx = jnp.asarray(p_idx_np, dtype=jnp.int32)

# choose recon once
recon = reconstruct_u_from_SG_jax_p1 if (P == N_KL+1) else reconstruct_u_from_SG_jax

# bc mask once
bc_mask_np = np.ones(P*num_nodes, dtype=np.float32)
bc_mask_np[bc_idx] = 0.0
BC_MASK = jnp.asarray(bc_mask_np)

# Cache for make_F_SG_core
_F_SG_CORE_CACHE = {}
@jax.jit
def F_SG_apply(Ucoeff, f0, f1, t, sqrt_lam, params):
    Phi = jnp.asarray(params["eigvecs_grid"])            # (N_KL, Nx+1, Ny+1)

    Ucoeff = jnp.asarray(Ucoeff)
    f0 = jnp.asarray(f0); f1 = jnp.asarray(f1)
    t = jnp.asarray(t, dtype=jnp.int32)
    sqrt_lam = jnp.asarray(sqrt_lam)

    def recon_to_F(xi_vec):
        u = recon(Ucoeff, xi_vec, params).reshape(Nx+1, Ny+1)

        w = sqrt_lam * xi_vec
        # Phi is now (N_KL, Nx+1, Ny+1) vertex-centred; result is already nodal
        alpha_grid = f0 + f1 * jnp.tensordot(w, Phi, axes=(0, 0))

        q = build_beam_q_grid_jax(u, Xg, Yg, alpha_grid, x, y, T_abl, params, t)
        return assemble_F_from_qgrid_jax(q, params)  # (num_nodes,)

    F_stack = jax.vmap(recon_to_F)(xi_samples)  # (K, num_nodes)
    c0 = jnp.einsum("k,kj->j", xi_weights, F_stack)
    cm = jnp.einsum("k,kj,km->mj", xi_weights, F_stack, xi_samples)

    F_SG = jnp.zeros((P, num_nodes), dtype=F_stack.dtype)
    F_SG = F_SG.at[0, :].set(c0)
    F_SG = F_SG.at[p_idx, :].set(cm)

    return F_SG.reshape(-1)

import numpy as np
import jax.numpy as jnp

def forcing_dF_from_dphi(params, sqrt_lam_c,t, Ucoeff, dphi_reshaped):
    """
    Directional derivative of the KL-weighted forcing w.r.t. eigenvectors.

    Uses the same forcing construction as make_F_SG_core, but swaps:
      eigvecs_grid := dphi
    and uses the usual sqrt_lam weights.

    Inputs
    ------
    params : dict containing at least:
        - "sqrt_lam"    : (N_KL,) weights (sqrt eigenvalues)
        - "eigvecs_grid": (N_KL, Nx, Ny) eigenvectors on grid (not used directly here)
    dphi_reshaped : (Nx, Ny, N_KL)  (or whatever your phi_grid layout is)
    """

    # dphi_reshaped expected (Nx,Ny,N_KL) -> make eigvecs_grid as (N_KL,Nx,Ny)
    if dphi_reshaped.ndim != 3:
        raise ValueError(f"dphi_reshaped must be 3D (Nx,Ny,N_KL). Got {dphi_reshaped.shape}")

    dphi_grid = np.transpose(np.asarray(dphi_reshaped, float), (2, 0, 1))  # (N_KL,Nx,Ny)

    # Create a shallow copy of params with swapped eigenvectors
    params_dphi = dict(params)
    params_dphi["eigvecs_grid"] = jnp.asarray(dphi_grid)

    # Build forcing operator with swapped eigenvectors
    F_dphi_core = make_F_SG_core(params_dphi, f0=0.0, f1=1.0, t=t, sqrt_lam=sqrt_lam_c)

    return np.asarray(F_dphi_core(Ucoeff), dtype=float)
def make_F_SG_core(params, f0, f1, t, sqrt_lam=None, phi=None):

    global _F_SG_CORE_CACHE

    # Skip cache when f0/f1 are JAX tracers (inside jax.grad/jit) or
    # non-scalar arrays — both are unhashable and will error in future JAX.
    _cacheable = (not isinstance(f0, jax.core.Tracer)
                  and not isinstance(f1, jax.core.Tracer)
                  and np.ndim(f0) == 0
                  and np.ndim(f1) == 0)

    if _cacheable:
        cache_key = (
            float(f0), float(f1),
            params.get("Nx"), params.get("Ny"),
            tuple(np.asarray(sqrt_lam).flatten()) if sqrt_lam is not None else None,
            id(params.get("eigvecs_grid"))
        )
        if cache_key in _F_SG_CORE_CACHE:
            return _F_SG_CORE_CACHE[cache_key]
   
    # ---------------------------------------
    # Unpack params once
    # ---------------------------------------
    Nx = int(params["Nx"])
    Ny = int(params["Ny"])
    Xg = jnp.asarray(params["Xg"])
    Yg = jnp.asarray(params["Yg"])

    if sqrt_lam is None:
        sqrt_lam = params["sqrt_lam"]   # shape (N_KL,)
    else:
        sqrt_lam = sqrt_lam
    sqrt_lam = jnp.asarray(sqrt_lam)

    Phi = jnp.asarray(params["eigvecs_grid"])  # (N_KL, Nx+1, Ny+1) or (N_KL, Nx, Ny)
    idx_of = params["idx_of"]

    xi_samples = jnp.asarray(params["xi_samples"])  # (K, N_KL)
    xi_weights = jnp.asarray(params["xi_weights"])  # (K,)
    K_samp, N_dim = xi_samples.shape
    assert N_dim == N_KL

    num_nodes = (Nx + 1) * (Ny + 1)

    # ---------------------------------------
    # Precompute p_idx once and stash as JAX array
    # ---------------------------------------
    if "p_idx" in params:
        p_idx = jnp.asarray(params["p_idx"], dtype=jnp.int32)
    else:
        p_idx_np = np.empty(N_KL, dtype=int)
        for m in range(N_KL):
            e = [0] * N_KL
            e[m] = 1
            p_idx_np[m] = idx_of[tuple(e)]
        p_idx = jnp.asarray(p_idx_np, dtype=jnp.int32)
        params["p_idx"] = np.asarray(p_idx_np, dtype=int)  # optional cache

    # ---------------------------------------
    # Choose reconstructor once
    # ---------------------------------------
    if P == N_KL + 1:
        recon = reconstruct_u_from_SG_jax_p1
    else:
        recon = reconstruct_u_from_SG_jax

    f0_jax = jnp.asarray(f0)
    f1_jax = jnp.asarray(f1)

    # ---------------------------------------
    # Build the core JAX function
    # ---------------------------------------
    def F_SG_core(Ucoeff):
        """
        Pure JAX function: Ucoeff -> F_SG(Ucoeff) with shape (P * num_nodes,)
        """
        Ucoeff = jnp.asarray(Ucoeff)

        def recon_to_F(xi_vec):
            # 1) reconstruct u(x,y; ξ)
            u = recon(Ucoeff, xi_vec, params).reshape(Nx + 1, Ny + 1)

            # 2) α(x,y; ξ) = f0 + f1 * Σ_m sqrt(λ_m) φ_m(x,y) ξ_m
            w = sqrt_lam * xi_vec              # (N_KL,)
            # Phi is now (N_KL, Nx+1, Ny+1) vertex-centred; result is already nodal
            alpha_grid = f0_jax + f1_jax * jnp.tensordot(w, Phi, axes=(0, 0))  # (Nx+1, Ny+1)

            # 3) beam source q(x,y; ξ)
            q = build_beam_q_grid_jax(
                u, Xg, Yg, alpha_grid, x, y, T_abl, params,t
            )  # (Nx+1, Ny+1)

            # 4) assemble FE RHS F(ξ)
            return assemble_F_from_qgrid_jax(q, params)  # (num_nodes,)

        # Evaluate F at all ξ samples
        F_stack = jax.vmap(recon_to_F)(xi_samples)  # (K, num_nodes)

        # First-order Hermite projection
        c0 = jnp.einsum("k,kj->j", xi_weights, F_stack)                # (num_nodes,)
        cm = jnp.einsum("k,kj,km->mj", xi_weights, F_stack, xi_samples)  # (N_KL, num_nodes)


        # Pack SG vector
        F_SG = jnp.zeros((P, num_nodes), dtype=F_stack.dtype)
        F_SG = F_SG.at[0, :].set(c0)
        F_SG = F_SG.at[p_idx, :].set(cm)

        return F_SG.reshape(-1)  # (P * num_nodes,)

    if _cacheable:
        _F_SG_CORE_CACHE[cache_key] = F_SG_core

    return F_SG_core

def clear_f_sg_cache():
    """Clear the F_SG_core cache."""
    global _F_SG_CORE_CACHE
    _F_SG_CORE_CACHE.clear()
   # print("  [F_SG cache cleared]")

import jax
import jax.numpy as jnp

# Cache for build_SG_operators
_SG_OPERATORS_CACHE = {}
def build_SG_operators(sigma, ell, frozen, theta_kappa=None, kappa_param=None):
    """Build SG operators with caching."""
   
    # ──────────────────────────────────────────────────────────────
    # FIX: Ensure theta_kappa is proper array for SmoothKappa
    # ──────────────────────────────────────────────────────────────
    if theta_kappa is None:
        # Use defaults from kappa_param
        if hasattr(kappa_param_obs, 'kappa_values'):
            # LayeredKappa: array of layer values
            theta_kappa_use = np.asarray(kappa_param_obs.kappa_values, dtype=float)
        else:
            # SmoothKappa: [kappa0, strength]
            theta_kappa_use = np.array([kappa_param_obs.kappa0, kappa_param.strength], dtype=float)
    else:
        # Convert whatever was passed to proper 1D array
        theta_kappa_use = np.atleast_1d(np.asarray(theta_kappa, dtype=float))
   
    # Validate shape for SmoothKappa
    if hasattr(kappa_param_obs, 'kappa0'):  # SmoothKappa detected
        if theta_kappa_use.size == 1:
            # Only kappa0 provided, add default strength
            theta_kappa_use = np.array([theta_kappa_use[0], kappa_param.strength], dtype=float)
        elif theta_kappa_use.size != 2:
            raise ValueError(f"SmoothKappa requires 2 parameters [kappa0, strength], got {theta_kappa_use.size}")
   
    print(f"[build_SG_operators] theta_kappa = {theta_kappa_use} (shape={theta_kappa_use.shape})")
   
    # Build cache key
    cache_key = (tuple(theta_kappa_use.ravel()), frozen)
   
    if cache_key in _SG_OPERATORS_CACHE:
        print(f"  → Cache HIT")
        return _SG_OPERATORS_CACHE[cache_key]
   
    print(f"  → Cache MISS, computing operators...")
   
    # ──────────────────────────────────────────────────────────────
    # Compute KL derivatives with PROPER array
    # ──────────────────────────────────────────────────────────────
    diff = SPDEKLDifferentiator(Nx, Ny, Lx, Ly, N_KL, kappa_param)
    res = diff.derivatives(theta_kappa_use)  # ← Now always 1D array

    eigvals_trunc = np.asarray(res.eigvals, float)     # (N_KL,)
    phi_flat      = np.asarray(res.eigvecs, float)     # (n, N_KL)
    dlambda_all   = np.asarray(res.dlambda, float)     # (N_KL, n_theta)
    dphi_all      = np.asarray(res.dphi, float)        # (n, N_KL, n_theta)
    # BUG FIX: use a LOCAL variable here to avoid overwriting the global xi_sample
    # that phi_one_step uses during the forward time-stepping. Using global xi_sample
    # here introduced randomness between FD perturbation calls.
    _xi_viz = np.random.randn(N_KL)

    coeff = np.sqrt(eigvals_trunc) * _xi_viz           # (N_KL,)
    k_fluct = phi_flat @ coeff                         # (n,)

    # reshape for plotting
    k_fluct_grid = k_fluct.reshape(Nx + 1, Ny + 1)     # vertex-centred: n == (Nx+1)*(Ny+1)
    k_values = k0 + k1 * k_fluct_grid

    plt.figure()
    plt.imshow(k_values.T, origin="lower", aspect="auto")
    plt.colorbar()
    plt.xlabel("x index")
    plt.ylabel("y index")
    plt.show()
    # sanity
    if phi_flat.ndim != 2:
        raise ValueError(f"Expected eigvecs 2D (n, N_KL), got {phi_flat.shape}")
    n, r = phi_flat.shape
    if r != N_KL:
        raise ValueError(f"N_KL mismatch: got {r}, expected {N_KL}")
    if n != (Nx + 1) * (Ny + 1):
        raise ValueError(f"Expected n=(Nx+1)*(Ny+1)={(Nx+1)*(Ny+1)}, got n={n}")

    # reshape eigenvectors to vertex grid (Nx+1, Ny+1, N_KL)
    phi_grid = phi_flat.reshape(Nx + 1, Ny + 1, N_KL)

    # choose which kappa component you want derivatives for
    i_theta = 0
    dlambda_trunc = dlambda_all                         # (N_KL,)
    dphi_grid     = dphi_all[:, :, i_theta].reshape(Nx + 1, Ny + 1, N_KL)   # (Nx+1,Ny+1,N_KL)

    kappa_params_update = {
        "sqrt_lam":        jnp.asarray(np.sqrt(eigvals_trunc)),
        "eigvecs_grid":    jnp.asarray(np.transpose(phi_grid, (2, 0, 1))),  # (N_KL,Nx,Ny)
        "f0":              float(f0),
        "f1":              float(f1),
        "eigvals_trunc":   eigvals_trunc,
        "eigvecs_reshaped": phi_grid,          # (Nx+1,Ny+1,N_KL)
    }
    if not frozen:
        params.update(kappa_params_update)

   
    phi_gradients = []
    for m in range(N_KL):
        phi = phi_grid[:, :, m]
        phi_x, phi_y = np.gradient(phi, dx, dy)
        phi_gradients.append((phi, phi_x, phi_y))

    K_kl_global = form_K_kl_global(eigvals_trunc, phi_grid, phi_gradients)
    M_kl_global = form_M_kl_global(eigvals_trunc, phi_grid, phi_gradients)

    K0_global = form_k0_global(1, 1)
    M0_global = form_M0_global(1, 1)

    K_SG_K0 = form_K_SG_K0(K0_global)
    M_SG_M0 = form_M_SG_M0(M0_global)

    K_SG_K1, M_SG_M1 = form_KM_SG_K1(K_kl_global, M_kl_global)

    _SG_OPERATORS_CACHE[cache_key] = (K_SG_K0, K_SG_K1, M_SG_M0, M_SG_M1, K_kl_global, M_kl_global, kappa_params_update)

    return K_SG_K0, K_SG_K1, M_SG_M0, M_SG_M1, K_kl_global, M_kl_global, kappa_params_update

def clear_sg_operators_cache():
    """Clear the SG operators cache."""
    global _SG_OPERATORS_CACHE
    _SG_OPERATORS_CACHE.clear()
   # print("  [SG operators cache cleared]")
# Build SG operators once for the chosen sigma, ell

#U_obs, U_history_obs, u_mean, u_variance,M_SG_M0, M_SG_M1, K_SG_K0, K_SG_K1,dK1_dell, dM1_dell, eigvals, eigvecs, deigvals = solve_nonlinear_system_fast(k0,k1,m0,m1,f0,f1,sigma, ell, xi_sample)

#U_obs, U_history_obs, u_mean, u_variance,M_SG_M0, M_SG_M1, K_SG_K0, K_SG_K1,dK1_dell, dM1_dell, eigvals, eigvecs, deigvals = solve_nonlinear_system(k0,k1,m0,m1,f0,f1,sigma, ell, xi_sample)
#T_obs = U_history_obs[-1,:num_nodes].reshape((Nx + 1, Ny + 1))
def sample_solution(Ucoeff, multi_idx, xi_sample):
    u = np.zeros(num_nodes)
    for k, alpha in enumerate(multi_idx):
        psi = eval_psi(xi_sample, alpha)
        u += psi * Ucoeff[k*num_nodes:(k+1)*num_nodes]
    return u

def build_linear_operators(k0, k1, m0, m1, M_SG_M0, M_SG_M1, K_SG_K0,K_SG_K1):
    # SG mass & stiffness for given k0,k1,m0,m1
    M_SG = m0 * M_SG_M0 + m1 * M_SG_M1
    K_SG = k0 * K_SG_K0 + k1 * K_SG_K1

    zero_vec = np.zeros(M_SG.shape[0])
    M_bc, _ = enforce_dirichlet_sym(M_SG.copy(), zero_vec, bc_idx, np.zeros_like(bc_idx, float))
    K_bc, _ = enforce_dirichlet_sym(K_SG.copy(), zero_vec, bc_idx, np.zeros_like(bc_idx, float))

    # System matrix for backward Euler in "mass form"
    A = (M_bc + dt * K_bc).tocsc()
    solve_A = factorized(A)     # sparse LU (or Cholesky for SPD)
   # solve_A = None
    return M_bc, K_bc, solve_A

# LU factorization cache for build_linear_operators_cached
_LU_CACHE = {}

"""
Replacement for build_linear_operators_cached.

WHY THE ORIGINAL IS SLOW
------------------------
The original calls scipy.sparse.linalg.factorized(A) which runs SuperLU on the
full (P * num_nodes) x (P * num_nodes) system — ~103k x 103k for P=61, Nx=Ny=40.
SuperLU fill-in on this coupled system is enormous: the stochastic coupling
(kron(G_m, K_kl[m])) creates off-diagonal blocks that SuperLU must eliminate,
producing a dense factor.

WHAT WE EXPLOIT INSTEAD
------------------------
The system matrix has the structure:

    A = I_P ⊗ A_phys  +  (stochastic coupling terms)

where A_phys = (m0*M0 + dt*k0*K0) is the single physical-space matrix
(num_nodes x num_nodes, ~1.7k x 1.7k for Nx=Ny=40).

The dominant part is block-diagonal: P identical copies of A_phys on the diagonal.
The stochastic coupling (k1/m1 terms) is a small perturbation in practice.

This suggests two cheap preconditioners, both implemented below:

1. BLOCK-DIAGONAL ILU  (recommended, default)
   Factor A_phys once with ILU(0). Apply the same factor to each of the P
   blocks independently. Cost: one ~1.7k ILU instead of one ~103k LU.
   Factor speedup vs SuperLU: typically 100-500x.

2. BLOCK-DIAGONAL EXACT SOLVE  (most accurate, use if ILU gives slow GMRES)
   Factor A_phys once with splu (exact sparse LU on the small physical matrix).
   Apply to each block. Cost: one ~1.7k LU. Still ~60x cheaper than full LU
   because the physical problem is much smaller and has no stochastic fill-in.

Both are returned as a callable solve_A(r) -> x compatible with the existing
LinearOperator wrapper in phi_one_step and adjoint_one_step.

USAGE  (drop-in replacement)
-----
Replace the existing build_linear_operators_cached call with this version.
The returned (M_bc, K_bc, solve_A) triple has the same interface as before;
solve_A is now a callable instead of None.
"""

from scipy.sparse.linalg import spilu, splu, LinearOperator
from scipy.sparse import eye as speye, kron as spkron
import numpy as np


# --------------------------------------------------------------------------- #
# Cache — keyed on the physical parameters only, not the full matrix bytes.   #
# The physical A_phys changes when k0/m0/dt change; P-scaling does not matter.#
# --------------------------------------------------------------------------- #
_LU_CACHE = {}


def _build_A_phys(k0, m0, K0_global, M0_global):
    """
    Assemble the physical-space system matrix:
        A_phys = m0 * M0_global + dt * k0 * K0_global
    with Dirichlet BCs enforced on the physical nodes only.

    This is (num_nodes x num_nodes), much smaller than the full SG system.
    """
    from scipy.sparse import csc_matrix
    A_phys = (m0 * M0_global + dt * k0 * K0_global).tocsc()

    # Enforce Dirichlet on physical indices (bphys, not the SG-expanded bc_idx).
    # We only need the diagonal=1 / off-diagonal=0 treatment on the small matrix.
    A_phys = A_phys.tolil()
    for i in bphys:                    # bphys = physical Dirichlet node list
        A_phys.rows[i] = [i]
        A_phys.data[i] = [1.0]
        col = A_phys.getcol(i).nonzero()[0]
        for r in col:
            if r != i:
                A_phys[r, i] = 0.0
    return csc_matrix(A_phys)


def build_linear_operators_cached(
    k0, k1, m0, m1,
    M_SG_M0, M_SG_M1,
    K_SG_K0, K_SG_K1,
    precond="block_ilu",     # "block_ilu" | "block_lu" | "none"
    drop_tol=1e-4,           # ILU drop tolerance (only used for block_ilu)
):
    """
    Build SG linear operators and a cheap block-diagonal preconditioner.

    Parameters
    ----------
    precond : str
        "block_ilu"  — ILU(0) on the physical block (recommended)
        "block_lu"   — exact sparse LU on the physical block (more accurate,
                       still ~60x cheaper than full SuperLU on the SG system)
        "none"       — returns solve_A=None (original behaviour)
    drop_tol : float
        ILU drop tolerance passed to spilu. Smaller = more accurate but slower.
        1e-4 is a good default; try 1e-3 if build is still too slow.

    Returns
    -------
    M_bc, K_bc : CSR sparse matrices (full SG size, with BCs)
    solve_A    : callable r -> x  (preconditioner application)
                 or None if precond="none"
    """
    global _LU_CACHE

    cache_key = (k0, k1, m0, m1, dt, precond, drop_tol,
                 M_SG_M0.data.tobytes()[:100],
                 K_SG_K0.data.tobytes()[:100])

    if cache_key in _LU_CACHE:
        print("  [Using cached block preconditioner]")
        return _LU_CACHE[cache_key]

    print(f"  [Building block-diagonal preconditioner ({precond})...]")

    # ------------------------------------------------------------------ #
    # 1. Full SG matrices (needed for M_bc, K_bc which phi_one_step uses  #
    #    to form the RHS — they are NOT used for the preconditioner)       #
    # ------------------------------------------------------------------ #
    M_SG = m0 * M_SG_M0 + m1 * M_SG_M1
    K_SG = k0 * K_SG_K0 + k1 * K_SG_K1

    zero_vec = np.zeros(M_SG.shape[0])
    M_bc, _ = enforce_dirichlet_sym(
        M_SG.copy(), zero_vec, bc_idx, np.zeros_like(bc_idx, float)
    )
    K_bc, _ = enforce_dirichlet_sym(
        K_SG.copy(), zero_vec, bc_idx, np.zeros_like(bc_idx, float)
    )

    # ------------------------------------------------------------------ #
    # 2. Build the preconditioner on the PHYSICAL block only              #
    # ------------------------------------------------------------------ #
    if precond == "none":
        solve_A = None

    else:
        # Extract the physical-space base matrices from the SG Kronecker products.
        # K_SG_K0 = I_P ⊗ K0_global  →  K0_global is the top-left num_nodes block.
        # Similarly for M_SG_M0.
        n = num_nodes
        K0_phys = K_SG_K0[:n, :n]   # CSR slice: the physical stiffness block
        M0_phys = M_SG_M0[:n, :n]   # CSR slice: the physical mass block

        # Physical system matrix (no stochastic coupling, no k1/m1 perturbation).
        # This is the dominant diagonal block of A = M_bc + dt*K_bc.
        A_phys = (m0 * M0_phys + dt * k0 * K0_phys).tocsc()

        # Apply Dirichlet on the physical boundary nodes
        A_phys = A_phys.tolil()
        for i in bphys:
            A_phys.rows[i] = [i]
            A_phys.data[i] = [1.0]
        A_phys = A_phys.tocsc()
        # Zero out the columns too (symmetric enforcement)
        A_phys = A_phys.T.tolil()
        for i in bphys:
            A_phys.rows[i] = [i]
            A_phys.data[i] = [1.0]
        A_phys = A_phys.T.tocsc()

        if precond == "block_ilu":
            # ILU(0) factorisation of the physical block.
            # spilu is much faster than splu and produces a good approximate inverse
            # for elliptic problems. drop_tol controls accuracy vs speed.
            factor = spilu(
                A_phys,
                drop_tol=drop_tol,
                fill_factor=10,        # allow up to 10x fill vs original nnz
                diag_pivot_thresh=0.1, # partial pivoting for stability
            )
            print(f"    ILU factor nnz: {factor.nnz if hasattr(factor,'nnz') else 'n/a'}")

        elif precond == "block_lu":
            # Exact sparse LU on the small physical block.
            # More accurate than ILU; num_nodes ~1.7k so this is fast.
            factor = splu(A_phys)
            print(f"    splu complete on {n}x{n} physical block")

        else:
            raise ValueError(f"Unknown precond='{precond}'")

        # ------------------------------------------------------------------ #
        # 3. Wrap as a block-diagonal apply: same factor applied to each of  #
        #    the P blocks of the SG residual vector.                          #
        #                                                                     #
        #    r has shape (P * num_nodes,).  Reshape to (P, num_nodes),        #
        #    apply factor.solve to each row, reshape back.                    #
        #                                                                     #
        #    This is exact for the I_P ⊗ A_phys part and approximate for     #
        #    the k1/m1 coupling — exactly what GMRES needs to converge fast.  #
        # ------------------------------------------------------------------ #
        def solve_A(r):
            R = np.asarray(r, dtype=float).reshape(P, num_nodes)
            X = np.empty_like(R)
            for p in range(P):
                X[p] = factor.solve(R[p])
            return X.ravel()

    result = (M_bc, K_bc, solve_A)
    _LU_CACHE[cache_key] = result
    return result


def clear_lu_cache():
    """Clear the preconditioner cache (call when mesh, dt, or kappa changes)."""
    global _LU_CACHE
    _LU_CACHE.clear()
    print("  [LU/ILU cache cleared]")

def clear_lu_cache():
    """Clear the LU factorization cache (call when mesh or dt changes)."""
    global _LU_CACHE
    _LU_CACHE.clear()
    print("  [LU cache cleared]")
import numpy as np
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# 1) Pick surface nodes from a time-varying height y = h(x)
# ------------------------------------------------------------
def build_surface_nodes_from_height(x, y_grid, h):
    x = np.asarray(x)
    y_grid = np.asarray(y_grid)
    h = np.asarray(h)

    Nx = x.size - 1
    Ny = y_grid.size - 1

    # For each x_i, choose the y_j minimizing |y_j - h_i|
    j_star = np.abs(y_grid[None, :] - h[:, None]).argmin(axis=1)
    node_idx = np.arange(Nx+1) * (Ny+1) + j_star
    return node_idx.astype(int), x.astype(float), y_grid[j_star].astype(float)


import numpy as np
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt

# ---------- Operator (supports sensor above/below surface) ----------
def build_radiometer_operator(x_surf, y_surf, node_idx_surf, num_nodes,
                              center, angle_deg, aggregate=False, weights='uniform',
                              direction='auto'):

    xc, yc = center
    x_surf = np.asarray(x_surf, float)
    y_surf = np.asarray(y_surf, float)
    node_idx_surf = np.asarray(node_idx_surf, int)

    if direction == 'auto':
        direction = 'down' if np.nanmedian(y_surf) < yc else 'up'

    dx = x_surf - xc
    dy = y_surf - yc

    # Points must lie "in front of" the sensor in the view direction
    if direction == 'down':
        in_front = dy < 0.0
        denom = np.maximum(np.abs(dy), 1e-16)   # angle to -y axis
    else:  # 'up'
        in_front = dy > 0.0
        denom = np.maximum(np.abs(dy), 1e-16)   # angle to +y axis

    theta = np.arctan2(np.abs(dx), denom)
    fov_mask = (theta <= np.deg2rad(angle_deg)) & in_front

    sel_idx = node_idx_surf[fov_mask]
    dx_sel = dx[fov_mask]
    dy_sel = dy[fov_mask]

    if sel_idx.size == 0:
        return csr_matrix((0, num_nodes)), fov_mask, None

    # Weights across visible points
    if weights == 'uniform':
        w = np.ones(sel_idx.size, float)
    elif weights == 'cosine':
        r = np.sqrt(dx_sel**2 + dy_sel**2)
        # cosine w.r.t. view axis (down: -y; up: +y)
        cosang = ( -dy_sel if direction == 'down' else dy_sel ) / np.maximum(r, 1e-16)
        w = np.clip(cosang, 0.0, None)
    elif weights == 'inverse_square':
        r2 = dx_sel**2 + dy_sel**2
        w = 1.0 / np.maximum(r2, 1e-16)
    else:
        w = np.ones(sel_idx.size, float)

    if aggregate:
        w_norm = w / np.maximum(w.sum(), 1e-16)
        row = np.zeros(num_nodes)
        row[sel_idx] = w_norm
        H = csr_matrix(row.reshape(1, -1))
        return H, fov_mask, w_norm
    else:
        rows = np.arange(sel_idx.size)
        H = csr_matrix((np.ones(sel_idx.size), (rows, sel_idx)),
                       shape=(sel_idx.size, num_nodes))
        return H, fov_mask, None


# ---------- Plot (draw wedge in the correct direction) ----------
def plot_radiometer_p(x, Lx, Ly, x_surf, y_surf, H, selected_mask,
                    center, angle_deg, title=None, show_spy=True,
                    direction='auto'):
    x = np.asarray(x); x_surf = np.asarray(x_surf); y_surf = np.asarray(y_surf)
    xc, yc = center
    if direction == 'auto':
        direction = 'down' if np.nanmedian(y_surf) < yc else 'up'

    # A) Geometry
    plt.figure(figsize=(6,5))
    # domain rectangle
    plt.plot([0, Lx, Lx, 0, 0], [0, 0, Ly, Ly, 0])
    # surface polyline + selected
    plt.plot(x_surf, y_surf)
    if selected_mask is not None and np.any(selected_mask):
        plt.scatter(x_surf[selected_mask], y_surf[selected_mask], s=30)
    # sensor
    plt.scatter([xc], [yc])

    # FOV wedge edges (extend to a convenient y-limit)
    if direction == 'down':
        # extend to min surface y or bottom boundary
        y_end = min(np.nanmin(y_surf), 0.0)
        dy_extent = yc - y_end
        dx_edge = np.tan(np.deg2rad(angle_deg)) * max(dy_extent, 0.0)
        x_left,  x_right = xc - dx_edge, xc + dx_edge
        plt.plot([xc, x_left],  [yc, yc - dy_extent])
        plt.plot([xc, x_right], [yc, yc - dy_extent])
    else:  # 'up'
        y_end = max(np.nanmax(y_surf), Ly)
        dy_extent = y_end - yc
        dx_edge = np.tan(np.deg2rad(angle_deg)) * max(dy_extent, 0.0)
        x_left,  x_right = xc - dx_edge, xc + dx_edge
        plt.plot([xc, x_left],  [yc, yc + dy_extent])
        plt.plot([xc, x_right], [yc, yc + dy_extent])

    plt.title(title or f"Radiometer (center={center}, half-angle={angle_deg}°, dir={direction})")
    plt.xlabel("x"); plt.ylabel("y")
    plt.xlim(-0.02*Lx, 1.02*Lx); plt.ylim(-0.02*Ly, 1.5*Ly)
    plt.tight_layout()





# --- helpers from previous answer (unchanged) ---
def _ray_dir(angle_deg, direction):
    a = np.deg2rad(angle_deg)
    if direction == 'down':
        return (-np.sin(a), -np.cos(a)), (+np.sin(a), -np.cos(a))
    else:
        return (-np.sin(a), +np.cos(a)), (+np.sin(a), +np.cos(a))

def _ray_segment_intersection(p, r, q1, q2):
    r = np.asarray(r, float); s = np.asarray(q2, float) - np.asarray(q1, float)
    qp = np.asarray(q1, float) - np.asarray(p, float)
    cross = lambda a,b: a[0]*b[1] - a[1]*b[0]
    denom = cross(r, s)
    if np.isclose(denom, 0.0): return None, None
    t = cross(qp, s) / denom     # along ray
    u = cross(qp, r) / denom     # along segment
    if t >= 0.0 and 0.0 <= u <= 1.0: return t, u
    return None, None

def _first_hit_on_polyline(p, r, xs, ys):
    best_t = np.inf; hit = (None, None); which_seg = None; u_best = None
    for i in range(len(xs)-1):
        (t,u) = _ray_segment_intersection(p, r, (xs[i],ys[i]), (xs[i+1],ys[i+1]))
        if t is not None and t < best_t:
            best_t = t; hit = (p[0]+t*r[0], p[1]+t*r[1]); which_seg=i; u_best=u
    return hit, which_seg, u_best

def _interp_on_seg(xs, ys, i, u):
    return (1-u)*xs[i] + u*xs[i+1], (1-u)*ys[i] + u*ys[i+1]

# --- the “cool” plotter ---
def plot_radiometer_cool(
    x, Lx, Ly, x_surf, y_surf, H, selected_mask,
    center, angle_deg, title=None, show_spy=False,
    direction='auto', shade_domain=True, shade_alpha=0.12,
    inset=False, inset_zoom=3.0, savepath=None
):
    x = np.asarray(x); xs = np.asarray(x_surf); ys = np.asarray(y_surf)
    xc, yc = center
    if direction == 'auto':
        direction = 'down' if np.nanmedian(ys) < yc else 'up'

    fig, ax = plt.subplots(figsize=(7.2, 5.6))

    # 1) domain outline
    ax.plot([0,Lx,Lx,0,0], [0,0,Ly,Ly,0], lw=1.2)

    # 2) shade material under the surface
    if shade_domain:
        ax.fill_between(xs, 0.0, ys, alpha=shade_alpha, linewidth=0)

    # 3) surface & selected points
    ax.plot(xs, ys, lw=1.8)
    if selected_mask is not None and np.any(selected_mask):
        # visible segment highlighted
        vis_x = xs[selected_mask]; vis_y = ys[selected_mask]
        ax.plot(vis_x, vis_y, lw=3.0, ls=(0,(5,2)))
        ax.scatter(vis_x, vis_y, s=18, zorder=3)

    # 4) sensor
    ax.scatter([xc],[yc], s=50, zorder=4)
    ax.annotate("sensor", (xc, yc), xytext=(8, 10), textcoords='offset points')

    # 5) FOV edges (clipped to surface) + wedge fill under surface
    dL, dR = _ray_dir(angle_deg, direction); p=(xc,yc)
    (hitL, iL, uL) = _first_hit_on_polyline(p, dL, xs, ys)
    (hitR, iR, uR) = _first_hit_on_polyline(p, dR, xs, ys)

    # Build polygon of wedge clipped by surface between the two hits
    if all(h is not None for h in hitL) and all(h is not None for h in hitR):
        # indices spanning the surface between both hits
        # figure out left/right ordering along x
        xL, yL = hitL; xR, yR = hitR
        # map hits to exact points on polyline (use seg + u for robustness)
        xL, yL = _interp_on_seg(xs, ys, iL, uL)
        xR, yR = _interp_on_seg(xs, ys, iR, uR)

        # collect surface points between the two hits
        if xL <= xR:
            i_start, i_end = iL, iR
            surf_poly_x = [xL] + xs[i_start+1:i_end+1].tolist() + [xR]
            surf_poly_y = [yL] + ys[i_start+1:i_end+1].tolist() + [yR]
        else:
            i_start, i_end = iR, iL
            surf_poly_x = [xR] + xs[i_start+1:i_end+1].tolist() + [xL]
            surf_poly_y = [yR] + ys[i_start+1:i_end+1].tolist() + [yL]

        # draw clipped edge rays
        ax.plot([xc, xL], [yc, yL], lw=1.5)
        ax.plot([xc, xR], [yc, yR], lw=1.5)

        # fill wedge polygon (sensor -> left hit -> along surface -> right hit -> sensor)
        wedge_x = [xc, xL] + surf_poly_x[1:-1] + [xR, xc]
        wedge_y = [yc, yL] + surf_poly_y[1:-1] + [yR, yc]
        ax.fill(wedge_x, wedge_y, alpha=0.10, hatch='///', linewidth=0)

        # angle arc label
        arc_r = 0.06 * max(Lx, Ly)
        # draw two small arcs (one per half-angle)
        a = angle_deg
        if direction == 'down':
            # arcs centered at sensor, around -y axis
            arc1 = Arc((xc,yc), 2*arc_r, 2*arc_r, angle=0, theta1=270-a, theta2=270)
            arc2 = Arc((xc,yc), 2*arc_r, 2*arc_r, angle=0, theta1=270,   theta2=270+a)
        else:
            arc1 = Arc((xc,yc), 2*arc_r, 2*arc_r, angle=0, theta1=90-a, theta2=90)
            arc2 = Arc((xc,yc), 2*arc_r, 2*arc_r, angle=0, theta1=90,   theta2=90+a)
        for arc in (arc1, arc2): ax.add_patch(arc)
        ax.annotate(f"{angle_deg}°", (xc, yc), xytext=(10, -12),
                    textcoords='offset points')

    # 6) axes polish
    ax.set_title(title or f"Radiometer (dir={direction}, half-angle={angle_deg}°)")
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.set_xlim(-0.02*Lx, 1.02*Lx); ax.set_ylim(-0.02*Ly, 1.5*Ly)
    ax.set_aspect('equal', adjustable='box')
    ax.grid(alpha=0.25)
    fig.tight_layout()


    # 8) optional spy figure
    if show_spy:
        fig2, ax2 = plt.subplots(figsize=(6, 2.8))
        ax2.spy(H, markersize=2)
        ax2.set_title("Sparse pattern of H"); fig2.tight_layout()

    if savepath:
        fig.savefig(savepath, dpi=200)

def visualize_radiometer_at_timestep(
    T_mean_nodes, T_var_nodes, T_abl,
    center=(Lx/2, Ly+0.005), angle_deg=20.0,
    aggregate=True, weights='cosine',
    title_prefix="Radiometer"
):
    # --- ensure 2D (Nx+1, Ny+1) for vmap-over-x ---
    Tm = np.asarray(T_mean_nodes).reshape((Nx+1, Ny+1), order='C')
    Tv = np.asarray(T_var_nodes).reshape((Nx+1, Ny+1), order='C')

    # call your extractor with 2D arrays (JAX will accept numpy inputs)
    x_nodes_out, h_mean, _, _ = extract_interface_h_from_SG_jax(
        jax.numpy.asarray(Tm),
        jax.numpy.asarray(Tv),
        T_abl,
        jax.numpy.asarray(x),
        jax.numpy.asarray(y)
    )

    # back to numpy
    x_nodes_out = np.asarray(x_nodes_out)
    h_mean      = np.asarray(h_mean)

    # Snap to grid + build H + plot (unchanged)
    surf_idx, x_surf, y_surf = build_surface_nodes_from_height(x_nodes_out, y, h_mean)
    H, seen_mask, w_norm = build_radiometer_operator(
        x_surf, y_surf, surf_idx, num_nodes,
        center=center, angle_deg=angle_deg,
        aggregate=aggregate, weights=weights
    )
    #plot_radiometer(x, Lx, Ly, x_surf, y_surf, H, seen_mask,
    #                center=center, angle_deg=angle_deg,
    #                title=f"{title_prefix}", show_spy=True)
    plot_radiometer_cool(
        x, Lx, Ly, x_surf, y_surf, H, seen_mask,
        center=center, angle_deg=angle_deg,
        direction='down',           # sensor above surface
        title=f"Radiometer",
        show_spy=False,
        inset=False,
        savepath=None               # or "radiometer_t{:03d}.png".format(t)
    )


    return H, seen_mask, w_norm, x_surf, y_surf

angle_deg = 80.0
weights = 'cosine'      # 'uniform' | 'cosine' | 'inverse_square'
aggregate = True
import jax
import jax.numpy as jnp
import jax
import jax.numpy as jnp

def get_visible_surface_points_jax(
    x_surf, y_surf,
    center=(0.05, 0.105),
    angle_deg=20.0,
    fov_smooth=2.0,
    direction='down',
    weights='cosine',
    min_inside=0.5,
    min_weight=0.0
):

    xc, yc = center
    dx = x_surf - xc
    dy = y_surf - yc

    # Viewing geometry
    if direction == 'down':
        theta = jnp.arctan2(jnp.abs(dx), -dy + 1e-12)
        along = jnp.maximum((-dy) / (jnp.sqrt(dx**2 + dy**2) + 1e-12), 0.0)
    else:
        theta = jnp.arctan2(jnp.abs(dx),  dy + 1e-12)
        along = jnp.maximum(( dy) / (jnp.sqrt(dx**2 + dy**2) + 1e-12), 0.0)

    theta_max = jnp.deg2rad(angle_deg)
    smooth = jnp.deg2rad(fov_smooth)

    # Smooth FOV mask
    inside = jax.nn.sigmoid((theta_max - theta) / smooth)

    # Raw weighting
    if weights == 'uniform':
        w_raw = inside
    elif weights == 'cosine':
        w_raw = inside * along
    elif weights == 'inverse_square':
        r2 = dx**2 + dy**2
        w_raw = inside / (r2 + 1e-12)
    else:
        w_raw = inside

    # Visible subset
    seen_mask = (inside >= min_inside) & (w_raw > min_weight)
    idx_vis = jnp.nonzero(seen_mask, size=seen_mask.size, fill_value=-1)[0]
    idx_vis = idx_vis[idx_vis >= 0]

    # Extract coordinates and normalize weights
    x_vis = x_surf[idx_vis]
    y_vis = y_surf[idx_vis]
    w_vis = w_raw[idx_vis]
    w_vis = w_vis / (jnp.sum(w_vis) + 1e-12)

    return idx_vis, x_vis, y_vis, w_vis, seen_mask


from scipy.sparse.linalg import LinearOperator, gmres
def build_unit_element_stiffness(hx, hy):
    g = 1/np.sqrt(3); pts = [(-g,-g),(g,-g),(-g,g),(g,g)]
    Ke = np.zeros((4,4))
    detJ = (hx*hy)/4.0; invJx = 2.0/hx; invJy = 2.0/hy
    for xi,eta in pts:
        dN_dxi  = np.array([-(1-eta), (1-eta), -(1+eta), (1+eta)])*0.25
        dN_deta = np.array([-(1-xi), -(1+xi), (1-xi), (1+xi)])*0.25
        dN_dx, dN_dy = invJx*dN_dxi, invJy*dN_deta
        B = np.vstack([dN_dx, dN_dy])
        Ke += (B.T @ B) * detJ
    return 0.5*(Ke+Ke.T)
def assemble_from_element_weights_fast(k_elem, Nx, Ny, hx, hy):
    Ke_unit = build_unit_element_stiffness(hx, hy)
    ii, jj = np.meshgrid(np.arange(Nx), np.arange(Ny), indexing='ij')
    ii, jj = ii.ravel(), jj.ravel()
    n0 = ii*(Ny+1)+jj; n1=n0+1; n2=(ii+1)*(Ny+1)+jj; n3=n2+1
    node_idx = np.stack([n0,n1,n2,n3], axis=1)  # (N_elem, 4)
    # Ke per element: (N_elem, 4, 4) = Ke_unit[None] * k_elem.ravel()[:,None,None]
    Ke = Ke_unit[None] * k_elem.ravel()[:, None, None]
    rows = np.repeat(node_idx[:, :, None], 4, axis=2).ravel()
    cols = np.repeat(node_idx[:, None, :], 4, axis=1).ravel()
    data = Ke.ravel()
    K = coo_matrix((data, (rows, cols)), shape=((Nx+1)*(Ny+1),)*2).tocsr()
    return 0.5*(K+K.T)
def assemble_from_element_weights(k_elem, Nx, Ny, hx, hy):
    num_nodes = (Nx+1)*(Ny+1)
    Ke_unit = build_unit_element_stiffness(hx, hy)
    rows = np.empty(Nx*Ny*16, dtype=np.int64)
    cols = np.empty(Nx*Ny*16, dtype=np.int64)
    data = np.empty(Nx*Ny*16, dtype=np.float64)
    nid = lambda i,j: i*(Ny+1) + j
    t = 0
    for i in range(Nx):
        for j in range(Ny):
            nodes = [nid(i,j), nid(i+1,j), nid(i,j+1), nid(i+1,j+1)]
            Ke = Ke_unit * float(k_elem[i,j])
            for a,I in enumerate(nodes):
                base = t + 4*a
                rows[base:base+4] = I
                cols[base:base+4] = nodes
                data[base:base+4] = Ke[a,:]
            t += 16
    K = coo_matrix((data,(rows,cols)), shape=(num_nodes,num_nodes)).tocsr()
    return 0.5*(K+K.T)
def apply_I_kron_K0u(K0u, v):
    V = v.reshape(P, num_nodes)
    Y = np.empty_like(V)
    for p in range(P):
        Y[p] = K0u @ V[p]
    return Y.reshape(-1)
def build_unit_element_mass(hx, hy):
    """
    2D bilinear Q1 element mass matrix with k=1 using 2x2 Gauss quadrature.
    """
    g = 1/np.sqrt(3)
    pts = [(-g, -g), (g, -g), (-g, g), (g, g)]
    Me = np.zeros((4, 4), dtype=float)

    detJ = (hx * hy) / 4.0
    for xi, eta in pts:
        N = 0.25 * np.array([
            (1 - xi) * (1 - eta),
            (1 + xi) * (1 - eta),
            (1 - xi) * (1 + eta),
            (1 + xi) * (1 + eta),
        ])
        Me += np.outer(N, N) * detJ

    return 0.5 * (Me + Me.T)


def assemble_mass_from_element_weights(m_elem, Nx, Ny, hx, hy):
    """
    Assemble a global mass matrix whose element (i,j) uses scalar weight m_elem[i,j].
    """
    num_nodes = (Nx + 1) * (Ny + 1)
    Me_unit = build_unit_element_mass(hx, hy)

    rows = np.empty(Nx * Ny * 16, dtype=np.int64)
    cols = np.empty(Nx * Ny * 16, dtype=np.int64)
    data = np.empty(Nx * Ny * 16, dtype=np.float64)

    nid = lambda i, j: i * (Ny + 1) + j  # node index
    t = 0
    for i in range(Nx):
        for j in range(Ny):
            nodes = [nid(i, j), nid(i+1, j), nid(i, j+1), nid(i+1, j+1)]
            Me = Me_unit * float(m_elem[i, j])
            for a, I in enumerate(nodes):
                base = t + 4 * a
                rows[base:base+4] = I
                cols[base:base+4] = nodes
                data[base:base+4] = Me[a, :]
            t += 16

    M = coo_matrix((data, (rows, cols)), shape=(num_nodes, num_nodes)).tocsr()
    return 0.5 * (M + M.T)

def apply_mass(M0u, v,m1,M_SG_M1):
    # mean part: I ⊗ M0u
    y = apply_I_kron_M0u(M0u, v)
    # stochastic part (linear in m1) stays as before:
    y += m1 * (M_SG_M1 @ v)
    return y
def build_unit_element_mass(hx, hy):
    """
    2D bilinear Q1 element mass matrix with k=1 using 2x2 Gauss quadrature.
    """
    g = 1/np.sqrt(3)
    pts = [(-g, -g), (g, -g), (-g, g), (g, g)]
    Me = np.zeros((4, 4), dtype=float)

    detJ = (hx * hy) / 4.0
    for xi, eta in pts:
        N = 0.25 * np.array([
            (1 - xi) * (1 - eta),
            (1 + xi) * (1 - eta),
            (1 - xi) * (1 + eta),
            (1 + xi) * (1 + eta),
        ])
        Me += np.outer(N, N) * detJ

    return 0.5 * (Me + Me.T)


def assemble_mass_from_element_weights(m_elem, Nx, Ny, hx, hy):
    """
    Assemble a global mass matrix whose element (i,j) uses scalar weight m_elem[i,j].
    """
    num_nodes = (Nx + 1) * (Ny + 1)
    Me_unit = build_unit_element_mass(hx, hy)

    rows = np.empty(Nx * Ny * 16, dtype=np.int64)
    cols = np.empty(Nx * Ny * 16, dtype=np.int64)
    data = np.empty(Nx * Ny * 16, dtype=np.float64)

    nid = lambda i, j: i * (Ny + 1) + j  # node index
    t = 0
    for i in range(Nx):
        for j in range(Ny):
            nodes = [nid(i, j), nid(i+1, j), nid(i, j+1), nid(i+1, j+1)]
            Me = Me_unit * float(m_elem[i, j])
            for a, I in enumerate(nodes):
                base = t + 4 * a
                rows[base:base+4] = I
                cols[base:base+4] = nodes
                data[base:base+4] = Me[a, :]
            t += 16

    M = coo_matrix((data, (rows, cols)), shape=(num_nodes, num_nodes)).tocsr()
    return 0.5 * (M + M.T)




def apply_I_kron_M0u(M0u, v):
    """
    Block-diagonal SG application: (I_P ⊗ M0u) v.
    """
    V = v.reshape(P, num_nodes)
    Y = np.empty_like(V)
    for p in range(P):
        Y[p] = M0u @ V[p]
    return Y.reshape(-1)

from scipy.sparse.linalg import LinearOperator, gmres
import numpy as np
import time

def _clear_all_kappa_caches():
    """
    Clear EVERY cache that depends on theta_kappa.

    When kappa changes the eigenvectors change, which invalidates:
      - _SG_OPERATORS_CACHE  (K_kl / M_kl built from eigvecs)
      - _K_mode_base / _M_mode_base  (mode stiffness/mass bases)
      - _F_mean_base / _F_mode_base  (forcing projections ∫f φ_m dΩ)
      - _LU_CACHE            (preconditioner factored from SG matrices)
      - _F_SG_CORE_CACHE     (jitted nonlinear source closures)
    """
    clear_f_sg_cache()
    clear_sg_operators_cache()
    clear_lu_cache()
    global _K_mode_base, _M_mode_base
    _K_mode_base = None
    _M_mode_base = None
    global _F_mean_base, _F_mode_base, _F_cache_t
    _F_mean_base = None
    _F_mode_base = None
    _F_cache_t   = None


def compute_J_perturbed(U0, U_target, solid_p, melt_p, tk_p):
    #_clear_all_kappa_caches()
    clear_f_sg_cache()
    res = run_forward(U0, solid_p, melt_p, ell, theta_kappa=tk_p, frozen=False)
    U_p = res[0][-1]
    return 0.5 * np.sum((U_p - U_target)**2)


def forcing_param_grads_numpy(Ucoeff, mu, f0, f1,t,
                              K_SG_K0, K_SG_K1, M_SG_M0, M_SG_M1):
    # SG matrices are no longer needed here; keep signature for compatibility.
    U_jax  = jnp.asarray(Ucoeff)
    mu_jax = jnp.asarray(mu)

    g0, g1 = forcing_param_grads_jax(U_jax, mu_jax, f0, f1,t)
    return float(g0), float(g1)
def apply_K1_vjp_k1(self, v, grad_output):
    """
    Compute the vector-Jacobian product: grad_output^T @ d(apply_K1)/d(k1)
    Returns gradient w.r.t. k1 (nodal values).
   
    Args:
        v: input vector (P * num_nodes,)
        grad_output: gradient from loss w.r.t. K1@v (P * num_nodes,)
   
    Returns:
        grad_k1: gradient w.r.t. k1 nodal weights (num_nodes,)
    """
    V = v.reshape(self.P, self.num_nodes)
    G = grad_output.reshape(self.P, self.num_nodes)
   
    w = self.k1_sqrt
    # d(sqrt(k1))/d(k1) = 1/(2*sqrt(k1))
    dw_dk1 = np.where(w > 1e-15, 0.5 / w, 0.0)
   
    V_scaled = V * w[None, :]  # w * v for each mode
    grad_w = np.zeros(self.num_nodes)
   
    for m in range(self.N_KL):
        gd = self.G_data[m]
        if len(gd['p']) == 0:
            continue
       
        K_m = self.K_kl[m]
       
        # Precompute for efficiency
        unique_q = np.unique(gd['q'])
        unique_p = np.unique(gd['p'])
       
        # K_m @ (w * V[q]) for each unique q
        Kv_cache = {q: K_m @ V_scaled[q] for q in unique_q}
       
        for i, (p, q, g_val) in enumerate(zip(gd['p'], gd['q'], gd['g'])):
            # Term 1: derivative of outer w: delta_w * (K_m @ (w * v_q))
            # Contribution: g_val * G[p] * (K_m @ (w * v_q))
            grad_w += g_val * G[p] * Kv_cache[q]
           
            # Term 2: derivative of inner w: w * (K_m @ (delta_w * v_q))
            # Via transpose: g_val * (K_m.T @ (w * G[p])) * V[q]
            grad_w += g_val * (K_m.T @ (w * G[p])) * V[q]
   
    # Chain rule: grad_k1 = grad_w * dw/dk1
    grad_k1 = grad_w * dw_dk1
   
    return grad_k1


def apply_M1_vjp_m1(self, v, grad_output):
    """
    Compute the vector-Jacobian product: grad_output^T @ d(apply_M1)/d(m1)
    Returns gradient w.r.t. m1 (nodal values).
    """
    V = v.reshape(self.P, self.num_nodes)
    G = grad_output.reshape(self.P, self.num_nodes)
   
    w = self.m1_sqrt
    dw_dm1 = np.where(w > 1e-15, 0.5 / w, 0.0)
   
    V_scaled = V * w[None, :]
    grad_w = np.zeros(self.num_nodes)
   
    for m in range(self.N_KL):
        gd = self.G_data[m]
        if len(gd['p']) == 0:
            continue
       
        M_m = self.M_kl[m]
       
        unique_q = np.unique(gd['q'])
        Mv_cache = {q: M_m @ V_scaled[q] for q in unique_q}
       
        for i, (p, q, g_val) in enumerate(zip(gd['p'], gd['q'], gd['g'])):
            grad_w += g_val * G[p] * Mv_cache[q]
            grad_w += g_val * (M_m.T @ (w * G[p])) * V[q]
   
    grad_m1 = grad_w * dw_dm1
   
    return grad_m1

@jax.jit
def forcing_param_grads_jax(Ucoeff_jax, mu_jax, f0, f1,t):
    """
    Returns (mu^T dF/df0, mu^T dF/df1) for the current Ucoeff and mu.
    """
    def scalar_obj(f0_local, f1_local):
        # Build F_SG_core with the LOCAL f0, f1 so JAX can trace through them
        F_SG_core = make_F_SG_core(params, f0_local, f1_local, t)
        F_SG = F_SG_core(Ucoeff_jax)
        return jnp.vdot(mu_jax, F_SG)  # scalar
    #clear_f_sg_cache()
    g_f0, g_f1 = jax.grad(scalar_obj, argnums=(0, 1))(f0, f1)
    return g_f0, g_f1
@jax.jit
def forcing_param_grads_spatial_jax(Ucoeff_jax, mu_jax, f0_nodal_jax, f1, t):
    """
    Returns (g_f0_nodal, g_f1):
      g_f0_nodal — d(mu^T F)/d(f0_nodal), same shape as f0_nodal_jax (Nx+1, Ny+1)
      g_f1       — d(mu^T F)/d(f1), scalar

    Uses the full spatial f0_nodal field, matching the forward solve which passes
    a phase-blended nodal f0 array rather than a scalar mean.
    """
    def scalar_obj(f0_loc, f1_loc):
        F_SG_core = make_F_SG_core(params, f0_loc, f1_loc, t)
        F_SG = F_SG_core(Ucoeff_jax)
        return jnp.vdot(mu_jax, F_SG)
    return jax.grad(scalar_obj, argnums=(0, 1))(f0_nodal_jax, jnp.asarray(f1))

def adjoint_grad_all_phase(
    U_history, Mu_history, solid_prop, melt_prop,
    K_SG_K1, M_SG_M1,
    forcing_param_grads_numpy, spatial_op=None,
    freeze_phase=False,
    vap_prop=None,
):
    """
    Gradients for solid/melt/vapour parameters in the new phase formulation.

    KEY FIX — correct three-phase sensitivity weights
    --------------------------------------------------
    The forward blending for every property uses the additive-correction form:

        prop_eff = prop_s + (prop_m - prop_s)*S + (prop_v - prop_m)*V
                 = prop_s*(1-S)  +  prop_m*(S-V)  +  prop_v*V

    So the sensitivities are:
        d(prop_eff)/d(prop_s) = (1 - S)
        d(prop_eff)/d(prop_m) = (S - V)      ← was wrongly S*(1-V)
        d(prop_eff)/d(prop_v) = V

    w_m = S - V can be negative near the vap front where V > S; that is
    mathematically correct (vapour is displacing melt, so k0_m has less
    influence there) and is the primary source of the 10-30 % discrepancy
    in the melt temperature range.

    The same weights are reused for K0, M0, K1, M1 because all four
    properties share the same blending formula.
    """
    if vap_prop is None:
        vap_prop = melt_prop

    g = dict(
        k0_s=0.0, k0_m=0.0,
        k1_s=0.0, k1_m=0.0,
        m0_s=0.0, m0_m=0.0,
        m1_s=0.0, m1_m=0.0,
        f0_s=0.0, f0_m=0.0,
        f1_s=0.0, f1_m=0.0,
        k0_v=0.0, k1_v=0.0,
        m0_v=0.0, m1_v=0.0,
        f0_v=0.0, f1_v=0.0,
        rho_vap0 = 0.0, rho_vap1 = 0.0
    )

    if freeze_phase:
        U_modes0 = U_history[1].reshape(P, num_nodes)
        Tmean0 = U_modes0[0].reshape(Nx + 1, Ny + 1)
        _, S_elem_frozen, _ = melt_fraction_from_Tmean(Tmean0)

    for n in range(time_steps - 1):
        u_np1 = U_history[n+1].copy()
        u_n   = U_history[n].copy()
        mu_n  = Mu_history[n].copy()
        du    = u_np1 - u_n

        # ------------------------------------------------------------------
        # Phase fields at time n+1
        # ------------------------------------------------------------------
        if freeze_phase:
            S_elem = S_elem_frozen
        else:
            U_modes = u_n.reshape(P, num_nodes)
            Tmean_nodes = U_modes[0].reshape(Nx + 1, Ny + 1)
            _, S_elem, _ = melt_fraction_from_Tmean(Tmean_nodes)

        Sbar = float(np.mean(S_elem))

        _, V_elem, dV_dT_elem_n = vap_fraction_from_Tmean(
            u_n.reshape(P, num_nodes)[0].reshape(Nx + 1, Ny + 1)
        )
        Vbar = float(np.mean(V_elem))

        # ------------------------------------------------------------------
        # Correct sensitivity weights — same for all properties
        #   forward: eff = s*(1-S) + m*(S-V) + v*V
        # ------------------------------------------------------------------
        w_s = 1.0 - S_elem          # d(eff)/d(prop_s),  always in [0,1]
        w_m = S_elem - V_elem        # d(eff)/d(prop_m),  can be negative
        w_v = V_elem                 # d(eff)/d(prop_v),  always in [0,1]
        w_rhovap0 = dV_dT_elem_n
        w_rhovap1 = dV_dT_elem_n

        # ------------------------------------------------------------------
        # K0 gradients
        # ------------------------------------------------------------------
        K_dk0_s = assemble_from_element_weights_fast(w_s, Nx, Ny, dx, dy)
        K_dk0_m = assemble_from_element_weights_fast(w_m, Nx, Ny, dx, dy)
        K_dk0_v = assemble_from_element_weights_fast(w_v, Nx, Ny, dx, dy)

        g["k0_s"] -= dt * float(mu_n @ apply_I_kron_K0u(K_dk0_s, u_np1))
        g["k0_m"] -= dt * float(mu_n @ apply_I_kron_K0u(K_dk0_m, u_np1))
        g["k0_v"] -= dt * float(mu_n @ apply_I_kron_K0u(K_dk0_v, u_np1))

        # ------------------------------------------------------------------
        # K1 gradients
        # ------------------------------------------------------------------
        k1_elem = (solid_prop["k1"] * w_s
                   + melt_prop["k1"] * (w_s + w_m)   # = k1_s + (k1_m-k1_s)*S
                   )
        # Recompute exactly as the forward does:
        k1_elem = (solid_prop["k1"]
                   + (melt_prop["k1"] - solid_prop["k1"]) * S_elem
                   + (vap_prop["k1"]  - melt_prop["k1"])  * V_elem)
        m1_elem = (solid_prop["m1"]
                   + (melt_prop["m1"] - solid_prop["m1"]) * S_elem
                   + (vap_prop["m1"]  - melt_prop["m1"])  * V_elem
                   +vap_prop["rho_vap1"] * dV_dT_elem_n )

        if spatial_op is not None:
            spatial_op.update_weights(k1_elem, m1_elem)

            # VJP returns nodal gradient (num_nodes,) — convert to element space
            grad_k1_nodes = spatial_op.apply_K1_vjp_k1(u_np1, mu_n)
            grad_k1_elem  = node_to_elem_mean(
                grad_k1_nodes.reshape(Nx + 1, Ny + 1), Nx, Ny
            )  # (Nx, Ny)

            g["k1_s"] -= dt * float(np.sum(grad_k1_elem * w_s))
            g["k1_m"] -= dt * float(np.sum(grad_k1_elem * w_m))
            g["k1_v"] -= dt * float(np.sum(grad_k1_elem * w_v))

        else:
            Ku1 = K_SG_K1 @ u_np1
            gk1_total = -dt * float(mu_n @ Ku1)
            g["k1_s"] += (1.0 - Sbar) * gk1_total
            g["k1_m"] += Sbar * gk1_total

        # ------------------------------------------------------------------
        # M0 gradients
        # ------------------------------------------------------------------
        M_dm0_s = assemble_mass_from_element_weights(w_s, Nx, Ny, dx, dy)
        M_dm0_m = assemble_mass_from_element_weights(w_m, Nx, Ny, dx, dy)
        M_dm0_v = assemble_mass_from_element_weights(w_v, Nx, Ny, dx, dy)
        M_rhovap0 = assemble_mass_from_element_weights(w_rhovap0, Nx, Ny, dx, dy)
        M_rhovap1 = assemble_mass_from_element_weights(w_rhovap1, Nx, Ny, dx, dy)

        g["m0_s"] -= float(mu_n @ apply_I_kron_M0u(M_dm0_s, du))
        g["m0_m"] -= float(mu_n @ apply_I_kron_M0u(M_dm0_m, du))
        g["m0_v"] -= float(mu_n @ apply_I_kron_M0u(M_dm0_v, du))
        g["rho_vap0"] -= float(mu_n @ apply_I_kron_M0u(M_rhovap0, du))

        # ------------------------------------------------------------------
        # M1 gradients
        # ------------------------------------------------------------------
        if spatial_op is not None:
            grad_m1_nodes = spatial_op.apply_M1_vjp_m1(du, mu_n)
            grad_m1_elem  = node_to_elem_mean(
                grad_m1_nodes.reshape(Nx + 1, Ny + 1), Nx, Ny
            )  # (Nx, Ny)

            g["m1_s"] -= float(np.sum(grad_m1_elem * w_s))
            g["m1_m"] -= float(np.sum(grad_m1_elem * w_m))
            g["m1_v"] -= float(np.sum(grad_m1_elem * w_v))
            g["rho_vap1"] -= float(np.sum(grad_m1_elem*w_rhovap1))

        else:
            MU1_du = M_SG_M1 @ du
            gm1_total = -float(mu_n @ MU1_du)
            g["m1_s"] += (1.0 - Sbar) * gm1_total
            g["m1_m"] += Sbar * gm1_total

        # ------------------------------------------------------------------
        # Forcing gradients — spatially resolved chain rule through f0_nodal
        # ------------------------------------------------------------------
        # Reconstruct f0_nodal exactly as the forward does (element → nodal)
        f0_elem_adj = (solid_prop["f0"]
                       + (melt_prop["f0"] - solid_prop["f0"]) * S_elem
                       + (vap_prop["f0"]  - melt_prop["f0"])  * V_elem)
        f0_nodal_adj = elem_to_node_weights(f0_elem_adj, Nx, Ny).reshape(Nx + 1, Ny + 1)
        # f1 enters as a uniform scalar multiplier of the KL field — scalar blend is correct
        f1_eff = (solid_prop["f1"]
                  + (melt_prop["f1"] - solid_prop["f1"]) * Sbar
                  + (vap_prop["f1"]  - melt_prop["f1"])  * Vbar)

        g_f0_nodal, dJ_df1_eff = forcing_param_grads_spatial_jax(
            jnp.asarray(u_n), jnp.asarray(mu_n),
            jnp.asarray(f0_nodal_adj), f1_eff, n
        )
        g_f0_nodal = dt * np.asarray(g_f0_nodal)  # shape (Nx+1, Ny+1)
        gf1_eff = dt * float(dJ_df1_eff)

        # Phase-sensitivity weights on the nodal grid
        V_nodal = elem_to_node_weights(V_elem, Nx, Ny).reshape(Nx + 1, Ny + 1)
        S_nodal = elem_to_node_weights(S_elem, Nx, Ny).reshape(Nx + 1, Ny + 1)

        # Correct chain rule: dJ/df0_v = sum_i g_f0_nodal_i * V_nodal_i
        g["f0_v"] += float(np.sum(g_f0_nodal * V_nodal))
        g["f0_m"] += float(np.sum(g_f0_nodal * (S_nodal - V_nodal)))
        g["f0_s"] += float(np.sum(g_f0_nodal * (1.0 - S_nodal)))
        g["f1_s"] += (1.0 - Sbar) * gf1_eff
        g["f1_m"] += (Sbar - Vbar) * gf1_eff
        g["f1_v"] += Vbar          * gf1_eff

    return g

def phi_one_step(t,
    u_prev,
    solid_prop, melt_prop, vap_prop, ell, sigma, xi_sample,
    M_bc, K_bc, solve_A,                 # (you can remove if truly unused)
    K_SG_K0, K_SG_K1, M_SG_M0, M_SG_M1,
    F_SG_numpy,spatial_op,
    max_picard=40, tol_picard=1e-8, omega=1.0,
    alpha_k=0.0, alpha_m=0.0,            # <- make explicit (optional)
    freeze_phase=False,                # <- if True, compute S once from u_prev
):
    start = time.time()
    """
    One implicit step with Picard iterations.
    Phase-dependent mean operators via melt_fraction_from_Tmean(Tmean).

    - k0,m0: spatial blend per element using S_elem
    - k1,m1,f0,f1: global blend using Sbar = mean(S_elem) (cheap approximation)
    - optional temperature dependence: k += alpha_k * Tavg, m += alpha_m * Tavg
    - vap_prop: if provided, a third phase (vapour) is blended above T_vap_lo/hi
    """

    u_prev_loc = np.asarray(u_prev, dtype=float).copy()
    u_prev_loc[bc_idx] = 0.0

    # Picard initial guess
    U_k = u_prev_loc.copy()

    # Optional: freeze phase at start of step (for speed / stability)
    if freeze_phase:
        U_modes0 = U_k.reshape(P, num_nodes)
        Tmean_nodes0 = U_modes0[0].reshape(Nx + 1, Ny + 1)
        _, S_elem_frozen = melt_fraction_from_Tmean(Tmean_nodes0)

    t0 = time.monotonic()
    for it in range(max_picard):

        # ---- phase field from current mean (unless frozen) ----
        if freeze_phase:
            S_elem = S_elem_frozen
            # still need the nodal mean for vap_fraction below
            U_modes = U_k.reshape(P, num_nodes)
            Tmean_nodes = U_modes[0].reshape(Nx + 1, Ny + 1)
            # dS_dT_elem is zero when phase is frozen (no linearisation of S)
            dS_dT_elem = np.zeros_like(S_elem)
        else:
            U_modes = u_prev_loc.reshape(P, num_nodes)
            Tmean_nodes = U_modes[0].reshape(Nx + 1, Ny + 1)
            _, S_elem, dS_dT_elem = melt_fraction_from_Tmean(Tmean_nodes)

        Sbar = float(np.mean(S_elem))

        # ---- vaporisation fraction at current mean temperature ----
        _, V_elem, dV_dT_elem = vap_fraction_from_Tmean(Tmean_nodes)
        Vbar = float(np.mean(V_elem))

        # ---- effective (global) source + stochastic amplitudes ----
        # Three-phase blend: solid -> melt -> vapour
        f0_eff = (solid_prop["f0"] + (melt_prop["f0"] - solid_prop["f0"]) * Sbar
                  + (vap_prop["f0"]  - melt_prop["f0"])  * Vbar)
        f1_eff = (solid_prop["f1"] + (melt_prop["f1"] - solid_prop["f1"]) * Sbar
                  + (vap_prop["f1"]  - melt_prop["f1"])  * Vbar)
        k1_elem = (solid_prop["k1"] + (melt_prop["k1"] - solid_prop["k1"]) * S_elem
                   + (vap_prop["k1"]  - melt_prop["k1"])  * V_elem)
        m1_elem = (solid_prop["m1"] + (melt_prop["m1"] - solid_prop["m1"]) * S_elem
                   + (vap_prop["m1"]  - melt_prop["m1"])  * V_elem)+  vap_prop["rho_vap1"] * dV_dT_elem

        #k1_full = np.tile(elem_to_node_weights(k1_elem, Nx, Ny), P)
        #m1_full = np.tile(elem_to_node_weights(m1_elem, Nx, Ny), P)
        spatial_op.update_weights(k1_elem, m1_elem)
        # ---- assemble mean operators K0u, M0u (spatial) ----
        k0_elem = solid_prop["k0"] + (melt_prop["k0"] - solid_prop["k0"]) * S_elem
        f0_elem = solid_prop["f0"] + (melt_prop["f0"] - solid_prop["f0"]) * S_elem

        # base phase-blended volumetric heat capacity
        m0_elem = solid_prop["m0"] + (melt_prop["m0"] - solid_prop["m0"]) * S_elem

        # add latent heat as apparent heat capacity in mushy zone
        m0_elem = m0_elem + rhoL * dS_dT_elem

        # ---- vapour-phase blending for k0 and m0 ----
        k0_elem = k0_elem + (vap_prop["k0"] - melt_prop["k0"]) * V_elem
        f0_elem = f0_elem + (vap_prop["f0"] - melt_prop["f0"]) * V_elem
        f0_nodal = elem_to_node_weights(f0_elem, Nx, Ny).reshape(Nx + 1, Ny + 1)
        m0_elem = m0_elem + (vap_prop["m0"] - melt_prop["m0"]) * V_elem
        # add latent heat of vaporisation as apparent heat capacity at the vapour front
        m0_elem = m0_elem + vap_prop["rho_vap0"] * dV_dT_elem
        # Optional extra temperature dependence (element-averaged mean temperature)
        if (alpha_k != 0.0) or (alpha_m != 0.0):
            Tmean = U_k[:num_nodes].reshape((Nx + 1, Ny + 1))
            Tavg_elem = 0.25 * (
                Tmean[:-1, :-1] + Tmean[1:, :-1] + Tmean[:-1, 1:] + Tmean[1:, 1:]
            )
            if alpha_k != 0.0:
                k0_elem = np.maximum(k0_elem + alpha_k * Tavg_elem, 1e-12)
            if alpha_m != 0.0:
                m0_elem = np.maximum(m0_elem + alpha_m * Tavg_elem, 1e-12)

        K0u = assemble_from_element_weights_fast(k0_elem, Nx, Ny, dx, dy)
        M0u = assemble_mass_from_element_weights(m0_elem, Nx, Ny, dx, dy)

        # ---- nonlinear SG source F(U_k) ----
        # Prefer: make F depend on f0_eff,f1_eff
        #try:def phi_one
        F_SG = F_SG_numpy(u_prev_loc, f0_nodal, f1_eff,t)
        #except TypeError:
            # fallback: assume F_SG_numpy already closes over f0,f1
            # (in that case you must rebuild the closure outside per step)
           # F_SG = F_SG_numpy(u_prev_loc)
           # print("exception")

        F_SG = np.asarray(F_SG, dtype=float)
        F_SG[bc_idx] = 0.0

        # ---- RHS: M(u) u_prev + dt * F ----
        #rhs = apply_mass(M0u, u_prev_loc, m1_full, M_SG_M1) + dt * F_SG
        rhs = apply_I_kron_M0u(M0u, u_prev) + spatial_op.apply_M1(u_prev) + dt * F_SG
        rhs[bc_idx] = 0.0

        # ---- operator matvec ----
        def A_matvec(v):
            v = np.asarray(v, dtype=float).copy()
            v[bc_idx] = 0.0

            #y = apply_mass(M0u, v, m1_full, M_SG_M1)
            y = apply_I_kron_M0u(M0u, v)

            y += dt * apply_I_kron_K0u(K0u, v)
           # y += dt * k1_eff * (K_SG_K1 @ v)
            y += spatial_op.apply_M1(v)
            #plt.imshow(m1_elem)
            #plt.imshow(spatial_op.apply_M1(v)[:num_nodes].reshape((Nx+1,Ny+1)))

            #plt.colorbar()
            #plt.show()
            #y += dt * (k1_full * (K_SG_K1 @ v))
            y += dt * spatial_op.apply_K1(v)
            y[bc_idx] = 0.0
            return y

        Aop = LinearOperator(
            shape=(P * num_nodes, P * num_nodes),
            matvec=A_matvec,
            dtype=float,
        )

        # Preconditioner:
        # If you have a good constant preconditioner, you can always use it.
        # If solve_A corresponds to a *fixed* operator but your phase makes A vary,
        # it may still help, but it’s approximate.
        M_prec = None
        if solve_A is not None:
            M_prec = LinearOperator(
                shape=(P * num_nodes, P * num_nodes),
                matvec=lambda r: solve_A(r),
                dtype=float,
            )
        U_raw, info = gmres(
            Aop,
            rhs,
            x0=U_k,
            rtol=1e-7,
            restart=60,
        #    M=M_prec,
        )
        U_raw = np.asarray(U_raw, dtype=float)
        U_raw[bc_idx] = 0.0

        if info != 0:
            print(f"warning: GMRES info={info} in phi_one_step Picard it {it}")

        # Picard damping
        U_next = (1.0 - omega) * U_k + omega * U_raw
        U_next[bc_idx] = 0.0

        # convergence
        diff = np.linalg.norm(U_next - U_k)
        denom = max(1.0, np.linalg.norm(U_k))
        rel_diff = diff / denom

        U_k = U_next
      #  print(rel_diff)
        if rel_diff < tol_picard:
            break

    t1 = time.monotonic()

    U_new = np.asarray(U_k, dtype=float)
    U_new[bc_idx] = 0.0
    #print(np.max(U_k[:num_nodes]))
    #print("nonmean L2:", np.linalg.norm(U_modes[1:]))
    #print("max var:", np.max(np.sum(U_modes[1:]**2, axis=0)))

    def plot():
        plt.figure(figsize=(6, 5))
        plt.imshow(f0_nodal[:num_nodes].reshape((Nx +1, Ny+1 )).T,
                    extent=[0, Lx, 0, Ly], origin='lower', cmap='magma')
        plt.colorbar(label='kW')
        plt.title('Temperature')
        plt.figure(figsize=(6, 5))
        plt.imshow(U_new[:num_nodes].reshape((Nx + 1, Ny + 1)).T,
                    extent=[0, Lx, 0, Ly], origin='lower', cmap='magma')
        plt.colorbar(label='kW')
        plt.title('Temperature')
        plt.xlabel('x'); plt.ylabel('y'); plt.show()
        plt.figure(figsize=(6, 5))
        plt.imshow(F_SG[:num_nodes].reshape((Nx + 1, Ny + 1)).T,
                    extent=[0, Lx, 0, Ly], origin='lower', cmap='magma')
        plt.colorbar(label='kW')
        plt.title('Source term')
        plt.xlabel('x'); plt.ylabel('y'); plt.show()
        u_variance = np.zeros(num_nodes)
        for mode in range(1, P):
            u_variance += U_k[mode * num_nodes:(mode + 1) * num_nodes] ** 2
        u_variance = u_variance.reshape((Nx + 1, Ny + 1))

        plt.figure(figsize=(6, 5))
        plt.imshow(u_variance.T, extent=[0, Lx, 0, Ly], origin='lower', cmap='magma')
        plt.colorbar(label='Temperature Variance'); plt.title("Temperature Variance at Final Time")
        plt.xlabel("x"); plt.ylabel("y"); plt.show()
    if t% 200==0:
    #
        plot()
    u_mean = U_new[:num_nodes]

    u_variance = np.zeros(num_nodes)
    for mode in range(1, P):
        u_variance += U_new[mode * num_nodes:(mode + 1) * num_nodes] ** 2
    u_variance = u_variance.reshape((Nx + 1, Ny + 1))

   # H, seen_mask, w_norm, x_surf, y_surf = visualize_radiometer_at_timestep(
   #     u_mean, u_variance,
   #     T_abl=500.0,                # or whatever you use
    #    center=(Lx/2, Ly+0.02),
    #    angle_deg=20.0,
    #    aggregate=True,
    #    weights='cosine',
    #    title_prefix=f"Radiometer"
   # )
# --- after you have U_new, u_mean, u_variance computed in phi_one_step ---
   
    # reshape the mean temperature field to (Nx+1, Ny+1) in JAX
    T_mean_nodes = jnp.asarray(u_mean).reshape((Nx + 1, Ny + 1))
    T_var_nodes  = jnp.asarray(u_variance)      # if your interface helper wants it
    x_nodes      = jnp.asarray(x)               # shape (Nx+1,)
    y_nodes      = jnp.asarray(y)               # shape (Ny+1,)

    # 1) Get the interface y = h(x)
    x_surf, h_mean, _, _ = extract_interface_h_from_SG_jax(
        T_mean_nodes, T_var_nodes, 500.0, x_nodes, y_nodes
    )
    x_surf = jnp.asarray(x_surf).reshape((-1,))   # (Nx+1,)
    y_surf = jnp.asarray(h_mean ).reshape((-1,))  # (Nx+1,)

    x_vis, y_vis, w_vis = np.zeros(10), np.zeros(10),  np.zeros(10)
   # print(x_vis, y_vis)
    end = time.time()
    smin_i, *_ = softmin_depth(u_mean.reshape(Nx + 1, Ny + 1), y, T_abl, 10, 0.0005)
    #print(smin_i)
    return U_new, u_mean, u_variance, x_vis, y_vis,w_vis, K_SG_K0, K_SG_K1, M_SG_M0, M_SG_M1
def compute_full_adjoint_corrections(
    mu, U_np1, U_n,
    d2S_dT2_elem, d2V_dT2_elem,
    dS_dT_elem_np1, dV_dT_elem_np1,
    solid_prop, melt_prop, vap_prop,
    Nx, Ny, dx, dy, dt, rhoL, P, num_nodes
):
    correction = np.zeros(P * num_nodes, dtype=float)

    U_np1_modes = U_np1.reshape(P, num_nodes)
    U_n_modes   = U_n.reshape(P, num_nodes)
    mu_modes    = mu.reshape(P, num_nodes)
    du_modes    = U_np1_modes - U_n_modes

    # --- element-wise (U^{n+1} - U^n) . mu, summed over modes ---
    du_dot_mu_elem = np.zeros((Nx, Ny))
    for p in range(P):
        du_p_e = node_to_elem_mean(du_modes[p].reshape(Nx+1, Ny+1), Nx, Ny)
        mu_p_e = node_to_elem_mean(mu_modes[p].reshape(Nx+1, Ny+1), Nx, Ny)
        du_dot_mu_elem += du_p_e * mu_p_e

    # --- element-wise grad(U^{n+1}) . grad(mu), summed over modes ---
    grad_dot_elem = np.zeros((Nx, Ny))
    for p in range(P):
        u_p  = U_np1_modes[p].reshape(Nx+1, Ny+1)
        mu_p = mu_modes[p].reshape(Nx+1, Ny+1)
        ux  = 0.5*((u_p[1:,:-1]-u_p[:-1,:-1]) + (u_p[1:,1:]-u_p[:-1,1:]))/dx
        uy  = 0.5*((u_p[:-1,1:]-u_p[:-1,:-1]) + (u_p[1:,1:]-u_p[1:,:-1]))/dy
        mx  = 0.5*((mu_p[1:,:-1]-mu_p[:-1,:-1]) + (mu_p[1:,1:]-mu_p[:-1,1:]))/dx
        my  = 0.5*((mu_p[:-1,1:]-mu_p[:-1,:-1]) + (mu_p[1:,1:]-mu_p[1:,:-1]))/dy
        grad_dot_elem += ux*mx + uy*my

    # --- element weights ---
    w_mass  = (rhoL * d2S_dT2_elem
               + vap_prop["rho_vap0"] * d2V_dT2_elem) * du_dot_mu_elem

    w_stiff = ((melt_prop["k0"] - solid_prop["k0"]) * dS_dT_elem_np1
               + (vap_prop["k0"] - melt_prop["k0"]) * dV_dT_elem_np1) * grad_dot_elem

    # --- nodal assembly: each element weight w_e spreads to its 4 corner nodes ---
    # This is elem_to_node_weights scaled by element area (dx*dy/4 per corner)
    mass_nodal  = elem_to_node_weights(w_mass,  Nx, Ny) * (dx * dy / 4.0)
    stiff_nodal = elem_to_node_weights(w_stiff, Nx, Ny) * (dx * dy / 4.0)

    # Only mean mode (index 0) is affected
    correction[:num_nodes] = mass_nodal.ravel() + dt * stiff_nodal.ravel()

    return correction
def node_to_elem_mean(coeff_nodes, Nx, Ny):
    """
    Average a nodal field (Nx+1, Ny+1) down to element centres (Nx, Ny).
    Each element value is the mean of its 4 corner nodes.
    Mirrors the element-averaging already used in melt_fraction_from_Tmean.
    """
    c = coeff_nodes.reshape(Nx + 1, Ny + 1)
    return 0.25 * (c[:-1, :-1] + c[1:, :-1] + c[:-1, 1:] + c[1:, 1:])
def adjoint_one_step(t, solid_prop, melt_prop, vap_prop, U_np1, lam_np1, U_n,
                     M_SG_M0, M_SG_M1,
                     K_SG_K0, K_SG_K1,
                     M_bc, K_bc, solve_A,
                     
                     spatial_op,params,
                     mu_prev=None,
                     do_dot_test=False,
                     F_SG_core=None, freeze_phase=False,
                     ):
    """
    One adjoint step using spatially-varying k1/m1 via spatial_op.

    Forward system:  A(U_{n+1}) * U_{n+1} = M(U_{n+1}) * U_n + dt * F(U_n)
    where            A = M(U_{n+1}) + dt * K(U_{n+1})

    Residual as function of U_n:
        R(U_n) = A(U_{n+1})*U_{n+1}  -  M(U_{n+1})*U_n  -  dt*F(U_n)
        dR/dU_n = -M(U_{n+1}) - dt * J_F(U_n)^T

    So:  lam_n = -dR/dU_n^T * mu = M(U_{n+1})^T * mu + dt * J_F(U_n)^T * mu

    KEY FIX: lam_n must use M evaluated at n+1 (not n). The old code
    recomputed phase at time n, updated spatial_op to time-n weights, and
    assembled M0u_n — all wrong. M(U_{n+1}) and spatial_op are already
    correct at n+1 from the GMRES setup above; they are reused directly.
    """
    if vap_prop is None:
        vap_prop = melt_prop

    bc_mask_np = np.ones(P * num_nodes, dtype=np.float32)
    bc_mask_np[bc_idx] = 0.0
    BC_MASK = jnp.asarray(bc_mask_np)

    # ------------------------------------------------------------------
    # Phase at time n+1 — used for A matrix and for lam_n
    # ------------------------------------------------------------------
    if freeze_phase:
        S_elem_np1 = S_elem_frozen
        dS_dT_elem_np1 = np.zeros_like(S_elem_frozen)
    else:
        U_modes_np1 = U_n.reshape(P, num_nodes)
        Tmean_nodes_np1 = U_modes_np1[0].reshape(Nx + 1, Ny + 1)
        _, S_elem_np1, dS_dT_elem_np1 = melt_fraction_from_Tmean(Tmean_nodes_np1)

    Sbar_np1 = float(np.mean(S_elem_np1))

    _, V_elem_np1, dV_dT_elem_np1 = vap_fraction_from_Tmean(
        U_n.reshape(P, num_nodes)[0].reshape(Nx + 1, Ny + 1)
    )
    Vbar_np1 = float(np.mean(V_elem_np1))

    # ---- effective amplitudes at n+1 (used for both A^T and forcing VJP) ----
    f0_elem_np1 = (solid_prop["f0"] + (melt_prop["f0"] - solid_prop["f0"]) * S_elem_np1
                  + (vap_prop["f0"] - melt_prop["f0"]) * V_elem_np1)
    f1_eff_np1 = (solid_prop["f1"] + (melt_prop["f1"] - solid_prop["f1"]) * Sbar_np1
                  + (vap_prop["f1"] - melt_prop["f1"]) * Vbar_np1)
    f0_nodal = elem_to_node_weights(f0_elem_np1, Nx, Ny).reshape(Nx + 1, Ny + 1)

    # ---- spatial operator weights at n+1 ----
    k1_elem_np1 = (solid_prop["k1"] + (melt_prop["k1"] - solid_prop["k1"]) * S_elem_np1
                   + (vap_prop["k1"] - melt_prop["k1"]) * V_elem_np1)
    m1_elem_np1 = (solid_prop["m1"] + (melt_prop["m1"] - solid_prop["m1"]) * S_elem_np1
                   + (vap_prop["m1"] - melt_prop["m1"]) * V_elem_np1) + vap_prop["rho_vap1"] * dV_dT_elem_np1

    spatial_op.update_weights(k1_elem_np1, m1_elem_np1)

    # ---- mean operators at n+1 ----
    k0_elem_np1 = (solid_prop["k0"] + (melt_prop["k0"] - solid_prop["k0"]) * S_elem_np1
                   + (vap_prop["k0"] - melt_prop["k0"]) * V_elem_np1)
    m0_elem_np1 = (solid_prop["m0"] + (melt_prop["m0"] - solid_prop["m0"]) * S_elem_np1
                   + (vap_prop["m0"] - melt_prop["m0"]) * V_elem_np1)
    m0_elem_np1 = m0_elem_np1 + rhoL * dS_dT_elem_np1 + vap_prop["rho_vap0"] * dV_dT_elem_np1

    K0u_np1 = assemble_from_element_weights_fast(k0_elem_np1, Nx, Ny, dx, dy)
    M0u_np1 = assemble_mass_from_element_weights(m0_elem_np1, Nx, Ny, dx, dy)

    @jax.jit
    def JTmu(U, mu, f0, f1, t, sqrt_lam, params_dict):
        def scalar_fn(U_):
            F = F_SG_apply(U_, f0, f1, t, sqrt_lam, params_dict)
            F = F * BC_MASK
            return jnp.vdot(mu, F)
        return jax.grad(scalar_fn)(U)

    # ------------------------------------------------------------------
    # Bug 2 fix: d(M(U_{n+1}))/dU_{n+1} * (U_{n+1} - U_n) correction
    # This is the missing linearisation of the apparent heat capacity.
    # Preassembled once here so matvec_adj doesn't rebuild it every iteration.
    # ------------------------------------------------------------------
    Tm = (T_melt_lo )
    z_nodes = (Tmean_nodes_np1 - Tm) / Delta_melt
    tanh_z  = np.tanh(z_nodes)
    d2S_dT2_nodes = -tanh_z * (1.0 - tanh_z**2) / (Delta_melt**2)
    d2S_dT2_elem = 0.25 * (
        d2S_dT2_nodes[:-1, :-1] + d2S_dT2_nodes[1:, :-1] +
        d2S_dT2_nodes[:-1, 1:]  + d2S_dT2_nodes[1:, 1:]
    )

    Tv_mid = (T_vap_lo )
    z_v_nodes = (Tmean_nodes_np1 - Tv_mid) / Delta_vap
    tanh_zv = np.tanh(z_v_nodes)
    d2V_dT2_nodes = -tanh_zv * (1.0 - tanh_zv**2) / (Delta_vap**2)
    d2V_dT2_elem = 0.25 * (
        d2V_dT2_nodes[:-1, :-1] + d2V_dT2_nodes[1:, :-1] +
        d2V_dT2_nodes[:-1, 1:]  + d2V_dT2_nodes[1:, 1:]
    )

    du_mean = (U_np1 - U_n).reshape(P, num_nodes)[0]
    du_mean_elem = node_to_elem_mean(du_mean.reshape(Nx + 1, Ny + 1), Nx, Ny)

   # correction_field = (rhoL * d2S_dT2_elem + vap_prop["rho_vap0"] * d2V_dT2_elem) * du_mean_elem
    #M_corr = assemble_mass_from_element_weights(correction_field, Nx, Ny, dx, dy)

    # ------------------------------------------------------------------
    # Adjoint operator A(U_{n+1})^T
    # ------------------------------------------------------------------
    def matvec_adj(x):
        x = np.asarray(x, dtype=float)
        x_bc = x.copy()
        x_bc[bc_idx] = 0.0

        y  = apply_I_kron_M0u(M0u_np1, x_bc)
        y += spatial_op.apply_M1(x_bc)
        y += dt * apply_I_kron_K0u(K0u_np1, x_bc)
        y += dt * spatial_op.apply_K1(x_bc)
      #  y[:num_nodes] += M_corr @ x_bc[:num_nodes]

        y[bc_idx] = 0.0
        return y

    A_adj = LinearOperator(
        shape=(P * num_nodes, P * num_nodes),
        matvec=matvec_adj,
        dtype=float
    )

    b = np.asarray(lam_np1, dtype=float).copy()
    b[bc_idx] = 0.0

    x0 = None
    if mu_prev is not None:
        x0 = np.asarray(mu_prev, dtype=float).copy()
        x0[bc_idx] = 0.0

    mu, info = gmres(A_adj, b, rtol=1e-7, restart=60, x0=x0)
    if info != 0:
        print(f"Warning: adjoint GMRES info={info}")

    mu = np.asarray(mu, dtype=float)
    mu[bc_idx] = 0.0

    # ------------------------------------------------------------------
    # Compute lam_n = (dR/dU_n)^T * (-mu)
    #   = M(U_{n+1})^T * mu  +  dt * J_F(U_n)^T * mu
    #
    # CRITICAL: use M0u_np1 and spatial_op (already at n+1 weights).
    # Do NOT recompute phase at n or update spatial_op — that was the bug.
    # ------------------------------------------------------------------
    lam_n = apply_I_kron_M0u(M0u_np1, mu) + spatial_op.apply_M1(mu)

    #adj_src = JTmu(jnp.asarray(U_n), jnp.asarray(mu),
    #               f0_nodal, f1_eff_np1, t, params["sqrt_lam"], params)
    #lam_n += dt * np.asarray(adj_src, float)
    # --- NEW: full nonlinear correction terms ---
    lam_n -= compute_full_adjoint_corrections(
        mu          = mu,
        U_np1       = U_np1,
        U_n         = U_n,
        d2S_dT2_elem     = d2S_dT2_elem,      # already computed above in adjoint_one_step
        d2V_dT2_elem     = d2V_dT2_elem,      # already computed above
        dS_dT_elem_np1   = dS_dT_elem_np1,    # already computed above
        dV_dT_elem_np1   = dV_dT_elem_np1,    # already computed above
        solid_prop  = solid_prop,
        melt_prop   = melt_prop,
        vap_prop    = vap_prop,
        Nx=Nx, Ny=Ny, dx=dx, dy=dy,
        dt=dt, rhoL=rhoL, P=P, num_nodes=num_nodes,
    )

    lam_n[bc_idx] = 0.0
    lam_variance = np.zeros(num_nodes)
    for mode in range(1, P):
        lam_variance += lam_n[mode * num_nodes:(mode + 1) * num_nodes] ** 2
    lam_variance = lam_variance.reshape((Nx + 1, Ny + 1))

    def plot():
        plt.figure(figsize=(6, 5))
        plt.imshow(lam_n[:num_nodes].reshape((Nx + 1, Ny + 1)).T,
                   extent=[0, Lx, 0, Ly], origin='lower', cmap='magma')
        plt.colorbar(label='kW')
        plt.title('Adjoint')
        plt.xlabel('x'); plt.ylabel('y'); plt.show()
        plt.figure(figsize=(6, 5))
        plt.imshow(lam_variance.T,
                   extent=[0, Lx, 0, Ly], origin='lower', cmap='magma')
        plt.colorbar(label='kW')
        plt.title('Adjoint var')
        plt.xlabel('x'); plt.ylabel('y'); plt.show()
    #plot()

    return lam_n, mu
def forward_step(t,u_prev,M_bc, K_bc, solve_A,solid_prop,melt_prop,vap_prop,K_SG_K0, K_SG_K1, M_SG_M0, M_SG_M1,F_SG_numpy,spatial_op):
    """
    Convenience wrapper around phi_one_step for the dot test.
    Assumes k0, k1, m0, m1, f0, f1, sigma, ell, xi_sample are defined globally.
    """
    return phi_one_step(t,
        u_prev,
        solid_prop,melt_prop, vap_prop, ell, sigma,
        xi_sample,
        M_bc, K_bc, solve_A,K_SG_K0, K_SG_K1, M_SG_M0, M_SG_M1,F_SG_numpy,spatial_op)
def adjoint_step(t,U_np1, lam_np1, U_n,  # ADD U_n
                 M_bc, K_bc, solve_A, solid_prop, melt_prop, vap_prop,
                 K_SG_K0, K_SG_K1, M_SG_M0, M_SG_M1,
                 spatial_op, params, # NEW: Add spatial_op parameter
                 mu_prev=None,
                 F_SG_core=None):
    return adjoint_one_step(t,solid_prop,melt_prop,vap_prop,
        U_np1=U_np1,
        lam_np1=lam_np1,
        U_n=U_n,  # Pass U_n
        M_SG_M0=M_SG_M0,
        M_SG_M1=M_SG_M1,
        K_SG_K0=K_SG_K0,
        K_SG_K1=K_SG_K1,
        M_bc=M_bc, K_bc=K_bc, solve_A=solve_A,
        params=params,
        spatial_op=spatial_op,  # Pass spatial_op
        mu_prev=mu_prev,
        F_SG_core=F_SG_core
    )
time_steps = int(T_final / dt)
print(time_steps)
ndofs = P * num_nodes

def run_forward(U0,solid_prop,melt_prop,vap_prop,ell,frozen=False,theta_kappa=None, kappa_param=None):
    """
    Run forward in time and store U_history.
    theta_kappa: per-layer kappa values for SPDE field (replaces ell for KL modes)
    """
    time0 = time.monotonic()
    K_SG_K0, K_SG_K1, M_SG_M0, M_SG_M1, K_kl_global, M_kl_global, kappa_params_update = build_SG_operators(sigma, ell, frozen, theta_kappa=theta_kappa, kappa_param=kappa_param)
    # Local snapshot: isolates this call's eigvecs from any subsequent build_SG_operators
    # call that would otherwise overwrite the global params dict.
    local_params = {**params, **kappa_params_update}
    U0_loc = U0.copy()

    print(vap_prop)
    print(theta_kappa)
    print(kappa_param)

    U_history = np.zeros((time_steps, n_dofs), dtype=np.float64)    
    U_history[0, :] = U0_loc
    x_vis_hist = [None] * time_steps
    y_vis_hist = [None] * time_steps
    w_vis_hist = [None] * time_steps

    # Use CACHED version to avoid 23s LU factorization on repeated calls!
    time1 = time.monotonic()
    #print("K and M ", time1-time0)
    k0_eff = solid_prop["k0"]  # or some combination of solid/melt
    k1_eff = solid_prop["k1"]
    m0_eff = solid_prop["m0"]
    m1_eff = solid_prop["m1"]

    M_bc, K_bc, solve_A = build_linear_operators_cached(
        k0_eff, k1_eff, m0_eff, m1_eff,
        M_SG_M0, M_SG_M1, K_SG_K0, K_SG_K1
    )

    M_bc, K_bc, solve_A = build_linear_operators_cached(k0, k1, m0, m1, M_SG_M0, M_SG_M1, K_SG_K0,K_SG_K1)
    spatial_op = SpatialK1M1Operator(
    K_kl_global, M_kl_global, G_list, P, num_nodes, Nx, Ny
    )
    clear_f_sg_cache()  # keep if you have caching elsewhere
    #print(solid_prop, melt_prop, ell)
    def F_SG_jax(Ucoeff, f0, f1,t, sqrt_lam=None):
        # Use local_params (snapshot for this kappa) so FD perturbation calls
        # with different kappa values don't corrupt each other's eigvecs.
        core = make_F_SG_core(local_params, f0, f1, t, sqrt_lam=sqrt_lam)
        return core(Ucoeff)

    F_SG_jitted = jax.jit(F_SG_jax)

    def F_SG_numpy(U,f0,f1,t):
        # use global f0,f1 or pass them from caller
        return np.asarray(F_SG_jitted(U, f0, f1, t,None), dtype=float)


    #def F_SG_numpy(Ucoeff, params, f0, f1):
        # Ucoeff is already NumPy; just call the jitted function
    #    F_jax = F_SG_jitted(Ucoeff)
    #    return np.asarray(F_jax, dtype=float)
    #def F_bc_jax(U):
    #   F = F_SG_jax(U, f0, f1)       # or F_SG_jitted(U) if you prefer, but pick ONE
   #     return F.at[bc_idx].set(0.0)  # P F(U)

   # def F_SG_numpy(Ucoeff, params, f0, f1):
    ##    F_jax = F_bc_jax(jnp.asarray(Ucoeff))
    #    return np.asarray(F_jax, dtype=float)

    U_history[0, :] = U0_loc

    for n in range(1, time_steps):
        #print(n)
        U_history[n, :], u_mean, u_variance, x_vis, y_vis,w_vis, K_SG_K0, K_SG_K1, M_SG_M0, M_SG_M1 = forward_step(n,U_history[n-1, :], M_bc, K_bc, solve_A,solid_prop, melt_prop,vap_prop,K_SG_K0, K_SG_K1, M_SG_M0, M_SG_M1,F_SG_numpy,spatial_op)
        x_vis_hist[n] = np.asarray(x_vis)
        y_vis_hist[n] = np.asarray(y_vis)
        w_vis_hist[n] = np.asarray(w_vis)
       # if n==1:
       #     x_vis_hist[0], y_vis_hist[0] = np.asarray(x_vis), np.asarray(y_vis)

    return (U_history, u_mean, u_variance,
            M_bc, K_bc, solve_A,
            x_vis_hist, y_vis_hist, w_vis_hist,
            K_SG_K0, K_SG_K1, M_SG_M0, M_SG_M1,
            spatial_op,
            local_params)  # local_params: kappa-specific snapshot, safe for adjoint

def run_adjoint(U_history, M_bc, K_bc, solve_A, K_SG_K0, K_SG_K1,
                M_SG_M0, M_SG_M1, T_obs, u_mean, u_variance,
                x_vis_hist, y_vis_hist, w_vis_hist, solid_prop, melt_prop, vap_prop,
                spatial_op,  # NEW: Add spatial_op parameter
                F_SG_core=None):
    """
    Backward adjoint sweep.
   
    Convention: Mu_history[n] stores the μ from solving A^T μ = λ_{n+1}
                This μ is used for gradients of the step n → n+1
    """    
#    L_history = np.zeros_like(U_history)
    Mu_history = np.zeros_like(U_history)
   
    # Terminal condition
    #L_history[-1, :] = likelihood_visible_gradient(
    #    U_history[-1, :], T_obs, u_mean, u_variance,
    #    x_vis, y_vis, x, y, dx, dy, num_obs,
    #    variance_depends_on_U=True
    #)
    #if x_vis_hist[-1] is not None and len(x_vis_hist[-1]) > 0 and w_vis_hist[-1] is not None:
    #    L_current = radiometer_likelihood_weighted_grad(
    #        U_history[-1, :], T_obs_hist[-1],
    #        x_vis_hist[-1], y_vis_hist[-1], w_vis_hist[-1],
    #        x, y, sigma_obs, num_obs
    #    )
    #else:
    #    L_current = 0.0  # No observation at final time
    #L_history[-1, :] = U_history[-1, :].copy()
    # Backward sweep - start from time_steps - 2 down to 0
    # This computes λ_n and μ_n for n = time_steps-2, ..., 0
    L_current = np.zeros_like(U_history[-1])
    L_current[:num_nodes] = T_obs_mean[-1] - U_history[-1,:num_nodes]
   # print("hist of visible x: ", x_vis_hist)
   # print("hist of visible y: ", y_vis_hist)

   # print("hist of visible w: ", w_vis_hist)

    for n in range(time_steps - 2, -1, -1):
       
        U_np1 = U_history[n + 1, :]
        U_n = U_history[n, :]
        lam_np1 = L_current

        # Warm start: use μ from previous iteration (if available)
        # Note: Mu_history[n+1] was set in the previous iteration of this loop
        if n < time_steps - 2:
            mu_prev = Mu_history[n + 1, :]
        else:
            mu_prev = None  # First iteration, no warm start
       
        lam_n, mu_n = adjoint_step(n,
            U_np1, lam_np1, U_n,
            M_bc, K_bc, solve_A,
            solid_prop, melt_prop, vap_prop,params,
            K_SG_K0, K_SG_K1, M_SG_M0, M_SG_M1,
            spatial_op,  # Pass spatial_op
            mu_prev=mu_prev,
            F_SG_core=F_SG_core
        )
        #if x_vis_hist[n] is not None and len(x_vis_hist[n]) > 0:
        #if x_vis_hist[n] is not None and len(x_vis_hist[n]) > 0 and w_vis_hist[n] is not None:
            #lam_n += radiometer_likelihood_weighted_grad(
            #    U_history[n, :], T_obs_hist[n],
            #    x_vis_hist[n], y_vis_hist[n], w_vis_hist[n],
            #    x, y, sigma_obs, num_obs
            #)

            #lam_n[:num_nodes] += T_obs_mean[n] - U_history[n,:num_nodes]
        L_current = lam_n
        Mu_history[n, :] = mu_n  # μ from A^T μ = λ_{n+1}
   
    return Mu_history
def sample_solution(Ucoeff, multi_idx, xi_sample):
    u = np.zeros(num_nodes)
    for k, alpha in enumerate(multi_idx):
        psi = eval_psi(xi_sample, alpha)
        u += psi * Ucoeff[k*num_nodes:(k+1)*num_nodes]
    return u


n_dofs = P * num_nodes
U0 = np.zeros(n_dofs)
U0[bc_idx] = 0.0   # respect Dirichlet

U_obs, u_mean_obs, u_variance_obs, M_bc, K_bc, solve_A, x_vis_hist, y_vis_hist,w_vis_hist, K_SG_K0, K_SG_K1, M_SG_M0, M_SG_M1, spatial_op_obs, _obs_local_params = run_forward(
    U0, SOLID_obs,MELT_obs, VAP_obs,ell, theta_kappa = theta_kappa_obs, kappa_param = kappa_param_obs
)
for k in range(1, min(P, 4)):
    uk0 = U_obs[0, k*num_nodes:(k+1)*num_nodes]
    uk1 = U_obs[1, k*num_nodes:(k+1)*num_nodes]
    print(f"k={k}: u_k(t=0) norm={np.linalg.norm(uk0):.3e}  u_k(t=1) norm={np.linalg.norm(uk1):.3e}")
#import sys
#from vis_forward import plot_all, plot_wellbore_contours
#import inf_layered_vap as slv
#plot_wellbore_contours(U_obs, slv)
# or pick specific timesteps:
#plot_wellbore_contours(U_obs, slv, timesteps=[1, 3], N_SAMP=4)


y_nodes = np.linspace(0.0, Ly, Ny + 1)
wz = _trap_weights(y_nodes)
#J_opt=0
eps_smooth = 10.0
beta=0.005
J_opt = 0.0
y_nodes = np.linspace(0.0, Ly, Ny + 1)
wz = _trap_weights(y_nodes)


# observed mean and variance of depth across realisations
#h_obs_hist      = depth_ensemble[:, :].mean(axis=0)          # (time_steps,)
#sigma2_obs_hist = depth_ensemble[:, :].var(axis=0, ddof=1)   # (time_steps,)

h_obs_hist      = np.zeros(time_steps)
sigma2_obs_hist = np.zeros(time_steps)

for t in range(1, time_steps):
    u_t = U_obs[t]
    u_mean_2d = u_t[:num_nodes].reshape(Nx+1, Ny+1)
    h_u0, _, w_softmin, _, dH = softmin_depth(u_mean_2d, y_nodes, T_abl, eps_smooth, beta)
    g_flat = (w_softmin[:, None] * wz[None, :] * dH).ravel()
    U_modes = u_t.reshape(P, num_nodes)
    h_obs_hist[t]      = h_u0
    sigma2_obs_hist[t] = float(np.sum((U_modes[1:] @ g_flat)**2))

print("h_obs_hist (analytical):      ", h_obs_hist)
print("sigma2_obs_hist (analytical): ", sigma2_obs_hist)
print("CHECK")

#T_obs_hist = make_T_obs_history_weighted(U_obs, x_vis_hist, y_vis_hist, w_vis_hist,
#                                 x, y, sigma_obs, num_obs,
#                                 multi_idx, eval_psi, N_KL)

Nt = U_obs.shape[0]
num_nodes = (Nx + 1) * (Ny + 1)

T_obs = np.zeros((Nt, num_nodes, num_obs))  # (time, node, sample)

rng = np.random.default_rng(0)

for i in range(num_obs):
    xi_sample = rng.standard_normal(N_KL)

    for t in range(Nt):
        T_obs[t, :, i] = sample_solution(U_obs[t, :], multi_idx, xi_sample)
T_obs_mean = T_obs.mean(axis=2)   # (Nt, num_nodes)
from depth_objective import run_adjoint_depth, compute_depth_objective_trajectory, validate_depth_adjoint_fd
from depth_objective import softmin_depth
import numpy as np
from depth_objective import _trap_weights

y_nodes = np.linspace(0.0, Ly, Ny + 1)
h_obs_hist = np.zeros(time_steps)
for t in range(time_steps):
    u2d = U_obs[t, :num_nodes].reshape(Nx+1, Ny+1)
    h_obs_hist[t], _, _, _, _ = softmin_depth(
        u2d, y_nodes, T_abl, eps=10.0, beta=0.005)
print("Hobs: ",h_obs_hist)
# then call with sigma_d matching your depth obs noise
J_total, J_hist = compute_depth_objective_trajectory(
    U_obs, Nx, Ny, Ly, num_nodes,
    T_abl=T_abl, eps=10.0, beta=2.0*dx,
    h_obs_hist=h_obs_hist, sigma_d=sigma_d,
)
print(J_hist)




import numpy as np
import copy
def make_phi_one_step(
    t,
    M_bc,
    K_bc,
    solve_A,
    solid_prop,
    melt_prop,
    vap_prop,
    K_SG_K0,
    K_SG_K1,
    M_SG_M0,
    M_SG_M1,
    F_SG,
    spatial_op,
):
    def phi(U_n):
        op = copy.deepcopy(spatial_op)
        out = forward_step(
            t,
            U_n,
            M_bc,
            K_bc,
            solve_A,
            solid_prop,
            melt_prop,
            vap_prop,
            K_SG_K0,
            K_SG_K1,
            M_SG_M0,
            M_SG_M1,
            F_SG,
            op,
        )
        U_np1 = out[0]   # assuming first return is next state
        return U_np1
    return phi
def make_adjoint_one_step(
    t,
    solid_prop,
    melt_prop,
    vap_prop,
    M_SG_M0,
    M_SG_M1,
    K_SG_K0,
    K_SG_K1,
    M_bc,
    K_bc,
    solve_A,
    spatial_op,
    params,
):
    def adj(U_n, U_np1, lam_np1):
        op = copy.deepcopy(spatial_op)
        return adjoint_one_step(
            t=t,
            solid_prop=solid_prop,
            melt_prop=melt_prop,
            vap_prop=vap_prop,
            U_np1=U_np1,
            lam_np1=lam_np1,
            U_n=U_n,
            M_SG_M0=M_SG_M0,
            M_SG_M1=M_SG_M1,
            K_SG_K0=K_SG_K0,
            K_SG_K1=K_SG_K1,
            M_bc=M_bc,
            K_bc=K_bc,
            solve_A=solve_A,
            spatial_op=op,
            params=params,
            mu_prev=None,
            F_SG_core=None,
        )
    return adj
def dot_test_one_step_wrapped(
    n,
    U_n,
    phi,
    adj,
    bc_idx,
    eps_list=([1e-4]),
    seed=0,
):
    rng = np.random.default_rng(seed)

    U_n = np.asarray(U_n, float).copy()
    ndofs = U_n.size

    v = rng.standard_normal(ndofs)
    w = rng.standard_normal(ndofs)
    v[bc_idx] = 0.0
    w[bc_idx] = 0.0

    if np.linalg.norm(v) > 0:
        v /= np.linalg.norm(v)
    if np.linalg.norm(w) > 0:
        w /= np.linalg.norm(w)

    U_np1 = np.asarray(phi(U_n.copy()), float)
    U_np1[bc_idx] = 0.0

    lam_n, *_ = adj(U_n.copy(), U_np1.copy(), w.copy())
    lam_n = np.asarray(lam_n, float)
    lam_n[bc_idx] = 0.0

    rhs = float(v @ lam_n)

    for eps in eps_list:
        Up = U_n + eps * v
        Um = U_n - eps * v
        Up[bc_idx] = 0.0
        Um[bc_idx] = 0.0

        Jv = (np.asarray(phi(Up)) - np.asarray(phi(Um))) / (2.0 * eps)
        Jv[bc_idx] = 0.0

        lhs = float(w @ Jv)
        abs_err = abs(lhs - rhs)
        rel_err = abs_err / max(1.0, abs(lhs), abs(rhs))

        print(
            f"[dot test step {n}] eps={eps:8.1e} "
            f"lhs={lhs:+.6e} rhs={rhs:+.6e} "
            f"abs={abs_err:.3e} rel={rel_err:.3e}"
        )
def F_SG_jax(Ucoeff, f0, f1,t, sqrt_lam=None):
    # Use local_params (snapshot for this kappa) so FD perturbation calls
    # with different kappa values don't corrupt each other's eigvecs.
    core = make_F_SG_core(params, f0, f1, t, sqrt_lam=sqrt_lam)
    return core(Ucoeff)

F_SG_jitted = jax.jit(F_SG_jax)

def F_SG_numpy(U,f0,f1,t):
    # use global f0,f1 or pass them from caller
    return np.asarray(F_SG_jitted(U, f0, f1, t,None), dtype=float)


phi = make_phi_one_step(
    t=0,
    M_bc=M_bc,
    K_bc=K_bc,
    solve_A=solve_A,
    solid_prop=SOLID,
    melt_prop=MELT,
    vap_prop=VAP,
    K_SG_K0=K_SG_K0,
    K_SG_K1=K_SG_K1,
    M_SG_M0=M_SG_M0,
    M_SG_M1=M_SG_M1,
    F_SG=F_SG_numpy,   # or whatever your source object is actually called
    spatial_op=spatial_op_obs,
)

adj = make_adjoint_one_step(
    t=0,
    solid_prop=SOLID,
    melt_prop=MELT,
    vap_prop=VAP,
    M_SG_M0=M_SG_M0,
    M_SG_M1=M_SG_M1,
    K_SG_K0=K_SG_K0,
    K_SG_K1=K_SG_K1,
    M_bc=M_bc,
    K_bc=K_bc,
    solve_A=solve_A,
    spatial_op=spatial_op_obs,
    params=params,
)
def build_source_coeffs_from_U(U, solid_prop, melt_prop, vap_prop):
    U_modes = U.reshape(P, num_nodes)
    Tmean_nodes = U_modes[0].reshape(Nx + 1, Ny + 1)

    _, S_elem, _ = melt_fraction_from_Tmean(Tmean_nodes)
    _, V_elem, _ = vap_fraction_from_Tmean(Tmean_nodes)

    Sbar = float(np.mean(S_elem))
    Vbar = float(np.mean(V_elem))

    f0_elem = (
        solid_prop["f0"]
        + (melt_prop["f0"] - solid_prop["f0"]) * S_elem
        + (vap_prop["f0"]  - melt_prop["f0"])  * V_elem
    )
    f1_eff = (
        solid_prop["f1"]
        + (melt_prop["f1"] - solid_prop["f1"]) * Sbar
        + (vap_prop["f1"]  - melt_prop["f1"])  * Vbar
    )
    f0_nodal = elem_to_node_weights(f0_elem, Nx, Ny).reshape(Nx + 1, Ny + 1)
    return f0_nodal, f1_eff, Tmean_nodes
def dot_test_source_direct_true(
    t,
    U,
    f0_nodal,
    f1_eff,
    params,
    bc_idx,
    eps=1e-6,
    seed=0,
):
    rng = np.random.default_rng(seed)

    U = np.asarray(U, float).copy()
    U[bc_idx] = 0.0

    v = rng.standard_normal(U.shape)
    w = rng.standard_normal(U.shape)
    v[bc_idx] = 0.0
    w[bc_idx] = 0.0

    if np.linalg.norm(v) > 0:
        v /= np.linalg.norm(v)
    if np.linalg.norm(w) > 0:
        w /= np.linalg.norm(w)

    BC_MASK = np.ones_like(U, dtype=np.float32)
    BC_MASK[bc_idx] = 0.0
    BC_MASK = jnp.asarray(BC_MASK)

    def F_map(U_):
        F = F_SG_apply(
            jnp.asarray(U_),
            jnp.asarray(f0_nodal),
            f1_eff,
            t,
            params["sqrt_lam"],
            params,
        )
        F = F * BC_MASK
        return np.asarray(F, float)

    Fp = F_map(U + eps * v)
    Fm = F_map(U - eps * v)
    Jv_fd = (Fp - Fm) / (2.0 * eps)
    lhs = float(w @ Jv_fd)

    @jax.jit
    def JTmu_true(U_, mu_, f0_, f1_, t_, sqrt_lam_, params_dict_):
        def scalar_fn(Ux):
            F = F_SG_apply(Ux, f0_, f1_, t_, sqrt_lam_, params_dict_)
            F = F * BC_MASK
            return jnp.vdot(mu_, F)
        return jax.grad(scalar_fn)(U_)

    JT_w = JTmu_true(
        jnp.asarray(U),
        jnp.asarray(w),
        jnp.asarray(f0_nodal),
        f1_eff,
        t,
        params["sqrt_lam"],
        params,
    )
    JT_w = np.asarray(JT_w, float)
    JT_w[bc_idx] = 0.0
    rhs = float(v @ JT_w)

    abs_err = abs(lhs - rhs)
    rel_err = abs_err / max(1.0, abs(lhs), abs(rhs))

    print(f"[source direct true] lhs={lhs:+.6e} rhs={rhs:+.6e} abs={abs_err:.3e} rel={rel_err:.3e}")
    return rel_err, lhs, rhs
run_dot_test=False
if run_dot_test ==True:
    for n in range(time_steps - 1):
        U_test = U_obs[n, :].copy()

        f0_nodal, f1_eff, _ = build_source_coeffs_from_U(
            U_test, SOLID, MELT, VAP
        )

        dot_test_source_direct_true(
            t=n,
            U=U_test,
            f0_nodal=f0_nodal,
            f1_eff=f1_eff,
            params=params,
            bc_idx=bc_idx,
            eps=0.1,
            seed=123 + n,
        )
    for n in range(time_steps - 1):

        phi_n = make_phi_one_step(
            t=n,
            M_bc=M_bc,
            K_bc=K_bc,
            solve_A=solve_A,
            solid_prop=SOLID,
            melt_prop=MELT,
            vap_prop=VAP,
            K_SG_K0=K_SG_K0,
            K_SG_K1=K_SG_K1,
            M_SG_M0=M_SG_M0,
            M_SG_M1=M_SG_M1,
            F_SG=F_SG_numpy,   # or your actual source object
            spatial_op=spatial_op_obs,
        )

        adj_n = make_adjoint_one_step(
            t=n,
            solid_prop=SOLID,
            melt_prop=MELT,
            vap_prop=VAP,
            M_SG_M0=M_SG_M0,
            M_SG_M1=M_SG_M1,
            K_SG_K0=K_SG_K0,
            K_SG_K1=K_SG_K1,
            M_bc=M_bc,
            K_bc=K_bc,
            solve_A=solve_A,
            spatial_op=spatial_op_obs,
            params=params,
        )

        dot_test_one_step_wrapped(
            n=n,
            U_n=U_obs[n, :].copy(),
            phi=phi_n,
            adj=adj_n,
            bc_idx=bc_idx,
            seed=123 + n,
        )


# ---------------------------------------------------------------------------
# Generate SG capability plots
# Add this block at the bottom of inf_layered_vap.py, after run_forward and
# make_T_obs_history_weighted have been called.
# ---------------------------------------------------------------------------
from plot_sg_capabilities import generate_plots

#generate_plots(
#    U_obs=U_obs,
#    T_obs_hist=T_obs_hist,
#    local_params=_obs_local_params,
#    y_vis_hist=y_vis_hist,
#    x_vis_hist=x_vis_hist,
#    w_vis_hist=w_vis_hist,
#    get_visible_weights_fn=get_visible_weights_from_xy,
#    eval_psi_fn=eval_psi,
#    multi_idx=multi_idx,
#    SOLID_obs=SOLID_obs,
#    Nx=Nx, Ny=Ny, Lx=Lx, Ly=Ly,
#    num_nodes=num_nodes, P=P, N_KL=N_KL,
#    T_abl=T_abl,
#    time_steps=time_steps, dt=dt,
#    T_melt_lo=T_melt_lo, T_melt_hi=T_melt_hi, Delta_melt=Delta_melt,  # <-- add these
#    T_vap_lo=T_vap_lo, T_vap_hi=T_vap_hi, Delta_vap=Delta_vap,
#)



# Visualize spatial weights (m1_elem, k1_elem, S_elem)
#Sbar_hist, k1_hist, m1_hist, Tmax_hist = visualize_spatial_weights_history(
#    U_obs, SOLID_obs, MELT_obs, melt_fraction_from_Tmean,
#    Nx, Ny, Lx, Ly, P, num_nodes,
#    save_dir="spatial_weights_plots"
#)

# Visualize variance field
#visualize_variance_field(
#    U_obs, Nx, Ny, Lx, Ly, P, num_nodes,
#    save_dir="variance_plots"
#)
#print("Observations history: ", T_obs_hist)

# In inf_layered_vap.py, after computing sigma2_obs_hist, add:
import matplotlib.pyplot as plt


J_opt = 0.0

wz = _trap_weights(y_nodes)
J_opt = 0.0
wz = _trap_weights(y_nodes)

for t in range(1, time_steps):
    u_t = U_obs[t]
    u_mean_2d = u_t[:num_nodes].reshape(Nx + 1, Ny + 1)

    # --- mean depth term ---
    h_u0, _, w_softmin, _, dH = softmin_depth(u_mean_2d, y_nodes, T_abl, eps_smooth, beta)
    g_flat = (w_softmin[:, np.newaxis] * wz[np.newaxis, :] * dH).ravel()

    r_mu = h_u0 - float(h_obs_hist[t])
    J_opt += 0.5 * r_mu**2 / float(sigma2_obs_hist[t])
    print(J_opt)
    # --- variance term ---
    U_modes = u_t.reshape(P, num_nodes)
    g_dot_uk = U_modes[1:] @ g_flat              # (P-1,)
    sigma2_pred = float(np.sum(g_dot_uk**2))
    r_var = sigma2_pred - float(sigma2_obs_hist[t])
    J_opt += 0.5 * r_var**2 *1e12
print("J: ", J_opt)

print("J opt: ", J_opt)
depth_obs_hist = make_depth_obs_history(
    U_obs,
    x, y,
    sigma_obs=1e-8, num_obs=50,
    multi_idx=multi_idx, eval_psi_func=eval_psi, N_KL=N_KL,
    T_abl=T_abl,
    restrict_to_visible=False,      # or True
    y_increases_with_depth=True,    # set to False if your y is negative downward
)
depth_mean = depth_obs_hist.mean(axis=1)
depth_var  = depth_obs_hist.var(axis=1, ddof=1)
print("depth mean", depth_mean)
print("depth var ", depth_var)




def build_KM_kl_dtheta_lists(
    eigvals_trunc,
    dlambda_dtheta,
    eigvecs_reshaped,
    dphi_dtheta=None,
):
    """
    Returns lists:
      Kd_list[m] : csr (num_nodes x num_nodes)   stiffness derivative for KL mode m
      Md_list[m] : csr (num_nodes x num_nodes)   mass derivative for KL mode m

    Includes:
      - eigenvalue contribution via dlambda_dtheta (your current 'grad_ell=True' path)
      - optional eigenvector contribution via dphi_dtheta (adds the missing term for kappa)

    Assumes global symbols exist (as in your code):
      Nx, Ny, N_KL, num_nodes, x, y, dx, dy, Lx, Ly,
      compute_local_stiffness, compute_local_mass_derivative_ell
    """

    eigvals_trunc   = np.asarray(eigvals_trunc, float)
    dlambda_dtheta  = np.asarray(dlambda_dtheta, float)

    N_KL_local = eigvals_trunc.shape[0]
    if dphi_dtheta is not None and dphi_dtheta.shape != eigvecs_reshaped.shape:
        raise ValueError(
            f"dphi_dtheta must match eigvecs_reshaped shape. "
            f"Got {dphi_dtheta.shape} vs {eigvecs_reshaped.shape}."
        )

    # ----------------------------
    # Local helper: Q1 mass matrix for coefficient field c(x,y)
    # (Rectangle element, 2×2 Gauss)
    # ----------------------------
    def _local_mass_rect_q1(x_coords, y_coords, coeff_func):
        xL, xR = float(np.min(x_coords)), float(np.max(x_coords))
        yB, yT = float(np.min(y_coords)), float(np.max(y_coords))
        hx = xR - xL
        hy = yT - yB
        if hx <= 0 or hy <= 0:
            raise ValueError("Non-positive element size.")

        xC = 0.5 * (xL + xR)
        yC = 0.5 * (yB + yT)

        g = 1.0 / np.sqrt(3.0)
        xis  = (-g, +g)
        etas = (-g, +g)

        # Node order consistent with your nodes_e:
        # 0:(xL,yB) 1:(xL,yT) 2:(xR,yB) 3:(xR,yT)
        def Nvals(xi, eta):
            return np.array([
                0.25*(1.0 - xi)*(1.0 - eta),  # (xL,yB)
                0.25*(1.0 - xi)*(1.0 + eta),  # (xL,yT)
                0.25*(1.0 + xi)*(1.0 - eta),  # (xR,yB)
                0.25*(1.0 + xi)*(1.0 + eta),  # (xR,yT)
            ], dtype=float)

        detJ = (hx * hy) / 4.0
        Me = np.zeros((4, 4), dtype=float)

        for xi in xis:
            for eta in etas:
                xp = xC + 0.5*hx*xi
                yp = yC + 0.5*hy*eta
                c = float(coeff_func(xp, yp))
                Nq = Nvals(xi, eta)
                Me += c * detJ * np.outer(Nq, Nq)

        return Me

    # ----------------------------
    # Precompute phi fields (and optionally dphi fields)
    # Note: your compute_local_stiffness currently uses only phi, not phi_x/phi_y,
    # but we keep the interface returning (phi,phi_x,phi_y) because your code expects it.
    # ----------------------------
    phi_cache = []
    for m in range(N_KL_local):
        phi = eigvecs_reshaped[:, :, m]
        phi_x, phi_y = np.gradient(phi, dx, dy)

        if dphi_dtheta is not None:
            dphi = dphi_dtheta[:, :, m]
            dphi_x, dphi_y = np.gradient(dphi, dx, dy)
        else:
            dphi = dphi_x = dphi_y = None

        phi_cache.append((phi, phi_x, phi_y, dphi, dphi_x, dphi_y))

    # ----------------------------
    # Build stiffness derivative lists
    # ----------------------------
    Kd_list = []
    for m in range(N_KL_local):
        phi, phi_x, phi_y, dphi, dphi_x, dphi_y = phi_cache[m]

        def phi_grad_func(xp, yp, _phi=phi, _phi_x=phi_x, _phi_y=phi_y):
            i = min(max(int(np.floor(xp / Lx * Nx)), 0), Nx-1)
            j = min(max(int(np.floor(yp / Ly * Ny)), 0), Ny-1)
            return _phi[i, j], _phi_x[i, j], _phi_y[i, j]

        K_kl_dtheta = lil_matrix((num_nodes, num_nodes))

        # (A) eigenvalue contribution: uses your grad_ell=True branch
        for ix in range(Nx):
            for iy in range(Ny):
                node1 = ix * (Ny + 1) + iy
                node2 = node1 + 1
                node3 = (ix + 1) * (Ny + 1) + iy
                node4 = node3 + 1
                nodes_e = [node1, node2, node3, node4]

                x_coords = np.array([x[ix], x[ix + 1], x[ix + 1], x[ix]], dtype=float)
                y_coords = np.array([y[iy], y[iy], y[iy + 1], y[iy + 1]], dtype=float)

                Ke_lam = compute_local_stiffness(
                    x_coords, y_coords,
                    k_func=lambda _x, _y: 0.0,
                    k1=1.0,
                    phi_grad_func=phi_grad_func,
                    m=m,
                    eigvals_trunc=eigvals_trunc,
                    deigvals_trunc_dell=dlambda_dtheta,
                    grad_ell=True,
                )

                for a in range(4):
                    ra = nodes_e[a]
                    for b in range(4):
                        K_kl_dtheta[ra, nodes_e[b]] += Ke_lam[a, b]

        # (B) eigenvector contribution: call base branch with phi := dphi
        # This adds k1*sqrt(lambda_m)*dphi_m inside k, which is the missing term.
        if dphi_dtheta is not None:
            zeros = np.zeros_like(dlambda_dtheta)

            def dphi_grad_func(xp, yp, _dphi=dphi, _dphi_x=dphi_x, _dphi_y=dphi_y):
                i = min(max(int(np.floor(xp / Lx * Nx)), 0), Nx-1)
                j = min(max(int(np.floor(yp / Ly * Ny)), 0), Ny-1)
                return _dphi[i, j], _dphi_x[i, j], _dphi_y[i, j]

            for ix in range(Nx):
                for iy in range(Ny):
                    node1 = ix * (Ny + 1) + iy
                    node2 = node1 + 1
                    node3 = (ix + 1) * (Ny + 1) + iy
                    node4 = node3 + 1
                    nodes_e = [node1, node2, node3, node4]

                    x_coords = np.array([x[ix], x[ix + 1], x[ix + 1], x[ix]], dtype=float)
                    y_coords = np.array([y[iy], y[iy], y[iy + 1], y[iy + 1]], dtype=float)

                    Ke_phi = compute_local_stiffness(
                        x_coords, y_coords,
                        k_func=lambda _x, _y: 0.0,
                        k1=1.0,
                        phi_grad_func=dphi_grad_func,     # <-- swapped
                        m=m,
                        eigvals_trunc=eigvals_trunc,
                        deigvals_trunc_dell=zeros,        # no eigenvalue part here
                        grad_ell=False,                   # <-- base branch
                    )

                    for a in range(4):
                        ra = nodes_e[a]
                        for b in range(4):
                            K_kl_dtheta[ra, nodes_e[b]] += Ke_phi[a, b]

        Kd_list.append(K_kl_dtheta.tocsr())

    # ----------------------------
    # Build mass derivative lists
    # ----------------------------
    Md_list = []
    for m in range(N_KL_local):
        phi, _, _, dphi, _, _ = phi_cache[m]

        def phi_func(xp, yp, _phi=phi):
            i = min(max(int(np.floor(xp / Lx * Nx)), 0), Nx-1)
            j = min(max(int(np.floor(yp / Ly * Ny)), 0), Ny-1)
            return _phi[i, j]

        M_kl_dtheta = lil_matrix((num_nodes, num_nodes))

        # (A) eigenvalue contribution: your existing routine
        for ix in range(Nx):
            for iy in range(Ny):
                node1 = ix * (Ny + 1) + iy
                node2 = node1 + 1
                node3 = (ix + 1) * (Ny + 1) + iy
                node4 = node3 + 1
                nodes_e = [node1, node2, node3, node4]

                x_coords = np.array([x[ix], x[ix + 1], x[ix + 1], x[ix]], dtype=float)
                y_coords = np.array([y[iy], y[iy], y[iy + 1], y[iy + 1]], dtype=float)

                Me_lam = compute_local_mass_derivative_ell(
                    x_coords, y_coords,
                    phi_func,
                    m, eigvals_trunc, dlambda_dtheta
                )

                for a in range(4):
                    ra = nodes_e[a]
                    for b in range(4):
                        M_kl_dtheta[ra, nodes_e[b]] += Me_lam[a, b]

        # (B) eigenvector contribution: add sqrt(lambda_m) * dphi_m term
        if dphi_dtheta is not None:
            w_m = np.sqrt(max(float(eigvals_trunc[m]), 0.0))

            def dphi_func(xp, yp, _dphi=dphi):
                i = min(max(int(np.floor(xp / Lx * Nx)), 0), Nx-1)
                j = min(max(int(np.floor(yp / Ly * Ny)), 0), Ny-1)
                return _dphi[i, j]

            for ix in range(Nx):
                for iy in range(Ny):
                    node1 = ix * (Ny + 1) + iy
                    node2 = node1 + 1
                    node3 = (ix + 1) * (Ny + 1) + iy
                    node4 = node3 + 1
                    nodes_e = [node1, node2, node3, node4]

                    x_coords = np.array([x[ix], x[ix + 1], x[ix + 1], x[ix]], dtype=float)
                    y_coords = np.array([y[iy], y[iy], y[iy + 1], y[iy + 1]], dtype=float)

                    Me_phi = _local_mass_rect_q1(
                        x_coords, y_coords,
                        coeff_func=lambda _x, _y, wm=w_m: wm * dphi_func(_x, _y)
                    )

                    for a in range(4):
                        ra = nodes_e[a]
                        for b in range(4):
                            M_kl_dtheta[ra, nodes_e[b]] += Me_phi[a, b]

        Md_list.append(M_kl_dtheta.tocsr())

    return Kd_list, Md_list


def build_KM_kl_dtheta_lists_all(
    eigvals_trunc,     # (N_KL,)
    dlambda_dtheta,    # (N_KL, n_theta)
    eigvecs_reshaped,  # (Nx+1, Ny+1, N_KL)
    dphi_dtheta,       # (Nx+1, Ny+1, N_KL, n_theta)
):
    """
    Build Kd and Md sparse matrix lists for ALL parameters at once.
    The phi_cache (gradients of eigenvectors) is computed once and reused
    across all n_theta parameters — that's the key saving vs calling
    build_KM_kl_dtheta_lists n_theta times separately.

    Returns
    -------
    Kd_lists : list of length n_theta, each element is a list of N_KL sparse matrices
    Md_lists : list of length n_theta, each element is a list of N_KL sparse matrices
    """
    n_theta = dlambda_dtheta.shape[1]
    Kd_lists = []
    Md_lists = []
    for l in range(n_theta):
        Kd_l, Md_l = build_KM_kl_dtheta_lists(
            eigvals_trunc=eigvals_trunc,
            dlambda_dtheta=dlambda_dtheta[:, l],
            eigvecs_reshaped=eigvecs_reshaped,
            dphi_dtheta=dphi_dtheta[:, :, :, l],
        )
        Kd_lists.append(Kd_l)
        Md_lists.append(Md_l)
    return Kd_lists, Md_lists
def apply_SG_dell_matrix_free(v, A_list):
    """
    Computes y = (sum_m kron(G_list[m], A_list[m])) v, without forming the kron sum.
    DOF ordering assumed: [mode0_nodes, mode1_nodes, ..., modeP-1_nodes]
    so v reshapes to (P, num_nodes).
    """
    V = v.reshape(P, num_nodes)  # (P, N)
    Y = np.zeros_like(V)

    # For each KL mode: y_i += sum_j G_ij * (A @ v_j)
    for m in range(N_KL):
        A = A_list[m]           # (N,N) sparse
        Gm = G_list[m]          # (P,P) sparse
        AV = np.vstack([A.dot(V[j]) for j in range(P)])  # (P,N)
        Y += Gm.dot(AV)         # (P,N)

    return Y.reshape(-1)
def apply_SG_dell_matrix_free(v, A_list):
    """
    Computes y = (sum_m kron(G_list[m], A_list[m])) @ v
    No sqrt-weight scaling — for derivative matrices that are already fully assembled.
    """
    V = v.reshape(P, num_nodes)
    Y = np.zeros_like(V)
    for m in range(N_KL):
        A  = A_list[m]
        Gm = G_list[m]
        AV = np.vstack([A.dot(V[j]) for j in range(P)])
        Y += Gm.dot(AV)
    return Y.reshape(-1)
def apply_SG_dell_matrix_free_weighted(v, A_list, weight_sqrt):
    """
    Computes y = D^{1/2} @ (sum_m kron(G_list[m], A_list[m])) @ D^{1/2} @ v
    """
    V = v.reshape(P, num_nodes)
    Y = np.zeros_like(V)
   
    # Scale input by sqrt weights
    V_scaled = V * weight_sqrt[None, :]

    for m in range(N_KL):
        A = A_list[m]
        Gm = G_list[m]
        AV = np.vstack([A.dot(V_scaled[j]) for j in range(P)])
        # Scale output by sqrt weights
        AV_scaled = AV * weight_sqrt[None, :]
        Y += Gm.dot(AV_scaled)

    return Y.reshape(-1)

def build_Kd_Md_lists_consistent(eigvals_trunc, dlambda_dkappa, dphi_dkappa,
                                  eigvecs_reshaped=None):
    """
    Build derivative matrices using the same Gauss quadrature as the forward operator.

    eigvecs_reshaped : (Nx+1, Ny+1, N_KL) — current eigenvectors at theta.
        Must be passed explicitly so that the dw * K_m^base term uses the
        correct base matrices (not a stale global cache from a different kappa).
    """
    global _K_mode_base, _M_mode_base

    if eigvecs_reshaped is None:
        # Fallback: use whatever is currently cached (legacy path, may be stale)
        if _K_mode_base is None:
            raise ValueError("build_Kd_Md_lists_consistent: eigvecs_reshaped must be "
                             "supplied when _K_mode_base is not yet built.")
    else:
        # Always rebuild from the supplied eigvecs so the base is consistent with
        # the kappa point at which derivatives are being evaluated.
        _K_mode_base, _M_mode_base = _assemble_mode_bases(
            eigvals_trunc, eigvecs_reshaped, phi_gradients=None
        )

    sqrt_lam = np.sqrt(np.maximum(eigvals_trunc, 0.0))
    dw_all   = 0.5 * dlambda_dkappa / np.maximum(sqrt_lam[:, None], 1e-30)
    n_theta  = dlambda_dkappa.shape[1]

    Kd_lists = []
    Md_lists = []
    for l in range(n_theta):
        # dphi contribution: assemble base matrices for dphi_l as if it were a new phi
        dphi_l = dphi_dkappa[:, :, :, l]   # (Nx+1, Ny+1, N_KL)
        K_dphi_base, M_dphi_base = _assemble_mode_bases(
            eigvals_trunc, dphi_l, phi_gradients=None
        )
        # dw contribution uses _K_mode_base rebuilt above (correct kappa point)
        Kd_l = [dw_all[m, l] * _K_mode_base[m]
                + sqrt_lam[m] * K_dphi_base[m]
                for m in range(N_KL)]
        Md_l = [dw_all[m, l] * _M_mode_base[m]
                + sqrt_lam[m] * M_dphi_base[m]
                for m in range(N_KL)]
        Kd_lists.append(Kd_l)
        Md_lists.append(Md_l)

    return Kd_lists, Md_lists
def compute_adjoint_grad_kappa_phase_matrixfree_all(
    U_hist, Mu_hist,
    solid_prop, melt_prop, vap_prop,
    eigvals_trunc,
    eigvecs_reshaped,
    dlambda_dkappa,
    dphi_dkappa,
    local_params,          # ← ADD
    freeze_phase=False,
    include_forcing_dphi=True,
    coo=None,
):
    """
    Compute dJ/dkappa for ALL kappa parameters simultaneously.
    Returns g_kappa : (n_theta,) gradient vector.
    """
    # Default: treat vapour same as melt (no effect when rho_vap=0)
    if vap_prop is None:
        vap_prop = melt_prop

    eigvals_trunc  = np.asarray(eigvals_trunc, float)   # (N_KL,)
    dlambda_dkappa = np.asarray(dlambda_dkappa, float)  # (N_KL, n_theta)
    n_theta        = dlambda_dkappa.shape[1]

    sqrt_lam = np.sqrt(np.maximum(eigvals_trunc, 0.0))  # (N_KL,)

    # dw[m, l] = 0.5 * dlambda[m,l] / sqrt_lam[m]
    dw_all = 0.5 * dlambda_dkappa / np.maximum(sqrt_lam[:, None], 1e-30)

    # Build Kd/Md derivative operator lists for all parameters at once.
    # Pass eigvecs_reshaped explicitly so the dw*K_m^base term is evaluated
    # at the correct kappa point, not a stale global cache.
    Kd_lists, Md_lists = build_Kd_Md_lists_consistent(
        eigvals_trunc,
        dlambda_dkappa,
        dphi_dkappa,
        eigvecs_reshaped=eigvecs_reshaped,
    )
    #Kd_lists, Md_lists = build_KM_kl_dtheta_lists_all(
    #    eigvals_trunc, dlambda_dkappa,
    #    eigvecs_reshaped,        # (Nx+1, Ny+1, N_KL)
    #    dphi_dkappa,             # (Nx+1, Ny+1, N_KL, n_theta)
    #)
    g_kappa = np.zeros(n_theta)

    # Optionally freeze phase at first step
    if freeze_phase:
        U_modes0  = np.asarray(U_hist[1], float).reshape(P, num_nodes)
        Tmean0    = U_modes0[0].reshape(Nx + 1, Ny + 1)
        _, S_elem_frozen, _ = melt_fraction_from_Tmean(Tmean0)
        _, V_elem_frozen, _ = vap_fraction_from_Tmean(Tmean0)
        Sbar_frozen = float(np.mean(S_elem_frozen))
        Vbar_frozen = float(np.mean(V_elem_frozen))

    for n in range(time_steps - 1):
        u_np1 = np.asarray(U_hist[n + 1], dtype=float).copy()
        u_n   = np.asarray(U_hist[n],     dtype=float).copy()
        mu_n  = np.asarray(Mu_hist[n],    dtype=float).copy()
        du    = u_np1 - u_n

        u_np1[bc_idx] = 0.0
        u_n[bc_idx]   = 0.0
        mu_n[bc_idx]  = 0.0

        # Phase fractions — consistent with forward solver (use u_np1)
        U_modes     = u_n.reshape(P, num_nodes)
        Tmean_nodes = U_modes[0].reshape(Nx + 1, Ny + 1)

        if freeze_phase:
            S_elem = S_elem_frozen
            V_elem = V_elem_frozen
            Sbar   = Sbar_frozen
            Vbar   = Vbar_frozen
        else:
            _, S_elem, _ = melt_fraction_from_Tmean(Tmean_nodes)
            _, V_elem, _ = vap_fraction_from_Tmean(Tmean_nodes)
            Sbar = float(np.mean(S_elem))
            Vbar = float(np.mean(V_elem))

        Sbar = float(np.mean(S_elem))

        # ---- vaporisation fraction at current mean temperature ----
        _, V_elem, dV_dT_elem = vap_fraction_from_Tmean(Tmean_nodes)
        Vbar = float(np.mean(V_elem))

        # ---- effective (global) source + stochastic amplitudes ----
        # Three-phase blend: solid -> melt -> vapour
        f0_eff = (solid_prop["f0"] + (melt_prop["f0"] - solid_prop["f0"]) * Sbar
                  + (vap_prop["f0"]  - melt_prop["f0"])  * Vbar)
        f1_eff = (solid_prop["f1"] + (melt_prop["f1"] - solid_prop["f1"]) * Sbar
                  + (vap_prop["f1"]  - melt_prop["f1"])  * Vbar)
        k1_elem = (solid_prop["k1"] + (melt_prop["k1"] - solid_prop["k1"]) * S_elem
                   + (vap_prop["k1"]  - melt_prop["k1"])  * V_elem)
        m1_elem = (solid_prop["m1"] + (melt_prop["m1"] - solid_prop["m1"]) * S_elem
                   + (vap_prop["m1"]  - melt_prop["m1"])  * V_elem)+  vap_prop["rho_vap1"] * dV_dT_elem


        # Three-phase f1 for forcing term
        f1_eff = (solid_prop["f1"]
                  + (melt_prop["f1"] - solid_prop["f1"]) * Sbar
                  + (vap_prop["f1"]  - melt_prop["f1"])  * Vbar)


                # k1/m1 sqrt weights — same symmetric weighting as apply_K1/apply_M1
        k1_nodes       = elem_to_node_weights(k1_elem, Nx, Ny)      # (num_nodes,)
        m1_nodes       = elem_to_node_weights(m1_elem, Nx, Ny)      # (num_nodes,)
        k1_sqrt_nodes  = np.sqrt(np.maximum(k1_nodes, 1e-30))       # (num_nodes,)
        m1_sqrt_nodes  = np.sqrt(np.maximum(m1_nodes, 1e-30))       # (num_nodes,)

        # Tile to SG space
        k1_sqrt_full = np.tile(k1_sqrt_nodes, P)   # (P*num_nodes,)
        m1_sqrt_full = np.tile(m1_sqrt_nodes, P)

        # Scale both u and mu by sqrt(k1), matching forward apply_K1:
        # K1 u = diag(k1_sqrt) * (Σ G_m⊗K_m) * diag(k1_sqrt) * u
        u_k = k1_sqrt_full * u_np1
        u_m = m1_sqrt_full * du
        mu_k = k1_sqrt_full * mu_n
        mu_m = m1_sqrt_full * mu_n


        for l in range(n_theta):

            # Forcing gradient (dF/dkappa via dw and dphi)
            if include_forcing_dphi:
                dw_l   = jnp.asarray(dw_all[:, l])          # (N_KL,)
                dphi_l = dphi_dkappa[:, :, :, l]            # (Nx+1, Ny+1, N_KL)
                F_dw_core = make_F_SG_core(local_params, f0=0.0, f1=1.0,   # ← was params
                                            t=n, sqrt_lam=dw_l)
                dF = np.asarray(F_dw_core(u_n), dtype=float)
                dF += forcing_dF_from_dphi(local_params, sqrt_lam, t=n,     # ← was params
                                            Ucoeff=u_n, dphi_reshaped=dphi_l)
                dF[bc_idx] = 0.0
                g_kappa[l] += dt * f1_eff * float(mu_n @ dF)

            # Stiffness and mass derivative contributions
            dK_u  = apply_SG_dell_matrix_free(u_k,  Kd_lists[l])
            dM_du = apply_SG_dell_matrix_free(u_m,  Md_lists[l])

            g_kappa[l] -= dt * float(mu_k @ dK_u)
            g_kappa[l] -=      float(mu_m @ dM_du)

    return g_kappa

Mu_hist, lam_hist, J_total, J_hist = run_adjoint_depth(
    np.zeros_like(U_obs), U_obs,M_bc, K_bc, solve_A,
    K_SG_K0, K_SG_K1, M_SG_M0, M_SG_M1,
    SOLID_obs, MELT_obs, VAP_obs, spatial_op_obs, params, bc_idx,
    Nx, Ny, Ly, num_nodes, P, T_abl, adjoint_one_step, np.zeros_like(sigma2_obs_hist),h_obs_hist = np.zeros_like(h_obs_hist), eps=eps_smooth, sigma_d=sigma_d
)
_diff = SPDEKLDifferentiator(Nx, Ny, Lx, Ly, N_KL, kappa_param_obs)
# update kappa_param values so derivatives are computed at kappa_cur
_res = _diff.derivatives(theta_kappa_obs)

eigvals_trunc = np.asarray(_res.eigvals, float)
eigvecs_grid  = np.asarray(_res.eigvecs, float).reshape(Nx+1, Ny+1, N_KL)
dlambda_all   = np.asarray(_res.dlambda, float)          # (N_KL, n_kappa)
dphi_all_grid = np.asarray(_res.dphi,    float).reshape(Nx+1, Ny+1, N_KL, -1)

g_phase = adjoint_grad_all_phase(
    np.zeros_like(U_obs), Mu_hist, SOLID, MELT,
    K_SG_K1=K_SG_K1, M_SG_M1=M_SG_M1,
    forcing_param_grads_numpy=forcing_param_grads_numpy,
    spatial_op=spatial_op_obs, freeze_phase=False, vap_prop = VAP
)
g_kappa_adj = compute_adjoint_grad_kappa_phase_matrixfree_all(
    np.zeros_like(U_obs), Mu_hist,
    SOLID, MELT, VAP,
    eigvals_trunc=eigvals_trunc,
    eigvecs_reshaped=eigvecs_grid,
    dlambda_dkappa=dlambda_all,
    dphi_dkappa=dphi_all_grid,
    local_params=params,    # ← ADD
    freeze_phase=False,
    include_forcing_dphi=True,
    coo=_diff.coo
)
#from plots_adjoint import generate_adjoint_plots
 
#generate_adjoint_plots(
#    Mu_hist             = Mu_hist,
#    U_obs               = U_obs,
#    g_phase             = g_phase,
#    g_kappa             = g_kappa_adj,
#    J_hist              = J_hist,
#    Nx=Nx, Ny=Ny, Lx=Lx, Ly=Ly,
#    num_nodes           = num_nodes,
#    P                   = P,
#    time_steps          = time_steps,
#    dt                  = dt,
#    kappa_param_names   = ["rho_vap0", "rho_vap1", "kappa_surface", "kappa_deep", "y_trans", "width"]
#,
#)


#dot_test_unsteady_phase()
print("U_obs vs U_hist diff:", np.abs(U_obs).max())


# ─────────────────────────────────────────────────────────────────────────────
# Scenario 2: LAB PRIOR on kappa only — the physically motivated case
# This is the one that matters for your paper's argument:
#   rho_vap params:  uninformative — these are constrained by mean depth alone
#                    (your Hessian analysis showed H_var is blind to them)
#   kappa_surface:   ±35% from lab GMRF fit
#   kappa_deep:      ±22% from lab core (competent granite, tighter)
#   y_trans, width:  uninformative — in-situ unknowns, depth data informs these
# ─────────────────────────────────────────────────────────────────────────────
#prior_lab_kappa = GaussianLogPrior(
#    theta_prior = theta_lab,
#    sigma_log   = np.array([
 #       np.inf,  # rho_vap0: H_var is blind to this — mean depth already fixed it
#        np.inf,  # rho_vap1: same
#        0.30,    # kappa_surface: ±35% from thin-section GMRF fit
#        0.20,    # kappa_deep:    ±22% from core sample GMRF fit
#        np.inf,  # y_trans: UNKNOWN in-situ — what depth data tells you
#        np.inf,  # width:   UNKNOWN in-situ — ditto
#    ]),
#)

# ─────────────────────────────────────────────────────────────────────────────
# Scenario 3: FULL PRIOR — including rho_vap from physical bounds
# rho_vap0 ~ 5e8 kg/m3 is the vapour density at ablation threshold.
# Literature range for rock vapour: order-of-magnitude uncertain.
# rho_vap1 controls temperature dependence — very poorly known.
# ─────────────────────────────────────────────────────────────────────────────
from run_inf_depth import GaussianLogPrior
prior_full = GaussianLogPrior(
    theta_prior = theta_lab,
    sigma_log   = np.array([
        1.0,    # rho_vap0: ±170% — order-of-magnitude uncertain
        0.4,    
        0.30,   # kappa_surface: ±35%
        0.20,   # kappa_deep:    ±22%
        0.50,   # y_trans: ±60% rough stratigraphic prior
        0.70,   # width:   ±100% highly uncertain
    ]),
)
def run_inf():
    iters=20
    theta_kappa_cur= theta_kappa_obs   # fix kappa at truth to test rho_vap0 alone
    # use obs physics as base; only rho_vap0 is the unknown
    VAP_base = {**VAP_obs, 'rho_vap0': VAP['rho_vap0'], 'rho_vap1': VAP['rho_vap1']}
    SOLID_cur, MELT_cur, VAP_cur = SOLID_obs.copy(), MELT_obs.copy(), VAP_base.copy()
    therm_cur = np.array([VAP_cur['rho_vap0'], VAP_cur['rho_vap1']], dtype=float)
    lr_kappa = 0

    lr_therm = [0,0]
    for i in range(iters):
        J_cur, g_therm_adj, g_kappa_adj = validate_depth_adjoint_fd(dx,dy,U_obs,
            run_forward,
            U0,
            SOLID_cur, MELT_cur, VAP_cur,
            ell, theta_kappa_cur,
            Nx, Ny, Ly, num_nodes, P, T_abl,
            adjoint_one_step,
            adjoint_grad_all_phase,
            forcing_param_grads_numpy,
            _clear_all_kappa_caches,#
            bc_idx, params,
            M_bc, K_bc, solve_A,
            K_SG_K0, K_SG_K1, M_SG_M0, M_SG_M1,
            sigma2_obs_hist,
            spatial_op_obs, h_obs_hist, kappa_param=kappa_param_init,
            compute_adjoint_grad_kappa_fn=compute_adjoint_grad_kappa_phase_matrixfree_all,
            Lx=Lx,
            N_KL=N_KL, sigma_d = sigma_d, eps_smooth=eps_smooth, run_fd_check=False)
        g_therm_arr = np.array([g_therm_adj.get('rho_vap0', 0.0),
                                g_therm_adj.get('rho_vap1', 0.0)], dtype=float)
        if i==0:
            lr_therm[0] = abs((0.6* VAP["rho_vap0"])/g_therm_arr[0] )
           # lr_therm[1] = abs((0.4*VAP["rho_vap1"])/g_therm_arr[1] )
        theta_kappa_cur -= g_kappa_adj * lr_kappa
        therm_cur       -= g_therm_arr * lr_therm
        VAP_cur["rho_vap0"] -= lr_therm[0]* g_therm_arr[0]
        VAP_cur["rho_vap1"] -= lr_therm[1]* g_therm_arr[1]

        print("iter: ", i, " j: " ,J_cur, " grad_kappa: ", g_kappa_adj, "grad_therm: ", g_therm_adj  )
        print("current kappa: ", theta_kappa_cur)
        print("current therm: ", therm_cur)

                                                                                                                                                    
def run_inf_lbfgs(mean_round_iters=30, var_round_iters=20, prior=None):
    from scipy.optimize import minimize

    OBJ_SCALE = 1e6   # rescale so gradients are O(1e-1) — allows L-BFGS-B line search to take proper steps

    VAP_base = {**VAP_obs,                                                                                                                                            
                'rho_vap0': VAP['rho_vap0'],              
                'rho_vap1': VAP['rho_vap1']}                                                                                                                          

    # ── Stage 1: mean-only, rho_vap0 + f0_v free ────────────────────────────────
    def obj_mean(x_log):
        rho0 = float(np.exp(x_log[0]))
        f0_v = float(np.exp(x_log[1]))
        VAP_cur = {**VAP_base, 'rho_vap0': rho0, 'f0': f0_v}

        J, g_therm_adj, _ = validate_depth_adjoint_fd(
            dx, dy, U_obs,
            run_forward, U0,
            SOLID_obs, MELT_obs, VAP_cur,
            ell, theta_kappa_obs,
            Nx, Ny, Ly, num_nodes, P, T_abl,
            adjoint_one_step,
            adjoint_grad_all_phase,
            forcing_param_grads_numpy,
            _clear_all_kappa_caches,
            bc_idx, params,
            M_bc, K_bc, solve_A,
            K_SG_K0, K_SG_K1, M_SG_M0, M_SG_M1,
            sigma2_obs_hist,
            spatial_op_obs, h_obs_hist,
            kappa_param=kappa_param_obs,
            compute_adjoint_grad_kappa_fn=compute_adjoint_grad_kappa_phase_matrixfree_all,
            Lx=Lx, N_KL=N_KL, sigma_d=sigma_d, eps_smooth=eps_smooth,
            run_fd_check=False, mean_only=True)

        g_log = np.array([
            g_therm_adj.get('rho_vap0', 0.0) * rho0,
            g_therm_adj.get('f0_v',     0.0) * f0_v,
        ])

        if prior is not None:
            theta1     = np.array([rho0, f0_v])
            theta1_ref = np.array([VAP['rho_vap0'], VAP['f0']])
            sigma1     = np.array([3.0, 0.5])
            log_ratio  = np.log(theta1 / theta1_ref)
            nll_p      = 0.5 * np.sum((log_ratio / sigma1) ** 2)
            grad_p     = log_ratio / sigma1 ** 2   # d(nll)/d(log theta)
         #   J         += nll_p
         #   g_log     += grad_p

        print(f"  [mean] J={J:.4e}  rho_vap0={rho0:.4e} (truth {VAP_obs['rho_vap0']:.2e})"
              f"  f0_v={f0_v:.4f} (truth {VAP_obs['f0']:.4f})  g={g_log}")
        return float(J) * OBJ_SCALE, g_log * OBJ_SCALE

    print("\n=== Stage 1: mean-only, rho_vap0 + f0_v ===")
    res1 = minimize(
        obj_mean,
        x0=np.log([VAP_base['rho_vap0'], VAP['f0']]),
        jac=True, method='L-BFGS-B',
        bounds=[
            (np.log(7e8),   np.log(1e10)),   # rho_vap0
            (np.log(0.05),  np.log(0.4)),    # f0_v
        ],
        options={'maxiter': mean_round_iters, 'ftol': 1e-10, 'gtol': 2e-4, 'maxcor':1})

    rho_vap0_fixed = float(np.exp(res1.x[0]))
    f0_v_fixed     = float(np.exp(res1.x[1]))
    print(f"  Stage 1 done: rho_vap0={rho_vap0_fixed:.4e}  f0_v={f0_v_fixed:.4f}"
          f"  J={res1.fun/OBJ_SCALE:.4e}  {res1.message}")

    # ── Stage 2: mean+variance, rho_vap1 + kappa free ────────────────────────
    def obj_var(x_log):
        rho1        = float(np.exp(x_log[0]))
        theta_kappa = np.exp(x_log[1:])
        kappa_cur   = SigmoidLayeredKappa(
            Ny=Ny, Ly=Ly,
            kappa_surface=float(theta_kappa[0]),
            kappa_deep=float(theta_kappa[1]),
            y_transition=float(theta_kappa[2]),
            width=float(theta_kappa[3])
        )
        VAP_cur = {**VAP_base, 'rho_vap0': rho_vap0_fixed, 'f0': f0_v_fixed, 'rho_vap1': rho1}
                                                                                                                                                                    
        J, g_therm_adj, g_kappa_adj = validate_depth_adjoint_fd(                                                                                                      
            dx, dy, U_obs,
            run_forward, U0,                                                                                                                                          
            SOLID_obs, MELT_obs, VAP_cur,                 
            ell, theta_kappa,
            Nx, Ny, Ly, num_nodes, P, T_abl,
            adjoint_one_step,                                                                                                                                         
            adjoint_grad_all_phase,
            forcing_param_grads_numpy,                                                                                                                                
            _clear_all_kappa_caches,                      
            bc_idx, params,
            M_bc, K_bc, solve_A,                                                                                                                                      
            K_SG_K0, K_SG_K1, M_SG_M0, M_SG_M1,
            sigma2_obs_hist,                                                                                                                                          
            spatial_op_obs, h_obs_hist,                   
            kappa_param=kappa_cur,                                                                                                                                    
            compute_adjoint_grad_kappa_fn=compute_adjoint_grad_kappa_phase_matrixfree_all,
            Lx=Lx, N_KL=N_KL, sigma_d=sigma_d, eps_smooth=eps_smooth,                                                                                                 
            run_fd_check=False, mean_only=False)                                                                                                                      

        # chain rule for all 5 parameters
        g_log = np.concatenate([
                [g_therm_adj.get('rho_vap1', 0.0) * rho1],
                g_kappa_adj * theta_kappa
        ])
        if prior is not None:
            theta_full = np.array([rho_vap0_fixed, rho1, *theta_kappa])
            nll_p, grad_p_theta = prior(theta_full)
            J += nll_p
            g_log += grad_p_theta[1:] * theta_full[1:]   # log-space chain rule, skip fixed rho_vap0
        print("g_log", g_log)                                                                                                                             
        print(f"  [var]  J={J:.4e}  rho_vap1={rho1:.4e} (truth {VAP_obs['rho_vap1']:.2e})"
            f"  kappa={theta_kappa.round(3)}")                                                                                                                      
        return float(J) * OBJ_SCALE, g_log * OBJ_SCALE
                                                                                                                                                                    
    x0_var = np.log([                                                                                                                                                 
        VAP_base['rho_vap1'],
        theta_kappa_init[0],   # kappa_surface                                                                                                                         
        theta_kappa_init[1],   # kappa_deep                
        theta_kappa_init[2],   # y_trans
        theta_kappa_init[3],   # width                                                                                                                                 
    ])
    bounds_var = [
        (np.log(1e5),       np.log(1e12)),       # rho_vap1
        (np.log(20),        np.log(150)),         # kappa_surface
        (np.log(20),        np.log(150)),         # kappa_deep
        (np.log(0.0*Ly),    np.log(Ly)),      # y_trans
        (np.log(0.01*Ly),   np.log(2.0*Ly)),      # width
    ]

    print("\n=== Stage 2: mean+variance, rho_vap1 + kappa ===")
    res2 = minimize(
        obj_var, x0_var, jac=True, method='L-BFGS-B',
        bounds=bounds_var,
        options={'maxiter': var_round_iters, 'ftol': 1e-15, 'gtol': 1e-8})

    theta2 = np.exp(res2.x)
    print(f"\nFinal result:")
    print(f"  rho_vap0   = {rho_vap0_fixed:.4e}  (truth {VAP_obs['rho_vap0']:.2e})")
    print(f"  f0_v       = {f0_v_fixed:.4f}      (truth {VAP_obs['f0']:.4f})")
    print(f"  rho_vap1   = {theta2[0]:.4e}  (truth {VAP_obs['rho_vap1']:.2e})")
    print(f"  kappa_surf = {theta2[1]:.4e}  (truth {theta_kappa_obs[0]:.2e})")
    print(f"  kappa_deep = {theta2[2]:.4e}  (truth {theta_kappa_obs[1]:.2e})")
    print(f"  y_trans    = {theta2[3]:.4e}  (truth {theta_kappa_obs[2]:.2e})")
    print(f"  width      = {theta2[4]:.4e}  (truth {theta_kappa_obs[3]:.2e})")
    print(f"  J_final    = {res2.fun/OBJ_SCALE:.4e}  {res2.message}")
    return {'stage1': res1, 'stage2': res2}

run_inf_lbfgs(prior=prior_full)

validate_depth_adjoint_fd(dx,dy,U_obs,
    run_forward,
    U0,
    SOLID, MELT, VAP,
    ell, theta_kappa_init,
    Nx, Ny, Ly, num_nodes, P, T_abl,
    adjoint_one_step,
    adjoint_grad_all_phase,
    forcing_param_grads_numpy,
    _clear_all_kappa_caches,#
    bc_idx, params,
    M_bc, K_bc, solve_A,    
    K_SG_K0, K_SG_K1, M_SG_M0, M_SG_M1,
    sigma2_obs_hist,
    spatial_op_obs, h_obs_hist, kappa_param=kappa_param_init,
    compute_adjoint_grad_kappa_fn=compute_adjoint_grad_kappa_phase_matrixfree_all,
    Lx=Lx,
    N_KL=N_KL, sigma_d = sigma_d, eps_smooth=eps_smooth,mean_only=True)



#run_inf()

prior_none = None
result = run_depth_inference(
    # data
    U_obs=U_obs,
    h_obs_hist=h_obs_hist,
    U0=U0,
    # starting point
    SOLID_init=SOLID,
    MELT_init=MELT,
    kappa_init=theta_kappa_init,
    VAP_init=VAP,
    ell=ell,
    prior=prior_none,
    # callables
    run_forward_fn=run_forward,
    run_adjoint_depth_fn=run_adjoint_depth,
    adjoint_one_step_fn=adjoint_one_step,
    adjoint_grad_all_phase_fn=adjoint_grad_all_phase,
    compute_adjoint_grad_kappa_fn=compute_adjoint_grad_kappa_phase_matrixfree_all,
    forcing_param_grads_numpy_fn=forcing_param_grads_numpy,
 #   make_depth_obs_history_fn=make_depth_obs_history,
    clear_caches_fn=_clear_all_kappa_caches,
    kappa_param=kappa_param_init,
    SPDEKLDifferentiator_cls=SPDEKLDifferentiator,
    # mesh
    bc_idx=bc_idx,
    params=params,
    Nx=Nx, Ny=Ny, Lx=Lx, Ly=Ly,
    N_KL=N_KL, num_nodes=num_nodes, P=P, time_steps=time_steps, sigma2_obs_hist= sigma2_obs_hist,
    # objective
    T_abl=T_abl,
    eps_smooth = eps_smooth,
    sigma_d=sigma_d,
    max_iter=50,
    beta=0.005
)
from run_inf_depth import validate_obj_and_grad
validate_obj_and_grad(
    U_obs, h_obs_hist, U0,
    SOLID, MELT, theta_kappa_init,
    VAP, ell,
    run_forward,
    run_adjoint_depth,
    adjoint_one_step,
    adjoint_grad_all_phase,
    compute_adjoint_grad_kappa_phase_matrixfree_all,
    forcing_param_grads_numpy,
    #make_depth_obs_history_fn,
    _clear_all_kappa_caches,
     kappa_param_init,
    SPDEKLDifferentiator,
    bc_idx, params,
    Nx, Ny, Lx, Ly, N_KL, num_nodes, P, time_steps, sigma2_obs_hist,
    T_abl=T_abl,
    eps_smooth=10.0,
    beta=None,
    sigma_d=sigma_d,
    evap_range=0.0,
    eps_fd=1e-4,
)
from scipy.optimize import minimize





plot_convergence(result, kappa_true=theta_kappa_obs)


# ============================================================================
# HESSIAN / IDENTIFIABILITY ANALYSIS
# Paste this block into inf_layered_vap.py after run_depth_inference
# ============================================================================
from hessian_depth import (
    gauss_newton_hessian,
    hessian_fd_of_gradient,
    identifiability_from_hessian,
    check_hessian_consistency,
    PARAM_NAMES,
    _eval_gradient,
)

# ── Converged parameters from final logged iteration ─────────────────────────
# rhovap_0=5e8  rhovap_1=9832405.4  k1=36.094  k2=24.895  k3=0.071  k4=0.010
theta_star = np.array([
    5.00000000e+08,   # rho_vap0
    9.83240540e+06,   # rho_vap1
    3.61054327e+01,   # kappa_surface
    2.48895419e+01,   # kappa_deep
    7.07845501e-02,   # y_trans
    1.00000000e-02,   # width
])

# ── Shared kwargs passed to _eval_gradient ───────────────────────────────────
# VAP_base is your VAP dict WITHOUT the rho_vap values (they come from theta).
# We pass VAP as VAP_base; _build_vap will overwrite rho_vap0/1 from theta.
_grad_kwargs = dict(
    U_obs=U_obs,
    h_obs_hist=h_obs_hist,
    sigma2_obs_hist=sigma2_obs_hist,
    U0=U0,
    SOLID=SOLID, MELT=MELT, VAP_base=VAP, ell=ell,
    run_forward_fn=run_forward,
    run_adjoint_depth_fn=run_adjoint_depth,
    adjoint_grad_all_phase_fn=adjoint_grad_all_phase,
    compute_adjoint_grad_kappa_fn=compute_adjoint_grad_kappa_phase_matrixfree_all,
    forcing_param_grads_numpy_fn=forcing_param_grads_numpy,
    SPDEKLDifferentiator_cls=SPDEKLDifferentiator,
    clear_caches_fn=_clear_all_kappa_caches,
    kappa_param=kappa_param_init,
    bc_idx=bc_idx, params=params,
    Nx=Nx, Ny=Ny, Lx=Lx, Ly=Ly,
    N_KL=N_KL, num_nodes=num_nodes, P=P, time_steps=time_steps,
    sigma_d=sigma_d,
    T_abl=T_abl, eps_smooth=eps_smooth, beta=0.005,
)

# ── 1. Gauss-Newton Hessian (cheap, SPD, no adjoints) ────────────────────────
print("\n" + "="*64)
print("  GAUSS-NEWTON HESSIAN  (FD of depth QoI, 2x6 forward solves)")
print("="*64)
H_gn, J_mat, J_mean, J_var= gauss_newton_hessian(
    theta_star,
    U_obs=U_obs, h_obs_hist=h_obs_hist, sigma2_obs_hist=sigma2_obs_hist, U0=U0,
    SOLID=SOLID, MELT=MELT, VAP_base=VAP, ell=ell,
    run_forward_fn=run_forward,
    SPDEKLDifferentiator_cls=SPDEKLDifferentiator,
    clear_caches_fn=_clear_all_kappa_caches,
    kappa_param=kappa_param_init,
    bc_idx=bc_idx, params=params,
    Nx=Nx, Ny=Ny, Lx=Lx, Ly=Ly,
    N_KL=N_KL, num_nodes=num_nodes, P=P, time_steps=time_steps,
    sigma_d=sigma_d, T_abl=T_abl, eps_smooth=eps_smooth, beta=0.005,
    verbose=True,
)
ident_gn = identifiability_from_hessian(H_gn, theta_star, PARAM_NAMES, verbose=True)
H_var = J_var[1:].T @ J_var[1:]
ident_var = identifiability_from_hessian(H_var, theta_star, PARAM_NAMES, verbose=True)
# ── 2. Full Hessian via FD of the adjoint gradient (2x6 forward+adjoint) ─────
print("\n" + "="*64)
print("  FULL HESSIAN  (FD of adjoint gradient, 2x6 grad evaluations)")
print("="*64)
H_full, g_star = hessian_fd_of_gradient(
    theta_star,
    grad_kwargs=_grad_kwargs,
    rel_step=1e-3,
    verbose=True,
)
ident_full = identifiability_from_hessian(H_full, theta_star, PARAM_NAMES, verbose=True)

# ── 3. Consistency check ─────────────────────────────────────────────────────
check_hessian_consistency(H_gn, H_full)

# ── 4. Decision table ────────────────────────────────────────────────────────
print("\n  PARAMETER DECISIONS (full Hessian):")
print(f"  {'param':>14s}  {'CR-std':>10s}  {'|theta|':>10s}  {'ratio':>8s}  action")
print("  " + "-" * 68)
for i, name in enumerate(PARAM_NAMES):
    cr  = ident_full["cr_std"][i]
    pri = abs(theta_star[i])
    r   = cr / max(pri, 1e-30)
    if r < 0.1:
        action = "infer freely"
    elif r < 1.0:
        action = "infer with informative prior"
    else:
        action = "FIX -- depth obs cannot identify"
    print(f"  {name:>14s}  {cr:>10.3e}  {pri:>10.3e}  {r:>8.2e}  {action}")

# ── 5. Save ───────────────────────────────────────────────────────────────────
np.savez(
    "hessian_results.npz",
    theta_star=theta_star,
    H_gn=H_gn,
    H_full=H_full,
    J_mat=J_mat,
    eigvals_gn=ident_gn["eigvals"],
    eigvals_full=ident_full["eigvals"],
    cr_std_gn=ident_gn["cr_std"],
    cr_std_full=ident_full["cr_std"],
    param_names=np.array(PARAM_NAMES),
)
print("\nSaved hessian_results.npz")

