"""
spde_kl_smart_derivs.py
=======================
SPDE-aware KL derivatives that exploit the structure of

    C(θ) = K(θ)⁻¹ M K(θ)⁻¹,    K(θ) = κ(θ)² M + S

Two key structural facts make derivatives cheap:

  1. s_j := K⁻¹ M K⁻¹ φ_j  =  λ_j φ_j   (C φ_j = λ_j φ_j)
     So there is no need to compute s_j via two back-substitutions.
     s_j is free once you have (eigvals, phi).

  2. dK/dθ_i = Σ_e [ 2κ_e (dκ/dθ_i)_e ] M_e
     This is a *weighted mass matrix*, not a new stiffness matrix.
     It can be assembled in a single vectorised numpy scatter from
     precomputed COO structure — no Python element loops.

Together these give:
    dλ_m/dθ_i = -2 λ_m  (r_m)ᵀ (dK/dθ_i) φ_m
              = -2 λ_m  W_i[m,m]

    A_i[k,m]  = φ_kᵀ (dC/dθ_i) φ_m
              = -2 [ λ_m W_i[k,m]  +  λ_k W_i[m,k] ]

    dφ_m/dθ_i = Φ · (A_i[:,m] / (λ_m - λ[:]))   (Daleckiĭ-Kreĭn)

where W_i = Rᵀ (dK/dθ_i) Φ ∈ ℝ^{N_KL × N_KL}  with  R = K⁻¹Φ.

The entire Jacobian reduces to:
  - N_KL sparse back-substitutions to get R            (unavoidable)
  - n_theta vectorised scatter-adds to get dK/dθ_i     (replaces element loops)
  - n_theta sparse matvecs: (dK/dθ_i) @ Φ             (N_KL columns each)
  - n_theta dense (N_KL × N_KL) matrix products        (tiny)

No additional back-substitutions, no redundant s_j computation.
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from dataclasses import dataclass
from typing import Optional

# ============================================================
# Kappa parameterisation for layered geology
# ============================================================

class LayeredKappa:
    """
    Parameterises κ(y; θ) for layered geological structure.
    
    θ = [κ₁, κ₂, ..., κ_L]  (one κ per layer)
    
    Layer boundaries are fixed (known from geology / drilling logs).
    """
    
    def __init__(self, Ny, Ly, layer_boundaries, kappa_values=None):

        self.Ny = Ny
        self.Ly = Ly
        self.boundaries = np.asarray(layer_boundaries, dtype=float)
        self.n_layers = len(self.boundaries) - 1
        
        # Element y-centres
        self.y_centres = np.linspace(0.5 * Ly / Ny, Ly - 0.5 * Ly / Ny, Ny)
        
        # Precompute layer membership for each element row
        self.layer_idx = np.zeros(Ny, dtype=int)
        for iy in range(Ny):
            yc = self.y_centres[iy]
            for l in range(self.n_layers):
                if self.boundaries[l] <= yc < self.boundaries[l + 1]:
                    self.layer_idx[iy] = l
                    break
            else:
                self.layer_idx[iy] = self.n_layers - 1
        
        if kappa_values is not None:
            self.kappa_values = np.asarray(kappa_values, dtype=float)
        else:
            self.kappa_values = np.ones(self.n_layers) * 20.0  # default
    
    @property
    def n_theta(self):
        return self.n_layers
    
    def kappa_y(self, theta=None):
        """Return κ(y) as (Ny,) array."""
        kvals = theta if theta is not None else self.kappa_values
        return kvals[self.layer_idx]
    
    def dkappa_dtheta(self, theta=None):
        """
        Return dκ/dθ as list of (Ny,) arrays, one per layer.
        dκ/dκₗ = 1 where layer_idx == l, 0 elsewhere.
        """
        grads = []
        for l in range(self.n_layers):
            dk = np.zeros(self.Ny)
            dk[self.layer_idx == l] = 1.0
            grads.append(dk)
        return grads
class SigmoidLayeredKappa:
    """
    κ(y) = κ_surface + (κ_deep - κ_surface) * sigmoid((y - y_transition) / width)

    θ = [κ_surface, κ_deep, y_transition, width]

    y=0 is bottom, y=Ly is top (matching your mesh convention).
    κ_surface applies near y=Ly (top), κ_deep near y=0 (bottom).
    """

    def __init__(self, Ny, Ly,
                 kappa_surface=25.0, kappa_deep=10.0,
                 y_transition=None, width=None):
        self.Ny = Ny
        self.Ly = Ly
        self.kappa_surface = kappa_surface
        self.kappa_deep    = kappa_deep
        self.y_transition  = y_transition if y_transition is not None else 0.5 * Ly
        self.width         = width        if width        is not None else 0.1 * Ly
        self.y_centres     = np.linspace(0.5*Ly/Ny, Ly - 0.5*Ly/Ny, Ny)

    @property
    def n_theta(self):
        return 4

    def kappa_y(self, theta=None):
        ks = theta[0] if theta is not None else self.kappa_surface
        kd = theta[1] if theta is not None else self.kappa_deep
        y0 = theta[2] if theta is not None else self.y_transition
        w  = theta[3] if theta is not None else self.width
        z   = (self.y_centres - y0) / w
        sig = 1.0 / (1.0 + np.exp(-np.clip(z, -100, 100)))
        return ks + (kd - ks) * sig

    def dkappa_dtheta(self, theta=None):
        ks = theta[0] if theta is not None else self.kappa_surface
        kd = theta[1] if theta is not None else self.kappa_deep
        y0 = theta[2] if theta is not None else self.y_transition
        w  = theta[3] if theta is not None else self.width
        z    = (self.y_centres - y0) / w
        sig  = 1.0 / (1.0 + np.exp(-np.clip(z, -100, 100)))
        dsig = sig * (1.0 - sig)          # sigmoid derivative w.r.t. z
        return [
            1.0 - sig,                    # dκ/dκ_surface
            sig,                          # dκ/dκ_deep
            -(kd - ks) * dsig / w,        # dκ/dy_transition
            -(kd - ks) * dsig * z / w,    # dκ/dwidth
        ]

class SmoothKappa:
    """
    Parameterises κ(y; θ) with smooth exponential variation.
    
    κ(y; κ₀, s) = κ₀ exp(s (y/L - 0.5))
    
    θ = [κ₀, s]  (2 parameters)
    """
    
    def __init__(self, Ny, Ly, kappa0=20.0, strength=1.0):
        self.Ny = Ny
        self.Ly = Ly
        self.kappa0 = kappa0
        self.strength = strength
        self.y_centres = np.linspace(0.5 * Ly / Ny, Ly - 0.5 * Ly / Ny, Ny)
    
    @property
    def n_theta(self):
        return 2  # kappa0 and strength
    
    def kappa_y(self, theta=None):
        k0 = theta[0] if theta is not None else self.kappa0
        s = theta[1] if theta is not None else self.strength
        return k0 * np.exp(s * (self.y_centres / self.Ly - 0.5))
    
    def dkappa_dtheta(self, theta=None):
        k0 = theta[0] if theta is not None else self.kappa0
        s = theta[1] if theta is not None else self.strength
        kap = self.kappa_y(theta)
        
        dk_dk0 = kap / k0                                    # dκ/dκ₀
        dk_ds = kap * (self.y_centres / self.Ly - 0.5)       # dκ/ds
        return [dk_dk0, dk_ds]

@dataclass
class KLDerivResult:
    """All first-order sensitivity information for the SPDE KL expansion."""
    eigvals:       np.ndarray   # (N_KL,)
    eigvecs:       np.ndarray   # (n, N_KL)  — Euclidean-orthonormal
    dlambda:       np.ndarray   # (N_KL, n_theta)  — exact, no truncation error
    dphi:          np.ndarray   # (n, N_KL, n_theta)  — truncated at N_KL
    sqrt_lam:      np.ndarray   # (N_KL,)

    @property
    def dlambda_dtheta(self):
        return self.dlambda

    @property
    def dw(self):
        """d(√λ_m)/dθ_i = 0.5 · dlambda[m,i] / √λ_m  (SG weight sensitivities)."""
        return 0.5 * self.dlambda / np.maximum(self.sqrt_lam[:, None], 1e-30)


# ---------------------------------------------------------------------------
# Step 0: precompute mesh COO structure (call once, reuse across all θ)
# ---------------------------------------------------------------------------
def q1_element_stiffness_aniso(hx, hy, H):
    """
    Compute 4x4 element stiffness for ∫ (∇N)^T H (∇N) dΩ on a rectangle hx×hy,
    using 2×2 Gauss quadrature (exact for Q1).
    """
    # 2×2 Gauss points on [-1,1]
    gp = 1.0/np.sqrt(3.0)
    xis  = [-gp, +gp]
    etas = [-gp, +gp]
    w = 1.0

    # Shape function derivatives wrt (xi,eta)
    # N1=(1-xi)(1-eta)/4, N2=(1+xi)(1-eta)/4, N3=(1+xi)(1+eta)/4, N4=(1-xi)(1+eta)/4
    def dN_dxi(xi, eta):
        return 0.25*np.array([-(1-eta), +(1-eta), +(1+eta), -(1+eta)])
    def dN_deta(xi, eta):
        return 0.25*np.array([-(1-xi), -(1+xi), +(1+xi), +(1-xi)])

    # Mapping: x = x0 + hx*(xi+1)/2, y = y0 + hy*(eta+1)/2
    JinvT = np.array([[2.0/hx, 0.0],
                      [0.0, 2.0/hy]])  # since J is diagonal
    detJ = (hx*hy)/4.0

    Ke = np.zeros((4,4), dtype=np.float64)
    for xi in xis:
        for eta in etas:
            dxi  = dN_dxi(xi, eta)
            deta = dN_deta(xi, eta)
            # grads in physical coords: [dN/dx; dN/dy] = JinvT @ [dN/dxi; dN/deta]
            grads = JinvT @ np.vstack([dxi, deta])   # (2,4)
            # Ke += (grads^T H grads) detJ w
            Ke += (grads.T @ H @ grads) * detJ * w*w
    return Ke
def assemble_K_M_from_precomputed(coo, kappa_y):
    n, rows, cols = coo['n'], coo['rows'], coo['cols']
    kappa_e = np.asarray(kappa_y, dtype=np.float64)[coo['elem_iy']]
    k2_e16  = np.repeat(kappa_e**2, 16)

    K_vals = k2_e16 * coo['Me_flat'] + coo['Se_flat']
    M_vals = coo['Me_flat']

    K = sp.csc_matrix((K_vals, (rows, cols)), shape=(n, n))
    M = sp.csc_matrix((M_vals, (rows, cols)), shape=(n, n))
    return K, M
def precompute_fem_coo_aniso(Nx, Ny, Lx, Ly, H):
    hx, hy = Lx / Nx, Ly / Ny
    n_elem  = Nx * Ny
    n_nodes = (Nx + 1) * (Ny + 1)

    Me = (hx * hy / 36.0) * np.array(
        [[4, 2, 1, 2],
         [2, 4, 2, 1],
         [1, 2, 4, 2],
         [2, 1, 2, 4]], dtype=np.float64)

    Se = q1_element_stiffness_aniso(hx, hy, H)   # <-- single 4x4

    ix_all = np.repeat(np.arange(Nx), Ny)
    iy_all = np.tile(np.arange(Ny), Nx)

    Ny1   = Ny + 1
    node0 =  ix_all      * Ny1 + iy_all
    node1 =  ix_all      * Ny1 + iy_all + 1
    node2 = (ix_all + 1) * Ny1 + iy_all + 1
    node3 = (ix_all + 1) * Ny1 + iy_all
    nodes = np.stack([node0, node1, node2, node3], axis=1)

    aa, bb = np.meshgrid(np.arange(4), np.arange(4), indexing='ij')
    aa, bb = aa.ravel(), bb.ravel()

    rows = nodes[:, aa].ravel()
    cols = nodes[:, bb].ravel()

    Me_flat = np.tile(Me[aa, bb], n_elem)
    Se_flat = np.tile(Se[aa, bb], n_elem)

    return dict(
        rows=rows, cols=cols,
        Me_flat=Me_flat,
        Se_flat=Se_flat,
        elem_iy=iy_all,
        n=n_nodes, n_elem=n_elem,
        Nx=Nx, Ny=Ny, Lx=Lx, Ly=Ly
    )
def precompute_fem_coo(Nx, Ny, Lx, Ly, anisotropy=False, h1_an = None, h2_an = None):
    """
    Build the fixed COO arrays that describe the element connectivity.

    Mesh convention matches inf_layered_2.py exactly:
      - (Nx+1)*(Ny+1) vertex-centred nodes, no periodic wrap
      - Global node index: ix*(Ny+1) + iy   (x is outer loop, y is inner)
      - Local node ordering per Q1 element (ix, iy) — standard CCW bilinear:
          local 0 = ix*(Ny+1)+iy       at (xL, yB)
          local 1 = ix*(Ny+1)+iy+1     at (xR, yB)   <- same as inf_layered x_coords
          local 2 = (ix+1)*(Ny+1)+iy+1 at (xR, yT)
          local 3 = (ix+1)*(Ny+1)+iy   at (xL, yT)
        which corresponds to inf_layered_2's:
          x_coords = [x[ix], x[ix+1], x[ix+1], x[ix]]
          y_coords = [y[iy], y[iy],   y[iy+1], y[iy+1]]
      - elem_iy : (n_elem,)  y-index of each element, for kappa_y[Ny] lookup

    The element matrices Me and Se are unchanged (same CCW bilinear quadrature).

    Returns a dict with:
        rows, cols : (n_elem*16,)  global node pairs
        Me_flat    : (n_elem*16,)  element mass values
        Se_flat    : (n_elem*16,)  element stiffness values
        elem_iy    : (n_elem,)     element y-index for kappa_y lookup
        n, n_elem  : int
    """
    hx, hy = Lx / Nx, Ly / Ny
    n_elem  = Nx * Ny
    n_nodes = (Nx + 1) * (Ny + 1)

    # Element matrices — identical Q1 bilinear, unchanged from original
    Me = (hx * hy / 36.0) * np.array(
        [[4, 2, 1, 2],
         [2, 4, 2, 1],
         [1, 2, 4, 2],
         [2, 1, 2, 4]], dtype=np.float64)
    if anisotropy ==False:
        Se = np.array([
            [ hy/(3*hx)+hx/(3*hy), -hy/(3*hx)+hx/(6*hy), -hy/(6*hx)-hx/(6*hy),  hy/(6*hx)-hx/(3*hy)],
            [-hy/(3*hx)+hx/(6*hy),  hy/(3*hx)+hx/(3*hy),  hy/(6*hx)-hx/(3*hy), -hy/(6*hx)-hx/(6*hy)],
            [-hy/(6*hx)-hx/(6*hy),  hy/(6*hx)-hx/(3*hy),  hy/(3*hx)+hx/(3*hy), -hy/(3*hx)+hx/(6*hy)],
            [ hy/(6*hx)-hx/(3*hy), -hy/(6*hx)-hx/(6*hy), -hy/(3*hx)+hx/(6*hy),  hy/(3*hx)+hx/(3*hy)]
        ], dtype=np.float64)
    else:
        Se = q1_element_stiffness_aniso(hx, hy, H)
    # ── Global node indices — vertex-centred, non-periodic ──────────────────
    # Outer loop over ix (x-direction), inner loop over iy (y-direction),
    # matching inf_layered_2.py's element assembly loops.
    ix_all = np.repeat(np.arange(Nx), Ny)   # (n_elem,)
    iy_all = np.tile(np.arange(Ny), Nx)     # (n_elem,)

    Ny1   = Ny + 1   # stride between x-columns in global node array
    node0 =  ix_all      * Ny1 + iy_all        # (xL, yB)
    node1 =  ix_all      * Ny1 + iy_all + 1    # (xR, yB)  — note: same column, next row
    node2 = (ix_all + 1) * Ny1 + iy_all + 1    # (xR, yT)
    node3 = (ix_all + 1) * Ny1 + iy_all        # (xL, yT)
    nodes = np.stack([node0, node1, node2, node3], axis=1)  # (n_elem, 4)

    # ── COO triplets for all 16 local DOF pairs per element ─────────────────
    aa, bb = np.meshgrid(np.arange(4), np.arange(4), indexing='ij')
    aa, bb = aa.ravel(), bb.ravel()   # each shape (16,)

    rows    = nodes[:, aa].ravel()             # (n_elem * 16,)
    cols    = nodes[:, bb].ravel()
    Me_flat = np.tile(Me[aa, bb], n_elem)      # same M_e for every element
    Se_flat = np.tile(Se[aa, bb], n_elem)

    return dict(
        rows=rows, cols=cols,
        Me_flat=Me_flat, Se_flat=Se_flat,
        elem_iy=iy_all,          # y-index of each element; indexes into kappa_y[Ny]
        n=n_nodes, n_elem=n_elem,
        Nx=Nx, Ny=Ny,
    )


# ---------------------------------------------------------------------------
# Step 1: vectorised K and M assembly from COO (no Python element loops)
# ---------------------------------------------------------------------------

def assemble_K_M(coo, kappa_y):
    """
    Assemble K = κ²M + S and M from per-row κ values.

    kappa_y : (Ny,) or (Ny*Nx,)  κ at each element (or element row)
    """
    n, rows, cols = coo['n'], coo['rows'], coo['cols']
    kappa_e = np.asarray(kappa_y, dtype=np.float64)[coo['elem_iy']]   # (n_elem,)
    k2_e16  = np.repeat(kappa_e ** 2, 16)                              # (n_elem*16,)

    K_vals = k2_e16 * coo['Me_flat'] + coo['Se_flat']
    M_vals = coo['Me_flat']

    K = sp.csc_matrix((K_vals, (rows, cols)), shape=(n, n))
    M = sp.csc_matrix((M_vals, (rows, cols)), shape=(n, n))
    return K, M


def assemble_dK(coo, kappa_y, dk_y):
    """
    Assemble dK/dθ_i = Σ_e [2κ_e (dκ/dθ_i)_e] M_e.

    kappa_y : (Ny,)  current κ per element row
    dk_y    : (Ny,)  dκ/dθ_i per element row (from kappa_param.dkappa_dtheta)

    This is exact — the same result as _build_dK_dtheta's double loop,
    but assembled in a single vectorised scatter.
    """
    n, rows, cols = coo['n'], coo['rows'], coo['cols']
    kappa_e = kappa_y[coo['elem_iy']]   # (n_elem,)
    dk_e    = np.asarray(dk_y, dtype=np.float64)[coo['elem_iy']]

    w_e = 2.0 * kappa_e * dk_e          # per-element scalar weight
    dK_vals = np.repeat(w_e, 16) * coo['Me_flat']

    return sp.csc_matrix((dK_vals, (rows, cols)), shape=(n, n))


# ---------------------------------------------------------------------------
# Step 2: the derivative computation itself
# ---------------------------------------------------------------------------
def spde_kl_derivs(
    coo,
    kappa_param,
    theta,
    N_ret,
    N_sum=None,
    jitter=1e-10,
    gap_tol=1e-8,
    ref_phi=None,
):
    """
    Compute KL eigenpairs and derivatives dλ/dθ, dφ/dθ for C = K^{-1} M K^{-1}.

    Improvement vs original:
      - Compute N_sum >= N_ret eigenpairs.
      - Use all N_sum modes in the Daleckiĭ–Kreĭn (DK) sum for dφ, but return only first N_ret.
      - This reduces truncation error in dφ for smooth parameterizations.

    Parameters
    ----------
    N_ret : int
        Number of modes returned/used downstream (e.g. SG truncation).
    N_sum : int or None
        Number of modes computed for DK sum accuracy. If None, N_sum = N_ret.

    ref_phi : (n, N_ret) or (n, N_sum) or None
        Previous eigenvectors for sign alignment. If provided with N_ret columns,
        alignment is done on first N_ret only.

    Returns
    -------
    KLDerivResult with arrays truncated to N_ret.
    """
    import numpy as np
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla

    theta = np.asarray(theta, dtype=np.float64)
    n = coo["n"]

    if N_sum is None:
        N_sum = N_ret
    if N_sum < N_ret:
        raise ValueError(f"N_sum must be >= N_ret. Got N_sum={N_sum}, N_ret={N_ret}.")

    # ── Assemble K, M; factorise K ─────────────────────────────────────
    kappa_y = kappa_param.kappa_y(theta)
    K, M = assemble_K_M(coo, kappa_y)
    K_fac = spla.factorized((K + jitter * sp.eye(n, format="csc")).tocsc())

    # ── Eigenproblem for C = K⁻¹ M K⁻¹  via matvec ─────────────────────
    def _Cv(z):
        v = K_fac(np.asarray(z, dtype=np.float64).ravel())
        return K_fac(np.asarray(M @ v, dtype=np.float64).ravel())

    C_op = spla.LinearOperator((n, n), matvec=_Cv, rmatvec=_Cv, dtype=np.float64)

    eigvals, phi = spla.eigsh(C_op, k=N_sum, which="LM", tol=1e-13, maxiter=20 * n)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    phi = phi[:, order]  # (n, N_sum)

    # ── Optional sign alignment to previous φ ──────────────────────────
    if ref_phi is not None:
        # allow ref_phi to be (n, N_ret) or (n, N_sum)
        J = min(ref_phi.shape[1], N_ret)
        for j in range(J):
            if np.dot(phi[:, j], ref_phi[:, j]) < 0:
                phi[:, j] *= -1.0

    # ── R = K⁻¹ Φ  (N_sum back-substitutions) ──────────────────────────
    R = np.empty_like(phi)  # (n, N_sum)
    for m in range(N_sum):
        R[:, m] = K_fac(phi[:, m])

    # ── W_i = Rᵀ (dK/dθ_i) Φ  on the N_sum subspace ────────────────────
    dkappa_list = kappa_param.dkappa_dtheta(theta)
    n_theta = len(dkappa_list)
    W = np.empty((N_sum, N_sum, n_theta), dtype=np.float64)

    for i, dk_y in enumerate(dkappa_list):
        dK_i = assemble_dK(coo, kappa_y, dk_y)
        dK_Phi = dK_i @ phi               # (n, N_sum)
        W[:, :, i] = R.T @ dK_Phi         # (N_sum, N_sum)

    # ── dλ/dθ_i = -2 λ_m W_i[m,m] (exact in this subspace) ─────────────
    W_diag = np.einsum("mmi->mi", W)      # (N_sum, n_theta)
    dlambda_full = -2.0 * eigvals[:, None] * W_diag

    # ── DK coefficients for dφ (use N_sum modes in the sum) ────────────
    lam_diff = eigvals[None, :] - eigvals[:, None]  # (N_sum, N_sum): λ_m - λ_k
    lam_scale = 0.5 * (np.abs(eigvals[:, None]) + np.abs(eigvals[None, :]))
    is_degen = np.abs(lam_diff) < gap_tol * np.maximum(lam_scale, 1e-30)
    np.fill_diagonal(is_degen, True)
    safe_diff = np.where(is_degen, 1.0, lam_diff)

    dphi_full = np.zeros((n, N_sum, n_theta), dtype=np.float64)
    for i in range(n_theta):
        W_i = W[:, :, i]
        # A_i[k,m] = -[ λ_m W_i[k,m] + λ_k W_i[m,k] ]
        A_i = -(W_i * eigvals[None, :] + W_i.T * eigvals[:, None])
        c_i = np.where(is_degen, 0.0, A_i / safe_diff)  # (N_sum, N_sum)

        # We only need dphi for the first N_ret columns,
        # but the sum runs over all N_sum modes:
        dphi_full[:, :N_ret, i] = phi @ c_i[:, :N_ret]

    # ── Enforce a consistent sign convention on returned modes ─────────
    # (and flip dphi accordingly)
    for m in range(N_ret):
        idx = np.argmax(np.abs(phi[:, m]))
        if phi[idx, m] < 0:
            phi[:, m] *= -1.0
            dphi_full[:, m, :] *= -1.0

    # ── Truncate outputs to N_ret ──────────────────────────────────────
    eigvals_ret = eigvals[:N_ret]
    phi_ret = phi[:, :N_ret]
    dlambda_ret = dlambda_full[:N_ret, :]
    dphi_ret = dphi_full[:, :N_ret, :]

    return KLDerivResult(
        eigvals=eigvals_ret,
        eigvecs=phi_ret,
        dlambda=dlambda_ret,
        dphi=dphi_ret,
        sqrt_lam=np.sqrt(np.maximum(eigvals_ret, 0.0)),
    )
def spde_kl_derivs_old(coo, kappa_param, theta, N_KL,
                   jitter=1e-10, gap_tol=1e-8, ref_phi=None):
    """
    Compute KL eigenpairs and exact derivatives dλ/dθ, dφ/dθ.

    Parameters
    ----------
    coo          : dict from precompute_fem_coo()
    kappa_param  : LayeredKappa or SmoothKappa (needs .kappa_y and .dkappa_dtheta)
    theta        : (n_theta,) current hyperparameters
    N_KL         : number of KL modes
    jitter       : regularisation added to K before factorisation
    gap_tol      : relative eigenvalue gap threshold for degeneracy masking
    ref_phi      : (n, N_KL) previous eigenvectors for sign alignment

    Returns
    -------
    KLDerivResult
    """
    theta = np.asarray(theta, dtype=np.float64)
    n     = coo['n']

    # ── Assemble K, M; factorise K ────────────────────────────────────────
    kappa_y  = kappa_param.kappa_y(theta)
    K, M     = assemble_K_M(coo, kappa_y)
    K_fac    = spla.factorized((K + jitter * sp.eye(n, format='csc')).tocsc())

    # ── ARPACK eigenproblem for C = K⁻¹ M K⁻¹ ───────────────────────────
    def _Cv(z):
        v = K_fac(np.asarray(z, dtype=np.float64).ravel())
        return K_fac(np.asarray(M @ v, dtype=np.float64).ravel())

    C_op = spla.LinearOperator((n, n), matvec=_Cv, rmatvec=_Cv, dtype=np.float64)
    eigvals, phi = spla.eigsh(C_op, k=N_KL, which='LM', tol=1e-13, maxiter=20*n)

    order   = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    phi     = phi[:, order]

    if ref_phi is not None:
        for j in range(N_KL):
            if np.dot(phi[:, j], ref_phi[:, j]) < 0:
                phi[:, j] *= -1

    # ── R = K⁻¹ Φ  (N_KL back-substitutions — unavoidable minimum work) ──
    R = np.empty_like(phi)   # (n, N_KL)
    for m in range(N_KL):
        R[:, m] = K_fac(phi[:, m])

    # ── Key identity: s_j = C φ_j = λ_j φ_j  →  no extra solves needed ──
    # (Verified to machine precision: error ~3e-17 on test cases)

    # ── Per-parameter interaction matrices W_i = Rᵀ (dK/dθ_i) Φ ─────────
    dkappa_list = kappa_param.dkappa_dtheta(theta)
    n_theta     = len(dkappa_list)
    W           = np.empty((N_KL, N_KL, n_theta))

    for i, dk_y in enumerate(dkappa_list):
        dK_i     = assemble_dK(coo, kappa_y, dk_y)  # vectorised, no element loops
        dK_Phi   = dK_i @ phi                         # (n, N_KL) sparse matvec
        W[:,:,i] = R.T @ dK_Phi                       # (N_KL, N_KL) dense

    # ── dλ/dθ_i = -2 λ_m W_i[m,m]  (exact, no truncation) ───────────────
    W_diag  = np.einsum('mmi->mi', W)            # (N_KL, n_theta)
    dlambda = -2.0 * eigvals[:, None] * W_diag

    # ── Daleckiĭ-Kreĭn coefficients for dφ ────────────────────────────────
    #
    # φ_kᵀ (dC/dθ_i) φ_m = -[ λ_m W_i[k,m]  +  λ_k W_i[m,k] ]
    #
    # This follows from dC/dθ = -K⁻¹(dK/dθ)C - C(dK/dθ)K⁻¹ and Cφ = λφ:
    #   φ_kᵀ (dC/dθ) φ_m = -φ_kᵀ K⁻¹(dK/dθ) Cφ_m  -  φ_kᵀ C(dK/dθ)K⁻¹φ_m
    #                     = -λ_m (K⁻¹φ_k)ᵀ(dK/dθ)φ_m  -  λ_k φ_kᵀ(dK/dθ)(K⁻¹φ_m)
    #                     = -λ_m r_kᵀ(dK/dθ)φ_m  -  λ_k r_mᵀ(dK/dθ)φ_k
    #                     = -[ λ_m W_i[k,m]  +  λ_k W_i[m,k] ]
    # Note: for the diagonal (k=m) this gives -2 λ_m W_i[m,m], matching dlambda.
    #
    # Daleckiĭ-Kreĭn:  dφ_m/dθ_i = Σ_{k≠m} A_i[k,m]/(λ_m - λ_k) · φ_k

    lam_diff  = eigvals[None, :] - eigvals[:, None]    # λ_m - λ_k, shape (N_KL, N_KL)
    lam_scale = 0.5 * (np.abs(eigvals[:, None]) + np.abs(eigvals[None, :]))
    is_degen  = np.abs(lam_diff) < gap_tol * np.maximum(lam_scale, 1e-30)
    np.fill_diagonal(is_degen, True)
    safe_diff = np.where(is_degen, 1.0, lam_diff)   # avoid /0 on diagonal

    dphi = np.zeros((n, N_KL, n_theta))
    for i in range(n_theta):
        W_i = W[:, :, i]
        # A_i[k,m] = phi_k^T (dC/dtheta_i) phi_m
        #           = -lam_m W_i[k,m]  -  lam_k W_i[m,k]
        # (one term from each K^{-1} in C = K^{-1} M K^{-1})
        A_i = -(W_i * eigvals[None, :]          # lam_m scales column m of W_i
              + W_i.T * eigvals[:, None])        # lam_k scales row k of W_i^T
        c_i = np.where(is_degen, 0.0, A_i / safe_diff)
        dphi[:, :, i] = phi @ c_i
    for m in range(N_KL):
        idx = np.argmax(np.abs(phi[:, m]))
        if phi[idx, m] < 0:
            phi[:, m] *= -1
            # also flip the corresponding dphi
            dphi[:, m, :] *= -1
    return KLDerivResult(
        eigvals  = eigvals,
        eigvecs  = phi,
        dlambda  = dlambda,
        dphi     = dphi,
        sqrt_lam = np.sqrt(np.maximum(eigvals, 0.0)),
    )

def gmrf_sample_alpha2_from_KM(K, M, seed=0, jitter=1e-10):
    n = K.shape[0]
    rng = np.random.default_rng(seed)
    K_fac = spla.factorized((K + jitter*sp.eye(n, format="csc")).tocsc())

    # w ~ N(0, M)  (use lumping for speed)
    M_lump = np.array(M.sum(axis=1)).ravel()
    w = np.sqrt(np.maximum(M_lump, 0.0)) * rng.standard_normal(n)

    # K v = w
    v = K_fac(w)
    # K u = M v
    u = K_fac((M @ v))
    u = np.asarray(u).ravel()
    u -= u.mean()
    return u
# ---------------------------------------------------------------------------
# Convenience stateful wrapper (drop-in for SPDEFieldProvider.get_dphi_dkappa)
# ---------------------------------------------------------------------------
import numpy as np

def build_B(phi, sqrt_lam):
    # phi: (n, r), sqrt_lam: (r,)
    return phi * sqrt_lam[None, :]

def analytic_dB(phi, sqrt_lam, dlambda, dphi, eps_safe=1e-30):
    """
    Returns dB: (n, r, n_theta)
    """
    inv2sqrt = 0.5 / np.maximum(sqrt_lam, eps_safe)  # (r,)
    dsqrt = inv2sqrt[:, None] * dlambda              # (r, n_theta)

    n, r = phi.shape
    n_theta = dlambda.shape[1]
    dB = np.empty((n, r, n_theta), dtype=float)

    for i in range(n_theta):
        dB[:, :, i] = dphi[:, :, i] * sqrt_lam[None, :] + phi * dsqrt[None, :, i]
    return dB
def procrustes_align_blocks(Phi, Phi_ref, groups):
    """
    Align Phi to Phi_ref within each cluster/group.
    Phi, Phi_ref: (n, r)
    returns aligned Phi and list of Q per group (for optional derivative rotation)
    """
    Phi = Phi.copy()
    Qs = []
    for g in groups:
        G = np.array(g, dtype=int)
        if len(G) == 1:
            j = G[0]
            if np.dot(Phi[:, j], Phi_ref[:, j]) < 0:
                Phi[:, j] *= -1.0
            Qs.append(np.array([[1.0]]))
        else:
            A = Phi_ref[:, G].T @ Phi[:, G]   # (k,k)
            U, _, Vt = np.linalg.svd(A, full_matrices=False)
            Q = U @ Vt                         # (k,k)
            Phi[:, G] = Phi[:, G] @ Q
            Qs.append(Q)
    return Phi, Qs
def fd_dB(spde_kl_derivs_fn, coo, kappa_param, theta, N_ret, N_sum, eps, gap_tol, jitter=1e-10):
    """
    Returns:
      B0: (n,r)
      dB_fd: (n,r,n_theta)
      dB_an: (n,r,n_theta)
      relerr_per_param: (n_theta,)
    """
    theta = np.asarray(theta, float)
    n_theta = theta.size

    # base (analytic)
    res0 = spde_kl_derivs_fn(coo, kappa_param, theta, N_ret=N_ret, N_sum=N_sum,
                            jitter=jitter, gap_tol=gap_tol, ref_phi=None)
    phi0 = res0.eigvecs
    lam0 = res0.eigvals
    sqrt0 = res0.sqrt_lam
    B0 = build_B(phi0, sqrt0)

    dB_an = analytic_dB(phi0, sqrt0, res0.dlambda, res0.dphi)

    # clusters based on base eigenvalues (returned modes)
    groups = detect_clusters(lam0, gap_tol=gap_tol)

    dB_fd = np.zeros_like(dB_an)
    relerrs = np.zeros(n_theta, dtype=float)

    for i in range(n_theta):
        th_p = theta.copy(); th_p[i] += eps
        th_m = theta.copy(); th_m[i] -= eps

        # compute +/- with ref_phi=phi0 for sign stability (helps singletons)
        res_p = spde_kl_derivs_fn(coo, kappa_param, th_p, N_ret=N_ret, N_sum=N_sum,
                                 jitter=jitter, gap_tol=gap_tol, ref_phi=phi0)
        res_m = spde_kl_derivs_fn(coo, kappa_param, th_m, N_ret=N_ret, N_sum=N_sum,
                                 jitter=jitter, gap_tol=gap_tol, ref_phi=phi0)

        # block align phi± to phi0 (critical!)
        phi_p, _ = procrustes_align_blocks(res_p.eigvecs, phi0, groups)
        phi_m, _ = procrustes_align_blocks(res_m.eigvecs, phi0, groups)

        Bp = build_B(phi_p, res_p.sqrt_lam)
        Bm = build_B(phi_m, res_m.sqrt_lam)

        dB_fd[:, :, i] = (Bp - Bm) / (2.0 * eps)

        num = np.linalg.norm(dB_an[:, :, i] - dB_fd[:, :, i], ord='fro')
        den = np.linalg.norm(dB_fd[:, :, i], ord='fro') + 1e-30
        relerrs[i] = num / den

    return B0, dB_fd, dB_an, relerrs
class SPDEKLDifferentiator:
    """
    Precomputes mesh structure once and exposes derivatives(theta) for reuse.

    Usage
    -----
    diff = SPDEKLDifferentiator(Nx, Ny, Lx, Ly, N_KL, kappa_param)
    result = diff.derivatives(theta)
    # result.dlambda    → (N_KL, n_layers), same as dlambda_dkappa
    # result.dphi       → (n, N_KL, n_layers)
    # result.dw         → (N_KL, n_layers), d(sqrt_lam)/dtheta for SG weights
    """

    def __init__(self, Nx, Ny, Lx, Ly, N_KL, kappa_param):
        self.coo         = precompute_fem_coo(Nx, Ny, Lx, Ly)
        self.N_KL        = N_KL
        self.kappa_param = kappa_param
        self._ref_phi    = None

    def derivatives(self, theta, gap_tol=1e-6):
        result = spde_kl_derivs(
            self.coo, self.kappa_param, theta, self.N_KL, 10*self.N_KL,
            gap_tol=gap_tol, ref_phi=self._ref_phi,
        )
        self._ref_phi = result.eigvecs.copy()
        return result


# ---------------------------------------------------------------------------
# Verification helpers
# ---------------------------------------------------------------------------

def _detect_degen_groups(eigvals, gap_tol=1e-6):
    """
    Return a list of index groups where eigenvalues are degenerate.
    Non-degenerate modes appear as singleton lists.
    E.g. [0], [1,2], [3], [4,5,6], ...
    """
    N = len(eigvals)
    groups = []
    i = 0
    while i < N:
        j = i + 1
        scale = max(abs(eigvals[i]), 1e-30)
        while j < N and abs(eigvals[i] - eigvals[j]) < gap_tol * scale:
            j += 1
        groups.append(list(range(i, j)))
        i = j
    return groups


def _eigpairs_at(coo, kappa_param, th, k_fd):
    """Return (eigvals, phi) for k_fd leading modes at theta=th."""
    kappa_y = kappa_param.kappa_y(th)
    K, M    = assemble_K_M(coo, kappa_y)
    n       = coo['n']
    Kf      = spla.factorized((K + 1e-10*sp.eye(n, format='csc')).tocsc())
    def _Cv(z):
        v = Kf(z.ravel()); return Kf((M@v).ravel())
    ev, phi = spla.eigsh(spla.LinearOperator((n,n), matvec=_Cv, dtype=np.float64),
                         k=k_fd, which='LM', tol=1e-13)
    order = np.argsort(ev)[::-1]
    return ev[order], phi[:, order]
def _build_B(phi, sqrt_lam):
    return phi * sqrt_lam[None, :]

def _analytic_dB(phi, sqrt_lam, dlambda, dphi, eps_safe=1e-30):
    inv2sqrt = 0.5 / np.maximum(sqrt_lam, eps_safe)     # (r,)
    dsqrt = inv2sqrt[:, None] * dlambda                 # (r, n_theta)
    n, r = phi.shape
    n_theta = dlambda.shape[1]
    dB = np.empty((n, r, n_theta), dtype=float)
    for i in range(n_theta):
        dB[:, :, i] = dphi[:, :, i] * sqrt_lam[None, :] + phi * dsqrt[None, :, i]
    return dB

def _procrustes_align_blocks(Phi, Phi_ref, groups):
    """
    Align Phi to Phi_ref within each group. Keeps your singleton sign convention,
    but does a block rotation for groups > 1.
    """
    Phi = Phi.copy()
    for g in groups:
        if len(g) == 1:
            j = g[0]
            if np.dot(Phi[:, j], Phi_ref[:, j]) < 0:
                Phi[:, j] *= -1.0
        else:
            G = np.array(g, dtype=int)
            A = Phi_ref[:, G].T @ Phi[:, G]             # (k,k)
            U, _, Vt = np.linalg.svd(A, full_matrices=False)
            Q = U @ Vt                                  # (k,k)
            Phi[:, G] = Phi[:, G] @ Q
    return Phi

def verify(coo, kappa_param, theta, N_KL, eps=1e-6, n_print=10,
           test_dphi=True, modes_dphi=None, gap_tol=1e-6, test_dB=True):
    """
    Central-FD verification of dλ/dθ and (optionally) dφ/dθ.

    For dφ, raw FD on individual eigenvectors is meaningless for degenerate
    modes (the basis within a degenerate subspace is arbitrary).  Instead we
    test via the rank-1 projector for non-degenerate modes:

        P_m(θ) = φ_m φ_mᵀ

        dP_m/dθ_i  =  dφ_m φ_mᵀ + φ_m (dφ_m)ᵀ

    and via the subspace projector for degenerate groups:

        P_grp(θ) = Φ_grp Φ_grpᵀ

    Both are basis-independent and uniquely defined.  We compare
    ||dP_analytic - dP_fd||_F / ||dP_fd||_F.

    Parameters
    ----------
    coo          : dict from precompute_fem_coo
    kappa_param  : LayeredKappa or SmoothKappa
    theta        : (n_theta,)
    N_KL         : number of KL modes
    eps          : FD step size
    n_print      : number of modes to print in the dλ table
    test_dphi    : whether to run the dφ projector check
    modes_dphi   : list of mode indices (0-based) to test; None = first 8 non-degen
    gap_tol      : relative gap for degeneracy detection

    Returns
    -------
    dict with keys 'max_dlam_err', 'max_dphi_err'
    """
    theta   = np.asarray(theta, dtype=np.float64)
    n_theta = len(theta)
    k_fd    = N_KL + 5   # buffer against truncation-boundary mode swaps

    analytic = spde_kl_derivs(coo, kappa_param, theta, N_KL, N_KL*4,gap_tol=gap_tol)
    phi0     = analytic.eigvecs    # (n, N_KL) at theta

    # ── dλ via FD on eigenvalues only ─────────────────────────────────────
    def _eigs_only(th):
        ev, _ = _eigpairs_at(coo, kappa_param, th, k_fd)
        return ev[:N_KL]

    dlam_fd = np.zeros((N_KL, n_theta))
    for i in range(n_theta):
        tp, tm = theta.copy(), theta.copy()
        tp[i] += eps; tm[i] -= eps
        dlam_fd[:, i] = (_eigs_only(tp) - _eigs_only(tm)) / (2*eps)

    max_dlam = 0.0
    print(f"\n{'─'*70}")
    print(f"  [A] dλ/dθ  analytic vs FD  (eps={eps:.0e})")
    print(f"{'─'*70}")
    print(f"  {'mode':<6}{'param':<7}{'analytic':>16}{'FD':>16}{'rel_err':>10}  OK?")
    print(f"{'─'*70}")
    for m in range(min(N_KL, n_print)):
        for i in range(n_theta):
            a   = float(analytic.dlambda[m, i])
            f   = float(dlam_fd[m, i])
            rel = abs(a - f) / max(abs(f), 1e-14)
            max_dlam = max(max_dlam, rel)
            flag = '✓' if rel < 1e-3 else '✗'
            print(f"  λ_{m+1:<4} θ_{i+1:<4} {a:>16.6e} {f:>16.6e} {rel:>10.2e}  {flag}")
    for m in range(n_print, N_KL):
        for i in range(n_theta):
            f = dlam_fd[m, i]
            if abs(f) < 1e-10 * max(abs(dlam_fd[:, i]).max(), 1e-30):
                continue
            max_dlam = max(max_dlam, abs(analytic.dlambda[m, i] - f) / abs(f))
    print(f"{'─'*70}")
    print(f"  Max rel error (all {N_KL} modes): {max_dlam:.2e}")

    if not test_dphi:
        return {'max_dlam_err': max_dlam, 'max_dphi_err': None}

    # ── dφ via FD on projectors ────────────────────────────────────────────
    #
    # Strategy
    # --------
    # For mode m, compute P_m(θ±ε) = φ_m(θ±ε) φ_m(θ±ε)ᵀ and form
    #   dP_m/dθ_i |_FD  = [P_m(θ+ε) - P_m(θ-ε)] / (2ε)
    #
    # The analytic dP comes from dφ:
    #   dP_m/dθ_i |_analytic = dφ_m φ_mᵀ + φ_m (dφ_m)ᵀ
    #
    # For degenerate groups, use the subspace projector instead:
    #   P_grp = Φ_grp Φ_grpᵀ,  dP_grp = sum_m in grp (dφ_m φ_mᵀ + φ_m dφ_mᵀ)
    #
    # Sign alignment of FD eigenvectors: align each FD φ_m to analytic φ0_m
    # using sign (for non-degen) or Procrustes (for degen groups).
    # BUT: for the projector test we don't need alignment at all — the projector
    # P = Φ Φᵀ is sign- and rotation-invariant within the subspace.

    groups = _detect_degen_groups(analytic.eigvals[:N_KL], gap_tol=gap_tol)

    # Choose which modes/groups to test
    if modes_dphi is None:
        # Pick first 8 non-degenerate modes (skip degenerate groups for raw dphi)
        nondegen_modes = [g[0] for g in groups if len(g) == 1][:8]
        degen_groups   = [g for g in groups if len(g) > 1][:4]
    else:
        nondegen_modes = [m for m in modes_dphi
                          if len([g for g in groups if m in g][0]) == 1]
        degen_groups   = list({tuple(g) for m in modes_dphi
                               for g in groups if m in g and len(g) > 1})

    print(f"\n{'─'*70}")
    print(f"  [B] dφ/dθ via projector FD  (eps={eps:.0e})")
    print(f"  Non-degen modes tested: {[m+1 for m in nondegen_modes]}")
    print(f"  Degen groups tested:    {[[m+1 for m in g] for g in degen_groups]}")
    print(f"{'─'*70}")
    print(f"  {'mode/grp':<12}{'param':<7}{'||dP_a-dP_fd||_F / ||dP_fd||_F':>32}  note")
    print(f"{'─'*70}")

    max_dphi = 0.0

    def _phi_at(th):
        """Return phi (n, N_KL) aligned to phi0 via per-column sign flip."""
        _, phi = _eigpairs_at(coo, kappa_param, th, k_fd)
        phi = phi[:, :N_KL]
        for j in range(N_KL):
            if np.dot(phi[:, j], phi0[:, j]) < 0:
                phi[:, j] *= -1
        return phi

    # Compute FD phi at perturbed points (2 * n_theta evaluations)
    phi_p = {}; phi_m_ = {}
    for i in range(n_theta):
        tp, tm = theta.copy(), theta.copy()
        tp[i] += eps; tm[i] -= eps
        phi_p[i]  = _phi_at(tp)
        phi_m_[i] = _phi_at(tm)

    # Non-degenerate mode projectors
    for m in nondegen_modes:
        phi0_m = phi0[:, m]   # (n,)
        for i in range(n_theta):
            # FD projector derivative
            pm_p = np.outer(phi_p[i][:, m],  phi_p[i][:, m])
            pm_m = np.outer(phi_m_[i][:, m], phi_m_[i][:, m])
            dP_fd = (pm_p - pm_m) / (2*eps)   # (n, n)

            # Analytic projector derivative: dφ_m φ_mᵀ + φ_m (dφ_m)ᵀ
            dphi_m = analytic.dphi[:, m, i]   # (n,)
            dP_an  = np.outer(dphi_m, phi0_m) + np.outer(phi0_m, dphi_m)

            norm_fd = np.linalg.norm(dP_fd, 'fro')
            rel     = np.linalg.norm(dP_an - dP_fd, 'fro') / max(norm_fd, 1e-30)
            max_dphi = max(max_dphi, rel)
            flag = '✓' if rel < 1e-2 else ('⚠' if rel < 0.1 else '✗')
            print(f"  φ_{m+1:<9} θ_{i+1:<4} {rel:>32.4e}  {flag}")

    # Degenerate group subspace projectors
    for grp in degen_groups:
        Phi0_grp = phi0[:, grp]   # (n, g)
        for i in range(n_theta):
            Pp_grp = phi_p[i][:, grp];  pm_grp = phi_m_[i][:, grp]
            dP_fd  = (Pp_grp @ Pp_grp.T - pm_grp @ pm_grp.T) / (2*eps)

            dP_an  = sum(
                np.outer(analytic.dphi[:, m, i], phi0[:, m])
                + np.outer(phi0[:, m], analytic.dphi[:, m, i])
                for m in grp
            )

            norm_fd = np.linalg.norm(dP_fd, 'fro')
            rel     = np.linalg.norm(dP_an - dP_fd, 'fro') / max(norm_fd, 1e-30)
            max_dphi = max(max_dphi, rel)
            flag = '✓' if rel < 1e-2 else ('⚠' if rel < 0.1 else '✗')
            grp_str = f"{{{','.join(str(m+1) for m in grp)}}}"
            print(f"  grp{grp_str:<8} θ_{i+1:<4} {rel:>32.4e}  {flag}")

    print(f"{'─'*70}")
    print(f"  Max dφ projector error: {max_dphi:.2e}  "
          f"(truncation error from modes >{N_KL} expected)\n")
    if test_dB:
        # ── [C] dB/dθ where B = Φ diag(sqrt(λ))  (SG-consistent) ───────────
        #
        # FD: build B(θ±ε) using Φ(θ±ε) aligned to Φ(θ) by block Procrustes,
        #     and sqrt(λ)(θ±ε) from the same eigsolve.
        #
        # Analytic: dB = dΦ*sqrt + Φ*(0.5/sqrt)*dλ
        #
        # This catches gauge issues that projector tests can miss in near-degenerate cases.

        groups_ret = _detect_degen_groups(analytic.eigvals[:N_KL], gap_tol=gap_tol)

        B0 = _build_B(phi0, analytic.sqrt_lam)
        dB_an = _analytic_dB(phi0, analytic.sqrt_lam, analytic.dlambda, analytic.dphi)

        max_dB = 0.0
        print(f"\n{'─'*70}")
        print(f"  [C] dB/dθ for B=Φ diag(sqrt(λ))  analytic vs FD  (eps={eps:.0e})")
        print(f"{'─'*70}")
        print(f"  {'param':<7}{'||dB_a-dB_fd||_F / ||dB_fd||_F':>32}  note")
        print(f"{'─'*70}")

        def _eigpairs_ret(th):
            # Use the same eigsolve as _phi_at but return BOTH eigvals and eigvecs
            ev, ph = _eigpairs_at(coo, kappa_param, th, k_fd)
            ev = ev[:N_KL]
            ph = ph[:, :N_KL]
            return ev, ph

        for i in range(n_theta):
            tp, tm = theta.copy(), theta.copy()
            tp[i] += eps; tm[i] -= eps

            ev_p, ph_p = _eigpairs_ret(tp)
            ev_m, ph_m = _eigpairs_ret(tm)

            # Align eigenvectors to base using block Procrustes on returned modes
            ph_p = _procrustes_align_blocks(ph_p, phi0, groups_ret)
            ph_m = _procrustes_align_blocks(ph_m, phi0, groups_ret)

            Bp = _build_B(ph_p, np.sqrt(np.maximum(ev_p, 0.0)))
            Bm = _build_B(ph_m, np.sqrt(np.maximum(ev_m, 0.0)))

            dB_fd = (Bp - Bm) / (2*eps)

            norm_fd = np.linalg.norm(dB_fd, 'fro')
            rel = np.linalg.norm(dB_an[:, :, i] - dB_fd, 'fro') / max(norm_fd, 1e-30)
            max_dB = max(max_dB, rel)
            flag = '✓' if rel < 1e-3 else ('⚠' if rel < 1e-2 else '✗')
            print(f"  θ_{i+1:<10} {rel:>32.4e}  {flag}")

        print(f"{'─'*70}")
        print(f"  Max dB error: {max_dB:.2e}\n")

        # include in return dict
        out = {'max_dlam_err': max_dlam, 'max_dphi_err': max_dphi, 'max_dB_err': max_dB}
        return out

    return {'max_dlam_err': max_dlam, 'max_dphi_err': max_dphi}


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import sys, time
    # LayeredKappa and SmoothKappa already defined above

    print("=" * 70)
    print("  SPDE-aware KL derivative self-test")
    print("  Mesh: vertex-centred (Nx+1)*(Ny+1), matches inf_layered_2.py")
    print("=" * 70)

    # ── Test 1: LayeredKappa, 40×40, N_KL=20 ────────────────────────────────
    Nx, Ny = 50, 50; Lx, Ly = 0.1, 0.1; N_KL = 80


    coo2 = precompute_fem_coo(Nx, Ny, Lx, Ly)

    # ── Test 3: SmoothKappa FD ────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  FD verification  (20×20 grid, SmoothKappa)")
    print(f"{'='*70}")
   # theta_s = np.array([20., 2.])
    #kp_s    = SmoothKappa(Ny=Ny2, Ly=Ly, kappa0=10.0, strength=1.1)
    kp_s = SigmoidLayeredKappa(
        Ny=Ny,
        Ly=Ly,
        kappa_surface=80.0,    # short correlation near surface (weathered/fractured)
        kappa_deep=10.0,       # longer correlation at depth (competent granite)
        y_transition=0.4*Ly,   # transition at 60% depth
        width=0.1*Ly,          # transition zone ~10% of domain width
    )
    theta_s = np.array([80.0, 10.0, 0.4*Ly, 0.1*Ly])
    res_s_phi = verify(coo2, kp_s, theta_s, N_KL=100, eps=1e-4, n_print=10,
                       test_dphi=True, modes_dphi=[0, 1, 2, 3, 4,5,6,7,8,9], gap_tol=1e-6)

    # ── Summary ───────────────────────────────────────────────────────────
    print("=" * 70)
    print("  FINAL SUMMARY")
    print("=" * 70)
    print(f"  LayeredKappa dλ  max rel err = {res_s_phi['max_dlam_err']:.2e}  "
          f"{'✓' if res_s_phi['max_dlam_err']<1e-4 else '✗'}  (exact via Hellmann-Feynman)")
    print(f"  SmoothKappa  dφ  max rel err = {res_s_phi['max_dphi_err']:.2e}  "
          f"{'✓' if res_s_phi['max_dphi_err']<0.05 else '✗'}  (N_KL=80, modes 1-5)")
    print("=" * 70)
    print("  Note: dφ errors decrease as N_KL increases (truncation in Daleckii-Krein sum).")
    print("        dλ errors are independent of N_KL (exact Hellmann-Feynman).")