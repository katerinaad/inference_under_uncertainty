import numpy as np
from stable_eigh_test import SPDEKLDifferentiator, SigmoidLayeredKappa

# ---------------------------------------------------------------------------
# Trapezoidal weights
# ---------------------------------------------------------------------------

def _trap_weights(y_nodes):
    dz = np.diff(y_nodes)
    w = np.empty_like(y_nodes)
    w[0] = 0.5 * dz[0]
    w[1:-1] = 0.5 * (dz[:-1] + dz[1:])
    w[-1] = 0.5 * dz[-1]
    return w


# ---------------------------------------------------------------------------
# Forward: soft-min depth
# ---------------------------------------------------------------------------

def softmin_depth(u_mean_2d, y_nodes, T_abl, eps, beta, evap_range=0.0):
    """
    Parameters
    ----------
    u_mean_2d : (Nx+1, Ny+1) mean temperature field
    y_nodes : (Ny+1,)
    T_abl : ablation threshold (K)
    eps : arctan smoothing width (K), ~5-10 K recommended
    beta : soft-min sharpness over x-columns (m), ~2-5*dx recommended
    evap_range : transition width above T_abl (K), default 0

    Returns
    -------
    smin : float soft-min surface height
    surf : (Nx+1,) surface height per x-column
    w_softmin : (Nx+1,) soft-min weights
    H : (Nx+1, Ny+1) arctan indicator
    dH : (Nx+1, Ny+1) dH/dT
    """
    #print(y_nodes, T_abl, eps, beta, evap_range)
    wz = _trap_weights(y_nodes)
    ytop = float(y_nodes[-1])
    #beta=0.005
    evap_range = 0
    beta = 0.0005
    phi = u_mean_2d - T_abl - evap_range
    phi_eps = phi / eps
    H = 0.5 * (1.0 + (2.0 / np.pi) * np.arctan(phi_eps))
    dH = (1.0 / (np.pi * eps)) / (1.0 + phi_eps ** 2)

    thickness = H @ wz          # (Nx+1,)
    surf = ytop - thickness     # (Nx+1,)

    z_ref = float(surf.min())
    exps = np.exp(np.clip(-(surf - z_ref) / beta, -100.0, 0.0))
    esum = float(exps.sum())
    smin = z_ref - beta * np.log(esum)
    w_softmin = exps / esum

    return smin, surf, w_softmin, H, dH


# ---------------------------------------------------------------------------
# Adjoint seed: d(NLL_t)/dT
# ---------------------------------------------------------------------------

def _depth_adjoint_seed_mean_var(u_mean_2d, y_nodes, smin_obs, T_abl, eps, beta,
                                  sigma_d, evap_range=0.0):
    """
    Returns (Nx+1, Ny+1) gradient of 0.5*(smin-smin_obs)²/sigma_d² w.r.t. T.
    """
    smin, surf, w_softmin, H, dH = softmin_depth(
        u_mean_2d, y_nodes, T_abl, eps, beta, evap_range
    )

    residual = (smin - smin_obs) / (sigma_d ** 2)  # scalar, scaled by 1/sigma_d²

    wz = _trap_weights(y_nodes)
    grad = w_softmin[:, np.newaxis] * wz[np.newaxis, :] * dH  # (Nx+1, Ny+1)

    return -residual * grad


# ---------------------------------------------------------------------------
# Public: scalar NLL over trajectory
# ---------------------------------------------------------------------------

def compute_depth_objective_trajectory(U_hist, Nx, Ny, Ly, num_nodes, T_abl,
                                        eps=50.0, beta=None, evap_range=0.0,
                                        h_obs_hist=None, sigma_d=None):
    """
    Negative log-likelihood of depth observations under diagonal Gaussian.

    J = Σ_t 0.5 * (smin(U_t) - d_obs_t)² / sigma_d²

    Parameters
    ----------
    h_obs_hist : (time_steps,) observed depths (required)
    sigma_d : depth observation noise std dev (m)

    Returns
    -------
    J_total : float
    J_hist : (time_steps,) per-step NLL contributions (0 at t=0)
    """
    if beta is None:
        beta = 2.0 * Ly / Nx
    if h_obs_hist is None:
        raise ValueError(
            "h_obs_hist is required. Pass observed depth mean. "
            "For a self-consistency test compute smin_traj from U_hist first."
        )

    inv_var = 1.0 / (sigma_d ** 2)
    y_nodes = np.linspace(0.0, Ly, Ny + 1)
    time_steps = U_hist.shape[0]
    J_hist = np.zeros(time_steps)

    for t in range(1, time_steps):
        u_mean = U_hist[t, :num_nodes].reshape(Nx + 1, Ny + 1)
        smin, _, _, _, _ = softmin_depth(u_mean, y_nodes, T_abl, eps, beta,
                                          evap_range)
        r = smin - float(h_obs_hist[t])
        J_hist[t] = 0.5 * inv_var * r ** 2
        J_hist[t] = r
    return float(np.sum(J_hist)), J_hist


def _vjp_g_wrt_u_mean(u_mean_2d, y_nodes, T_abl, eps, beta, p_flat, evap_range=0.0):
    """
    Computes grad_u = (∂g/∂u_mean)^T p, where:
    g_ij = w_softmin[i] * wz[j] * dH_ij
    and p_flat is a vector with same shape as g_flat (num_nodes,).
    Returns grad_u_flat (num_nodes,).
    """
    Nx1, Ny1 = u_mean_2d.shape
    wz = _trap_weights(y_nodes)
    ytop = float(y_nodes[-1])

    # forward pieces (same as softmin_depth)
    phi = u_mean_2d - T_abl - evap_range
    q = phi / eps

    H = 0.5 * (1.0 + (2.0 / np.pi) * np.arctan(q))
    dH = (1.0 / (np.pi * eps)) / (1.0 + q**2)

    thickness = H @ wz      # (Nx+1,)
    surf = ytop - thickness  # (Nx+1,)

    # softmin weights w over x-columns (treat z_ref as a numeric stabilizer: no grad through it)
    z_ref = float(surf.min())
    a = -(surf - z_ref) / beta
    a = np.clip(a, -100.0, 0.0)
    exps = np.exp(a)
    w = exps / float(exps.sum())  # (Nx+1,)

    # reshape p
    p = p_flat.reshape(Nx1, Ny1)  # (Nx+1, Ny+1)

    # g_ij = w_i * wz_j * dH_ij
    # accumulate adjoints
    # dL/dw_i = sum_j p_ij * wz_j * dH_ij
    dL_dw = np.sum(p * (wz[None, :] * dH), axis=1)  # (Nx+1,)

    # dL/ddH_ij += p_ij * w_i * wz_j
    dL_ddH = p * (w[:, None] * wz[None, :])  # (Nx+1, Ny+1)

    # backprop through w = softmax(a), a = -(surf - z_ref)/beta
    # dL/da = (diag(w)-w w^T) dL/dw
    w_dot = float(np.dot(w, dL_dw))
    dL_da = w * (dL_dw - w_dot)          # (Nx+1,)
    dL_dsurf = -(1.0 / beta) * dL_da     # (Nx+1,)

    # surf = ytop - thickness
    dL_dthickness = -dL_dsurf  # (Nx+1,)

    # thickness = H @ wz => dL/dH_ij += dL/dthickness_i * wz_j
    dL_dH = dL_dthickness[:, None] * wz[None, :]  # (Nx+1, Ny+1)

    # H depends on u: dH/du_mean = dH (already computed)
    # dH (the derivative) depends on u too; need ddH/du_mean
    # dH = (1/(pi*eps)) * 1/(1+q^2), q=phi/eps
    # ddH/du = (-2/(pi*eps^2)) * q/(1+q^2)^2
    ddH_du = (-2.0 / (np.pi * eps**2)) * (q / (1.0 + q**2)**2)

    # total grad wrt u_mean: from H path + from dH path
    grad_u = dL_dH * dH + dL_ddH * ddH_du
    return grad_u.ravel()


def depth_adjoint_seed(U_t, h_obs_t, sigma2_obs_t, Nx, Ny, Ly, num_nodes, P,
                        T_abl, eps, beta, bc_idx, evap_range=0.0,
                        include_mean_term=False, sigma_d=None, mean_only=False):
    beta = 0.005
    y_nodes = np.linspace(0.0, Ly, Ny + 1)
    wz = _trap_weights(y_nodes)

    u_mean_2d = U_t[:num_nodes].reshape(Nx + 1, Ny + 1)
    h_u0, _, w_softmin, _, dH = softmin_depth(u_mean_2d, y_nodes, T_abl, eps, beta, evap_range)
    g_flat = (w_softmin[:, None] * wz[None, :] * dH).ravel()

    lam = np.zeros(P * num_nodes, dtype=float)

    # --- optional mean-depth term (only if your J includes it) ---
    if mean_only==True:
        r_mu = (h_u0 - float(h_obs_t))
        # J_mu = 0.5 * r_mu^2 / sigma_d^2 => dJ/du0 = r_mu/sigma_d^2 * dh/du0
        # and dh/du0 = -g_flat (because surf = ytop - H@wz)
        lam[:num_nodes] += -(r_mu) * g_flat / max(1e-20, float(sigma2_obs_t))

    # --- variance term ---
    U_modes = U_t.reshape(P, num_nodes)
    a = U_modes[1:] @ g_flat  # a_k = g^T u_k, shape (P-1,)
    sigma2_pred = float(np.sum(a**2))
    r_var = sigma2_pred - float(sigma2_obs_t)

    # mode blocks (this part you had right)
    if mean_only==False:
        for k in range(1, P):
            lam[k*num_nodes:(k+1)*num_nodes] = (2.0 * r_var * a[k-1]) * g_flat *1e10

        # NEW: mean block contribution from g(u0)
        # dJ/dg = 2 r_var * sum_k a_k u_k
        s_vec = (a[:, None] * U_modes[1:]).sum(axis=0)  # sum_k a_k u_k, shape (num_nodes,)
        p_g = 2.0 * r_var * s_vec  # = dJ/dg (without 1e10 scale)
        lam[:num_nodes] += _vjp_g_wrt_u_mean(
            u_mean_2d, y_nodes, T_abl, eps, beta, p_g, evap_range=evap_range) *1e10
    #print(np.max(lam))
    # (optional) enforce BC
    lam[bc_idx] = 0.0
    return lam


def depth_adjoint_seed_prev(U_t, h_obs_t, sigma2_obs_t, Nx, Ny, Ly, num_nodes, P,
                             T_abl, eps, beta, bc_idx, evap_range=0.0):
    """
    Adjoint seed for linearised depth objective at timestep t.
    Seeds k=0 block from mean depth residual,
    seeds k>=1 blocks from variance residual.
    """
    y_nodes = np.linspace(0.0, Ly, Ny + 1)
    wz = _trap_weights(y_nodes)

    u_mean_2d = U_t[:num_nodes].reshape(Nx + 1, Ny + 1)
    h_u0, _, w_softmin, _, dH = softmin_depth(u_mean_2d, y_nodes, T_abl, eps, beta, evap_range)
    g_flat = (w_softmin[:, np.newaxis] * wz[np.newaxis, :] * dH).ravel()

    lam = np.zeros(P * num_nodes, dtype=float)

    # --- mean block: dJ/du_0 ---
    r_mu = (h_u0 - float(h_obs_t))
    lam[:num_nodes] = -r_mu * g_flat / float(sigma2_obs_t)
    # print(np.max(lam[:num_nodes]))
    # --- mode blocks: dJ/du_k for k>=1 ---
    U_modes = U_t.reshape(P, num_nodes)
    g_dot_uk = U_modes[1:] @ g_flat  # (P-1,)
    sigma2_pred = float(np.sum(g_dot_uk**2))
    r_var = sigma2_pred - float(sigma2_obs_t)

    #for k in range(1, P):
    #    lam[k*num_nodes:(k+1)*num_nodes] = 2.0 * r_var * g_dot_uk[k-1] * g_flat
    #    print(np.max(2.0 * r_var * g_dot_uk[k-1] * g_flat))
    return lam


# ---------------------------------------------------------------------------
# Public: full adjoint sweep
# ---------------------------------------------------------------------------


def run_adjoint_depth(U_obs,
                      U_hist,
                      M_bc, K_bc, solve_A,
                      K_SG_K0, K_SG_K1, M_SG_M0, M_SG_M1,
                      solid_prop, melt_prop, vap_prop, spatial_op, params, bc_idx,
                      Nx, Ny, Ly, num_nodes, P, T_abl,
                      adjoint_one_step_fn,
                      sigma2_obs_hist,
                      eps=None,
                      beta=None,
                      evap_range=0.0,
                      h_obs_hist=None,
                      sigma_d=None,
                      mean_only=False
                      ):
    """
    Backward adjoint sweep for the NLL depth objective.

    h_obs_hist : (time_steps,) observed depths (required for non-trivial gradient)
    sigma_d : depth observation noise std dev (m)
    """
    if beta is None:
        beta = 2.0 * Ly / Nx
    if h_obs_hist is None:
        raise ValueError("h_obs_hist is required.")
    #print("HOBSHIST", h_obs_hist)
    y_nodes = np.linspace(0.0, Ly, Ny + 1)
    time_steps = U_hist.shape[0]
    smin_obs = np.asarray(h_obs_hist, float)

    # ---- compute smin trajectory for J ------------------------------------
    smin_traj = np.zeros(time_steps)
    for t in range(time_steps):
        u_mean = U_hist[t, :num_nodes].reshape(Nx + 1, Ny + 1)
        smin_traj[t], _, _, _, _ = softmin_depth(u_mean, y_nodes, T_abl,
                                                  eps, beta, evap_range)

    J_hist = np.zeros(time_steps)
    J_total = 0.0

    # ---- backward sweep ---------------------------------------------------
    Mu_hist = np.zeros_like(U_hist)
    lam_hist = np.zeros_like(U_hist)

    L_current = depth_adjoint_seed(
        U_hist[-1], h_obs_hist[-1], sigma2_obs_hist[-1],
        Nx, Ny, Ly, num_nodes, P,
        T_abl, eps, beta, bc_idx, evap_range, sigma_d=sigma_d, mean_only=mean_only
    )

    for n in range(time_steps - 2, -1, -1):
        mu_prev = Mu_hist[n + 1, :] if n < time_steps - 2 else None

        lam_n, mu_n = adjoint_one_step_fn(
            n,
            solid_prop, melt_prop, vap_prop,
            U_hist[n + 1, :], L_current, U_hist[n, :],
            M_SG_M0, M_SG_M1,
            K_SG_K0, K_SG_K1,
            M_bc, K_bc, solve_A,
            spatial_op, params,
            mu_prev=mu_prev,
        )

        if n >= 1:  # J only accumulates from t=1, seed is zero at t=0
            stage_seed = depth_adjoint_seed(
                U_hist[n], h_obs_hist[n], sigma2_obs_hist[n],
                Nx, Ny, Ly, num_nodes, P,
                T_abl, eps, beta, bc_idx, evap_range, mean_only=mean_only
            )
            lam_n = lam_n + stage_seed
            # lam_n = lam_n + stage_seed
        L_current = lam_n
        Mu_hist[n, :] = mu_n
        lam_hist[n, :] = lam_n

    return Mu_hist, lam_hist, J_total, J_hist


# ---------------------------------------------------------------------------
# Public: FD validation
# ---------------------------------------------------------------------------

def validate_depth_adjoint_fd(dx, dy, U_obs_passed,
                               run_forward_fn,
                               U0,
                               solid_prop, melt_prop, vap_prop,
                               ell, theta_kappa,
                               Nx, Ny, Ly, num_nodes, P, T_abl,
                               adjoint_one_step_fn,
                               adjoint_grad_all_phase_fn,
                               forcing_param_grads_numpy_fn,
                               clear_caches_fn,
                               bc_idx, params,
                               M_bc, K_bc, solve_A,
                               K_SG_K0, K_SG_K1, M_SG_M0, M_SG_M1,
                               sigma2_obs_hist,
                               spatial_op,
                               h_obs_hist,  # required: observed depth history
                               eps_smooth=10.0,
                               beta=None,
                               evap_range=0.0,
                               sigma_d=None,
                               eps_fd=5e-5,
                               verbose=True,
                               # kappa gradient: pass these from the call site in inf_layered_vap.py
                               kappa_param=None,           # LayeredKappa / SmoothKappa instance
                               compute_adjoint_grad_kappa_fn=None,  # = compute_adjoint_grad_kappa_phase_matrixfree
                               Lx=None,    # domain width (needed to build SPDEKLDifferentiator)
                               N_KL=None,  # number of KL modes, 
                               run_fd_check = True , mean_only=True
                               ):
    U_obs = U_obs_passed.copy()
    print("UOBS", np.max(U_obs))

    U0_clean = U0.copy()  # <-- save this
    y_nodes = np.linspace(0.0, Ly, Ny + 1)

    # 1. Base forward solve
    result0 = run_forward_fn(U0_clean, solid_prop, melt_prop, vap_prop, ell,
                             theta_kappa=theta_kappa, kappa_param=kappa_param)
    (U_hist0, _, _, M_bc0, K_bc0, solve_A0, _, _, _,
     K0_K0, K0_K1, M0_M0, M0_M1, sp0, local_params0) = result0

    time_steps = U_hist0.shape[0]

    print("UOBS", np.max(U_obs))

    def _J_of_hist(U_hist):
        J = 0.0
        wz = _trap_weights(y_nodes)
        beta = 0.005
        for t in range(1, time_steps):
            u_t = U_hist[t]
            u_mean_2d = u_t[:num_nodes].reshape(Nx + 1, Ny + 1)

            # --- mean depth term ---
            h_u0, _, w_softmin, _, dH = softmin_depth(u_mean_2d, y_nodes, T_abl, eps_smooth, beta)
            g_flat = (w_softmin[:, np.newaxis] * wz[np.newaxis, :] * dH).ravel()

            r_mu = h_u0 - float(h_obs_hist[t])
            if mean_only ==True:    
                J += 0.5 * r_mu**2 / max(1e-20, sigma2_obs_hist[t])
            # --- variance term ---
            if mean_only==False:
                U_modes = u_t.reshape(P, num_nodes)
                g_dot_uk = U_modes[1:] @ g_flat  # (P-1,)
                sigma2_pred = float(np.sum(g_dot_uk**2))
                r_var = sigma2_pred - float(sigma2_obs_hist[t])
                J += 0.5 * (r_var**2) *1e10
        return J

    print("UOBS", np.max(U_obs))

    J_base = _J_of_hist(U_hist0)
    print("before adjoint")
    # 2. Adjoint sweep — use local_params0 (kappa-specific snapshot) so the
    # adjoint forcing VJP reads the same eigvecs as the forward solve did.
    Mu_hist, _, _, _ = run_adjoint_depth(U_obs,
                                          U_hist0, M_bc0, K_bc0, solve_A0,
                                          K0_K0, K0_K1, M0_M0, M0_M1,
                                          solid_prop, melt_prop, vap_prop, sp0, local_params0, bc_idx,
                                          Nx, Ny, Ly, num_nodes, P, T_abl,
                                          adjoint_one_step_fn, sigma2_obs_hist,
                                          eps=eps_smooth, beta=beta, evap_range=evap_range,
                                          h_obs_hist=h_obs_hist, sigma_d=sigma_d, mean_only=mean_only
                                          )
    print("after adjoint")
    # 3. Adjoint gradients
    g_phase = adjoint_grad_all_phase_fn(
        U_hist0, Mu_hist, solid_prop, melt_prop,
        K_SG_K1=K0_K1, M_SG_M1=M0_M1,
        forcing_param_grads_numpy=forcing_param_grads_numpy_fn,
        spatial_op=sp0, freeze_phase=False, vap_prop=vap_prop
    )
    print("grad all phase")
    # 3b. Kappa adjoint gradient
    # ------------------------------------------------------------------
    # We need dlambda (N_KL, n_layers) and dphi (n, N_KL, n_layers) for
    # ALL kappa layers — build_SG_operators only stores layer 0, so we
    # re-run SPDEKLDifferentiator here with the base theta_kappa.
    # ------------------------------------------------------------------
    g_kappa_adj = None
    if (kappa_param is not None and compute_adjoint_grad_kappa_fn is not None
            and theta_kappa is not None and Lx is not None and N_KL is not None):

        _diff = SPDEKLDifferentiator(Nx, Ny, Lx, Ly, N_KL, kappa_param)
        print("spde kl derivs returned")
        theta_kappa_arr = np.atleast_1d(np.asarray(theta_kappa, float))
        # Ensure SmoothKappa has 2 parameters
        if hasattr(_diff.kappa_param, 'kappa0') and theta_kappa_arr.size == 1:
            theta_kappa_arr = np.array([theta_kappa_arr[0], _diff.kappa_param.strength], dtype=float)
        _res = _diff.derivatives(theta_kappa_arr)
        eigvals_trunc = np.asarray(_res.eigvals, float)      # (N_KL,)
        eigvecs_reshaped = np.asarray(_res.eigvecs, float)   # (n, N_KL)
        dlambda_all = np.asarray(_res.dlambda, float)        # (N_KL, n_layers)
        dphi_all = np.asarray(_res.dphi, float)              # (n, N_KL, n_layers)

        # Reshape eigvecs to (Nx+1, Ny+1, N_KL) — vertex-centred grid
        eigvecs_grid = eigvecs_reshaped.reshape(Nx + 1, Ny + 1, N_KL)

        n_rf_params = dlambda_all.shape[1]
        g_kappa_adj = np.zeros(n_rf_params)

        dphi_all_grid = dphi_all.reshape(Nx+1, Ny+1, N_KL, n_rf_params)  # if not already

        g_kappa_adj = compute_adjoint_grad_kappa_fn(
            U_hist0, Mu_hist,
            solid_prop, melt_prop, vap_prop,
            eigvals_trunc=eigvals_trunc,
            eigvecs_reshaped=eigvecs_grid,
            dlambda_dkappa=dlambda_all,
            dphi_dkappa=dphi_all_grid,
            local_params=local_params0,  # ← ADD
            freeze_phase=False,
            include_forcing_dphi=True,
            coo=_diff.coo
        )
        print(g_kappa_adj)
        #if verbose:
        #    print("\n" + "-" * 72)
        #    print(f"{'Kappa param':<12} {'Adjoint':>18} {'FD (central)':>18} "
        #          f"{'Rel err':>12} Status")
        #    print("-" * 72)
    
    # 4. FD reference
    def _J_perturbed(solid_p, melt_p, vap_p, U0_clean):
        clear_caches_fn()
        return _J_of_hist(run_forward_fn(U0_clean, solid_p, melt_p, vap_p, ell,
                                         theta_kappa=theta_kappa, kappa_param=kappa_param)[0])

    #if verbose:
    #    print("\n" + "-" * 72)
    #    print(f"{'Parameter':<12} {'Adjoint':>18} {'FD (central)':>18} "
    #          f"{'Rel err':>12} Status")
    #    print("-" * 72)

    if run_fd_check==True:
        results = {}
        param_pairs = [
            ('m0_m', 'm0', 'melt'),
            ('rho_vap0', 'rho_vap0', 'vap'),
            ('rho_vap1', 'rho_vap1', 'vap')
        ]
        for param_name, key, which in param_pairs:
            g_adj = float(g_phase[param_name])
            base = solid_prop[key] if which == 'solid' else melt_prop[key] if which == 'melt' else vap_prop[key]
            h_step = eps_fd * max(abs(base), 1e-8)

            sp = dict(solid_prop); sm = dict(solid_prop)
            mp = dict(melt_prop);  mm = dict(melt_prop)
            vp = dict(vap_prop);   vm = dict(vap_prop)
            if which == 'solid':
                sp[key] = base + h_step; sm[key] = base - h_step
            elif which == 'melt':
                mp[key] = base + h_step; mm[key] = base - h_step
            else:  # 'vap'
                vp[key] = base + h_step; vm[key] = base - h_step

            g_fd = (_J_perturbed(sp, mp, vp, U0_clean) - _J_perturbed(sm, mm, vm, U0_clean)) / (2 * h_step)
            denom = max(abs(g_adj), abs(g_fd), 1e-30)
            rel_err = abs(g_adj - g_fd) / denom
            status = "✓" if rel_err < 1e-2 else ("⚠" if rel_err < 1e-1 else "✗")

            if verbose:
                print(f"{param_name:<12} {g_adj:>+18.6e} {g_fd:>+18.6e} "
                    f"{rel_err:>12.3e} {status}")
            results[param_name] = {'adj': g_adj, 'fd': g_fd, 'rel_err': rel_err}

        # 5. Kappa FD check
        # -----------------------------------------------------------------
        # For each kappa layer l, perturb theta_kappa[l] by ±h and re-run
        # the forward solve, then compare central-FD to g_kappa_adj[l].
        # -----------------------------------------------------------------
        if g_kappa_adj is not None and theta_kappa is not None:
            theta_kappa_base = np.asarray(theta_kappa, float).copy()
            print(theta_kappa_base)
            for l in range(len(g_kappa_adj)):
                # Purely relative step — avoids huge perturbations for small params
                # (e.g. width=0.01 or y_transition=0.06) and prevents tk_m going ≤ 0.
                abs_val = abs(theta_kappa_base[l])
                h_k = eps_fd * max(abs_val, 1e-8)
                while theta_kappa_base[l] - h_k <= 0 and h_k > 1e-15:
                    h_k *= 0.5
                tk_p = theta_kappa_base.copy(); tk_p[l] += h_k
                tk_m = theta_kappa_base.copy(); tk_m[l] -= h_k

                # Rebuild kappa_param with perturbed values
                kappa_param_p = SigmoidLayeredKappa(
                    Ny=Ny, Ly=Ly,
                    kappa_surface=tk_p[0], kappa_deep=tk_p[1],
                    y_transition=tk_p[2], width=tk_p[3]
                )
                kappa_param_m = SigmoidLayeredKappa(
                    Ny=Ny, Ly=Ly,
                    kappa_surface=tk_m[0], kappa_deep=tk_m[1],
                    y_transition=tk_m[2], width=tk_m[3]
                )

                clear_caches_fn()
                J_p = _J_of_hist(run_forward_fn(U0_clean, solid_prop, melt_prop, vap_prop, ell,
                                                theta_kappa=tk_p, kappa_param=kappa_param_p)[0])
                clear_caches_fn()
                J_m = _J_of_hist(run_forward_fn(U0_clean, solid_prop, melt_prop, vap_prop, ell,
                                                theta_kappa=tk_m, kappa_param=kappa_param_m)[0])

                g_fd_l = (J_p - J_m) / (2.0 * h_k)
                g_adj_l = float(g_kappa_adj[l])
                denom = max(abs(g_adj_l), abs(g_fd_l), 1e-30)
                rel_err = abs(g_adj_l - g_fd_l) / denom
                status = "✓" if rel_err < 1e-2 else ("⚠" if rel_err < 1e-1 else "✗")

                if verbose:
                    print(f"theta_kappa_{l+1:<7} {g_adj_l:>+18.6e} {g_fd_l:>+18.6e} "
                        f"{rel_err:>12.3e} {status}")
                results[f"theta_kappa_{l+1}"] = {
                    'adj': g_adj_l, 'fd': g_fd_l, 'rel_err': rel_err
                }

    if verbose:
        print("-" * 72)
    return J_base, g_phase, g_kappa_adj

# ---------------------------------------------------------------------------

