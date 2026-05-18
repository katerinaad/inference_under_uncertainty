"""run_inf_depth.py

Gradient-based inference of physical parameters from ablation-depth
observations using the PDE-constrained adjoint of the SG heat solver.

Parameter vector
----------------
theta = [rho_vap0, rho_vap1, kappa_surface, kappa_deep, y_trans, width]

Optimisation is performed internally in log-space so that the vastly
different parameter scales (1e8 vs 0.01) do not hurt L-BFGS-B conditioning.
The public interface always works in natural (positive) space.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import minimize

from depth_objective import run_adjoint_depth, softmin_depth, _trap_weights
from stable_eigh_test import SPDEKLDifferentiator, SigmoidLayeredKappa

PARAM_NAMES = ['rho_vap0', 'rho_vap1', 'kappa_surface', 'kappa_deep', 'y_trans', 'width']


# ---------------------------------------------------------------------------
# Prior
# ---------------------------------------------------------------------------

class GaussianLogPrior:
    """
    Isotropic Gaussian prior on log(theta_i).

        -log p(theta) = 0.5 * sum_i ((log theta_i - log mu_i) / sigma_log_i)^2

    Set sigma_log_i = np.inf for uninformative (flat) components.
    """

    def __init__(self, theta_prior, sigma_log):
        self._log_mu    = np.log(np.asarray(theta_prior, float))
        self._sigma_log = np.asarray(sigma_log, float)

    def __call__(self, theta):
        """Return (nll, grad_nll_wrt_theta) — both in natural parameter space."""
        theta      = np.asarray(theta, float)
        log_theta  = np.log(np.maximum(theta, 1e-300))
        finite     = np.isfinite(self._sigma_log)
        d          = np.where(finite, (log_theta - self._log_mu) / self._sigma_log, 0.0)
        nll        = 0.5 * float(np.dot(d, np.where(finite, d, 0.0)))
        grad_log   = np.where(finite, d / np.where(finite, self._sigma_log, 1.0), 0.0)
        grad_theta = grad_log / np.maximum(theta, 1e-300)
        return nll, grad_theta


# ---------------------------------------------------------------------------
# Build helpers
# ---------------------------------------------------------------------------

def _build_vap(vap_base, rho_vap0, rho_vap1):
    v = dict(vap_base)
    v['rho_vap0'] = float(rho_vap0)
    v['rho_vap1'] = float(rho_vap1)
    return v


def _build_kappa_param(theta_kappa, Ny, Ly):
    return SigmoidLayeredKappa(
        Ny=Ny, Ly=Ly,
        kappa_surface=float(theta_kappa[0]),
        kappa_deep=float(theta_kappa[1]),
        y_transition=float(theta_kappa[2]),
        width=float(theta_kappa[3]),
    )


# ---------------------------------------------------------------------------
# Objective (no gradient)
# ---------------------------------------------------------------------------

def _compute_J(U_hist, h_obs_hist, sigma2_obs_hist,
               Nx, Ny, Ly, num_nodes, P, T_abl,
               eps_smooth=10.0, beta=0.005, sigma_d=1e-3,
               var_weight_hist=None, debug=False):
    """
    J = sum_t [ 0.5*(h_mean - h_obs)^2/sigma_d^2
              + 0.5*(sigma2_pred - sigma2_obs)^2 * var_weight  (if var_weight_hist given) ]
    """
    y_nodes    = np.linspace(0.0, Ly, Ny + 1)
    time_steps = U_hist.shape[0]
    inv_var    = 1.0 / sigma_d ** 2
    J          = 0.0
    J_mean     = 0.0
    J_var      = 0.0

    for t in range(1, time_steps):
        U_t       = U_hist[t]
        u_mean_2d = U_t[:num_nodes].reshape(Nx + 1, Ny + 1)
        h_u0, _, w_softmin, _, dH = softmin_depth(u_mean_2d, y_nodes, T_abl, eps_smooth, beta)

        r_mu   = h_u0 - float(h_obs_hist[t])
        dJ_mu  = 0.5 * r_mu ** 2 * inv_var
        J      += dJ_mu
        J_mean += dJ_mu

        if var_weight_hist is not None and var_weight_hist[t] > 0.0:
            wz     = _trap_weights(y_nodes)
            g_flat = (w_softmin[:, None] * wz[None, :] * dH).ravel()
            U_modes = U_t.reshape(P, num_nodes)
            a       = U_modes[1:] @ g_flat
            sigma2_pred = float(np.sum(a ** 2))
            r_var  = sigma2_pred - float(sigma2_obs_hist[t])
            dJ_var = 0.5 * r_var ** 2 * var_weight_hist[t]
            J      += dJ_var
            J_var  += dJ_var
            if debug:
                print(f"    t={t}  h_pred={h_u0:.4f}  h_obs={float(h_obs_hist[t]):.4f}"
                      f"  r_mu={r_mu:.3e}  dJ_mu={dJ_mu:.3e}"
                      f"  sigma2_pred={sigma2_pred:.3e}  sigma2_obs={float(sigma2_obs_hist[t]):.3e}"
                      f"  r_var={r_var:.3e}  w={var_weight_hist[t]:.3e}  dJ_var={dJ_var:.3e}")
        elif debug:
            print(f"    t={t}  h_pred={h_u0:.4f}  h_obs={float(h_obs_hist[t]):.4f}"
                  f"  r_mu={r_mu:.3e}  dJ_mu={dJ_mu:.3e}  [var inactive]")

    if debug:
        print(f"  J_mean={J_mean:.6e}  J_var={J_var:.6e}  J_total={J:.6e}")

    return J


# ---------------------------------------------------------------------------
# Core gradient evaluation
# ---------------------------------------------------------------------------

def _eval_gradient(
    theta, *,
    U_obs, h_obs_hist, sigma2_obs_hist, U0,
    SOLID, MELT, VAP_base, ell,
    run_forward_fn,
    run_adjoint_depth_fn,
    adjoint_one_step_fn,
    adjoint_grad_all_phase_fn,
    compute_adjoint_grad_kappa_fn,
    forcing_param_grads_numpy_fn,
    SPDEKLDifferentiator_cls,
    clear_caches_fn,
    kappa_param,          # initial template — rebuilt from theta each call
    bc_idx, params,
    Nx, Ny, Lx, Ly, N_KL, num_nodes, P, time_steps,
    T_abl, eps_smooth=10.0, sigma_d=1e-8, beta=0.005,
    prior=None,
    use_variance=False, f_rel=0.3,
):
    """
    Evaluate J(theta) and its gradient via forward + adjoint.

    theta = [rho_vap0, rho_vap1, kappa_surface, kappa_deep, y_trans, width]

    Returns
    -------
    J        : float
    grad     : (6,) ndarray  dJ/dtheta in natural space
    """
    rho_vap0, rho_vap1 = theta[0], theta[1]
    theta_kappa         = theta[2:]

    VAP_cur   = _build_vap(VAP_base, rho_vap0, rho_vap1)
    kappa_cur = _build_kappa_param(theta_kappa, Ny, Ly)
    tk_arr    = np.asarray(theta_kappa, float)

    # ── Forward ──────────────────────────────────────────────────────────────
    fwd = run_forward_fn(U0, SOLID, MELT, VAP_cur, ell,
                         theta_kappa=tk_arr, kappa_param=kappa_cur)
    (U_hist, _, _,
     M_bc, K_bc, solve_A,
     _, _, _,
     K_SG_K0, K_SG_K1, M_SG_M0, M_SG_M1,
     spatial_op, local_params) = fwd

    # ── Variance weights (relative scaling: sigma_var = f_rel * sigma2_obs) ───
    var_weight_hist = None
    if use_variance:
        sigma2_max = float(np.max(sigma2_obs_hist))
        thresh = max(1e-10, 1e-2 * sigma2_max)   # 1% of max — excludes pre-ablation noise floor
        var_weight_hist = np.where(
            sigma2_obs_hist > thresh,
            1.0 / (f_rel * sigma2_obs_hist) ** 2,
            0.0,
        )
        var_weight_hist[0] = 0.0  # skip t=0 (initial condition)

    # debug on first 3 calls
    _eval_gradient._ncalls = getattr(_eval_gradient, '_ncalls', 0) + 1
    _dbg = (_eval_gradient._ncalls <= 3)
    if _dbg:
        print(f"\n--- _eval_gradient call #{_eval_gradient._ncalls} ---")
        print(f"  theta = {theta}")
        if use_variance:
            print(f"  sigma2_max={sigma2_max:.3e}  thresh={thresh:.3e}")
            print(f"  var_weight_hist = {var_weight_hist}")

    J = _compute_J(U_hist, h_obs_hist, sigma2_obs_hist,
                   Nx, Ny, Ly, num_nodes, P, T_abl, eps_smooth, beta,
                   sigma_d=sigma_d, var_weight_hist=var_weight_hist, debug=_dbg)

    # ── Adjoint sweep ─────────────────────────────────────────────────────────
    # sigma2_for_adj carries sigma_d^2 as the mean-term denominator per step.
    sigma2_for_adj = np.full(time_steps, sigma_d ** 2)
    Mu_hist, _, _, _ = run_adjoint_depth_fn(
        U_obs, U_hist,
        M_bc, K_bc, solve_A,
        K_SG_K0, K_SG_K1, M_SG_M0, M_SG_M1,
        SOLID, MELT, VAP_cur, spatial_op, local_params, bc_idx,
        Nx, Ny, Ly, num_nodes, P, T_abl,
        adjoint_one_step_fn,
        sigma2_for_adj,
        eps=eps_smooth, beta=beta,
        h_obs_hist=h_obs_hist, sigma_d=sigma_d,
        mean_only=(not use_variance),
        var_weight_hist=var_weight_hist,
        sigma2_obs_var_hist=sigma2_obs_hist,
    )

    # ── Phase-parameter gradients (rho_vap0, rho_vap1) ───────────────────────
    g_phase = adjoint_grad_all_phase_fn(
        U_hist, Mu_hist, SOLID, MELT,
        K_SG_K1=K_SG_K1, M_SG_M1=M_SG_M1,
        forcing_param_grads_numpy=forcing_param_grads_numpy_fn,
        spatial_op=spatial_op, freeze_phase=False, vap_prop=VAP_cur,
    )

    # ── Kappa gradients (kappa_surface, kappa_deep, y_trans, width) ──────────
    _diff         = SPDEKLDifferentiator_cls(Nx, Ny, Lx, Ly, N_KL, kappa_cur)
    _res          = _diff.derivatives(tk_arr)
    eigvals_trunc = np.asarray(_res.eigvals, float)
    eigvecs_grid  = np.asarray(_res.eigvecs, float).reshape(Nx + 1, Ny + 1, N_KL)
    dlambda_all   = np.asarray(_res.dlambda, float)
    dphi_all_grid = np.asarray(_res.dphi,    float).reshape(Nx + 1, Ny + 1, N_KL, -1)

    g_kappa = compute_adjoint_grad_kappa_fn(
        U_hist, Mu_hist, SOLID, MELT, VAP_cur,
        eigvals_trunc=eigvals_trunc,
        eigvecs_reshaped=eigvecs_grid,
        dlambda_dkappa=dlambda_all,
        dphi_dkappa=dphi_all_grid,
        local_params=local_params,
        freeze_phase=False,
        include_forcing_dphi=True,
        coo=_diff.coo,
    )

    grad = np.array([
        float(g_phase.get('rho_vap0', 0.0)),
        float(g_phase.get('rho_vap1', 0.0)),
        *g_kappa,
    ])

    # ── Prior ─────────────────────────────────────────────────────────────────
    if prior is not None:
        nll_p, grad_p = prior(theta)
        J    += nll_p
        grad += grad_p

    return J, grad


# ---------------------------------------------------------------------------
# Inference loop
# ---------------------------------------------------------------------------

def run_depth_inference(
    *,
    U_obs, h_obs_hist, U0,
    SOLID_init, MELT_init, kappa_init, VAP_init, ell,
    prior,
    run_forward_fn,
    run_adjoint_depth_fn,
    adjoint_one_step_fn,
    adjoint_grad_all_phase_fn,
    compute_adjoint_grad_kappa_fn,
    forcing_param_grads_numpy_fn,
    clear_caches_fn,
    kappa_param,
    SPDEKLDifferentiator_cls,
    bc_idx, params,
    Nx, Ny, Lx, Ly, N_KL, num_nodes, P, time_steps, sigma2_obs_hist,
    T_abl, eps_smooth=10.0, sigma_d=1e-8,
    max_iter=50, beta=0.005,
    use_variance=False, f_rel=0.3,
    bounds_override=None,
):
    """
    Minimise  J(theta)  w.r.t.

        theta = [rho_vap0, rho_vap1, kappa_surface, kappa_deep, y_trans, width]

    using L-BFGS-B in log-space (s = log theta) for scale-invariant conditioning.

    Returns
    -------
    result : dict with keys
        theta_hist      (n_iter, 6) — theta at each objective call
        J_hist          (n_iter,)
        grad_norm_hist  (n_iter,)
        theta_final     (6,)
        theta0          (6,)
        J_final         float
        success         bool
        message         str
        PARAM_NAMES     list[str]
    """
    _eval_gradient._ncalls = 0   # reset debug counter for this run

    theta0 = np.array([
        VAP_init['rho_vap0'],
        VAP_init['rho_vap1'],
        *kappa_init,
    ], dtype=float)

    _grad_kw = dict(
        U_obs=U_obs, h_obs_hist=h_obs_hist, sigma2_obs_hist=sigma2_obs_hist, U0=U0,
        SOLID=SOLID_init, MELT=MELT_init, VAP_base=VAP_init, ell=ell,
        run_forward_fn=run_forward_fn,
        run_adjoint_depth_fn=run_adjoint_depth_fn,
        adjoint_one_step_fn=adjoint_one_step_fn,
        adjoint_grad_all_phase_fn=adjoint_grad_all_phase_fn,
        compute_adjoint_grad_kappa_fn=compute_adjoint_grad_kappa_fn,
        forcing_param_grads_numpy_fn=forcing_param_grads_numpy_fn,
        SPDEKLDifferentiator_cls=SPDEKLDifferentiator_cls,
        clear_caches_fn=clear_caches_fn,
        kappa_param=kappa_param,
        bc_idx=bc_idx, params=params,
        Nx=Nx, Ny=Ny, Lx=Lx, Ly=Ly,
        N_KL=N_KL, num_nodes=num_nodes, P=P, time_steps=time_steps,
        T_abl=T_abl, eps_smooth=eps_smooth, sigma_d=sigma_d, beta=beta,
        prior=prior,
        use_variance=use_variance, f_rel=f_rel,
    )

    history  = {'theta': [], 'J': [], 'grad_norm': []}
    itr      = [0]

    def _fg_log(s):
        theta = np.exp(s)
        clear_caches_fn()
        J, g_theta = _eval_gradient(theta, **_grad_kw)
        g_s        = g_theta * theta       # chain rule: dJ/ds_i = theta_i * dJ/dtheta_i
        history['theta'].append(theta.copy())
        history['J'].append(float(J))
        history['grad_norm'].append(float(np.linalg.norm(g_s)))
        itr[0] += 1
        param_str = '  '.join(f'{n}={v:.3g}' for n, v in zip(PARAM_NAMES, theta))
        print(f"  [{itr[0]:3d}]  J={J:.6e}  |g|={history['grad_norm'][-1]:.3e}  {param_str}")
        return float(J), g_s

    s0 = np.log(theta0)

    # Bounds in log-space: (log(lower), log(upper))
    bounds_log = [
        (np.log(1e3),  np.log(1e13)),          # rho_vap0
        (np.log(1e2),  np.log(1e12)),          # rho_vap1
        (np.log(0.5),  np.log(2e3)),           # kappa_surface
        (np.log(0.5),  np.log(2e3)),           # kappa_deep
        (np.log(1e-4), np.log(0.99 * Ly)),    # y_trans
        (np.log(1e-4), np.log(0.5  * Ly)),    # width
    ]
    if bounds_override is not None:
        for i, b in enumerate(bounds_override):
            if b is not None:
                bounds_log[i] = b

    res = minimize(
        _fg_log,
        s0,
        method='L-BFGS-B',
        jac=True,
        bounds=bounds_log,
        options={'maxiter': max_iter, 'ftol': 1e-15, 'gtol': 1e-8, 'iprint': -1},
    )

    theta_final = np.exp(res.x)

    return {
        'theta_hist':     np.array(history['theta']),
        'J_hist':         np.array(history['J']),
        'grad_norm_hist': np.array(history['grad_norm']),
        'theta_final':    theta_final,
        'theta0':         theta0,
        'J_final':        float(res.fun),
        'success':        bool(res.success),
        'message':        res.message,
        'PARAM_NAMES':    PARAM_NAMES,
    }


# ---------------------------------------------------------------------------
# Two-phase inference
# ---------------------------------------------------------------------------

def run_two_phase_inference(
    *,
    # Phase 1 controls which parameters are free in the mean-only phase.
    # phase1_free: list of PARAM_NAMES to optimise in phase 1 (default: rho_vap0 only)
    phase1_free=('rho_vap0',),
    # phase2_free: list of PARAM_NAMES to optimise in phase 2 (default: all except rho_vap0)
    phase2_free=('rho_vap1', 'kappa_surface', 'kappa_deep', 'y_trans', 'width'),
    max_iter_phase1=50,
    max_iter_phase2=100,
    f_rel=0.3,
    **common_kw,
):
    """
    Two-phase adjoint inference to handle parameter degeneracy.

    Phase 1 (mean-only):
        Optimises `phase1_free` parameters while keeping the rest fixed.
        Uses only the mean-depth term so the stochastic variance gradient
        does not dominate.

    Phase 2 (mean + variance):
        Starts from the Phase 1 result.  Fixes the Phase 1 parameters at
        their converged values and optimises `phase2_free`.  Adding the
        variance term now works because the Phase 1 degeneracy is broken
        (rho_vap0 is at its true value so kappa=2000 no longer fits
        h_mean).

    common_kw: all keyword arguments accepted by run_depth_inference
               (U_obs, h_obs_hist, sigma2_obs_hist, …).
    """
    # ── Shared callables / mesh args ─────────────────────────────────────────
    Ly     = common_kw['Ly']
    VAP_init   = common_kw['VAP_init']
    kappa_init = common_kw['kappa_init']

    theta0_full = np.array([
        VAP_init['rho_vap0'],
        VAP_init['rho_vap1'],
        *kappa_init,
    ], dtype=float)

    def _pin_bounds(free_names, theta_ref):
        """Return bounds_override that fixes all params NOT in free_names."""
        bds = []
        for i, name in enumerate(PARAM_NAMES):
            if name in free_names:
                bds.append(None)          # use default bound
            else:
                v = float(np.log(theta_ref[i]))
                bds.append((v, v))        # equality bound → parameter frozen
        return bds

    # ── Phase 1: mean-only, only phase1_free params move ─────────────────────
    print(f"\n{'='*60}")
    print(f"  PHASE 1 (mean-only)  free: {list(phase1_free)}")
    print(f"{'='*60}")

    bds1 = _pin_bounds(phase1_free, theta0_full)
    result1 = run_depth_inference(
        **{k: v for k, v in common_kw.items()
           if k not in ('use_variance', 'f_rel', 'max_iter', 'bounds_override')},
        use_variance=False,
        f_rel=f_rel,
        max_iter=max_iter_phase1,
        bounds_override=bds1,
    )

    theta1 = result1['theta_final']
    print(f"\n  Phase 1 converged: "
          + '  '.join(f'{n}={v:.3g}' for n, v in zip(PARAM_NAMES, theta1)))

    # ── Phase 2: mean + variance, only phase2_free params move ───────────────
    print(f"\n{'='*60}")
    print(f"  PHASE 2 (mean+variance)  free: {list(phase2_free)}")
    print(f"{'='*60}")

    # Build new VAP_init and kappa_init from Phase 1 result
    vap2 = dict(VAP_init)
    vap2['rho_vap0'] = float(theta1[0])
    vap2['rho_vap1'] = float(theta1[1])
    kappa2 = list(theta1[2:])

    bds2 = _pin_bounds(phase2_free, theta1)

    kw2 = dict(common_kw)
    kw2['VAP_init']   = vap2
    kw2['kappa_init'] = kappa2
    # Rebuild kappa_param from the Phase 1 kappa values
    kw2['kappa_param'] = _build_kappa_param(kappa2, common_kw['Ny'], common_kw['Ly'])

    result2 = run_depth_inference(
        **{k: v for k, v in kw2.items()
           if k not in ('use_variance', 'f_rel', 'max_iter', 'bounds_override')},
        use_variance=True,
        f_rel=f_rel,
        max_iter=max_iter_phase2,
        bounds_override=bds2,
    )

    return {'phase1': result1, 'phase2': result2}


# ---------------------------------------------------------------------------
# Convergence diagnostics
# ---------------------------------------------------------------------------

def plot_convergence(result, kappa_true=None):
    """
    Plot J convergence and parameter trajectories.
    Saves to convergence.png and prints a summary table.

    kappa_true : (4,) array  [kappa_surface, kappa_deep, y_trans, width]
                 of the true kappa params (only kappa_true[0:4] are plotted;
                 rho_vap true values are not required).
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Objective
    ax = axes[0]
    J = result['J_hist']
    ax.semilogy(np.maximum(J - J.min() + 1e-30, 1e-30), label='J - J_min')
    ax.semilogy(result['grad_norm_hist'], ls='--', label='|∇J|_log')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Value (log scale)')
    ax.set_title('Convergence')
    ax.legend(fontsize=8)
    ax.grid(True, which='both', ls=':')

    # Parameter trajectories normalised by initial value
    ax = axes[1]
    theta_hist = result['theta_hist']
    theta0     = result['theta0']
    names      = result['PARAM_NAMES']

    if theta_hist.ndim == 2 and theta_hist.shape[0] > 0:
        for i, name in enumerate(names):
            norm = theta_hist[:, i] / max(abs(theta0[i]), 1e-300)
            ax.plot(norm, label=name, lw=1.5)

        if kappa_true is not None:
            kappa_true = np.asarray(kappa_true, float)
            for j, tv in enumerate(kappa_true):
                i = j + 2      # offset: rho_vap0, rho_vap1 are indices 0,1
                norm_true = tv / max(abs(theta0[i]), 1e-300)
                ax.axhline(norm_true, ls='--', lw=0.8,
                           color=f'C{i}', alpha=0.7, label=f'{names[i]} true')

    ax.axhline(1.0, ls=':', color='k', alpha=0.4, lw=0.8)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('theta / theta_0')
    ax.set_title('Parameter trajectories (normalised by initial)')
    ax.legend(fontsize=6, ncol=2)
    ax.grid(True, ls=':')

    plt.tight_layout()
    plt.savefig('convergence.png', dpi=120)
    plt.show()
    print('Saved convergence.png')

    # ── Summary table ─────────────────────────────────────────────────────────
    kappa_true_ext = [None, None] + (list(kappa_true) if kappa_true is not None else [None] * 4)
    theta_f = result['theta_final']
    print('\n' + '=' * 68)
    print(f"  {'Param':<16}  {'Init':>14}  {'Final':>14}  {'True':>14}")
    print('  ' + '-' * 64)
    for i, name in enumerate(names):
        tv  = kappa_true_ext[i]
        row = f"  {name:<16}  {theta0[i]:>14.4g}  {theta_f[i]:>14.4g}"
        row += f"  {tv:>14.4g}" if tv is not None else f"  {'—':>14}"
        print(row)
    print(f'\n  J_final = {result["J_final"]:.6e}')
    print(f'  {result["message"]}')
    print('=' * 68)


# ---------------------------------------------------------------------------
# Gradient validation (FD check of the full 6-D gradient)
# ---------------------------------------------------------------------------

def validate_obj_and_grad(
    U_obs, h_obs_hist, U0,
    SOLID, MELT, theta_kappa_init, VAP, ell,
    run_forward_fn,
    run_adjoint_depth_fn,
    adjoint_one_step_fn,
    adjoint_grad_all_phase_fn,
    compute_adjoint_grad_kappa_fn,
    forcing_param_grads_numpy_fn,
    clear_caches_fn,
    kappa_param,
    SPDEKLDifferentiator_cls,
    bc_idx, params,
    Nx, Ny, Lx, Ly, N_KL, num_nodes, P, time_steps, sigma2_obs_hist,
    T_abl, eps_smooth=10.0, beta=None, sigma_d=1e-8,
    evap_range=0.0, eps_fd=1e-4,
):
    """
    Central-difference FD check of grad_J w.r.t. theta at the initial point.

    Uses only forward solves for FD (no adjoint), then compares against the
    full adjoint gradient.
    """
    if beta is None:
        beta = 0.005

    theta0 = np.array([
        VAP['rho_vap0'],
        VAP['rho_vap1'],
        *theta_kappa_init,
    ], dtype=float)

    _grad_kw = dict(
        U_obs=U_obs, h_obs_hist=h_obs_hist, sigma2_obs_hist=sigma2_obs_hist, U0=U0,
        SOLID=SOLID, MELT=MELT, VAP_base=VAP, ell=ell,
        run_forward_fn=run_forward_fn,
        run_adjoint_depth_fn=run_adjoint_depth_fn,
        adjoint_one_step_fn=adjoint_one_step_fn,
        adjoint_grad_all_phase_fn=adjoint_grad_all_phase_fn,
        compute_adjoint_grad_kappa_fn=compute_adjoint_grad_kappa_fn,
        forcing_param_grads_numpy_fn=forcing_param_grads_numpy_fn,
        SPDEKLDifferentiator_cls=SPDEKLDifferentiator_cls,
        clear_caches_fn=clear_caches_fn,
        kappa_param=kappa_param,
        bc_idx=bc_idx, params=params,
        Nx=Nx, Ny=Ny, Lx=Lx, Ly=Ly,
        N_KL=N_KL, num_nodes=num_nodes, P=P, time_steps=time_steps,
        T_abl=T_abl, eps_smooth=eps_smooth, sigma_d=sigma_d, beta=beta,
        prior=None,
    )

    # Base point
    clear_caches_fn()
    J0, grad_adj = _eval_gradient(theta0, **_grad_kw)

    print('\n' + '-' * 76)
    print(f'  J0 = {J0:.6e}')
    print(f"  {'Parameter':<18} {'Adjoint':>16} {'FD central':>16} {'Rel err':>10}  Status")
    print('-' * 76)

    # FD: only re-run forward, not the full adjoint
    def _J_only(theta_pert):
        rho_vap0, rho_vap1 = theta_pert[0], theta_pert[1]
        tk    = theta_pert[2:]
        VAP_p = _build_vap(VAP, rho_vap0, rho_vap1)
        kp    = _build_kappa_param(tk, Ny, Ly)
        fwd   = run_forward_fn(U0, SOLID, MELT, VAP_p, ell,
                               theta_kappa=np.asarray(tk, float), kappa_param=kp)
        U_h   = fwd[0]
        return _compute_J(U_h, h_obs_hist, sigma2_obs_hist,
                          Nx, Ny, Ly, num_nodes, P, T_abl, eps_smooth, beta,
                          sigma_d=sigma_d)

    results = {}
    for i, name in enumerate(PARAM_NAMES):
        h   = eps_fd * max(abs(theta0[i]), 1e-8)
        tp  = theta0.copy(); tp[i] += h
        tm  = theta0.copy(); tm[i] -= h

        clear_caches_fn()
        Jp = _J_only(tp)
        clear_caches_fn()
        Jm = _J_only(tm)

        g_fd  = (Jp - Jm) / (2.0 * h)
        g_adj = float(grad_adj[i])
        denom = max(abs(g_adj), abs(g_fd), 1e-30)
        rel   = abs(g_adj - g_fd) / denom
        status = '✓' if rel < 1e-2 else ('⚠' if rel < 1e-1 else '✗')

        print(f"  {name:<18} {g_adj:>+16.6e} {g_fd:>+16.6e} {rel:>10.3e}  {status}")
        results[name] = {'adj': g_adj, 'fd': g_fd, 'rel_err': rel}

    print('-' * 76)
    return results
