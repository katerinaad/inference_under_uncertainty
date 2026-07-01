"""
scan_stage1_ridge.py
====================
Ridge discovery for Stage 1 (rho_vap0, f0_m) in log-space.

Adds two diagnostics ON TOP of the existing inf_layered_vap_exp.py globals:

  1. GN_HESSIAN  (4 forward solves)
     Compute the Jacobian of the depth-trajectory residuals w.r.t. log(rho)
     and log(f0) via central FD.  H_GN = J^T J gives:
       - condition number (how elongated the valley is)
       - SVD-based ridge direction (right singular vector of smallest σ)
       - Cramér-Rao bound for each parameter

  2. CONTOUR_SCAN  (n_rho × n_f0 forward solves, optional)
     Evaluate J on a 2D grid in log-space.  Saves to ridge_scan.npy and
     produces a filled-contour plot with the GN-eigenvectors overlaid.

Usage
-----
  Run after inf_layered_vap_exp.py has been fully initialised, e.g. by
  appending:

      from scan_stage1_ridge import run_ridge_diagnostics
      run_ridge_diagnostics()

  at the bottom of inf_layered_vap_exp.py.  All required globals (run_forward,
  U0, U_obs, SOLID_obs, MELT_obs, VAP_obs, VAP, MELT, h_obs_hist, etc.) must
  already exist in the calling namespace.

  Alternatively, run the file directly after inf_layered_vap_exp.py has been
  exec'd in the same process.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

# ─── We depend on globals from inf_layered_vap_exp.py ────────────────────────
# Import them lazily inside the function so this file is importable on its own.


def _eval_J_forward(rho0, f0_m,
                    VAP_base, MELT_base, run_forward_fn, clear_caches_fn,
                    U0, SOLID_obs, ell, theta_kappa_obs, kappa_param_obs,
                    h_obs_hist, sigma2_obs_hist,
                    Nx, Ny, Ly, num_nodes, T_abl, sigma_d, eps_smooth):
    """Run one forward solve and return (J_scalar, h_pred_array)."""
    from depth_objective import softmin_depth, _trap_weights

    VAP_cur  = {**VAP_base,  'rho_vap0': float(rho0)}
    MELT_cur = {**MELT_base, 'f0':        float(f0_m)}
    clear_caches_fn()
    U_hist, *_ = run_forward_fn(U0, SOLID_obs, MELT_cur, VAP_cur, ell,
                                theta_kappa=theta_kappa_obs,
                                kappa_param=kappa_param_obs)

    y_nodes  = np.linspace(0.0, Ly, Ny + 1)
    inv_var  = 1.0 / sigma_d ** 2
    T        = U_hist.shape[0]
    h_pred   = np.zeros(T)
    J        = 0.0
    for t in range(1, T):
        u_mean_2d = U_hist[t, :num_nodes].reshape(Nx + 1, Ny + 1)
        h_t, *_   = softmin_depth(u_mean_2d, y_nodes, T_abl, eps_smooth, 0.005)
        h_pred[t] = h_t
        r          = h_t - float(h_obs_hist[t])
        J         += 0.5 * r ** 2 * inv_var
    return J, h_pred


def run_ridge_diagnostics(
    *,
    # ── parameter point to analyse ──────────────────────────────────────────
    rho_stall=None,   # if None, read from stage1_log.npz
    f0_stall=None,
    # ── FD step (in log-space) ───────────────────────────────────────────────
    fd_h=0.05,        # 5% relative perturbation in log-space
    # ── 2D scan grid (set do_scan=False to skip the expensive part) ─────────
    do_scan=True,
    n_rho=10,
    n_f0=10,
    rho_lo=6e8,  rho_hi=2.0e9,
    f0_lo=0.35,  f0_hi=0.95,
    resume_scan=True,   # load ridge_scan.npy if it exists, fill missing cells
    # ── globals injected from inf_layered_vap_exp.py ──────────────────────────
    globals_dict=None,  # pass globals() from the caller; defaults to inspect
):
    """
    Parameters
    ----------
    rho_stall, f0_stall : float  (natural space)
        The parameter point at which to compute the GN Hessian.
        Defaults to the best-J point in stage1_log.npz.
    fd_h : float
        Central-difference step in log(θ): θ_+ = θ * exp(+fd_h).
    do_scan : bool
        Whether to run the 2D contour scan (expensive).
    n_rho, n_f0 : int
        Grid resolution for the scan.
    rho_lo/hi, f0_lo/hi : float
        Grid bounds (natural space).
    resume_scan : bool
        If ridge_scan.npy exists, resume rather than recompute.
    globals_dict : dict
        Pass globals() from the calling script so we can read the
        shared setup (run_forward, U0, …).  If None we try sys.modules.
    """
    import sys
    if globals_dict is None:
        # Try to pull from the top-level namespace of the calling module
        frame = sys._getframe(1)
        globals_dict = frame.f_globals

    # ── Unpack required globals ───────────────────────────────────────────────
    G = globals_dict
    run_forward_fn     = G['run_forward']
    clear_caches_fn    = G['_clear_all_kappa_caches']
    U0                 = G['U0']
    SOLID_obs          = G['SOLID_obs']
    VAP_base           = {**G['VAP_obs'],
                          'rho_vap0': G['VAP']['rho_vap0'],
                          'rho_vap1': G['VAP']['rho_vap1']}
    MELT_base          = {**G['MELT_obs'],
                          'f0': G['MELT']['f0']}
    ell                = G['ell']
    theta_kappa_obs    = G['theta_kappa_obs']
    kappa_param_obs    = G['kappa_param_obs']
    h_obs_hist         = G['h_obs_hist']
    sigma2_obs_hist    = G['sigma2_obs_hist']
    Nx, Ny, Ly         = G['Nx'], G['Ny'], G['Ly']
    num_nodes          = G['num_nodes']
    T_abl              = G['T_abl']
    sigma_d            = G['sigma_d']
    eps_smooth         = G['eps_smooth']
    rho_truth          = float(G['VAP_obs']['rho_vap0'])
    f0_truth           = float(G['MELT_obs']['f0'])

    fwd_kwargs = dict(
        VAP_base=VAP_base, MELT_base=MELT_base,
        run_forward_fn=run_forward_fn, clear_caches_fn=clear_caches_fn,
        U0=U0, SOLID_obs=SOLID_obs, ell=ell,
        theta_kappa_obs=theta_kappa_obs, kappa_param_obs=kappa_param_obs,
        h_obs_hist=h_obs_hist, sigma2_obs_hist=sigma2_obs_hist,
        Nx=Nx, Ny=Ny, Ly=Ly, num_nodes=num_nodes,
        T_abl=T_abl, sigma_d=sigma_d, eps_smooth=eps_smooth,
    )

    # ── Load stall point from npz if not supplied ─────────────────────────────
    if rho_stall is None or f0_stall is None:
        npz = np.load("stage1_log.npz", allow_pickle=True)
        best_idx   = np.argmin(npz['J'])
        rho_stall  = float(npz['rho_vap0'][best_idx])
        f0_stall   = float(npz['f0_v'][best_idx])
        print(f"  Using stall point from stage1_log.npz: "
              f"rho={rho_stall:.4e}  f0={f0_stall:.6f}")

    # ══════════════════════════════════════════════════════════════════════════
    # DIAGNOSTIC 1: Gauss-Newton Hessian  (4 forward solves)
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 64)
    print("  GN HESSIAN  —  4 forward solves at perturbed (rho, f0)")
    print("=" * 64)

    s0 = np.log(rho_stall)
    t0 = np.log(f0_stall)
    h  = fd_h

    rho_p = np.exp(s0 + h);   rho_m = np.exp(s0 - h)
    f0_p  = np.exp(t0 + h);   f0_m_ = np.exp(t0 - h)

    print(f"  Base:   rho={rho_stall:.4e}  f0={f0_stall:.6f}")
    print(f"  ±δ(log): h = {h:.3f}")
    print(f"  rho: [{rho_m:.4e}, {rho_p:.4e}]   f0: [{f0_m_:.6f}, {f0_p:.6f}]")
    print()

    print("  [1/4] rho+δ ...", end=' ', flush=True)
    J_p0, h_pp0 = _eval_J_forward(rho_p, f0_stall, **fwd_kwargs)
    print(f"J={J_p0:.4e}")

    print("  [2/4] rho-δ ...", end=' ', flush=True)
    J_m0, h_pm0 = _eval_J_forward(rho_m, f0_stall, **fwd_kwargs)
    print(f"J={J_m0:.4e}")

    print("  [3/4] f0+δ  ...", end=' ', flush=True)
    J_0p, h_p0p = _eval_J_forward(rho_stall, f0_p,  **fwd_kwargs)
    print(f"J={J_0p:.4e}")

    print("  [4/4] f0-δ  ...", end=' ', flush=True)
    J_0m, h_p0m = _eval_J_forward(rho_stall, f0_m_, **fwd_kwargs)
    print(f"J={J_0m:.4e}")

    # ── Jacobian of depth trajectory w.r.t. log(rho), log(f0) ─────────────────
    # dh(t)/d(log rho) ≈ [h(rho+δ,f0) - h(rho-δ,f0)] / (2h)
    dh_ds = (h_pp0 - h_pm0) / (2 * h)   # (T,)
    dh_dt = (h_p0p - h_p0m) / (2 * h)   # (T,)

    # Weighted Jacobian rows: r_i = (dh_i/dθ) / σ_d
    inv_sd = 1.0 / sigma_d
    J_mat  = np.column_stack([dh_ds[1:] * inv_sd,
                               dh_dt[1:] * inv_sd])   # (T-1, 2)

    # Gauss-Newton Hessian in log-space
    H_gn   = J_mat.T @ J_mat    # (2, 2), units: [dimensionless]

    # SVD of J_mat: right sing. vecs = parameter directions
    U_sv, sv, Vt = np.linalg.svd(J_mat, full_matrices=False)
    eigvals_gn, eigvecs_gn = np.linalg.eigh(H_gn)

    cond = sv[0] / max(sv[1], 1e-30)   # condition number of Jacobian

    print(f"\n  Jacobian singular values:  σ₁={sv[0]:.4e}  σ₂={sv[1]:.4e}")
    print(f"  Condition number (σ₁/σ₂) = {cond:.1f}")
    print(f"\n  GN Hessian H_GN (log-space):")
    print(f"    [[{H_gn[0,0]:.4e}, {H_gn[0,1]:.4e}],")
    print(f"     [{H_gn[1,0]:.4e}, {H_gn[1,1]:.4e}]]")
    print(f"\n  GN eigenvalues: λ_min={eigvals_gn[0]:.4e}  λ_max={eigvals_gn[1]:.4e}")
    print(f"  GN condition number (λ_max/λ_min) = {eigvals_gn[1]/max(eigvals_gn[0],1e-30):.1f}")

    # Ridge direction = right singular vector of SMALLEST σ (most insensitive direction)
    ridge_hat  = Vt[1]   # right singular vector for σ₂ (smallest), shape (2,)
    print(f"\n  Ridge direction (right svec of σ_min) in (log ρ, log f0):")
    print(f"    v_ridge = [{ridge_hat[0]:.4f}, {ridge_hat[1]:.4f}]")
    print(f"    → if ρ increases by 1% (δ log ρ = 0.01),")
    print(f"      f0 must change by {ridge_hat[1]/ridge_hat[0]*0.01*100:.3f}% to stay on ridge")

    # Cramér-Rao lower bound: std(log θ_i) ≥ sqrt([H_GN^{-1}]_ii)
    try:
        H_inv = np.linalg.inv(H_gn)
        cr_std_s = np.sqrt(max(H_inv[0, 0], 0.0))   # std of log(rho)
        cr_std_t = np.sqrt(max(H_inv[1, 1], 0.0))   # std of log(f0)
        print(f"\n  Cramér-Rao bounds (from GN Hessian):")
        print(f"    std(log ρ)  ≥ {cr_std_s:.4f}  → relative error in ρ  ≥ {cr_std_s*100:.1f}%")
        print(f"    std(log f0) ≥ {cr_std_t:.4f}  → relative error in f0 ≥ {cr_std_t*100:.1f}%")
    except np.linalg.LinAlgError:
        print("  H_GN is singular — parameters are completely unidentifiable!")
        H_inv = np.full((2, 2), np.inf)
        cr_std_s = cr_std_t = np.inf

    # Angle between ridge direction and stall→truth vector
    vec_to_truth = np.array([np.log(rho_truth) - np.log(rho_stall),
                              np.log(f0_truth)  - np.log(f0_stall)])
    cos_angle = np.dot(vec_to_truth, ridge_hat) / (
        np.linalg.norm(vec_to_truth) * np.linalg.norm(ridge_hat) + 1e-30)
    angle_deg = np.degrees(np.arccos(np.clip(abs(cos_angle), 0, 1)))
    print(f"\n  Direction stall→truth (log-space): {vec_to_truth}")
    print(f"  Angle between ridge and stall→truth: {angle_deg:.1f}°")
    print(f"  (0° = fully on ridge, 90° = perpendicular to ridge)")

    # ── Save GN results ───────────────────────────────────────────────────────
    np.savez("stage1_gn_hessian.npz",
             H_gn=H_gn, J_mat=J_mat, sv=sv, Vt=Vt,
             ridge_hat=ridge_hat, cond=cond,
             rho_stall=rho_stall, f0_stall=f0_stall,
             rho_truth=rho_truth, f0_truth=f0_truth)
    print("\n  Saved stage1_gn_hessian.npz")

    # ══════════════════════════════════════════════════════════════════════════
    # DIAGNOSTIC 2: 2D contour scan  (n_rho * n_f0 forward solves)
    # ══════════════════════════════════════════════════════════════════════════
    J_grid     = None
    rho_grid   = np.logspace(np.log10(rho_lo), np.log10(rho_hi), n_rho)
    f0_grid    = np.linspace(f0_lo, f0_hi, n_f0)
    scan_file  = "ridge_scan.npy"

    if do_scan:
        print("\n" + "=" * 64)
        print(f"  2D CONTOUR SCAN  ({n_rho}×{n_f0} = {n_rho*n_f0} forward solves)")
        print(f"  rho ∈ [{rho_lo:.2e}, {rho_hi:.2e}]  (log-spaced)")
        print(f"  f0  ∈ [{f0_lo:.3f}, {f0_hi:.3f}]  (linear)")
        print("=" * 64)

        if resume_scan and os.path.exists(scan_file):
            J_grid = np.load(scan_file)
            print(f"  Resuming from {scan_file} "
                  f"({np.isfinite(J_grid).sum()} of {n_rho*n_f0} cells done)")
        else:
            J_grid = np.full((n_rho, n_f0), np.nan)

        n_total = n_rho * n_f0
        done    = 0
        for i, rho in enumerate(rho_grid):
            for j, f0 in enumerate(f0_grid):
                if np.isfinite(J_grid[i, j]):
                    done += 1
                    continue
                done += 1
                print(f"  [{done:3d}/{n_total}] rho={rho:.3e}  f0={f0:.3f} ...",
                      end=' ', flush=True)
                J_val, _ = _eval_J_forward(rho, f0, **fwd_kwargs)
                J_grid[i, j] = J_val
                print(f"J={J_val:.4e}")
                np.save(scan_file, J_grid)   # checkpoint after every cell

        np.save(scan_file, J_grid)
        np.save("ridge_scan_axes.npy",
                np.array([rho_grid, f0_grid], dtype=object))
        print(f"\n  Saved {scan_file} and ridge_scan_axes.npy")

    # ══════════════════════════════════════════════════════════════════════════
    # PLOT
    # ══════════════════════════════════════════════════════════════════════════
    # Load optimisation path
    npz   = np.load("stage1_log.npz", allow_pickle=True)
    s_path = np.log10(npz['rho_vap0'])
    t_path = np.log10(npz['f0_v'])
    g_path = npz['g']          # log-space gradients
    J_path = npz['J']

    if J_grid is not None:
        _plot_scan_with_gn(
            J_grid, rho_grid, f0_grid,
            rho_stall, f0_stall, rho_truth, f0_truth,
            ridge_hat, sv, H_gn,
            s_path, t_path, g_path, J_path,
        )
    else:
        _plot_gn_only(
            rho_stall, f0_stall, rho_truth, f0_truth,
            ridge_hat, sv,
            s_path, t_path, g_path, J_path,
        )

    print("\nRidge diagnostics complete.")
    return dict(H_gn=H_gn, sv=sv, Vt=Vt, ridge_hat=ridge_hat,
                cond=cond, J_grid=J_grid,
                rho_grid=rho_grid, f0_grid=f0_grid)


def _plot_scan_with_gn(J_grid, rho_grid, f0_grid,
                       rho_stall, f0_stall, rho_truth, f0_truth,
                       ridge_hat, sv, H_gn,
                       s_path, t_path, g_path, J_path):
    """2D filled contour + optimisation path + GN eigenvectors."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # ── left: contour ─────────────────────────────────────────────────────────
    ax = axes[0]
    RHO_mg, F0_mg = np.meshgrid(np.log10(rho_grid), np.log10(f0_grid), indexing='ij')
    J_log = np.log10(np.clip(J_grid, 1.0, None))

    pcm = ax.contourf(RHO_mg, F0_mg, J_log, levels=25, cmap='viridis_r')
    ax.contour(RHO_mg, F0_mg, J_log, levels=15, colors='k', linewidths=0.4, alpha=0.5)
    plt.colorbar(pcm, ax=ax, label='log₁₀(J)')

    # Optimisation path
    ax.plot(s_path, t_path, 'o-', color='white', lw=1.5, ms=4, zorder=6, label='optim path')

    # Key points
    ax.plot(np.log10(rho_stall), np.log10(f0_stall), '*', color='red',
            ms=14, zorder=10, label=f'stall (J={J_path.min():.1e})')
    ax.plot(np.log10(rho_truth), np.log10(f0_truth), 'D', color='lime',
            ms=10, zorder=10, label='truth')

    # GN eigenvectors at stall point (ridge = most insensitive direction)
    s0, t0 = np.log10(rho_stall), np.log10(f0_stall)
    _, eigvecs_gn = np.linalg.eigh(H_gn)
    for k in range(2):
        vec = eigvecs_gn[:, k] / np.log(10)   # ln → log10 scale
        length = 0.25 / max(abs(vec).max(), 1e-10)
        color  = 'cyan' if k == 0 else 'yellow'
        label  = f'ridge (λ_min={np.linalg.eigvalsh(H_gn)[k]:.1e})' if k==0 \
                 else f'steep (λ_max={np.linalg.eigvalsh(H_gn)[k]:.1e})'
        ax.annotate("", xy=(s0 + length*vec[0], t0 + length*vec[1]),
                    xytext=(s0 - length*vec[0], t0 - length*vec[1]),
                    arrowprops=dict(arrowstyle="<->", color=color, lw=2.0, mutation_scale=10))
        ax.text(s0 + length*vec[0], t0 + length*vec[1], label,
                color=color, fontsize=8, ha='left')

    ax.set_xlabel('log₁₀(ρ_vap0)')
    ax.set_ylabel('log₁₀(f0)')
    ax.set_title('J landscape (contour) + GN eigenvectors\ncyan=ridge, yellow=steep')
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(True, ls=':', alpha=0.3)

    # ── right: J along ridge and steep directions from stall ──────────────────
    ax2 = axes[1]
    n_pts = 30
    alphas = np.linspace(-0.5, 0.5, n_pts)   # log-space offset along direction

    _, eigvecs_gn_nat = np.linalg.eigh(H_gn)  # in natural log-space
    J_along_ridge = []
    J_along_steep = []

    rho_ridge = rho_stall * np.exp(alphas * eigvecs_gn_nat[0, 0])
    f0_ridge  = f0_stall  * np.exp(alphas * eigvecs_gn_nat[1, 0])
    rho_steep = rho_stall * np.exp(alphas * eigvecs_gn_nat[0, 1])
    f0_steep  = f0_stall  * np.exp(alphas * eigvecs_gn_nat[1, 1])

    # Interpolate from J_grid (bilinear)
    from scipy.interpolate import RegularGridInterpolator
    interp = RegularGridInterpolator(
        (np.log(rho_grid), np.log(f0_grid)), J_grid,
        method='linear', bounds_error=False, fill_value=np.nan)

    pts_ridge = np.column_stack([np.log(rho_ridge), np.log(f0_ridge)])
    pts_steep = np.column_stack([np.log(rho_steep), np.log(f0_steep)])
    J_ridge   = interp(pts_ridge)
    J_steep   = interp(pts_steep)

    ax2.plot(alphas, J_ridge, 'c-o', ms=4, lw=2, label='along ridge (λ_min direction)')
    ax2.plot(alphas, J_steep, 'y-o', ms=4, lw=2, label='along steep (λ_max direction)')
    ax2.axvline(0, color='red', ls='--', lw=1.5, label='stall point')
    ax2.set_xlabel('Offset along direction (log-space units)')
    ax2.set_ylabel('J')
    ax2.set_title('J cross-sections through stall point\n(flat ridge vs steep valley)')
    ax2.legend(fontsize=9)
    ax2.grid(True, ls=':')

    plt.tight_layout()
    plt.savefig('ridge_scan_contour.png', dpi=150)
    print("  Saved ridge_scan_contour.png")
    plt.close(fig)


def _plot_gn_only(rho_stall, f0_stall, rho_truth, f0_truth,
                  ridge_hat, sv, s_path, t_path, g_path, J_path):
    """Plot just the optimisation path with GN ridge direction (no scan needed)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.plot(s_path, t_path, 'o-', color='royalblue', lw=1.5, ms=5, label='optim path')
    ax.plot(np.log10(rho_stall), np.log10(f0_stall), '*', color='red',
            ms=14, zorder=10, label=f'stall (J={J_path.min():.1e})')
    ax.plot(np.log10(rho_truth), np.log10(f0_truth), 'D', color='green',
            ms=10, zorder=10, label='truth')

    # Ridge direction arrow (in log10 scale)
    s0, t0  = np.log10(rho_stall), np.log10(f0_stall)
    rh10    = ridge_hat / np.log(10)
    length  = 0.30
    ax.annotate("", xy=(s0 + length*rh10[0], t0 + length*rh10[1]),
                xytext=(s0 - length*rh10[0], t0 - length*rh10[1]),
                arrowprops=dict(arrowstyle="<->", color='cyan', lw=2.5, mutation_scale=12))
    ax.text(s0 + length*rh10[0] + 0.01, t0 + length*rh10[1],
            f'ridge\nσ₁/σ₂={sv[0]/max(sv[1],1e-30):.0f}', color='cyan', fontsize=9)

    # Gradient arrows at each iterate
    for i in range(len(s_path)):
        gs = g_path[i, 0]; gt = g_path[i, 1]
        gn = max(np.sqrt(gs**2 + gt**2), 1e-15)
        sc = 0.07
        ax.annotate("", xy=(s_path[i] - sc*gs/gn/np.log(10),
                             t_path[i] - sc*gt/gn/np.log(10)),
                    xytext=(s_path[i], t_path[i]),
                    arrowprops=dict(arrowstyle="-|>", color='orange',
                                    lw=0.8, mutation_scale=7))

    ax.set_xlabel('log₁₀(ρ_vap0)');  ax.set_ylabel('log₁₀(f0)')
    ax.set_title(f'Optimisation path + GN ridge direction\n'
                 f'(σ₁/σ₂ = {sv[0]/max(sv[1],1e-30):.0f} → valley elongation)')
    ax.legend(fontsize=9);  ax.grid(True, ls=':')

    ax2 = axes[1]
    ax2.semilogy(J_path, 'o-', color='royalblue', lw=2, ms=5)
    ax2.axhline(J_path.min(), color='red', ls='--', lw=1.5,
                label=f'stall J={J_path.min():.1e}')
    ax2.set_xlabel('Iteration');  ax2.set_ylabel('J')
    ax2.set_title('Cost function history');  ax2.legend();  ax2.grid(True, which='both', ls=':')

    plt.tight_layout()
    plt.savefig('ridge_gn_only.png', dpi=150)
    print("  Saved ridge_gn_only.png")
    plt.close(fig)


# ─── Optional: call directly if appended to inf_layered_vap_exp.py ────────────
if __name__ == "__main__":
    # This branch runs if you exec() this file from inside inf_layered_vap_exp.py
    # or if you call it as a module that gets exec-imported.
    # Typically you'd do:
    #
    #     exec(open("scan_stage1_ridge.py").read())
    #     run_ridge_diagnostics(do_scan=True, n_rho=8, n_f0=8)
    #
    # at the bottom of inf_layered_vap_exp.py.
    print("scan_stage1_ridge.py: run_ridge_diagnostics() is available.")
    print("Call it with the globals from inf_layered_vap_exp.py:")
    print("  run_ridge_diagnostics(globals_dict=globals(), do_scan=False)  # GN only")
    print("  run_ridge_diagnostics(globals_dict=globals(), do_scan=True,   # + scan")
    print("                        n_rho=8, n_f0=8)")
