"""
scan_stage2_landscape.py
========================
2D objective landscape scan over (rho_melt1_surf, rho_melt1_deep) for the
Stage 2 variance objective.

Evaluates J_fwd (forward-only, no adjoint) on a log-spaced grid to show:
  - whether a global minimum exists at the true values (5e7, 3e8)
  - whether the swapped initialisation (3e8, 5e7) is a local minimum
  - the ridge / non-identifiable manifold connecting them
  - the convex region around each mode

Cost: n_surf * n_deep forward solves (no adjoint).  Each forward solve is
roughly half the cost of a full obj_var call.  Checkpoints after every cell
so the scan can be interrupted and resumed.

Usage (append to inf_layered_vap_exp.py or call after exec):
    from scan_stage2_landscape import run_landscape_scan
    run_landscape_scan(globals_dict=globals())
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os


# ─────────────────────────────────────────────────────────────────────────────
# Forward-only J evaluator (no adjoint)
# Replicates the _J_of_hist logic from depth_objective.py
# ─────────────────────────────────────────────────────────────────────────────

def _eval_J_fwd(rho_melt1_surf, rho_melt1_deep,
                rho_vap0_fixed, f0_m_fixed, theta_kappa, kappa_param,
                run_forward_fn, clear_caches_fn,
                U0, SOLID_obs, MELT_base, VAP_base,
                sigmoid_rho_fn, y_trans, width,
                h_obs_hist, sigma2_obs_hist,
                Nx, Ny, Ly, num_nodes, P, N_KL, T_abl,
                eps_smooth, sigma_d):
    """One forward solve + J computation (no adjoint)."""
    from depth_objective import softmin_depth, _trap_weights

    MELT_cur = dict(MELT_base)
    MELT_cur['rho_melt1_s'] = float(rho_melt1_surf)
    MELT_cur['rho_melt1_d'] = float(rho_melt1_deep)
    MELT_cur['f0']          = float(f0_m_fixed)
    MELT_cur['rho_melt1']   = sigmoid_rho_fn(MELT_cur, y_trans, width)

    VAP_cur = dict(VAP_base)
    VAP_cur['rho_vap0'] = float(rho_vap0_fixed)

    clear_caches_fn()
    try:
        U_hist, *_ = run_forward_fn(U0, SOLID_obs, MELT_cur, VAP_cur,
                                    theta_kappa=theta_kappa,
                                    kappa_param=kappa_param)
    except Exception as e:
        print(f"  forward solve failed: {e}")
        return np.nan, None, None

    y_nodes    = np.linspace(0.0, Ly, Ny + 1)
    wz         = _trap_weights(y_nodes)
    beta       = 0.005
    time_steps = U_hist.shape[0]

    J          = 0.0
    h_pred     = np.zeros(time_steps)
    s2_pred    = np.zeros(time_steps)

    for t in range(1, time_steps):
        u_t       = U_hist[t]
        u_mean_2d = u_t[:num_nodes].reshape(Nx + 1, Ny + 1)

        if not np.isfinite(u_mean_2d).all():
            return np.nan, None, None

        # mean depth
        h_u0, _, w_sm, _, dH = softmin_depth(u_mean_2d, y_nodes, T_abl, eps_smooth, beta)
        g_flat = (w_sm[:, np.newaxis] * wz[np.newaxis, :] * dH).ravel()

        r_mu        = h_u0 - float(h_obs_hist[t])
        obs_var     = max(float(sigma2_obs_hist[t]), 1e-20)
        J          += 0.5 * r_mu ** 2 / obs_var
        h_pred[t]   = float(h_u0)

        # variance depth
        U_modes     = u_t.reshape(P, num_nodes)
        g_dot_uk    = U_modes[1:] @ g_flat      # (P-1,)
        sigma2_p    = float(np.sum(g_dot_uk ** 2))
        r_var       = sigma2_p - float(sigma2_obs_hist[t])
        J          += 0.5 * r_var ** 2 * 1e10   # matches obj_var scaling
        s2_pred[t]  = sigma2_p

    return float(J), h_pred, s2_pred


# ─────────────────────────────────────────────────────────────────────────────
# Main scan function
# ─────────────────────────────────────────────────────────────────────────────

def run_landscape_scan(
    *,
    # grid extent in physical space (log-spaced)
    rho_surf_lo = 4e7,   rho_surf_hi = 4e8,
    rho_deep_lo = 4e7,   rho_deep_hi = 4e8,
    n_surf      = 7,
    n_deep      = 7,
    # fix kappa and other params — None = read from stage2_log.npz (best iter)
    theta_kappa_fixed = None,
    # checkpoint file
    scan_file   = "stage2_landscape.npy",
    resume      = True,
    # plot output
    out_png     = "stage2_landscape.png",
    # globals from inf_layered_vap_exp.py
    globals_dict = None,
):
    import sys
    if globals_dict is None:
        globals_dict = sys._getframe(1).f_globals
    G = globals_dict

    # ── unpack required globals ───────────────────────────────────────────────
    run_forward_fn    = G['run_forward']
    clear_caches_fn   = G['_clear_all_kappa_caches']
    U0                = G['U0']
    SOLID_obs         = G['SOLID_obs']
    MELT_obs          = G['MELT_obs']
    VAP_obs           = G['VAP_obs']
    VAP               = G['VAP']
    MELT              = G['MELT']
    h_obs_hist        = G['h_obs_hist']
    sigma2_obs_hist   = G['sigma2_obs_hist']
    Nx, Ny, Ly        = G['Nx'], G['Ny'], G['Ly']
    num_nodes         = G['num_nodes']
    P, N_KL           = G['P'], G['N_KL']
    T_abl             = G['T_abl']
    sigma_d           = G['sigma_d']
    eps_smooth        = G['eps_smooth']
    sigmoid_rho_fn    = G['_sigmoid_rho_field']
    ell               = G['ell']

    # truth values (for plotting)
    rho_surf_truth = float(MELT_obs.get('rho_melt1_s', MELT_obs.get('rho_melt1', 5e7)))
    rho_deep_truth = float(MELT_obs.get('rho_melt1_d', MELT_obs.get('rho_melt1', 3e8)))

    # ── load stage1 result to fix rho_vap0 and f0_m ──────────────────────────
    if os.path.exists("stage1_log.npz"):
        s1 = np.load("stage1_log.npz")
        if len(s1['rho_vap0']) > 0:
            rho_vap0_fixed = float(s1['rho_vap0'][-1])
            f0_m_fixed     = float(s1['f0_m'][-1])
        else:
            rho_vap0_fixed = float(s1['rho_truth'])
            f0_m_fixed     = float(s1['f0_truth'])
        print(f"  [scan] rho_vap0_fixed={rho_vap0_fixed:.3e}  f0_m_fixed={f0_m_fixed:.4f}")
    else:
        rho_vap0_fixed = float(VAP_obs['rho_vap0'])
        f0_m_fixed     = float(MELT_obs['f0'])
        print(f"  [scan] stage1_log.npz not found — using truth: "
              f"rho_vap0={rho_vap0_fixed:.3e}  f0_m={f0_m_fixed:.4f}")

    # ── fix kappa parameters ──────────────────────────────────────────────────
    if theta_kappa_fixed is not None:
        theta_kappa = np.asarray(theta_kappa_fixed)
        print(f"  [scan] theta_kappa (user-supplied): {theta_kappa}")
    elif os.path.exists("stage2_log.npz"):
        s2 = np.load("stage2_log.npz")
        if len(s2['theta_kappa']) > 0:
            best_i      = int(np.argmin(s2['J']))
            theta_kappa = s2['theta_kappa'][best_i]
            print(f"  [scan] theta_kappa from best stage2 iter {best_i+1}: {theta_kappa.round(3)}")
        else:
            theta_kappa = G['theta_kappa_obs']
            print(f"  [scan] stage2_log empty — using truth kappa: {theta_kappa}")
    else:
        theta_kappa = G['theta_kappa_obs']
        print(f"  [scan] no stage2_log — using truth kappa: {theta_kappa}")

    y_trans = float(theta_kappa[2])
    width   = float(theta_kappa[3])

    from stable_eigh_test import SigmoidLayeredKappa
    kappa_cur = SigmoidLayeredKappa(
        Ny=Ny, Ly=Ly,
        kappa_surface = float(theta_kappa[0]),
        kappa_deep    = float(theta_kappa[1]),
        y_transition  = y_trans,
        width         = width,
    )

    MELT_base = dict(MELT_obs)
    MELT_base['f0'] = f0_m_fixed

    VAP_base = dict(VAP_obs)
    VAP_base['rho_vap0'] = rho_vap0_fixed
    VAP_base['rho_vap1'] = VAP.get('rho_vap1', VAP_obs.get('rho_vap1', 0.0))

    fwd_kwargs = dict(
        rho_vap0_fixed  = rho_vap0_fixed,
        f0_m_fixed      = f0_m_fixed,
        theta_kappa     = theta_kappa,
        kappa_param     = kappa_cur,
        run_forward_fn  = run_forward_fn,
        clear_caches_fn = clear_caches_fn,
        U0              = U0,
        SOLID_obs       = SOLID_obs,
        MELT_base       = MELT_base,
        VAP_base        = VAP_base,
        sigmoid_rho_fn  = sigmoid_rho_fn,
        y_trans         = y_trans,
        width           = width,
        h_obs_hist      = h_obs_hist,
        sigma2_obs_hist = sigma2_obs_hist,
        Nx=Nx, Ny=Ny, Ly=Ly, num_nodes=num_nodes, P=P, N_KL=N_KL,
        T_abl=T_abl, eps_smooth=eps_smooth, sigma_d=sigma_d,
    )

    # ── build grid ────────────────────────────────────────────────────────────
    rho_surf_grid = np.logspace(np.log10(rho_surf_lo), np.log10(rho_surf_hi), n_surf)
    rho_deep_grid = np.logspace(np.log10(rho_deep_lo), np.log10(rho_deep_hi), n_deep)

    print(f"\n  [scan] grid: {n_surf}×{n_deep} = {n_surf*n_deep} forward solves")
    print(f"  rho_surf ∈ [{rho_surf_lo:.2e}, {rho_surf_hi:.2e}] (log-spaced)")
    print(f"  rho_deep ∈ [{rho_deep_lo:.2e}, {rho_deep_hi:.2e}] (log-spaced)")
    print(f"  Truth: rho_surf={rho_surf_truth:.2e}  rho_deep={rho_deep_truth:.2e}")

    # ── load or init grid ─────────────────────────────────────────────────────
    axes_file = scan_file.replace('.npy', '_axes.npy')
    if resume and os.path.exists(scan_file) and os.path.exists(axes_file):
        J_grid   = np.load(scan_file)
        axes     = np.load(axes_file, allow_pickle=True)
        done_count = int(np.isfinite(J_grid).sum())
        print(f"  Resuming: {done_count}/{n_surf*n_deep} cells already computed")
        if J_grid.shape != (n_surf, n_deep):
            print("  WARNING: saved grid shape mismatch — starting fresh")
            J_grid = np.full((n_surf, n_deep), np.nan)
    else:
        J_grid = np.full((n_surf, n_deep), np.nan)

    np.save(axes_file, np.array([rho_surf_grid, rho_deep_grid], dtype=object))

    # ── scan ──────────────────────────────────────────────────────────────────
    total   = n_surf * n_deep
    cell_no = 0
    for i, rs in enumerate(rho_surf_grid):
        for j, rd in enumerate(rho_deep_grid):
            cell_no += 1
            if np.isfinite(J_grid[i, j]):
                continue
            print(f"  [{cell_no:3d}/{total}] rho_surf={rs:.3e}  rho_deep={rd:.3e} ...",
                  end='  ', flush=True)
            J_val, _, _ = _eval_J_fwd(rs, rd, **fwd_kwargs)
            J_grid[i, j] = J_val
            print(f"J = {J_val:.4e}")
            np.save(scan_file, J_grid)   # checkpoint every cell

    np.save(scan_file, J_grid)
    print(f"\n  Saved {scan_file}")

    # ── find best and report ──────────────────────────────────────────────────
    finite_mask = np.isfinite(J_grid)
    if finite_mask.any():
        best_flat = np.nanargmin(J_grid)
        bi, bj    = np.unravel_index(best_flat, J_grid.shape)
        print(f"\n  Best J = {J_grid[bi, bj]:.4e}  at "
              f"rho_surf={rho_surf_grid[bi]:.3e}  rho_deep={rho_deep_grid[bj]:.3e}")
        print(f"  Truth:              rho_surf={rho_surf_truth:.3e}  rho_deep={rho_deep_truth:.3e}")

    # ── plot ──────────────────────────────────────────────────────────────────
    _plot_landscape(J_grid, rho_surf_grid, rho_deep_grid,
                    rho_surf_truth, rho_deep_truth, out_png)

    return J_grid, rho_surf_grid, rho_deep_grid


def _plot_landscape(J_grid, rho_surf_grid, rho_deep_grid,
                    rho_surf_truth, rho_deep_truth, out_png):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    log_rs = np.log10(rho_surf_grid)
    log_rd = np.log10(rho_deep_grid)
    RS, RD = np.meshgrid(log_rs, log_rd, indexing='ij')

    # clip J for display (log scale, floor at a sensible minimum)
    J_disp = np.where(np.isfinite(J_grid) & (J_grid > 0), J_grid, np.nan)
    J_log  = np.log10(J_disp)

    # ── left: filled contour ─────────────────────────────────────────────────
    ax = axes[0]
    vmin = np.nanmin(J_log); vmax = np.nanmax(J_log)
    pcm  = ax.contourf(RS, RD, J_log, levels=20, cmap='viridis_r', vmin=vmin, vmax=vmax)
    ax.contour(RS, RD, J_log, levels=15, colors='k', linewidths=0.4, alpha=0.4)
    plt.colorbar(pcm, ax=ax, label='log₁₀(J)')

    # swap-symmetry diagonal (rho_surf = rho_deep)
    diag = np.linspace(min(log_rs.min(), log_rd.min()),
                       max(log_rs.max(), log_rd.max()), 100)
    ax.plot(diag, diag, 'w--', lw=1.2, alpha=0.7, label='rho_surf = rho_deep')

    # truth and swap point
    ax.plot(np.log10(rho_surf_truth), np.log10(rho_deep_truth),
            'D', color='lime', ms=11, zorder=10, label=f'truth ({rho_surf_truth:.1e},{rho_deep_truth:.1e})')
    ax.plot(np.log10(rho_deep_truth), np.log10(rho_surf_truth),
            'x', color='red',  ms=11, mew=2.5, zorder=10, label='swap of truth')

    # best grid point
    if np.isfinite(J_grid).any():
        bi, bj = np.unravel_index(np.nanargmin(J_grid), J_grid.shape)
        ax.plot(log_rs[bi], log_rd[bj], '*', color='orange', ms=14,
                zorder=11, label=f'grid min J={J_grid[bi,bj]:.2e}')

    ax.set_xlabel('log₁₀(rho_melt1_surf)')
    ax.set_ylabel('log₁₀(rho_melt1_deep)')
    ax.set_title('J landscape: variance objective\n(rho_melt1_surf vs rho_melt1_deep)')
    ax.legend(fontsize=8, loc='upper right')
    ax.set_aspect('equal')
    ax.grid(True, ls=':', alpha=0.3)

    # ── right: 1D slices through truth ───────────────────────────────────────
    ax2 = axes[1]
    from scipy.interpolate import RegularGridInterpolator
    interp = RegularGridInterpolator(
        (log_rs, log_rd), J_log,
        method='linear', bounds_error=False, fill_value=np.nan)

    alphas = np.linspace(log_rs.min(), log_rs.max(), 200)

    # slice along rho_surf (rho_deep fixed at truth)
    pts_surf = np.column_stack([alphas,
                                np.full_like(alphas, np.log10(rho_deep_truth))])
    J_surf_slice = interp(pts_surf)

    # slice along rho_deep (rho_surf fixed at truth)
    pts_deep = np.column_stack([np.full_like(alphas, np.log10(rho_surf_truth)),
                                alphas])
    J_deep_slice = interp(pts_deep)

    # slice along swap diagonal (rho_surf = rho_deep)
    pts_diag = np.column_stack([alphas, alphas])
    J_diag_slice = interp(pts_diag)

    ax2.plot(alphas, J_surf_slice, 'b-',  lw=2, label='vary rho_surf (rho_deep=truth)')
    ax2.plot(alphas, J_deep_slice, 'g-',  lw=2, label='vary rho_deep (rho_surf=truth)')
    ax2.plot(alphas, J_diag_slice, 'r--', lw=2, label='diagonal (rho_surf=rho_deep)')
    ax2.axvline(np.log10(rho_surf_truth), color='lime', ls=':', lw=1.5, label='truth rho_surf')
    ax2.axvline(np.log10(rho_deep_truth), color='orange', ls=':', lw=1.5, label='truth rho_deep')
    ax2.set_xlabel('log₁₀(parameter value)')
    ax2.set_ylabel('log₁₀(J)')
    ax2.set_title('1D cross-sections through truth\n(convex = single bowl in that direction)')
    ax2.legend(fontsize=8)
    ax2.grid(True, ls=':')

    plt.tight_layout()
    plt.savefig(out_png, dpi=150, bbox_inches='tight')
    print(f"  Saved {out_png}")
    plt.close(fig)
