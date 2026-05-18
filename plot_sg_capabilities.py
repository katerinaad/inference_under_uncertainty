"""
plot_sg_capabilities.py
=======================
Standalone plotting module for the SG solver.

Call from inf_layered_vap.py after the forward solve:

    from plot_sg_capabilities import generate_plots
    generate_plots(
        U_obs=U_obs,
        T_obs_hist=T_obs_hist,
        local_params=_obs_local_params,
        x_vis_hist=x_vis_hist,
        y_vis_hist=y_vis_hist,
        w_vis_hist=w_vis_hist,
        get_visible_weights_fn=get_visible_weights_from_xy,
        multi_idx=multi_idx,
        eval_psi_fn=eval_psi,
        SOLID_obs=SOLID_obs,
        Nx=Nx, Ny=Ny, Lx=Lx, Ly=Ly,
        num_nodes=num_nodes, P=P, N_KL=N_KL,
        time_steps=time_steps, dt=dt,
        T_abl=T_abl,
        T_melt_lo=T_melt_lo, T_melt_hi=T_melt_hi, Delta_melt=50.0,
        T_vap_lo=T_vap_lo,   T_vap_hi=T_vap_hi,   Delta_vap=Delta_vap,
    )
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

# ---------------------------------------------------------------------------
# Style  — clean scientific / print aesthetic
# ---------------------------------------------------------------------------
PALETTE = {
    "bg":     "#fafaf8",          # warm off-white page
    "panel":  "#ffffff",          # pure white axes
    "border": "#c8c8c0",          # light warm grey spine
    "text":   "#1a1a18",          # near-black ink
    "muted":  "#6b6b62",          # mid warm grey for ticks / labels
    "blue":   "#2166ac",          # muted steel blue  (ColorBrewer RdBu)
    "orange": "#d6604d",          # terracotta / brick red
    "green":  "#4d9a6a",          # forest green
    "red":    "#b2182b",          # deep crimson
    "purple": "#6a3d9a",          # aubergine
}

_RC = {
    # backgrounds
    "figure.facecolor":   PALETTE["bg"],
    "axes.facecolor":     PALETTE["panel"],
    # spines
    "axes.edgecolor":     PALETTE["border"],
    "axes.linewidth":     0.8,
    # labels & ticks
    "axes.labelcolor":    PALETTE["text"],
    "axes.titlecolor":    PALETTE["text"],
    "xtick.color":        PALETTE["muted"],
    "ytick.color":        PALETTE["muted"],
    "xtick.major.width":  0.7,
    "ytick.major.width":  0.7,
    "xtick.major.size":   3.5,
    "ytick.major.size":   3.5,
    "xtick.direction":    "out",
    "ytick.direction":    "out",
    # text
    "text.color":         PALETTE["text"],
    # grid
    "axes.grid":          True,
    "grid.color":         "#e4e4de",
    "grid.linewidth":     0.5,
    "grid.linestyle":     "--",
    # typography
    "font.family":        "serif",
    "font.serif":         ["DejaVu Serif", "Georgia", "Times New Roman", "serif"],
    "mathtext.fontset":   "dejavuserif",
    "font.size":          9,
    "axes.titlesize":     10,
    "axes.labelsize":     9,
    # legend
    "legend.framealpha":  0.9,
    "legend.edgecolor":   PALETTE["border"],
    "legend.fontsize":    8,
    # resolution
    "figure.dpi":         150,
    "savefig.dpi":        200,
    "savefig.bbox":       "tight",
    "savefig.facecolor":  PALETTE["bg"],
    # lines
    "lines.linewidth":    1.4,
    "patch.linewidth":    0.6,
}


def _cbar(fig, im, ax, label):
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.ax.yaxis.set_tick_params(color=PALETTE["muted"], width=0.6)
    cb.outline.set_edgecolor(PALETTE["border"])
    cb.outline.set_linewidth(0.7)
    cb.set_label(label, color=PALETTE["muted"], fontsize=8)
    plt.setp(cb.ax.yaxis.get_ticklabels(), color=PALETTE["muted"], fontsize=7)
    return cb


def _title(ax, txt):
    ax.set_title(txt, color=PALETTE["text"], fontsize=10,
                 pad=6, loc="left", fontweight="bold")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _vap_mean_var(U_obs_t, Nx, Ny, num_nodes, P,
                  T_vap_lo, T_vap_hi, Delta_vap):
    """
    Compute E[V] and Var[V] for the vaporisation fraction field at one
    timestep, propagating uncertainty through the tanh nonlinearity via
    a first-order Taylor expansion around the SG mean field.

        V(T) = 0.5 * (1 + tanh((T - T_vap_mid) / Delta))

    Mean:     E[V] ≈ V(u_0)
    Variance: Var[V] ≈ (dV/dT)^2 * Var[T]
                     = (dV/dT)^2 * sum_{k=1}^{P-1} u_k^2

    This is the standard delta-method / moment propagation through a
    deterministic nonlinearity.  It is exact to first order and becomes
    more accurate as the stochastic perturbations are small relative to
    the transition width Delta_vap.

    Returns
    -------
    V_mean  : (Nx+1, Ny+1)  mean vaporisation fraction field
    V_var   : (Nx+1, Ny+1)  variance of vaporisation fraction field
    """
    Tv_mid = 0.5 * (T_vap_lo + T_vap_hi)
    Delta  = max(Delta_vap, 1e-12)

    blk    = U_obs_t.reshape(P, num_nodes)
    u0     = blk[0].reshape(Nx + 1, Ny + 1)          # SG mean temperature
    T_var  = np.sum(blk[1:] ** 2, axis=0).reshape(Nx + 1, Ny + 1)  # Var[T]

    z      = (u0 - Tv_mid) / Delta
    V_mean = 0.5 * (1.0 + np.tanh(z))                # E[V] ≈ V(u_0)

    dVdT   = 0.5 * (1.0 - np.tanh(z) ** 2) / Delta   # dV/dT at u_0
    V_var  = dVdT ** 2 * T_var                         # Var[V] via delta method

    return V_mean, V_var


# ---------------------------------------------------------------------------
# Figure 1 — Temperature mean + variance at 3 snapshots
# ---------------------------------------------------------------------------
def _fig1_mean_variance(U_obs, Nx, Ny, num_nodes, P, time_steps, dt, Lx, Ly,
                        out="fig1_mean_variance.png"):
    snap_idx = [1, time_steps // 2, time_steps - 1]

    means, vars_ = [], []
    for t in snap_idx:
        blk = U_obs[t].reshape(P, num_nodes)
        means.append(blk[0].reshape(Nx + 1, Ny + 1))
        vars_.append(np.sum(blk[1:] ** 2, axis=0).reshape(Nx + 1, Ny + 1))

    vmin_m = min(m.min() for m in means)
    vmax_m = max(m.max() for m in means)
    vmax_v = max(v.max() for v in vars_) or 1.0
    extent = [0, Lx * 1e3, 0, Ly * 1e3]

    fig, axes = plt.subplots(2, 3, figsize=(13, 6))
    fig.suptitle("SG Solver Output: Mean & Variance Fields",
                 color=PALETTE["text"], fontsize=12, fontweight="bold", y=1.01)

    for col, (t, mean_2d, var_2d) in enumerate(zip(snap_idx, means, vars_)):
        t_label = f"t = {t * dt:.3f} s"

        ax = axes[0, col]
        im = ax.imshow(mean_2d.T, origin="lower", aspect="auto",
                       extent=extent, cmap="inferno", vmin=vmin_m, vmax=vmax_m)
        _title(ax, f"Mean  [{t_label}]")
        ax.set_xlabel("x (mm)")
        ax.set_ylabel("y (mm)" if col == 0 else "")
        _cbar(fig, im, ax, "T (K)")

        ax = axes[1, col]
        im = ax.imshow(var_2d.T, origin="lower", aspect="auto",
                       extent=extent, cmap="plasma", vmin=0.0, vmax=vmax_v)
        _title(ax, f"Variance  [{t_label}]")
        ax.set_xlabel("x (mm)")
        ax.set_ylabel("y (mm)" if col == 0 else "")
        _cbar(fig, im, ax, "Var(T) (K^2)")

    fig.tight_layout()
    fig.savefig(out)
    print(f"  [fig1] saved {out}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2 — KL eigenfunctions & conductivity realisations
# ---------------------------------------------------------------------------
def _fig2_kl_realisations(local_params, Nx, Ny, Lx, Ly, k0, k1, N_KL,
                           n_samples=4, seed=42,
                           out="fig2_kl_realisations.png"):
    phi_grid = np.array(local_params["eigvecs_grid"])
    sqrt_lam = np.array(local_params["sqrt_lam"])

    rng     = np.random.default_rng(seed)
    n_modes = min(N_KL, 4)
    n_cols  = max(n_modes, n_samples)
    extent  = [0, Lx * 1e3, 0, Ly * 1e3]

    fig, axes = plt.subplots(2, n_cols, figsize=(3.5 * n_cols, 6))
    if n_cols == 1:
        axes = axes[:, np.newaxis]
    fig.suptitle("KL Eigenfunctions & Conductivity Realisations",
                 color=PALETTE["text"], fontsize=12, fontweight="bold", y=1.01)

    for m in range(n_cols):
        ax = axes[0, m]
        if m < n_modes:
            phi  = phi_grid[m]
            vabs = max(abs(phi.min()), abs(phi.max())) or 1.0
            norm = TwoSlopeNorm(vmin=-vabs, vcenter=0, vmax=vabs)
            im   = ax.imshow(phi.T, origin="lower", aspect="auto",
                             extent=extent, cmap="RdBu_r", norm=norm)
            _title(ax, f"phi_{m+1}(x,y)  [lam={sqrt_lam[m]**2:.2e}]")
            _cbar(fig, im, ax, "")
            ax.set_xlabel("x (mm)")
            ax.set_ylabel("y (mm)" if m == 0 else "")
        else:
            ax.set_visible(False)

    k_all = []
    for _ in range(n_samples):
        xi      = rng.standard_normal(N_KL)
        k_fluct = np.einsum("m,mij->ij", sqrt_lam * xi, phi_grid)
        k_all.append(k0 + k1 * k_fluct)

    kmin = min(k.min() for k in k_all)
    kmax = max(k.max() for k in k_all)

    for s in range(n_cols):
        ax = axes[1, s]
        if s < n_samples:
            im = ax.imshow(k_all[s].T, origin="lower", aspect="auto",
                           extent=extent, cmap="viridis", vmin=kmin, vmax=kmax)
            _title(ax, f"k(x,y; xi_{s+1})")
            _cbar(fig, im, ax, "k (W/mK)")
            ax.set_xlabel("x (mm)")
            ax.set_ylabel("y (mm)" if s == 0 else "")
        else:
            ax.set_visible(False)

    fig.tight_layout()
    fig.savefig(out)
    print(f"  [fig2] saved {out}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 4 — Radiometer observation vs SG prediction
# ---------------------------------------------------------------------------
def _fig4_radiometer_prediction(T_obs_hist, U_obs, num_nodes, P,
                                 x_vis_hist, y_vis_hist, w_vis_hist,
                                 x, y, time_steps, dt,
                                 get_visible_weights_fn,
                                 out="fig4_radiometer_prediction.png"):
    times = np.arange(time_steps) * dt

    obs_med  = np.nanmedian(T_obs_hist, axis=1)
    obs_p25  = np.nanpercentile(T_obs_hist, 25, axis=1)
    obs_p75  = np.nanpercentile(T_obs_hist, 75, axis=1)
    obs_mask = obs_med != 0.0

    pred_mean = np.zeros(time_steps)
    pred_var  = np.zeros(time_steps)
    for n in range(1, time_steps):
        if x_vis_hist[n] is None or len(x_vis_hist[n]) == 0:
            continue
        vis_idx, weights = get_visible_weights_fn(
            x_vis_hist[n], y_vis_hist[n], w_vis_hist[n], x, y)
        proj         = U_obs[n].reshape(P, num_nodes)[:, vis_idx] @ weights
        pred_mean[n] = proj[0]
        pred_var[n]  = float(np.sum(proj[1:] ** 2))

    p2s = 2.0 * np.sqrt(np.maximum(pred_var, 0.0))

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.fill_between(times[obs_mask], obs_p25[obs_mask], obs_p75[obs_mask],
                    alpha=0.25, color=PALETTE["orange"])
    ax.plot(times[obs_mask], obs_med[obs_mask], lw=1.5,
            color=PALETTE["orange"], label="Obs. median  (IQR shaded)")
    ax.fill_between(times, pred_mean - p2s, pred_mean + p2s,
                    alpha=0.20, color=PALETTE["blue"])
    ax.plot(times, pred_mean, lw=2.0, color=PALETTE["blue"],
            label="SG mean  (+/-2 sigma shaded)")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("Radiometer temperature (K)")
    ax.legend(fontsize=9, framealpha=0.88, edgecolor=PALETTE["border"])
    ax.grid(True, alpha=0.3)
    _title(ax, "Radiometer Observation vs SG Prediction")
    fig.tight_layout()
    fig.savefig(out)
    print(f"  [fig4] saved {out}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 6 — Ablation surface: depth measurements as dots + interpolated lines
# ---------------------------------------------------------------------------
def _fig6_depth_curves(U_obs, Nx, Ny, num_nodes, P, N_KL,
                        multi_idx, eval_psi_fn,
                        Lx, Ly, T_abl, dt,
                        eps=10.0, beta=0.005,
                        n_samples=30, seed=7,
                        snap_indices=None,
                        out="fig6_depth_curves.png"):
    """
    Plots the per-column ablation surface surf(x) with:
      - dots at each x-node  (the actual depth "measurements")
      - a faint smooth line connecting them (subtle visual guide)
      - grey min/max envelope across all samples
      - dashed white mean-field surface from u_0 only

    Each panel is one timestep.  Active columns only (where T >= T_abl).
    """
    from depth_objective import softmin_depth

    time_steps = U_obs.shape[0]
    if snap_indices is None:
        snap_indices = [time_steps // 3,
                        2 * time_steps // 3,
                        time_steps - 1]

    x_nodes = np.linspace(0, Lx, Nx + 1)
    y_nodes = np.linspace(0, Ly, Ny + 1)

    rng      = np.random.default_rng(seed)
    xi_draws = rng.standard_normal((n_samples, N_KL))

    sample_colors = [PALETTE["red"], PALETTE["orange"], PALETTE["green"],
                     PALETTE["blue"], PALETTE["purple"]]

    n_panels = len(snap_indices)
    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 5),
                             sharey=True)
    if n_panels == 1:
        axes = [axes]

    fig.suptitle(f"Ablation surface — {n_samples} SG samples  "
                 f"(dots = nodal measurements, lines = interpolant)",
                 color=PALETTE["text"], fontsize=12, fontweight="bold")

    for ax, t in zip(axes, snap_indices):
        U_blk = U_obs[t].reshape(P, num_nodes)

        # SG mean-field surface from u_0 only
        _, surf_mean_field, _, _, _ = softmin_depth(
            U_blk[0].reshape(Nx + 1, Ny + 1), y_nodes, T_abl, eps, beta)

        # All sample surfaces
        surf_mat = np.zeros((n_samples, Nx + 1))
        for i, xi in enumerate(xi_draws):
            u = _reconstruct(xi, multi_idx, eval_psi_fn, U_blk)
            _, surf_i, _, _, _ = softmin_depth(
                u.reshape(Nx + 1, Ny + 1), y_nodes, T_abl, eps, beta)
            surf_mat[i] = surf_i

        # Mask to active columns only
        active = surf_mean_field < Ly * 0.999
        x_act  = x_nodes[active]

        surf_lo = surf_mat[:, active].min(axis=0)
        surf_hi = surf_mat[:, active].max(axis=0)

        # Envelope
        ax.fill_between(x_act, surf_lo, surf_hi,
                        color=PALETTE["muted"], alpha=0.15,
                        label="sample envelope")

        # Individual samples: subtle line + dots
        for i in range(n_samples):
            col   = sample_colors[i % len(sample_colors)]
            y_act = surf_mat[i][active]
            # Faint connecting line
            ax.plot(x_act, y_act, lw=0.6, color=col, alpha=0.25,
                    zorder=2)
            # Dots at each node — only label the first few
            ax.scatter(x_act, y_act, s=8, color=col, alpha=0.55,
                       zorder=3, linewidths=0,
                       label=f"sample $\\xi_{{{i+1}}}$" if i < 4
                             else "_nolegend_")

        # Mean-field: dashed line + slightly larger dots
        y_mf = surf_mean_field[active]
        ax.plot(x_act, y_mf, lw=1.5, ls="--",
                color=PALETTE["text"], alpha=0.9, zorder=4,
                label="mean-field $y^*(x)$")
        ax.scatter(x_act, y_mf, s=14, color=PALETTE["text"],
                   alpha=0.9, zorder=5, linewidths=0)

        ax.set_xlim(0, Lx)
        ax.set_ylim(0, Ly)
        ax.set_xlabel("$x$ (m)")
        if ax is axes[0]:
            ax.set_ylabel("$y$ (m)")
        ax.set_title(f"$t = {t * dt:.3f}$ s",
                     color=PALETTE["text"], fontsize=11, fontweight="bold")
        ax.grid(True, alpha=0.2)

        if ax is axes[0]:
            handles, labels = ax.get_legend_handles_labels()
            seen, h2, l2 = set(), [], []
            for h, l in zip(handles, labels):
                if l not in seen:
                    seen.add(l); h2.append(h); l2.append(l)
            ax.legend(h2, l2, fontsize=8, framealpha=0.88,
                      edgecolor=PALETTE["border"])

    fig.tight_layout()
    fig.savefig(out)
    print(f"  [fig6] saved {out}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Shared helper — vap fraction from a flat node vector
# ---------------------------------------------------------------------------
def _vap_field(u_flat, Nx, Ny, Tv_mid, Delta):
    """V(T) for a flat (num_nodes,) temperature vector -> (Nx+1, Ny+1)."""
    T = u_flat.reshape(Nx + 1, Ny + 1)
    return 0.5 * (1.0 + np.tanh((T - Tv_mid) / Delta))


def _reconstruct(xi, multi_idx, eval_psi_fn, U_blk):
    """Reconstruct one SG surrogate sample from xi draws."""
    return sum(eval_psi_fn(xi, alpha) * U_blk[k]
               for k, alpha in enumerate(multi_idx))


# ---------------------------------------------------------------------------
# Figure 7 — Vaporisation fraction: mean & variance at 3 snapshots
# ---------------------------------------------------------------------------
def _fig7_vap_mean_variance(U_obs, Nx, Ny, num_nodes, P, N_KL,
                             multi_idx, eval_psi_fn,
                             time_steps, dt, Lx, Ly,
                             T_vap_lo, T_vap_hi, Delta_vap,
                             n_samples=300, seed=0,
                             out="fig7_vap_mean_variance.png"):
    """
    2x3 panel mirroring fig1 but for the vaporisation fraction V(x,y).

    V = 0.5*(1 + tanh((T - T_vap_mid) / Delta_vap)) is nonlinear in T so
    E[V] != V(E[T]).  Both statistics are estimated by sampling the SG
    surrogate with n_samples xi draws.

    Top row    : E[V]    at t_early, t_mid, t_final
    Bottom row : Var[V]  at the same timesteps
    """
    Tv_mid   = 0.5 * (T_vap_lo + T_vap_hi)
    Delta    = max(Delta_vap, 1e-12)
    snap_idx = [1, time_steps // 2, time_steps - 1]
    extent   = [0, Lx * 1e3, 0, Ly * 1e3]

    rng      = np.random.default_rng(seed)
    xi_draws = rng.standard_normal((n_samples, N_KL))

    v_means, v_vars = [], []
    for t in snap_idx:
        U_blk = U_obs[t].reshape(P, num_nodes)
        acc   = np.zeros((Nx + 1, Ny + 1))
        acc2  = np.zeros((Nx + 1, Ny + 1))
        for xi in xi_draws:
            V    = _vap_field(_reconstruct(xi, multi_idx, eval_psi_fn, U_blk),
                              Nx, Ny, Tv_mid, Delta)
            acc  += V
            acc2 += V ** 2
        vm = acc / n_samples
        vv = np.maximum(acc2 / n_samples - vm ** 2, 0.0)
        v_means.append(vm)
        v_vars.append(vv)

    vmax_v = max(v.max() for v in v_vars) or 1.0
    extent  = [0, Lx * 1e3, 0, Ly * 1e3]

    fig, axes = plt.subplots(2, 3, figsize=(13, 6))
    fig.suptitle("Vaporisation Fraction: Mean & Variance Fields",
                 color=PALETTE["text"], fontsize=12, fontweight="bold", y=1.01)

    for col, (t, vm, vv) in enumerate(zip(snap_idx, v_means, v_vars)):
        t_label = f"t = {t * dt:.3f} s"

        ax = axes[0, col]
        im = ax.imshow(vm.T, origin="lower", aspect="auto",
                       extent=extent, cmap="hot", vmin=0.0, vmax=1.0)
        _title(ax, f"E[V]  [{t_label}]")
        ax.set_xlabel("x (mm)")
        ax.set_ylabel("y (mm)" if col == 0 else "")
        _cbar(fig, im, ax, "Vap fraction  [0,1]")

        ax = axes[1, col]
        im = ax.imshow(vv.T, origin="lower", aspect="auto",
                       extent=extent, cmap="plasma", vmin=0.0, vmax=vmax_v)
        _title(ax, f"Var[V]  [{t_label}]")
        ax.set_xlabel("x (mm)")
        ax.set_ylabel("y (mm)" if col == 0 else "")
        _cbar(fig, im, ax, "Var(V)")

    fig.tight_layout()
    fig.savefig(out)
    print(f"  [fig7] saved {out}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 8 — Wellbore surface: surf(x) from softmin_depth for SG samples
# ---------------------------------------------------------------------------
def _fig8_wellbore_contours(U_obs, Nx, Ny, num_nodes, P, N_KL,
                             multi_idx, eval_psi_fn,
                             Lx, Ly, T_abl, dt,
                             snap_indices=None, n_show=4, n_samples=50,
                             seed=0, out="fig8_wellbore_contours.png"):
    """
    Each panel shows y = surf(x) — the per-column ablation surface height
    from softmin_depth — for n_show individual xi draws plus the envelope
    and mean-field surface.

    Uses surf(x) directly rather than matplotlib contours, which produced
    closed loops around the whole hot region rather than just the surface.
    Columns where T never reaches T_abl (surf ≈ Ly) are masked out.
    """
    from depth_objective import softmin_depth

    time_steps = U_obs.shape[0]
    if snap_indices is None:
        snap_indices = [time_steps // 3,
                        2 * time_steps // 3,
                        time_steps - 1]

    x_nodes = np.linspace(0, Lx, Nx + 1)
    y_nodes = np.linspace(0, Ly, Ny + 1)

    rng      = np.random.default_rng(seed)
    xi_draws = rng.standard_normal((n_samples, N_KL))

    sample_colors = [PALETTE["red"], PALETTE["orange"],
                     PALETTE["green"], PALETTE["blue"], PALETTE["purple"]]

    n_panels = len(snap_indices)
    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 6),
                             sharey=True)
    if n_panels == 1:
        axes = [axes]

    fig.suptitle(
        f"Wellbore — ablation surface $T = T_{{\\mathrm{{abl}}}}$ "
        f"for $N_{{\\mathrm{{samp}}}} = {n_show}$ temperature-field samples",
        color=PALETTE["text"], fontsize=12, fontweight="bold"
    )

    for ax, t in zip(axes, snap_indices):
        U_blk = U_obs[t].reshape(P, num_nodes)

        # SG mean-field surface
        _, surf_mean_field, _, _, _ = softmin_depth(
            U_blk[0].reshape(Nx + 1, Ny + 1), y_nodes, T_abl,
            eps=10.0, beta=0.005)

        # All sample surfaces
        surf_mat = np.zeros((n_samples, Nx + 1))
        for i, xi in enumerate(xi_draws):
            u = _reconstruct(xi, multi_idx, eval_psi_fn, U_blk)
            _, surf_i, _, _, _ = softmin_depth(
                u.reshape(Nx + 1, Ny + 1), y_nodes, T_abl,
                eps=10.0, beta=0.005)
            surf_mat[i] = surf_i

        # Mask columns where T never reaches T_abl (surf stays near Ly)
        active = surf_mean_field < Ly * 0.999

        surf_lo = surf_mat[:, active].min(axis=0)
        surf_hi = surf_mat[:, active].max(axis=0)
        x_act   = x_nodes[active]

        # Spread envelope
        ax.fill_between(x_act, surf_lo, surf_hi,
                        color=PALETTE["muted"], alpha=0.3, label="spread")

        # Individual sample surfaces
        for i in range(min(n_show, n_samples)):
            col = sample_colors[i % len(sample_colors)]
            ax.plot(x_act, surf_mat[i][active],
                    lw=1.2, color=col, alpha=0.85,
                    label=f"sample $\\xi_{{{i+1}}}$")

        # Mean-field surface
        ax.plot(x_act, surf_mean_field[active],
                lw=2.0, ls="--", color=PALETTE["text"],
                label="mean $y^*(x)$")

        ax.set_xlim(0, Lx)
        ax.set_ylim(0, Ly)
        ax.set_xlabel("$x$ (m)")
        if ax is axes[0]:
            ax.set_ylabel("$y$ (m)")
        ax.set_title(f"$t = {t * dt:.3f}$ s",
                     color=PALETTE["text"], fontsize=11, fontweight="bold")
        ax.grid(True, alpha=0.2)

        ax.text(0.02, 0.05,
                f"$T_{{\\mathrm{{abl}}}} = {T_abl}$\n$N_{{\\mathrm{{samp}}}} = {n_show}$",
                transform=ax.transAxes, fontsize=8, va="bottom",
                color=PALETTE["text"],
                bbox=dict(boxstyle="round,pad=0.3",
                          facecolor=PALETTE["panel"],
                          edgecolor=PALETTE["border"]))

        if ax is axes[0]:
            handles, labels = ax.get_legend_handles_labels()
            seen, h2, l2 = set(), [], []
            for h, l in zip(handles, labels):
                if l not in seen:
                    seen.add(l); h2.append(h); l2.append(l)
            ax.legend(h2, l2, fontsize=8, framealpha=0.88,
                      edgecolor=PALETTE["border"], loc="upper right")

    fig.tight_layout()
    fig.savefig(out)
    print(f"  [fig8] saved {out}")
    plt.close(fig)

# ---------------------------------------------------------------------------
# Figure 9 — Melt & Vaporisation fronts: mean + samples
# ---------------------------------------------------------------------------
def _fig_melt_vap_fronts(
    U_obs, Nx, Ny, num_nodes, P, N_KL,
    multi_idx, eval_psi_fn,
    Lx, Ly, dt,
    T_melt_lo, T_melt_hi,
    T_vap_lo,  T_vap_hi,
    Delta_melt=50.0, Delta_vap=50.0,
    snap_indices=None,
    n_samples=50, seed=42,
    out="fig9_melt_vap_fronts.png",
):
    """
    For each snapshot panel, plot the melt and vaporisation iso-fraction
    contour at fraction = 0.5  (i.e. T = T_mid for each phase transition).

    Both the SG mean-field contour and individual surrogate samples are shown:
      - grey fill   : min/max envelope across all samples
      - coloured lines : up to n_show individual samples
      - dashed white   : mean-field front (from u_0 alone, no sampling)

    The melt front uses the softened indicator
        M(T) = 0.5*(1 + tanh((T - T_melt_mid) / Delta_melt))
    and the vap front uses the same form with T_vap_mid / Delta_vap.

    The "front" at each x-column is the y-position of the first row where the
    smoothed indicator exceeds 0.5  (i.e. T >= T_mid).  Columns that never
    reach T_mid are masked out.
    """
    def _phase_front(T_2d, T_mid, Delta, y_nodes):
        """
        Return the per-column front height y*(x) where
            phi(T) = 0.5*(1 + tanh((T - T_mid)/Delta)) >= 0.5
        i.e. the first y-index where T >= T_mid, interpolated linearly.
        Columns that never reach T_mid are returned as np.nan.
        """
        Nx1, Ny1 = T_2d.shape
        front = np.full(Nx1, np.nan)
        for ix in range(Nx1):
            col = T_2d[ix, :]           # temperature along this x-column
            above = np.where(col >= T_mid)[0]
            if above.size == 0:
                continue
            j = above[0]
            if j == 0:
                front[ix] = y_nodes[0]
            else:
                # linear interpolation between j-1 and j
                t_lo, t_hi = col[j - 1], col[j]
                dT = t_hi - t_lo
                if abs(dT) < 1e-12:
                    front[ix] = y_nodes[j]
                else:
                    alpha = (T_mid - t_lo) / dT
                    front[ix] = y_nodes[j - 1] + alpha * (y_nodes[j] - y_nodes[j - 1])
        return front

    time_steps = U_obs.shape[0]
    if snap_indices is None:
        snap_indices = [time_steps // 3,
                        2 * time_steps // 3,
                        time_steps - 1]

    T_melt_mid = (T_melt_lo + T_melt_hi)/2
    T_vap_mid  =  (T_vap_lo  + T_vap_hi)/2

    x_nodes = np.linspace(0, Lx, Nx + 1)
    y_nodes = np.linspace(0, Ly, Ny + 1)

    rng      = np.random.default_rng(seed)
    xi_draws = rng.standard_normal((n_samples, N_KL))

    # colour ramps defined per-phase inside the panel loop
    n_show = min(5, n_samples)

    n_panels = len(snap_indices)
    fig, axes = plt.subplots(1, n_panels,
                             figsize=(6 * n_panels, 5),
                             sharey=True)
    if n_panels == 1:
        axes = [axes]

    fig.suptitle(
        f"Melt & Vaporisation Fronts — "
        f"{n_samples} SG samples  (dashed = mean field)",
        color=PALETTE["text"], fontsize=12, fontweight="bold",
    )

    for ax, t in zip(axes, snap_indices):
        U_blk = U_obs[t].reshape(P, num_nodes)     # (P, num_nodes)
        u0    = U_blk[0].reshape(Nx + 1, Ny + 1)  # SG mean temperature

        # ---- mean-field fronts (from u_0 alone) ----------------------------
        mf_melt = _phase_front(u0, T_melt_mid, Delta_melt, y_nodes)
        mf_vap  = _phase_front(u0, T_vap_mid,  Delta_vap,  y_nodes)

        # ---- sample fronts -------------------------------------------------
        melt_mat = np.full((n_samples, Nx + 1), np.nan)
        vap_mat  = np.full((n_samples, Nx + 1), np.nan)
        for i, xi in enumerate(xi_draws):
            u_samp = _reconstruct(xi, multi_idx, eval_psi_fn, U_blk)
            T_samp = u_samp.reshape(Nx + 1, Ny + 1)
            melt_mat[i] = _phase_front(T_samp, T_melt_mid, Delta_melt, y_nodes)
            vap_mat[i]  = _phase_front(T_samp, T_vap_mid,  Delta_vap,  y_nodes)

        # ---- active masks --------------------------------------------------
        melt_active = ~np.isnan(mf_melt)
        vap_active  = ~np.isnan(mf_vap)

        # Per-phase sample-line colour ramps so melt and vap are never confused
        melt_sample_colors = ["#e08030", "#c0601a", "#a04010",
                              "#e0a060", "#c08040"]   # oranges/ambers
        vap_sample_colors  = ["#2166ac", "#3a85c8", "#1a4f8a",
                              "#5a9fd0", "#0d3060"]   # blues

        # ---- helper: draw one phase ----------------------------------------
        def _draw_phase(ax, x_nodes, mat, mf, active,
                        face_color, line_color, samp_colors,
                        label_pfx, z_base):
            if active.sum() == 0:
                return
            xv = x_nodes[active]

            # sample envelope (drawn at z_base)
            sample_lo = np.nanmin(mat[:, active], axis=0)
            sample_hi = np.nanmax(mat[:, active], axis=0)
            ax.fill_between(xv, sample_lo, sample_hi,
                            color=face_color, alpha=0.22,
                            zorder=z_base,
                            label=f"{label_pfx} envelope")

            # individual sample lines (z_base + 1)
            for i in range(n_show):
                col = samp_colors[i % len(samp_colors)]
                yv  = mat[i, active]
                valid = ~np.isnan(yv)
                if valid.sum() < 2:
                    continue
                ax.plot(xv[valid], yv[valid],
                        lw=1.0, color=col, alpha=0.60,
                        zorder=z_base + 1,
                        label=f"{label_pfx} $\\xi_{{{i+1}}}$" if i < 4
                              else "_nolegend_")

            # mean-field front (z_base + 2, thick dashed)
            yv_mf = mf[active]
            valid_mf = ~np.isnan(yv_mf)
            if valid_mf.sum() >= 2:
                ax.plot(xv[valid_mf], yv_mf[valid_mf],
                        lw=2.2, ls="--", color=line_color,
                        zorder=z_base + 2,
                        label=f"{label_pfx} mean front")

        # Draw melt first (lower z), vap on top (higher z)
        _draw_phase(ax, x_nodes, melt_mat, mf_melt, melt_active,
                    face_color=PALETTE["orange"],
                    line_color=PALETTE["orange"],
                    samp_colors=melt_sample_colors,
                    label_pfx="Melt", z_base=2)

        _draw_phase(ax, x_nodes, vap_mat, mf_vap, vap_active,
                    face_color=PALETTE["blue"],
                    line_color=PALETTE["blue"],
                    samp_colors=vap_sample_colors,
                    label_pfx="Vap", z_base=5)

        ax.set_xlim(0, Lx)
        ax.set_ylim(0, Ly)
        ax.set_xlabel("$x$ (m)")
        if ax is axes[0]:
            ax.set_ylabel("$y$ (m)")
        ax.set_title(f"$t = {t * dt:.3f}$ s",
                     color=PALETTE["text"], fontsize=11, fontweight="bold")
        ax.grid(True, alpha=0.2)

        # annotation box
        ax.text(
            0.02, 0.05,
            f"$T_{{\\rm melt}} = {T_melt_mid:.0f}$ K\n"
            f"$T_{{\\rm vap}} = {T_vap_mid:.0f}$ K\n"
            f"$N_{{\\rm samp}} = {n_samples}$",
            transform=ax.transAxes, fontsize=8, va="bottom",
            color=PALETTE["text"],
            bbox=dict(boxstyle="round,pad=0.3",
                      facecolor=PALETTE["panel"],
                      edgecolor=PALETTE["border"]),
        )

        if ax is axes[0]:
            handles, labels = ax.get_legend_handles_labels()
            seen, h2, l2 = set(), [], []
            for h, l in zip(handles, labels):
                if l not in seen:
                    seen.add(l); h2.append(h); l2.append(l)
            ax.legend(h2, l2, fontsize=8, framealpha=0.88,
                      edgecolor=PALETTE["border"], loc="upper right")

    fig.tight_layout()
    fig.savefig(out)
    print(f"  [fig9] saved {out}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def generate_plots(
    U_obs,
    T_obs_hist,
    local_params,
    x_vis_hist,
    y_vis_hist,
    w_vis_hist,
    get_visible_weights_fn,
    multi_idx,
    eval_psi_fn,
    SOLID_obs,
    Nx, Ny, Lx, Ly,
    num_nodes, P, N_KL,
    time_steps, dt,
    # phase-change thresholds
    T_abl=500.0,
    T_melt_lo=300.0, T_melt_hi=600.0, Delta_melt=50.0,
    T_vap_lo=300.0,  T_vap_hi=500.0,  Delta_vap=50.0,
    # sampling
    n_samples=300,
    depth_eps=10.0, depth_beta=0.005, depth_n_samples=30,
    x=None, y=None,
    figures=(1, 2, 4, 6, 7, 8, 9),
    mc_seed=0,
):
    """
    Generate SG capability figures.

    figures tuple controls which are produced:
      1  -- temperature mean & variance at 3 snapshots
      2  -- KL eigenfunctions + conductivity realisations
      4  -- radiometer obs vs SG prediction
      6  -- ablation depth vs time for surrogate samples
      7  -- vaporisation fraction mean & variance at 3 snapshots
      8  -- vaporisation wellbore: 4 sample fields + empirical mean & variance
      9  -- melt & vaporisation fronts: mean + samples at 3 snapshots
    """
    if x is None:
        x = np.linspace(0, Lx, Nx + 1)
    if y is None:
        y = np.linspace(0, Ly, Ny + 1)

    plt.rcParams.update(_RC)
    print("Generating SG capability plots...")

    if 1 in figures:
        _fig1_mean_variance(
            U_obs, Nx, Ny, num_nodes, P, time_steps, dt, Lx, Ly)

    if 2 in figures:
        _fig2_kl_realisations(
            local_params, Nx, Ny, Lx, Ly,
            k0=SOLID_obs["k0"], k1=SOLID_obs["k1"], N_KL=N_KL)

    if 4 in figures:
        _fig4_radiometer_prediction(
            T_obs_hist, U_obs, num_nodes, P,
            x_vis_hist, y_vis_hist, w_vis_hist,
            x, y, time_steps, dt,
            get_visible_weights_fn=get_visible_weights_fn)

    if 6 in figures:
        _fig6_depth_curves(
            U_obs, Nx, Ny, num_nodes, P, N_KL,
            multi_idx, eval_psi_fn,
            Lx, Ly, T_abl, dt,
            eps=depth_eps, beta=depth_beta,
            n_samples=depth_n_samples, seed=mc_seed)

    if 7 in figures:
        _fig7_vap_mean_variance(
            U_obs, Nx, Ny, num_nodes, P, N_KL,
            multi_idx, eval_psi_fn,
            time_steps, dt, Lx, Ly,
            T_vap_lo, T_vap_hi, Delta_vap,
            n_samples=n_samples, seed=mc_seed)

    if 8 in figures:
        _fig8_wellbore_contours(
            U_obs, Nx, Ny, num_nodes, P, N_KL,
            multi_idx, eval_psi_fn,
            Lx, Ly, T_abl, dt,
            n_show=4, n_samples=n_samples, seed=mc_seed)

    if 9 in figures:
        _fig_melt_vap_fronts(
            U_obs, Nx, Ny, num_nodes, P, N_KL,
            multi_idx, eval_psi_fn,
            Lx, Ly, dt,
            T_melt_lo=T_melt_lo, T_melt_hi=T_melt_hi, Delta_melt=Delta_melt,
            T_vap_lo=T_vap_lo,   T_vap_hi=T_vap_hi,   Delta_vap=Delta_vap,
            n_samples=n_samples, seed=mc_seed)

    print("Done.")