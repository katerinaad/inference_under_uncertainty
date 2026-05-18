"""
Visualization script for m1_elem (and k1_elem, S_elem) at every timestep.

Add this to your code or run it after the forward solve.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import os

def visualize_spatial_weights_history(U_history, solid_prop, melt_prop, 
                                       melt_fraction_from_Tmean,
                                       Nx, Ny, Lx, Ly, P, num_nodes,
                                       save_dir="spatial_weights_plots"):
    """
    Visualize S_elem, k1_elem, m1_elem at each timestep.
    
    Parameters
    ----------
    U_history : ndarray (time_steps, P*num_nodes)
        Solution history from forward solve
    solid_prop, melt_prop : dict
        Material properties
    melt_fraction_from_Tmean : callable
        Function to compute melt fraction from mean temperature
    """
    os.makedirs(save_dir, exist_ok=True)
    
    time_steps = U_history.shape[0]
    
    # Storage for plotting
    Sbar_history = []
    k1_mean_history = []
    m1_mean_history = []
    Tmax_history = []
    
    for n in range(time_steps):
        U_modes = U_history[n].reshape(P, num_nodes)
        Tmean_nodes = U_modes[0].reshape(Nx + 1, Ny + 1)
        
        # Compute melt fraction
        _, S_elem, _ = melt_fraction_from_Tmean(Tmean_nodes)
        
        # Compute blended properties
        k1_elem = solid_prop["k1"] + (melt_prop["k1"] - solid_prop["k1"]) * S_elem
        m1_elem = solid_prop["m1"] + (melt_prop["m1"] - solid_prop["m1"]) * S_elem
        
        Sbar = float(np.mean(S_elem))
        Sbar_history.append(Sbar)
        k1_mean_history.append(np.mean(k1_elem))
        m1_mean_history.append(np.mean(m1_elem))
        Tmax_history.append(np.max(Tmean_nodes))
        
        # Create figure with 4 subplots
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. Mean temperature
        ax = axes[0, 0]
        im = ax.imshow(Tmean_nodes.T, origin='lower', extent=[0, Lx, 0, Ly], 
                       cmap='hot', aspect='equal')
        plt.colorbar(im, ax=ax, label='Temperature')
        ax.set_title(f'Mean Temperature (max={Tmean_nodes.max():.1f})')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        
        # 2. Melt fraction S_elem
        ax = axes[0, 1]
        im = ax.imshow(S_elem.T, origin='lower', extent=[0, Lx, 0, Ly],
                       cmap='Blues', aspect='equal', vmin=0, vmax=1)
        plt.colorbar(im, ax=ax, label='Melt fraction S')
        ax.set_title(f'Melt Fraction S (mean={Sbar:.3f})')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        
        # 3. k1_elem
        ax = axes[1, 0]
        im = ax.imshow(k1_elem.T, origin='lower', extent=[0, Lx, 0, Ly],
                       cmap='viridis', aspect='equal',
                       vmin=melt_prop["k1"], vmax=solid_prop["k1"])
        plt.colorbar(im, ax=ax, label='k1')
        ax.set_title(f'k1_elem (mean={np.mean(k1_elem):.4f})')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        
        # 4. m1_elem
        ax = axes[1, 1]
        im = ax.imshow(m1_elem.T, origin='lower', extent=[0, Lx, 0, Ly],
                       cmap='plasma', aspect='equal',
                       vmin=melt_prop["m1"], vmax=solid_prop["m1"])
        plt.colorbar(im, ax=ax, label='m1')
        ax.set_title(f'm1_elem (mean={np.mean(m1_elem):.1f})')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        
        fig.suptitle(f'Timestep {n}/{time_steps-1}', fontsize=14)
        plt.tight_layout()
        
        # Save
        plt.savefig(os.path.join(save_dir, f'step_{n:03d}.png'), dpi=100)
        plt.close()
        
        print(f"Step {n:3d}: Tmax={Tmax_history[-1]:7.1f}, Sbar={Sbar:.4f}, "
              f"k1_mean={k1_mean_history[-1]:.5f}, m1_mean={m1_mean_history[-1]:.1f}")
    
    # Summary plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    t = np.arange(time_steps)
    
    axes[0, 0].plot(t, Tmax_history, 'r-')
    axes[0, 0].set_xlabel('Timestep')
    axes[0, 0].set_ylabel('Max Temperature')
    axes[0, 0].set_title('Maximum Temperature')
    axes[0, 0].grid(True)
    
    axes[0, 1].plot(t, Sbar_history, 'b-')
    axes[0, 1].set_xlabel('Timestep')
    axes[0, 1].set_ylabel('Mean Melt Fraction')
    axes[0, 1].set_title('Mean Melt Fraction (Sbar)')
    axes[0, 1].set_ylim([0, 1])
    axes[0, 1].grid(True)
    
    axes[1, 0].plot(t, k1_mean_history, 'g-')
    axes[1, 0].axhline(solid_prop["k1"], color='r', linestyle='--', label=f'solid k1={solid_prop["k1"]}')
    axes[1, 0].axhline(melt_prop["k1"], color='b', linestyle='--', label=f'melt k1={melt_prop["k1"]}')
    axes[1, 0].set_xlabel('Timestep')
    axes[1, 0].set_ylabel('Mean k1')
    axes[1, 0].set_title('Mean k1_elem')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    axes[1, 1].plot(t, m1_mean_history, 'm-')
    axes[1, 1].axhline(solid_prop["m1"], color='r', linestyle='--', label=f'solid m1={solid_prop["m1"]}')
    axes[1, 1].axhline(melt_prop["m1"], color='b', linestyle='--', label=f'melt m1={melt_prop["m1"]}')
    axes[1, 1].set_xlabel('Timestep')
    axes[1, 1].set_ylabel('Mean m1')
    axes[1, 1].set_title('Mean m1_elem')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'summary.png'), dpi=150)
    plt.close()
    
    print(f"\nPlots saved to {save_dir}/")
    return Sbar_history, k1_mean_history, m1_mean_history, Tmax_history


def visualize_variance_field(U_history, Nx, Ny, Lx, Ly, P, num_nodes,
                             save_dir="variance_plots"):
    """
    Visualize the variance field at each timestep.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    time_steps = U_history.shape[0]
    
    for n in range(time_steps):
        U_modes = U_history[n].reshape(P, num_nodes)
        
        # Mean temperature
        Tmean = U_modes[0].reshape(Nx + 1, Ny + 1)
        
        # Variance = sum of squared non-mean modes
        Tvar = np.sum(U_modes[1:]**2, axis=0).reshape(Nx + 1, Ny + 1)
        
        # Standard deviation
        Tstd = np.sqrt(Tvar)
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        im0 = axes[0].imshow(Tmean.T, origin='lower', extent=[0, Lx, 0, Ly],
                             cmap='hot', aspect='equal')
        plt.colorbar(im0, ax=axes[0], label='T')
        axes[0].set_title(f'Mean Temperature')
        
        im1 = axes[1].imshow(Tvar.T, origin='lower', extent=[0, Lx, 0, Ly],
                             cmap='magma', aspect='equal')
        plt.colorbar(im1, ax=axes[1], label='Var(T)')
        axes[1].set_title(f'Temperature Variance (max={Tvar.max():.2f})')
        
        im2 = axes[2].imshow(Tstd.T, origin='lower', extent=[0, Lx, 0, Ly],
                             cmap='magma', aspect='equal')
        plt.colorbar(im2, ax=axes[2], label='Std(T)')
        axes[2].set_title(f'Temperature Std Dev (max={Tstd.max():.2f})')
        
        fig.suptitle(f'Timestep {n}', fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'variance_step_{n:03d}.png'), dpi=100)
        plt.close()
        
        print(f"Step {n:3d}: Tmax={Tmean.max():.1f}, Var_max={Tvar.max():.2f}, Std_max={Tstd.max():.2f}")
    
    print(f"\nVariance plots saved to {save_dir}/")


# ============================================================================
# HOW TO USE IN YOUR CODE
# ============================================================================

"""
After run_forward, add:

    # Visualize spatial weights
    from visualize_spatial import visualize_spatial_weights_history, visualize_variance_field
    
    Sbar_hist, k1_hist, m1_hist, Tmax_hist = visualize_spatial_weights_history(
        U_hist, SOLID, MELT, melt_fraction_from_Tmean,
        Nx, Ny, Lx, Ly, P, num_nodes,
        save_dir="spatial_weights_plots"
    )
    
    # Visualize variance field
    visualize_variance_field(
        U_hist, Nx, Ny, Lx, Ly, P, num_nodes,
        save_dir="variance_plots"
    )

This will create:
- spatial_weights_plots/step_000.png ... step_NNN.png  (one per timestep)
- spatial_weights_plots/summary.png (overview of all timesteps)
- variance_plots/variance_step_000.png ... (variance field at each step)
"""


if __name__ == "__main__":
    print("This module provides visualization functions.")
    print("Import and call visualize_spatial_weights_history() after your forward solve.")