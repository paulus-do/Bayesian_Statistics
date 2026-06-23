# file name: main_test_2D.py
import numpy as np
import matplotlib.pyplot as plt
from time import perf_counter
import argparse
from src.data_import import load_and_prepare_data, load_and_prepare_data_2D
from src.MCMC import run_mcmc
from src.MCMC_MH import run_mcmc_mh
from src.FEMBasis import FEMBasis2D

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Run 2D MCMC with different sampling methods')
    parser.add_argument('-mh', '--metropolis-hastings', action='store_true',
                       help='Use Metropolis-Hastings algorithm (default: Gibbs sampling)')
    parser.add_argument('-n', '--n-iter', type=int, default=5000,
                       help='Number of MCMC iterations (default: 5000)')
    parser.add_argument('-b', '--n-burn', type=int, default=2500,
                       help='Number of burn-in iterations (default: 2500)')
    return parser.parse_args()

def plot_mcmc_results(data, B, samples, spline_basis, n_curves=50, seed=42):
    """
    Plot MCMC results including fitted curves and trace plots
    
    Parameters:
    -----------
    data : list of arrays
        Original data for each group
    B : array
        B-spline basis matrix
    samples : dict
        MCMC samples dictionary
    spline_basis : BSplineBasis
        B-spline basis object
    n_curves : int
        Number of posterior curves to plot
    seed : int
        Random seed for reproducibility
    """
    
    np.random.seed(seed)
    
    # Create figure
    fig = plt.figure(figsize=(15, 12))
    
    # 1. Fitted curves for first group
    ax1 = plt.subplot(2, 2, 1)
    c1, global_min, global_max = plot_fitted_curves(ax1, data, spline_basis, samples, B, group_idx=0, n_curves=n_curves)
    
    # 2. Original data scatter plot
    ax2 = plt.subplot(2, 2, 2)
    c2, _, _ = plot_data_scatter(ax2, data_2D, global_min, global_max)
    
    # 3. Trace plots for beta parameters
    ax3 = plt.subplot(2, 2, 3)
    plot_beta_traces(ax3, samples)
    
    # 4. Trace plots for variance parameters
    ax4 = plt.subplot(2, 2, 4)
    plot_variance_traces(ax4, samples)
    
    # Create shared colorbar for the contour plots
    #plt.tight_layout()
    
    # Add a single colorbar for both contour plots
    cbar_ax = fig.add_axes([0.92, 0.55, 0.02, 0.3])  # [left, bottom, width, height]
    fig.colorbar(c1, cax=cbar_ax, label='Temp')
    plt.subplots_adjust(hspace=0.4, wspace=0.3)
    plt.show()
        
    # NEW: Create comparison window for year-specific b_i vs year-averaged data
    plot_year_comparison(data_2D, B, samples, spline_basis)

def plot_year_comparison(data_2D, B, samples, fem, n_years=4):
    """
    Plot comparison between year-specific b_i values and year-averaged data
    
    Parameters:
    -----------
    data_2D : array
        2D data array with shape (time_points, x_dim, y_dim)
    B : array
        Basis matrix
    samples : dict
        MCMC samples dictionary
    fem : FEMBasis2D
        FEM basis object
    n_years : int
        Number of years to plot (default: first 4 years)
    """
    # Get year-specific b_i values
    x = np.linspace(2, 22, 60)
    y = np.linspace(33, 53, 60)
    X, Y = np.meshgrid(x, y)
    points = np.vstack([X.ravel(), Y.ravel()]).T

    phi_plot = fem.evaluate_basis(points)
    mean_b0 = np.stack(samples['b_0']).mean(axis=0)  # Shape: (num_groups, basis_size)
    
    # Create figure
    fig, axes = plt.subplots(2, n_years, figsize=(4*n_years, 8))
    fig.suptitle('Year-specific Comparison: b_i vs Year-averaged Data', fontsize=14)
    
    # Calculate global min/max for consistent color scaling
    all_fitted_values = []
    all_data_values = []
    
    for year_idx in range(n_years):
        # Calculate fitted curve for this year
        mean_curve = phi_plot @ (mean_b0[year_idx])
        
        # Reshape to match grid for contour plotting
        curve_reshaped = mean_curve.reshape(X.shape)
        
        all_fitted_values.append(curve_reshaped)
        all_data_values.append(data_2D[year_idx])
    
    # Calculate global min/max
    global_min = min(np.min(all_fitted_values), np.min(data_2D[:n_years]))
    global_max = max(np.max(all_fitted_values), np.max(data_2D[:n_years]))
    
    # Plot each year
    for year_idx in range(n_years):
        # Top row: Year-specific b_i (fitted curve)
        ax_fitted = axes[0, year_idx]
        mean_curve = phi_plot @ (mean_b0[year_idx])
        curve_reshaped = all_fitted_values[year_idx]
        
        c1 = ax_fitted.contourf(X, Y, curve_reshaped, levels=20, cmap='viridis',
                               vmin=global_min, vmax=global_max)
        ax_fitted.scatter(fem.nodes[:, 0], fem.nodes[:, 1], c='k', s=10, alpha=0.5)
        ax_fitted.set_title(f'Year {year_idx + 1}: Fitted Curve')
        ax_fitted.set_xlabel('X coordinate')
        ax_fitted.set_ylabel('Y coordinate')
        
        # Bottom row: Year-averaged data
        ax_data = axes[1, year_idx]
        
        # Create coordinate grid for this specific year's data
        x_coords = np.linspace(2, 22, data_2D[year_idx].shape[1])
        y_coords = np.linspace(33, 53, data_2D[year_idx].shape[0])
        X_data, Y_data = np.meshgrid(x_coords, y_coords)
        
        c2 = ax_data.contourf(X_data, Y_data, data_2D[year_idx], levels=20, cmap='viridis',
                             vmin=global_min, vmax=global_max)
        ax_data.set_title(f'Year {year_idx + 1}: Original Data')
        ax_data.set_xlabel('X coordinate')
        ax_data.set_ylabel('Y coordinate')
    
    # Add colorbar
    #plt.tight_layout()
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(c1, cax=cbar_ax, label='Temperature')
    plt.subplots_adjust(hspace=0.4, wspace=0.3)

    plt.show()

def plot_fitted_curves(ax, data, fem, samples, B, group_idx=0, n_curves=1):
    """Plot fitted curves for a specific group

    Parameters:
    - ax: matplotlib axis
    - data: original data
    - fem: FEMBasis2D object (used to evaluate basis at points)
    - samples: MCMC samples dict
    - B: B-spline basis matrix (or other basis matrix) (kept for compatibility)
    
    Returns:
    - contour object, global_min, global_max
    """
    x = np.linspace(2, 22, 60)
    y = np.linspace(33, 53, 60)
    X, Y = np.meshgrid(x, y)
    points = np.vstack([X.ravel(), Y.ravel()]).T

    phi_plot = fem.evaluate_basis(points)  # fem should be a FEMBasis2D object

    beta = np.mean(samples['beta'], axis=0)
        
    curve = phi_plot @ beta
    curve_unstack = curve.reshape(X.shape)
    
    # Calculate data range for consistent color scaling
    data_averaged = np.mean(data_2D, axis=0)
    global_min = min(np.min(curve_unstack), np.min(data_averaged))
    global_max = max(np.max(curve_unstack), np.max(data_averaged))
    
    c = ax.contourf(X, Y, curve_unstack, levels=20, cmap='viridis', 
                    vmin=global_min, vmax=global_max)
    ax.scatter(fem.nodes[:, 0], fem.nodes[:, 1], c='k', s=20)  # show basis nodes
    ax.set_title('Fitted Surface (Posterior Mean over all betas)')
    ax.set_xlabel('X coordinate')
    ax.set_ylabel('Y coordinate')
    
    return c, global_min, global_max

def plot_data_scatter(ax, data_2D, global_min=None, global_max=None):
    """Plot original data as scatter plot
    
    Parameters:
    - ax: matplotlib axis
    - data_2D: 2D data array with shape (time_points, x_dim, y_dim)
    - global_min, global_max: optional global limits for consistent coloring
    
    Returns:
    - contour object, global_min, global_max
    """
    # Create coordinate grid matching data dimensions
    x_coords = np.linspace(2, 22, data_2D[0].shape[1])
    y_coords = np.linspace(33, 53, data_2D[0].shape[0])
    X, Y = np.meshgrid(x_coords, y_coords)
    
    # Skip first year and average the remaining data
    data_averaged = np.mean(data_2D, axis=0)
    
    # Use global limits if provided, otherwise calculate from this data
    if global_min is None or global_max is None:
        global_min = np.min(data_averaged)
        global_max = np.max(data_averaged)
    
    scatter = ax.contourf(X, Y, data_averaged, levels=20, cmap='viridis', 
                         vmin=global_min, vmax=global_max)
    
    ax.set_title('Original Data (Average over all years)')
    ax.set_xlabel('X coordinate')
    ax.set_ylabel('Y coordinate')
    
    return scatter, global_min, global_max

def plot_beta_traces(ax, samples):
    """Plot trace plots for beta parameters"""
    beta_samples = samples['beta']
    n_beta = beta_samples.shape[1]
    
    # Plot only first 10 beta parameters to avoid clutter
    n_to_plot = min(10, n_beta)
    for i in range(n_to_plot):
        ax.plot(beta_samples[:, i], alpha=0.7, label=f'β{i}')
    
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Value')
    ax.set_title(f'Trace Plots - Beta Parameters (first {n_to_plot})')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

def plot_variance_traces(ax, samples):
    """Plot trace plots for variance parameters"""
    # Plot sigma_e
    ax.plot(samples['sigma_e'], alpha=0.7, label='σ_e', color='red')
    
    # If sigma_b is a matrix, plot its diagonal elements
    sigma_b_samples = samples['sigma_b']
    if sigma_b_samples.ndim == 3:
        n_b = sigma_b_samples.shape[1]
        # Plot first few diagonal elements
        n_to_plot = min(5, n_b)
        colors = plt.cm.Set1(np.linspace(0, 1, n_to_plot))
        for i in range(n_to_plot):
            ax.plot(sigma_b_samples[:, i, i], alpha=0.7, 
                   color=colors[i], label=f'σ_b[{i},{i}]')
    
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Value')
    ax.set_title('Trace Plots - Variance Parameters')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

# Usage in your main script
if __name__ == "__main__":
    data = load_and_prepare_data()
    data_2D = load_and_prepare_data_2D()

    # Parse command line arguments
    args = parse_arguments()
    
    print(f"Running MCMC with method: {'Metropolis-Hastings' if args.metropolis_hastings else 'Gibbs sampling'}")
    print(f"Iterations: {args.n_iter}, Burn-in: {args.n_burn}")
    
    # --- 1. Create FEM Basis ---
    domain = ((2, 33), (22, 53))
    K = 64  # number of basis nodes
    fem = FEMBasis2D.from_domain(domain, K)
    x = np.linspace(2, 22, 21)
    y = np.linspace(33, 53, 21)
    X, Y = np.meshgrid(x, y)
    points = np.vstack([X.ravel(), Y.ravel()]).T
    phi = fem.evaluate_basis(points)    

    # --- 2. Import CSV data ---
    data_stack = [data_2D[i].reshape(phi.shape[0]) for i in range(len(data_2D))]

    # --- 3. Define Priors ---
    priors = {
        'c_beta': 100.0,            # Vague prior for beta
        'c_epsilon': 0.01,          # Vague prior for sigma_e
        'd_epsilon': 0.01,          # Vague prior for sigma_e
        'eta_b': K + 2,    # Prior df for sigma_b (min for defined mean)
        'S_b': np.eye(K)   # Prior mean matrix for sigma_b
    }
    
    # --- 4. Run MCMC ---
    start_time = perf_counter()
    
    if args.metropolis_hastings:
        # Use Metropolis-Hastings
        samples = run_mcmc_mh(data_stack, phi, priors, n_iter=args.n_iter, n_burn=args.n_burn)
    else:
        # Use Gibbs sampling (default)
        samples = run_mcmc(data_stack, phi, priors, n_iter=args.n_iter, n_burn=args.n_burn)
    
    print("\n--- Run Completed ---")
    elapsed = perf_counter() - start_time
    print(f"Elapsed time: {elapsed:.2f}s")
    print(f"Beta shape: {samples['beta'][0].shape}")
    print(f"b_0 shape: {samples['b_0'].shape}")
    
    # --- 5. Plot results ---
    plot_mcmc_results(data, phi, samples, fem, n_curves=1)