import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.data_import import load_and_prepare_data
from src.BSpline import BSplineBasis
from src.MCMC import run_mcmc
import random
from src.FEMBasis import FEMBasis2D

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
    plot_mean_avg(data,samples,spline_basis)
    overview_means(data,B, samples, spline_basis)
    # Create figure and axis properly
    fig, ax = plt.subplots(figsize=(12, 8))
    plot_mean_all_points(ax, data, B, samples, spline_basis, group_idx=0, n_curves=50)
    plt.tight_layout()
    plt.show()
    fig = plt.figure(figsize=(15, 12))
    plt.figure(figsize=(15, 12), constrained_layout=True)
    # 1. Fitted curves for first group
    ax1 = plt.subplot(2, 2, 1)
    plot_fitted_curves(ax1, data, B, samples, spline_basis, group_idx=0, n_curves=n_curves)
    
    # 2. Fitted curves for second group (if available)
    ax2 = plt.subplot(2, 2, 2)
    plot_fitted_curves(ax2, data, B, samples, spline_basis, group_idx=1, n_curves=n_curves)
    
    # 3. Trace plots for beta parameters
    ax3 = plt.subplot(2, 2, 3)
    plot_beta_traces(ax3, samples)
    
    # 4. Trace plots for variance parameters
    ax4 = plt.subplot(2, 2, 4)
    plot_variance_traces(ax4, samples)
    
    plt.show()

    
    # Additional detailed plots
    #plot_detailed_traces(samples)
    #plot_posterior_distributions(samples)

def overview_means(data, B_old, samples, spline_basis, group_idx=0, n_curves=1):
    ax1 = plt.subplot(2, 2, 1)
    plot_fitted_curves(ax1, data, B_old, samples, spline_basis, group_idx=0, n_curves=n_curves)
    
    # 2. Fitted curves for second group (if available)
    ax2 = plt.subplot(2, 2, 2)
    plot_fitted_curves(ax2, data, B_old, samples, spline_basis, group_idx=1, n_curves=n_curves)
    
    # 3. Trace plots for beta parameters
    ax3 = plt.subplot(2, 2, 3)
    plot_fitted_curves(ax3, data, B_old, samples, spline_basis, group_idx=2, n_curves=n_curves)
    
    # 4. Trace plots for variance parameters
    ax4 = plt.subplot(2, 2, 4)
    plot_fitted_curves(ax4, data, B_old, samples, spline_basis, group_idx=3, n_curves=n_curves)
    
    plt.tight_layout()
    plt.show()
def plot_fitted_curves(ax, data, B_old, samples, spline_basis, group_idx=0, n_curves=1):
    """Plot fitted curves for a specific group"""
    
    # Get time points from spline basis
    ts, B = spline_basis.evaluate(n_points=200)
    # Randomly select posterior samples to plot
    n_samples = len(samples['beta'])
    selected_indices = np.random.choice(n_samples, n_curves, replace=False)

    # Plot original data
    ts_data=t_eval = np.linspace(30, 60, 30)
    if group_idx < len(data):
        group_data = data[group_idx]
        if group_data.ndim == 2:
            # If data has multiple observations per group
            for i in range(min(5, group_data.shape[0])):  # Plot first 5 observations
                ax.scatter(ts_data, group_data[i], alpha=0.3, s=10, color='gray')
        else:
            ax.scatter(ts_data, group_data, alpha=0.7, s=20, color='black', label=f'Data[{group_idx}]')
    
    # Plot posterior curves
    for idx in selected_indices:
        #beta = samples['beta'][idx]
        beta = np.mean(samples['beta'], axis=0)
        # Get random effects for this group
        if f'b_{group_idx}' in samples:
            b = samples[f'b_{group_idx}'][idx]
        else:
            # If no group-specific random effects, use first group or zero
            b = samples['b_0'][idx] if 'b_0' in samples else np.zeros_like(beta)
        
        # Calculate fitted curve
        curve = B.T @ beta
        
        # Plot with low alpha for posterior uncertainty
        ax.plot(ts, curve, alpha=1, color='red',label='mean over all b[i]')
    
    # Plot mean posterior curve
    '''
    mean_beta = np.mean(samples['beta'], axis=0)
    if f'b_{group_idx}' in samples:
        mean_b = np.mean(samples[f'b_{group_idx}'], axis=0)
    else:
        mean_b = np.mean(samples['b_0'], axis=0) if 'b_0' in samples else np.zeros_like(mean_beta)
    '''
    mean_b0 = np.stack(samples['b_0']).mean(axis=0)  # (25,15)
    mean_curve = B.T @ (mean_b0[group_idx])
    ax.plot(ts, mean_curve, 'r-', color = 'blue', linewidth=2, label=f'mean b[{group_idx}] for specific i = {group_idx}')
    ax.set_xlabel('Lon', fontsize = 15)
    ax.set_ylabel('Temperature', fontsize = 15)
    ax.set_title(f'Fitted Curves over data from year: {group_idx}', fontsize = 20, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

def plot_beta_traces(ax, samples):
    """Plot trace plots for beta parameters"""
    beta_samples = samples['beta']
    n_beta = beta_samples.shape[1]
    
    for i in range(n_beta):
        ax.plot(beta_samples[:, i], alpha=0.7, label=f'β{i}')
    
    ax.set_xlabel('Iteration', fontsize=15)
    ax.set_ylabel('Value', fontsize=15)
    ax.set_title('Trace Plots - Beta Parameters', fontsize=20, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

def plot_variance_traces(ax, samples):
    """Plot trace plots for variance parameters"""
    # Plot sigma_e
    ax.plot(samples['sigma_e'], alpha=0.7, label='σ_e', color='red')
    
    # If sigma_b is a matrix, plot its diagonal elements
    sigma_b_samples = samples['sigma_b']
    if sigma_b_samples.ndim == 3:
        n_b = sigma_b_samples.shape[1]
        for i in range(min(3, n_b)):  # Plot first 3 diagonal elements
            ax.plot(sigma_b_samples[:, i, i], alpha=0.7, label=f'σ_b[{i},{i}]')
    
    ax.set_xlabel('Iteration', fontsize=15)
    ax.set_ylabel('Value', fontsize=15)
    ax.set_title('Trace Plots - Variance Parameters', fontsize=20, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

def plot_detailed_traces(samples):
    """Create detailed trace plots in separate figure"""
    n_params = 0
    if 'beta' in samples:
        n_params += samples['beta'].shape[1]
    if 'sigma_e' in samples:
        n_params += 1
    if 'sigma_b' in samples and samples['sigma_b'].ndim == 3:
        n_params += min(4, samples['sigma_b'].shape[1])  # First 4 diagonal elements
    
    n_cols = 3
    n_rows = (n_params + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
    axes = axes.flatten() if n_rows > 1 else [axes]
    
    plot_idx = 0
    
    # Beta parameters
    if 'beta' in samples:
        beta_samples = samples['beta']
        for i in range(beta_samples.shape[1]):
            if plot_idx < len(axes):
                axes[plot_idx].plot(beta_samples[:, i])
                axes[plot_idx].set_title(f'Beta {i}')
                axes[plot_idx].set_xlabel('Iteration')
                axes[plot_idx].set_ylabel('Value')
                axes[plot_idx].grid(True, alpha=0.3)
                plot_idx += 1
    
    # sigma_e
    if 'sigma_e' in samples and plot_idx < len(axes):
        axes[plot_idx].plot(samples['sigma_e'])
        axes[plot_idx].set_title('Sigma_e')
        axes[plot_idx].set_xlabel('Iteration')
        axes[plot_idx].set_ylabel('Value')
        axes[plot_idx].grid(True, alpha=0.3)
        plot_idx += 1
    
    # sigma_b diagonal elements
    if 'sigma_b' in samples and samples['sigma_b'].ndim == 3 and plot_idx < len(axes):
        sigma_b_samples = samples['sigma_b']
        n_b = sigma_b_samples.shape[1]
        for i in range(min(4, n_b)):  # First 4 diagonal elements
            if plot_idx < len(axes):
                axes[plot_idx].plot(sigma_b_samples[:, i, i])
                axes[plot_idx].set_title(f'Sigma_b[{i},{i}]')
                axes[plot_idx].set_xlabel('Iteration')
                axes[plot_idx].set_ylabel('Value')
                axes[plot_idx].grid(True, alpha=0.3)
                plot_idx += 1
    
    # Hide empty subplots
    for i in range(plot_idx, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.show()

def plot_posterior_distributions(samples):
    """Plot posterior distributions of parameters"""
    n_params = 0
    if 'beta' in samples:
        n_params += samples['beta'].shape[1]
    if 'sigma_e' in samples:
        n_params += 1
    
    n_cols = 3
    n_rows = (n_params + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
    axes = axes.flatten() if n_rows > 1 else [axes]
    
    plot_idx = 0
    
    # Beta parameters
    if 'beta' in samples:
        beta_samples = samples['beta']
        for i in range(beta_samples.shape[1]):
            if plot_idx < len(axes):
                axes[plot_idx].hist(beta_samples[:, i], bins=50, alpha=0.7, density=True)
                axes[plot_idx].set_title(f'Beta {i} Posterior')
                axes[plot_idx].set_xlabel('Value')
                axes[plot_idx].set_ylabel('Density')
                plot_idx += 1
    
    # sigma_e
    if 'sigma_e' in samples and plot_idx < len(axes):
        axes[plot_idx].hist(samples['sigma_e'], bins=50, alpha=0.7, density=True, color='red')
        axes[plot_idx].set_title('Sigma_e Posterior')
        axes[plot_idx].set_xlabel('Value')
        axes[plot_idx].set_ylabel('Density')
        plot_idx += 1
    
    # Hide empty subplots
    for i in range(plot_idx, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.show()


def plot_mean_all_points(ax, data, B_old, samples, spline_basis, group_idx=0, n_curves=1000):
    """Plot fitted curves for a specific group"""
    
    # Get time points from spline basis
    ts, B = spline_basis.evaluate(n_points=200)
    
    # Randomly select posterior samples to plot
    n_samples = len(samples['beta'])
    selected_indices = np.random.choice(n_samples, n_curves, replace=False)

    #------plot all data points -------
    # Convert list to numpy array for easier manipulation
    data_array = np.array(data)  # Shape: (time_points, longitudes)
    
    # If years not provided, create sequential indices
    years = list(range(len(data_array)))
    
    # Get unique longitudes - map from 30-60 to 0-30 to match spline basis
    n_longitudes = data_array.shape[1]
    original_longitudes = np.linspace(30, 60, n_longitudes, endpoint=False)
    # Map longitudes from 30-60 range to 0-30 range
    mapped_longitudes = original_longitudes
    
    # Create color mapping
    cmap = plt.get_cmap('tab20')
    year_colors = {year: cmap(i % 20) for i, year in enumerate(years)}
    
    # Plot data efficiently - for each time point, plot all longitudes
    for time_idx, year in enumerate(years):
        # Get temperatures for all longitudes at this time point
        temps = data_array[time_idx]
        
        # Only add label for first time point to avoid duplicate legend entries
        label = str(year) if time_idx == 0 else ""
        
        # Plot all longitudes for this time point (mapped to 0-30 range)
        ax.scatter(original_longitudes, temps,
                   color=year_colors[year], label=label, alpha=0.7, s=20, zorder=5)
    
    # Plot posterior curves with transparency so data points are visible
    for idx in selected_indices:
        
        # Get random effects for this group
        #if f'b_{group_idx}' in samples:
        #    b = samples[f'b_{group_idx}'][idx]
        #else:
            # If no group-specific random effects, use first group or zero
        #b = samples['b_0'][idx] # if 'b_0' in samples else np.zeros_like(beta)
        #b = np.stack(samples['b_0']).mean(axis=0)  # (25,15)
        # Calculate fitted curve
        spl = random.randint(0, len(samples['b_0'])-1)
        yrs=random.randint(0,len(samples['b_0'][0])-1)
        b = samples['b_0'][spl][yrs]
        curve = B.T @ b
        current_year_color = year_colors[yrs]
        # Plot with transparency so data points are visible
        label = 'Posterior samples' if idx == selected_indices[0] else ""
        ax.plot(ts, curve, alpha=0.5, color=current_year_color, label=label, zorder=3)
    
    # Plot mean posterior curve
    mean_beta = np.mean(samples['beta'], axis=0)
    if f'b_{group_idx}' in samples:
        mean_b = np.mean(samples[f'b_{group_idx}'], axis=0)
    else:
        mean_b = np.mean(samples['b_0'], axis=0) if 'b_0' in samples else np.zeros_like(mean_beta)
    
    mean_curve = B.T @ mean_beta
    ax.plot(ts, mean_curve, 'r-', linewidth=3, label='Mean posterior', zorder=4)
    
    # Set labels and title
    ax.set_xlabel('Longitude (30-60)', fontsize = 18)
    ax.set_ylabel('Temperature', fontsize = 18)
    ax.set_title(f'Fitted Curves over all datapoints', fontsize = 25, fontweight='bold')
    
    # Create clean legend without duplicates
    handles, labels = ax.get_legend_handles_labels()
    unique_labels = []
    unique_handles = []
    for handle, label in zip(handles, labels):
        if label not in unique_labels and label != "":
            unique_labels.append(label)
            unique_handles.append(handle)
    
    if unique_handles:
        ax.legend(unique_handles, unique_labels, title="Year", loc='best')
    
    ax.grid(True, alpha=0.3)
# Usage in your main script

def plot_mean_avg(data, samples, spline_basis):
    mean_b0 = np.stack(samples['b_0']).mean(axis=0)  # (25,15)
    print('mean_b0:',mean_b0.shape)
    ts, B = spline_basis.evaluate(n_points=200)
    for n in range(mean_b0.shape[0]):
        coeffs = mean_b0[n]
        mean_curve = B.T @ coeffs
        plt.plot(ts, mean_curve, label=f'year:{n}')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    data = load_and_prepare_data()

     # --- 1. Create B-Spline Basis ---

    ############B-Spline Basis##############

    K=15 # number of basis functions. Calles "K" in Biostatistics paper
    spline_basis=BSplineBasis(t0=30,t1=60, n_basis=K,degree=4)
    ts, B= spline_basis.evaluate()

  

    # --- 2. Import CSV data ---


    # --- 3. Define Priors ---
    priors = {
        'c_beta': 100.0,            # Vague prior for beta
        'c_epsilon': 0.01,          # Vague prior for sigma_e
        'd_epsilon': 0.01,          # Vague prior for sigma_e
        'eta_b': K + 2,    # Prior df for sigma_b (min for defined mean)
        'S_b': np.eye(K)   # Prior mean matrix for sigma_b
    }
 # --- 4. Run MCMC ---
    samples = run_mcmc(data, B.T, priors, n_iter=5000, n_burn=2500)
    print("\n--- Generating Plots ---")
    plot_mcmc_results(data, B.T, samples, spline_basis, n_curves=1)