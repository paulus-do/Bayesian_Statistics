import numpy as np
from time import perf_counter
import pandas as pd
import pickle
import os
import sys
from src.data_import import load_and_prepare_data_2D
from src.MCMC import run_mcmc
from src.MCMC_MH import run_mcmc_mh
from src.FEMBasis import FEMBasis2D

def ensure_directory(directory):
    """Ensure directory exists, create if it doesn't."""
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")

def save_samples(samples, filename, results_dir='results'):
    """Save samples to file in multiple formats."""
    # Create results directory if it doesn't exist
    ensure_directory(results_dir)
    
    # Save as pickle (for full Python objects)
    pickle_path = os.path.join(results_dir, f"{filename}.pkl")
    with open(pickle_path, 'wb') as f:
        pickle.dump(samples, f)
    
    # Save key arrays as CSV for easier analysis
    if isinstance(samples, dict):
        # Save beta samples
        if 'beta' in samples:
            beta_df = pd.DataFrame(samples['beta'])
            csv_path = os.path.join(results_dir, f"{filename}_beta.csv")
            beta_df.to_csv(csv_path, index=False)
            print(f"  Saved beta to: {csv_path}")
        
        # Save sigma_b if it's a 2D array
        if 'sigma_b' in samples and hasattr(samples['sigma_b'], 'shape'):
            sigma_b = samples['sigma_b']
            if len(sigma_b.shape) == 2:
                sigma_df = pd.DataFrame(sigma_b)
                csv_path = os.path.join(results_dir, f"{filename}_sigma_b.csv")
                sigma_df.to_csv(csv_path, index=False)
                print(f"  Saved sigma_b to: {csv_path}")
    
    print(f"✓ Saved samples to: {pickle_path}")

if __name__ == "__main__":
    data_2D = load_and_prepare_data_2D()

    # Create results directory at the very beginning
    results_dir = 'results'
    ensure_directory(results_dir)
    
    # --- 1. Create FEM Basis ---
    print("=" * 60)
    print("Setting up FEM basis...")
    print("=" * 60)
    
    domain = ((2, 33), (22, 53))
    K = 64  # number of basis nodes
    fem = FEMBasis2D.from_domain(domain, K)
    x = np.linspace(2, 22, 21)
    y = np.linspace(33, 53, 21)
    X, Y = np.meshgrid(x, y)
    points = np.vstack([X.ravel(), Y.ravel()]).T
    phi = fem.evaluate_basis(points)
    
    # Save phi for plotting
    phi_path = os.path.join(results_dir, 'phi_matrix.npy')
    np.save(phi_path, phi)
    print(f"✓ Saved phi matrix to: {phi_path}")
    print(f"  Phi shape: {phi.shape}")

    # --- 2. Import CSV data ---
    print("\n" + "=" * 60)
    print("Loading data...")
    print("=" * 60)
    
    data_stack = [data_2D[i].reshape(phi.shape[0]) for i in range(len(data_2D))]
    
    # Save data for plotting
    data_path = os.path.join(results_dir, 'data_stack.npy')
    np.save(data_path, data_stack)
    print(f"✓ Saved data stack to: {data_path}")
    print(f"  Number of data points (years): {len(data_stack)}")

    # --- 3. Define Priors ---
    priors = {
        'c_beta': 100.0,            # Vague prior for beta
        'c_epsilon': 0.01,          # Vague prior for sigma_e
        'd_epsilon': 0.01,          # Vague prior for sigma_e
        'eta_b': K + 2,    # Prior df for sigma_b (min for defined mean)
        'S_b': np.eye(K)   # Prior mean matrix for sigma_b
    }
    
    # Save priors
    priors_path = os.path.join(results_dir, 'priors.pkl')
    with open(priors_path, 'wb') as f:
        pickle.dump(priors, f)
    print(f"✓ Saved priors to: {priors_path}")

    # --- 4. Run MCMC (Inverse Wishart) ---
    print("\n" + "=" * 60)
    print("Running MCMC with Inverse Wishart")
    print("=" * 60)
    
    start_time = perf_counter()
    samples_normal = run_mcmc(data_stack, phi, priors, n_iter=5000, n_burn=2500)
    elapsed = perf_counter() - start_time
    
    print(f"\n✓ MCMC (IW) completed in {elapsed:.2f}s")
    print(f"  Beta shape: {samples_normal['beta'].shape if 'beta' in samples_normal else 'N/A'}")
    
    # Save IW samples
    save_samples(samples_normal, 'samples_inverse_wishart', results_dir)
    
    # --- 5. Run MCMC (Metropolis-Hastings) ---
    print("\n" + "=" * 60)
    print("Running MCMC with Metropolis-Hastings")
    print("=" * 60)
    
    start_time = perf_counter()
    samples_mh = run_mcmc_mh(data_stack, phi, priors, n_iter=5000, n_burn=2500)
    elapsed = perf_counter() - start_time
    
    print(f"\n✓ MCMC (MH) completed in {elapsed:.2f}s")
    print(f"  Beta shape: {samples_mh['beta'].shape if 'beta' in samples_mh else 'N/A'}")
    
    # Save MH samples
    save_samples(samples_mh, 'samples_metropolis_hastings', results_dir)
    
    # --- 6. Save summary file ---
    summary_path = os.path.join(results_dir, 'sampling_summary.txt')
    with open(summary_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("MCMC Sampling Summary\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Timestamp: {pd.Timestamp.now()}\n\n")
        f.write(f"FEM Basis: K = {K}\n")
        f.write(f"Data points: {len(data_stack)}\n")
        f.write(f"Phi matrix shape: {phi.shape}\n\n")
        f.write("Priors:\n")
        for key, value in priors.items():
            if isinstance(value, np.ndarray):
                f.write(f"  {key}: shape {value.shape}\n")
            else:
                f.write(f"  {key}: {value}\n")
        f.write("\nFiles created:\n")
        f.write("  - phi_matrix.npy\n")
        f.write("  - data_stack.npy\n")
        f.write("  - priors.pkl\n")
        f.write("  - samples_inverse_wishart.pkl\n")
        f.write("  - samples_metropolis_hastings.pkl\n")
    
    print("\n" + "=" * 60)
    print("SAMPLING COMPLETE")
    print("=" * 60)
    print(f"\nAll results saved in: {os.path.abspath(results_dir)}")
    print(f"Summary file: {summary_path}")
    print("\nNow you can run: python -m testing.compare_IW_MH_plot")
    print("=" * 60)