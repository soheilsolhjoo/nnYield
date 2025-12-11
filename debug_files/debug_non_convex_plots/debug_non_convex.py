import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os

OUTPUT_DIR = "debug_non_convex_plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def non_convex_model_numpy(s11, s22, s12):
    """
    Numpy version of the Star-Shaped model.
    Formula: R(theta) = 1.0 - 0.25 * cos(4*theta)
    """
    r = np.sqrt(s11**2 + s22**2 + s12**2 + 1e-8)
    theta = np.arctan2(s22, s11)
    # theta = np.where(theta < 0, theta + 2*np.pi, theta) # Not needed for cos()
    
    R_yield = 1.0 - 0.25 * np.cos(4.0 * theta)
    se = 1.0 * (r / (R_yield + 1e-8))
    return se

def get_hessian_numerical(s11, s22, s12, epsilon=1e-4):
    """
    Calculates Hessian using Finite Differences (Central Difference).
    This guarantees a result even if Autodiff fails.
    """
    n = len(s11)
    inputs = np.stack([s11, s22, s12], axis=1)
    
    # 1. Calculate Value
    val = non_convex_model_numpy(s11, s22, s12)
    
    # 2. Calculate Gradient and Hessian
    hess = np.zeros((n, 3, 3))
    
    # We need to perturb each input dimension
    # H_ij = ( f(x+ei+ej) - f(x+ei-ej) - f(x-ei+ej) + f(x-ei-ej) ) / (4*eps*eps)
    # Diagonal: H_ii = ( f(x+2ei) - 2f(x) + f(x-2ei) ) / (4*eps*eps) ? 
    # Standard Central: f''(x) = (f(x+h) - 2f(x) + f(x-h)) / h^2
    
    # Let's iterate over dimensions [0=s11, 1=s22, 2=s12]
    for i in range(3):
        for j in range(i, 3):
            
            # Create perturbed inputs
            inp_pp = inputs.copy(); inp_pp[:, i] += epsilon; inp_pp[:, j] += epsilon
            inp_mm = inputs.copy(); inp_mm[:, i] -= epsilon; inp_mm[:, j] -= epsilon
            inp_pm = inputs.copy(); inp_pm[:, i] += epsilon; inp_pm[:, j] -= epsilon
            inp_mp = inputs.copy(); inp_mp[:, i] -= epsilon; inp_mp[:, j] += epsilon
            
            f_pp = non_convex_model_numpy(inp_pp[:,0], inp_pp[:,1], inp_pp[:,2])
            f_mm = non_convex_model_numpy(inp_mm[:,0], inp_mm[:,1], inp_mm[:,2])
            f_pm = non_convex_model_numpy(inp_pm[:,0], inp_pm[:,1], inp_pm[:,2])
            f_mp = non_convex_model_numpy(inp_mp[:,0], inp_mp[:,1], inp_mp[:,2])
            
            # Mixed Partial Derivative formula
            d2 = (f_pp - f_pm - f_mp + f_mm) / (4 * epsilon * epsilon)
            
            hess[:, i, j] = d2
            hess[:, j, i] = d2 # Symmetric

    # Eigenvalues
    eigs = np.linalg.eigvalsh(hess)
    min_eigs = eigs[:, 0]
    
    return val, min_eigs

def run_debug():
    print(f"Generating Non-Convex Debug Plots in '{OUTPUT_DIR}' using Finite Differences...")
    
    # ==========================================
    # 1. VISUALIZE THE BROKEN SHAPE (2D Slice)
    # ==========================================
    theta = np.linspace(0, 2*np.pi, 360)
    s11 = np.cos(theta)
    s22 = np.sin(theta)
    s12 = np.zeros_like(theta)
    
    se_vals, min_eigs = get_hessian_numerical(s11, s22, s12)
    radius = 1.0 / (se_vals + 1e-8)
    
    plt.figure(figsize=(7, 7))
    plt.plot(radius * s11, radius * s22, 'r-', linewidth=3, label='Bad Model')
    plt.plot(s11, s22, 'k--', label='Ref Circle', alpha=0.3)
    plt.title("The 'Star' Yield Surface (Visibly Non-Convex)")
    plt.axis('equal'); plt.grid(True, alpha=0.3); plt.legend()
    plt.savefig(os.path.join(OUTPUT_DIR, "bad_yield_locus.png")); plt.close()

    # ==========================================
    # 2. HISTOGRAM (The Smoking Gun)
    # ==========================================
    vecs = np.random.randn(5000, 3).astype(np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    _, hist_eigs = get_hessian_numerical(vecs[:,0], vecs[:,1], vecs[:,2])
    
    plt.figure(figsize=(8, 5))
    plt.hist(hist_eigs, bins=50, color='red', edgecolor='black')
    plt.axvline(0, color='black', linestyle='--', linewidth=2, label='Stability Limit')
    plt.yscale('log')
    plt.title("Histogram of Min Eigenvalues (Red = Unstable)")
    plt.xlabel("Eigenvalue"); plt.ylabel("Count"); plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(OUTPUT_DIR, "bad_histogram.png")); plt.close()

    # ==========================================
    # 3. 1D SLICE (Locating the Instability)
    # ==========================================
    plt.figure(figsize=(8, 4))
    plt.plot(theta/np.pi, min_eigs, 'k-', linewidth=1)
    plt.fill_between(theta/np.pi, min_eigs, 0, where=(min_eigs < -1e-5), color='red', alpha=0.5, label='Unstable')
    plt.fill_between(theta/np.pi, min_eigs, 0, where=(min_eigs >= -1e-5), color='green', alpha=0.3, label='Stable')
    plt.axhline(0, color='k', linestyle='--')
    plt.title("Stability Slice (Red = Unstable)")
    plt.xlabel(r"Theta ($\times \pi$)"); plt.ylabel("Min Eigenvalue")
    plt.savefig(os.path.join(OUTPUT_DIR, "bad_slice.png")); plt.close()

    # ==========================================
    # 4. BINARY STABILITY MAP
    # ==========================================
    res = 100
    T = np.linspace(0, 2*np.pi, res); P = np.linspace(0, np.pi, res)
    TT, PP = np.meshgrid(T, P)
    r=1.0; S12=r*np.cos(PP); Rp=r*np.sin(PP); S11=Rp*np.cos(TT); S22=Rp*np.sin(TT)
    
    _, grid_eigs = get_hessian_numerical(S11.flatten(), S22.flatten(), S12.flatten())
    grid_eigs = grid_eigs.reshape(TT.shape)
    
    # 1.0 (Green) = Stable, 0.0 (Red) = Unstable
    binary_map = np.where(grid_eigs >= -1e-4, 1.0, 0.0)
    
    plt.figure(figsize=(7, 6))
    cmap = mcolors.ListedColormap(['red', 'green'])
    plt.contourf(TT/np.pi, PP/np.pi, binary_map, levels=[-0.1, 0.5, 1.1], cmap=cmap)
    plt.colorbar(ticks=[0, 1], label="0=Unstable, 1=Stable")
    plt.title("Binary Stability Map"); plt.xlabel("Theta"); plt.ylabel("Phi")
    plt.savefig(os.path.join(OUTPUT_DIR, "bad_binary_map.png")); plt.close()

    print("Done. Check 'debug_non_convex_plots' for correct red/green validation.")

if __name__ == "__main__":
    run_debug()