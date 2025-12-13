import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os

# Configuration
OUTPUT_DIR = "debug_convexity_plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"Saving debug plots to: {OUTPUT_DIR}/")

def von_mises_stress(inputs):
    """
    Analytical Von Mises Stress (Ground Truth).
    Equivalent to Hill48 with F=G=H=0.5, N=1.5.
    """
    # inputs: [s11, s22, s12]
    s11 = inputs[:, 0]
    s22 = inputs[:, 1]
    s12 = inputs[:, 2]
    
    # Plane stress Von Mises Formula:
    # sigma_vm = sqrt(s11^2 - s11*s22 + s22^2 + 3*s12^2)
    # Note: 3*s12^2 comes from 2*N*s12^2 where N=1.5
    term = tf.square(s11) + tf.square(s22) - s11*s22 + 3*tf.square(s12)
    
    # Use same epsilon safety as model
    return tf.sqrt(tf.maximum(term, 1e-8))

def get_robust_hessian(func, s11, s22, s12):
    """
    Calculates Hessian using Finite Differences (Central Difference).
    Matches the logic in sanity_check.py exactly.
    """
    inputs = np.stack([s11, s22, s12], axis=1).astype(np.float32)
    inputs_tf = tf.constant(inputs)
    epsilon = 1e-3
    hess_cols = []
    
    for i in range(3): # Perturb each input dim
        vec = np.zeros((1, 3), dtype=np.float32); vec[0, i] = epsilon
        vec_tf = tf.constant(vec)
        
        # Grad at x + h
        with tf.GradientTape() as t1:
            t1.watch(inputs_tf)
            pos_inp = inputs_tf + vec_tf
            val_pos = func(pos_inp)
        grad_pos = t1.gradient(val_pos, pos_inp)
        
        # Grad at x - h
        with tf.GradientTape() as t2:
            t2.watch(inputs_tf)
            neg_inp = inputs_tf - vec_tf
            val_neg = func(neg_inp)
        grad_neg = t2.gradient(val_neg, neg_inp)
        
        hess_col = (grad_pos - grad_neg) / (2.0 * epsilon)
        hess_cols.append(hess_col)
    
    hess_mat = tf.stack(hess_cols, axis=2).numpy()
    eigs = np.linalg.eigvalsh(hess_mat)
    return eigs[:, 0] # Return Minimum Eigenvalue

def run_convexity_checks():
    print("Running Analytical Convexity Checks...")

    # =========================================================
    # 1. HISTOGRAM
    # =========================================================
    print("   -> Generating Histogram...")
    n_samples = 5000
    # Generate random points on unit sphere
    vecs = np.random.randn(n_samples, 3)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    
    min_eigs = get_robust_hessian(von_mises_stress, vecs[:,0], vecs[:,1], vecs[:,2])

    plt.figure(figsize=(8, 5))
    # Plot histogram
    plt.hist(min_eigs, bins=50, color='teal', edgecolor='black')
    plt.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero')
    
    # Stats
    stats_str = f"Min: {min_eigs.min():.2e}\nMax: {min_eigs.max():.2e}"
    plt.title(f"Reference Hessian Min Eigenvalues (Analytical)\n{stats_str}")
    plt.xlabel("Eigenvalue")
    plt.ylabel("Count")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.savefig(os.path.join(OUTPUT_DIR, "debug_histogram.png"))
    plt.close()

    # =========================================================
    # 2. STABILITY SLICE (Equator)
    # =========================================================
    print("   -> Generating Stability Slice...")
    theta = np.linspace(0, 2*np.pi, 200)
    s11 = np.cos(theta)
    s22 = np.sin(theta)
    s12 = np.zeros_like(theta) # Pure shape, no shear
    
    slice_eigs = get_robust_hessian(von_mises_stress, s11, s22, s12)
    
    plt.figure(figsize=(8, 4))
    plt.plot(theta/np.pi, slice_eigs, 'k-', linewidth=1)
    
    # Color regions (though analytical should be all green)
    plt.fill_between(theta/np.pi, slice_eigs, 0, where=(slice_eigs < -1e-5), color='red', alpha=0.5, label='Unstable')
    plt.fill_between(theta/np.pi, slice_eigs, 0, where=(slice_eigs >= -1e-5), color='green', alpha=0.3, label='Stable')
    
    plt.axhline(0, color='k', linestyle='--')
    plt.title("Reference Stability Slice (Equator)")
    plt.xlabel(r"Theta ($\times \pi$)")
    plt.ylabel("Min Eig")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(OUTPUT_DIR, "debug_slice_1d.png"))
    plt.close()

    # =========================================================
    # 3. BINARY MAP
    # =========================================================
    print("   -> Generating Binary Map...")
    res = 60 
    T = np.linspace(0, 2*np.pi, res)
    P = np.linspace(0, np.pi/2.0, res) # Symmetry: Upper hemisphere only
    TT, PP = np.meshgrid(T, P)
    
    r=1.0
    S12 = r * np.cos(PP)
    Rp  = r * np.sin(PP)
    S11 = Rp * np.cos(TT)
    S22 = Rp * np.sin(TT)
    
    grid_eigs = get_robust_hessian(von_mises_stress, S11.flatten(), S22.flatten(), S12.flatten())
    grid_eigs = grid_eigs.reshape(TT.shape)
    
    # Binary: 1=Green (Stable), 0=Red (Unstable)
    binary_map = np.where(grid_eigs >= -1e-5, 1.0, 0.0)
    
    plt.figure(figsize=(7, 6))
    cmap = mcolors.ListedColormap(['red', 'green'])
    # Plot contour
    plt.contourf(TT/np.pi, PP/np.pi, binary_map, levels=[-0.1, 0.5, 1.1], cmap=cmap)
    
    cbar = plt.colorbar(ticks=[0, 1])
    cbar.ax.set_yticklabels(['Unstable', 'Stable'])
    
    plt.title("Reference Binary Stability Map")
    plt.xlabel(r"Theta ($\times \pi$)")
    plt.ylabel(r"Phi ($\times \pi$)")
    
    # Match Sanity Check orientation (Pole at top)
    plt.gca().invert_yaxis()
    
    plt.savefig(os.path.join(OUTPUT_DIR, "debug_binary_map.png"))
    plt.close()

    print("Done. Check the 'debug_convexity_plots' folder.")

if __name__ == "__main__":
    run_convexity_checks()