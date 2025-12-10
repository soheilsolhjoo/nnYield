import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from .model import HomogeneousYieldModel
from .data_loader import YieldDataLoader

class SanityChecker:
    def __init__(self, config):
        self.config = config
        self.plot_dir = os.path.join(config['training']['save_dir'], config['experiment_name'], "plots")
        self.csv_dir = os.path.join(config['training']['save_dir'], config['experiment_name'], "csv_data")
        
        os.makedirs(self.plot_dir, exist_ok=True)
        os.makedirs(self.csv_dir, exist_ok=True)
        
        self.model = HomogeneousYieldModel(config)
        self._load_weights()
        self.exp_data = self._load_experiments()

    def _load_weights(self):
        weights_path = os.path.join(self.config['training']['save_dir'],
                                    self.config['experiment_name'], "best_model.weights.h5")
        dummy = tf.constant(np.random.randn(1, 3).astype(np.float32))
        self.model(dummy)
        try:
            self.model.load_weights(weights_path)
            print(f"SanityCheck: Loaded weights from {weights_path}")
        except Exception as e:
            print(f"Warning: Could not load weights ({e}).")

    def _load_experiments(self):
        path = self.config['data'].get('experimental_csv', None)
        if path and os.path.exists(path):
            return pd.read_csv(path)
        return None

    def _save_to_csv(self, dataframe, filename):
        save_path = os.path.join(self.csv_dir, filename)
        dataframe.to_csv(save_path, index=False)
        print(f"   -> Data exported to: {save_path}")

    # =========================================================================
    #  CORE CALCULATION ENGINE
    # =========================================================================
    def _get_predictions(self, s11, s22, s12):
        # 1. Neural Network (Float32 for TF)
        inputs = np.stack([s11, s22, s12], axis=1).astype(np.float32)
        inputs_tf = tf.constant(inputs)
        
        with tf.GradientTape() as tape:
            tape.watch(inputs_tf)
            val_nn = self.model(inputs_tf)
        grads_nn = tape.gradient(val_nn, inputs_tf)
        
        if grads_nn is None:
            grads_nn = tf.zeros_like(inputs_tf)
            
        # 2. Hessian (Finite Diff on NN)
        epsilon = 1e-4
        hess_list = []
        for j in range(3):
            vec = np.zeros((1, 3), dtype=np.float32); vec[0, j] = epsilon
            vec_tf = tf.constant(vec)
            
            with tf.GradientTape() as t1:
                t1.watch(inputs_tf)
                val_pos = self.model(inputs_tf + vec_tf)
            grad_pos = t1.gradient(val_pos, inputs_tf + vec_tf) # Grad w.r.t perturbed input
            if grad_pos is None: grad_pos = tf.zeros_like(inputs_tf)

            with tf.GradientTape() as t2:
                t2.watch(inputs_tf)
                val_neg = self.model(inputs_tf - vec_tf)
            grad_neg = t2.gradient(val_neg, inputs_tf - vec_tf)
            if grad_neg is None: grad_neg = tf.zeros_like(inputs_tf)
            
            hess_col = (grad_pos - grad_neg) / (2.0 * epsilon)
            hess_list.append(hess_col)
            
        hess_nn = tf.stack(hess_list, axis=2).numpy()
        val_nn = val_nn.numpy().flatten()
        grads_nn = grads_nn.numpy()

        # 3. Benchmark (Analytical - Force Float64 for Precision)
        # Cast inputs to float64 to match Finite Difference precision
        s11_64 = s11.astype(np.float64)
        s22_64 = s22.astype(np.float64)
        s12_64 = s12.astype(np.float64)
        
        phys = self.config.get('physics', {})
        F, G, H, N = phys.get('F', 0.5), phys.get('G', 0.5), phys.get('H', 0.5), phys.get('N', 1.5)
        
        # Hill48 Equivalent Stress
        term = F*s22_64**2 + G*s11_64**2 + H*(s11_64-s22_64)**2 + 2*N*s12_64**2
        val_vm = np.sqrt(np.maximum(term, 1e-16)) # Use tighter epsilon for float64
        
        # Analytical Derivatives (d_sigma_bar / d_sigma_ij)
        denom = val_vm 
        denom = np.where(denom < 1e-12, 1e-12, denom) # Avoid div/0
        
        dg_d11 = (G*s11_64 + H*(s11_64-s22_64)) / denom
        dg_d22 = (F*s22_64 - H*(s11_64-s22_64)) / denom
        dg_d12 = (2*N*s12_64) / denom
        
        grads_vm = np.stack([dg_d11, dg_d22, dg_d12], axis=1)
        
        return (val_nn, grads_nn, hess_nn), (val_vm, grads_vm)

    def check_benchmark_derivatives(self):
        """
        Verifies that the Analytical Benchmark Gradients match Finite Difference approximations.
        """
        print("Running Benchmark Derivative Audit...")
        
        # 1. Generate random stress states (Float64 for test)
        s = np.random.uniform(-2, 2, (10, 3)).astype(np.float64)
        s11, s22, s12 = s[:,0], s[:,1], s[:,2]
        
        # 2. Get Analytical Gradients
        _, (_, grad_analytical) = self._get_predictions(s11, s22, s12)
        
        # 3. Compute Finite Difference
        phys = self.config.get('physics', {})
        F, G, H, N = phys.get('F', 0.5), phys.get('G', 0.5), phys.get('H', 0.5), phys.get('N', 1.5)
        
        def hill48_func(s1, s2, s3):
            val_sq = F*s2**2 + G*s1**2 + H*(s1-s2)**2 + 2*N*s3**2
            return np.sqrt(np.maximum(val_sq, 1e-16))

        epsilon = 1e-4 # Tuned for stability
        grad_fd = np.zeros_like(grad_analytical)
        
        for i in range(len(s11)):
            # d/ds11
            v_p = hill48_func(s11[i]+epsilon, s22[i], s12[i])
            v_m = hill48_func(s11[i]-epsilon, s22[i], s12[i])
            grad_fd[i, 0] = (v_p - v_m) / (2*epsilon)
            
            # d/ds22
            v_p = hill48_func(s11[i], s22[i]+epsilon, s12[i])
            v_m = hill48_func(s11[i], s22[i]-epsilon, s12[i])
            grad_fd[i, 1] = (v_p - v_m) / (2*epsilon)
            
            # d/ds12
            v_p = hill48_func(s11[i], s22[i], s12[i]+epsilon)
            v_m = hill48_func(s11[i], s22[i], s12[i]-epsilon)
            grad_fd[i, 2] = (v_p - v_m) / (2*epsilon)

        # 4. Compare
        error = np.abs(grad_analytical - grad_fd)
        max_error = np.max(error)
        
        print(f"   Max discrepancy (Analytical vs FD): {max_error:.2e}")
        
        if max_error < 1e-3: # slightly relaxed tolerance for numerical noise
            print("   ✅ Benchmark Derivatives are consistent.")
        else:
            print("   ❌ Benchmark Derivatives have a BUG.")
            # Print first failure for debugging
            idx = np.unravel_index(np.argmax(error), error.shape)
            print(f"      Fail at sample {idx[0]}, component {idx[1]}")
            print(f"      Analytical: {grad_analytical[idx]:.5f}, FD: {grad_fd[idx]:.5f}")
    
    # -------------------------------------------------------------------------

    def _calc_r_values(self, grads, alpha_rad):
        G11, G22, G12 = grads[:, 0], grads[:, 1], grads[:, 2]
        sin_a, cos_a = np.sin(alpha_rad), np.cos(alpha_rad)
        d_eps_t = -(G11 + G22)
        d_eps_w = G11*(sin_a**2) + G22*(cos_a**2) - 2*G12*sin_a*cos_a
        return np.divide(d_eps_w, d_eps_t, out=np.zeros_like(d_eps_w), where=np.abs(d_eps_t)>1e-6)

    # =========================================================================
    #  CHECK 1: 2D YIELD LOCI SLICES
    # =========================================================================
    def check_2d_loci_slices(self):
        print("Running 2D Loci Slices Check...")
        ref_stress = self.config['model']['ref_stress']
        phys = self.config.get('physics', {})
        N = phys.get('N', 1.5)
        max_shear = ref_stress / np.sqrt(2*N)
        
        shear_ratios = [0.0, 0.4, 0.8, 0.95]
        colors = plt.cm.viridis(np.linspace(0, 1, len(shear_ratios)))
        theta = np.linspace(0, 2*np.pi, 360)
        
        plt.figure(figsize=(7, 7))
        
        # Plot Benchmark Dummy first for Legend Order
        plt.plot([], [], 'k:', linewidth=1.5, label='Benchmark')

        max_val_plotted = 0.0

        for i, ratio in enumerate(shear_ratios):
            current_s12_val = ratio * max_shear
            s11_in = np.cos(theta); s22_in = np.sin(theta); s12_in = np.full_like(theta, current_s12_val)
            
            (val_nn, _, _), (val_vm, _) = self._get_predictions(s11_in, s22_in, s12_in)
            rad_nn = ref_stress / (val_nn + 1e-8)
            rad_vm = ref_stress / (val_vm + 1e-8)
            
            # Track max value for tighter axis limits
            current_max = max(rad_nn.max(), rad_vm.max())
            if current_max > max_val_plotted: max_val_plotted = current_max
            
            plt.plot(rad_nn*s11_in, rad_nn*s22_in, color=colors[i], linewidth=2, label=f"Shear={ratio:.2f}")
            plt.plot(rad_vm*s11_in, rad_vm*s22_in, color='k', linestyle=':', linewidth=1.5, alpha=0.6)

        # Plot Max Shear Marker
        plt.scatter([0], [0], color='red', marker='x', s=100, label=f'Max Shear ({max_shear:.2f})', zorder=10)
        
        # Tighter Axis Limits
        limit = max_val_plotted * 1.1 # 10% margin
        plt.xlim(-limit, limit)
        plt.ylim(-limit, limit)
        
        plt.axis('equal'); plt.xlabel("S11"); plt.ylabel("S22")
        plt.grid(True, alpha=0.3)
        plt.title(f"Yield Loci Slices (Ref={ref_stress})")

        # Legend inside, letting matplotlib find the best empty spot
        plt.legend(loc='best', fontsize='small', framealpha=0.9)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plot_dir, "yield_loci_slices.png")); plt.close()

    # =========================================================================
    #  CHECK 2: RADIUS VS THETA
    # =========================================================================
    def check_radius_vs_theta(self):
        print("Running Radius vs Theta Check...")
        ref_stress = self.config['model']['ref_stress']
        theta = np.linspace(0, 2*np.pi, 360)
        s11_in = np.cos(theta); s22_in = np.sin(theta); s12_in = np.zeros_like(theta)
        
        (val_nn, _, _), (val_vm, _) = self._get_predictions(s11_in, s22_in, s12_in)
        r_nn = ref_stress / (val_nn + 1e-8)
        r_vm = ref_stress / (val_vm + 1e-8)
        
        plt.figure(figsize=(8, 5))
        plt.plot(theta/np.pi, r_vm, 'k--', label='Benchmark')
        plt.plot(theta/np.pi, r_nn, 'r-', label='NN Prediction')
        
        plt.xlabel(r"Theta ($\times \pi$ rad)")
        plt.ylabel("Yield Radius (MPa)")
        plt.title("Yield Radius vs Direction (Plane Stress)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        dmin, dmax = min(r_nn.min(), r_vm.min()), max(r_nn.max(), r_vm.max())
        margin = (dmax - dmin) * 0.2 if dmax != dmin else 0.1
        plt.ylim(dmin - margin, dmax + margin)
        
        plt.savefig(os.path.join(self.plot_dir, "radius_vs_theta.png")); plt.close()

    # =========================================================================
    #  CHECK 3: R-VALUES VS ANGLE
    # =========================================================================
    def check_r_values(self):
        print("Running R-value Check...")
        alpha_deg = np.linspace(0, 90, 91)
        alpha_rad = np.radians(alpha_deg)
        
        # 1. Base Uniaxial Vector
        u11 = np.cos(alpha_rad)**2
        u22 = np.sin(alpha_rad)**2
        u12 = np.sin(alpha_rad)*np.cos(alpha_rad)
        
        # 2. Scale to Hill48 Yield Surface (MATCH DATA LOADER)
        ref_stress = self.config['model']['ref_stress']
        phys = self.config.get('physics', {})
        F, G, H, N = phys.get('F', 0.5), phys.get('G', 0.5), phys.get('H', 0.5), phys.get('N', 1.5)
        C11, C22, C12, C66 = G+H, F+H, -2*H, 2*N
        
        hill_val = C11*u11**2 + C22*u22**2 + C12*u11*u22 + C66*u12**2
        scale = ref_stress / np.sqrt(hill_val + 1e-8)
        
        s11 = u11 * scale
        s22 = u22 * scale
        s12 = u12 * scale
        
        (_, grad_nn, _), (_, grad_vm) = self._get_predictions(s11, s22, s12)
        rv_nn = self._calc_r_values(grad_nn, alpha_rad)
        rv_vm = self._calc_r_values(grad_vm, alpha_rad)
        
        plt.figure(figsize=(7, 5))
        plt.plot(alpha_deg, rv_vm, 'k--', label='Benchmark')
        plt.plot(alpha_deg, rv_nn, 'r-', label='NN Prediction')
        
        all_vals = np.concatenate([rv_nn, rv_vm])
        valid = all_vals[np.abs(all_vals) < 10]
        if len(valid) > 0:
            dmin, dmax = valid.min(), valid.max()
            margin = (dmax - dmin) * 0.2 if dmax!=dmin else 0.5
            plt.ylim(max(0, dmin-margin), dmax+margin)
            
        plt.title("R-values"); plt.xlabel("Angle (deg)"); plt.ylabel("R-value"); plt.legend(); plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(self.plot_dir, "r_values.png")); plt.close()

    # =========================================================================
    #  CHECK 4: FULL DOMAIN HEATMAPS
    # =========================================================================
    def check_full_domain_benchmark(self):
        print("Running Full Domain Benchmark...")
        res_theta, res_phi = 100, 50
        theta = np.linspace(0, 2*np.pi, res_theta)
        phi = np.linspace(0, np.pi, res_phi)
        TT, PP = np.meshgrid(theta, phi)
        
        r=1.0; s12=r*np.cos(PP); r_p=r*np.sin(PP); s11=r_p*np.cos(TT); s22=r_p*np.sin(TT)
        flat_s11, flat_s22, flat_s12 = s11.flatten(), s22.flatten(), s12.flatten()
        
        (val_nn, grad_nn, hess_nn), (val_vm, grad_vm) = self._get_predictions(flat_s11, flat_s22, flat_s12)
        
        err_se = np.abs(val_nn - val_vm) / (val_vm + 1e-8)
        norm_nn = np.linalg.norm(grad_nn, axis=1)
        norm_vm = np.linalg.norm(grad_vm, axis=1)
        cosine = np.clip(np.sum(grad_nn*grad_vm, axis=1)/(norm_nn*norm_vm+1e-8), -1, 1)
        err_angle_rad = np.arccos(cosine)
        eigs = np.linalg.eigvalsh(hess_nn); min_eigs = eigs[:, 0]

        err_se = np.nan_to_num(err_se).reshape(TT.shape)
        err_angle_rad = np.nan_to_num(err_angle_rad).reshape(TT.shape)
        min_eigs = np.nan_to_num(min_eigs).reshape(TT.shape)

        # --- Save ANGLES (TT, PP), not STRESS (s11, s22) ---
        df = pd.DataFrame({
            'theta_rad': TT.flatten(), 
            'phi_rad': PP.flatten(), 
            'min_eig': min_eigs.flatten()
        })
        self._save_to_csv(df, "full_domain_metrics.csv")

        metrics = [
            (err_se, "SE Rel. Error", "se_error_map", 'viridis', "Rel Error"),
            (err_angle_rad, "Grad Angle Deviation", "grad_angle_map", 'magma', "Dev (Rad)"),
            (min_eigs, "Convexity (Min Eig)", "convexity_map", 'RdBu', "Min Eig")
        ]
        
        X_plot, Y_plot = TT / np.pi, PP / np.pi
        for data, title, fname, cmap, cbar_label in metrics:
            plt.figure(figsize=(8, 6))
            norm = None
            if cmap == 'RdBu':
                dmin, dmax = data.min(), data.max()
                if dmin >= 0: dmin = -1e-4
                if dmax <= 0: dmax = 1e-4
                norm = mcolors.TwoSlopeNorm(vmin=dmin, vcenter=0., vmax=dmax)
            
            cp = plt.contourf(X_plot, Y_plot, data, levels=50, cmap=cmap, norm=norm)
            cbar = plt.colorbar(cp); cbar.set_label(cbar_label)
            stats = f"Min: {data.min():.2e}\nMax: {data.max():.2e}"
            plt.text(0.02, 0.98, stats, transform=plt.gca().transAxes, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.8))
            plt.title(title); plt.xlabel(r"Theta ($\times \pi$)"); plt.ylabel(r"Phi ($\times \pi$)")
            plt.savefig(os.path.join(self.plot_dir, f"{fname}.png")); plt.close()
            
    # =========================================================================
    #  CHECK 5: CONVEXITY ANALYSIS
    # =========================================================================
    def check_convexity_detailed(self):
        print("Running Convexity Analysis (Robust Finite Differences)...")
        
        # --- INTERNAL HELPER: Finite Difference Hessian ---
        # Calculates H approx (Grad(x+h) - Grad(x-h)) / 2h
        # This bypasses 2nd-order Autodiff issues to guarantee visibility of non-convexity.
        def get_robust_hessian(s11, s22, s12):
            inputs = np.stack([s11, s22, s12], axis=1).astype(np.float32)
            inputs_tf = tf.constant(inputs)
            epsilon = 1e-3
            hess_cols = []
            
            for i in range(3): # Perturb each input dim
                # Vec = [0, 0, 0] with epsilon at i
                vec = np.zeros((1, 3), dtype=np.float32); vec[0, i] = epsilon
                vec_tf = tf.constant(vec)
                
                # Grad at x + h
                with tf.GradientTape() as t1:
                    t1.watch(inputs_tf)
                    pos_inp = inputs_tf + vec_tf
                    val_pos = self.model(pos_inp)
                grad_pos = t1.gradient(val_pos, pos_inp)
                if grad_pos is None: grad_pos = tf.zeros_like(inputs_tf)

                # Grad at x - h
                with tf.GradientTape() as t2:
                    t2.watch(inputs_tf)
                    neg_inp = inputs_tf - vec_tf
                    val_neg = self.model(neg_inp)
                grad_neg = t2.gradient(val_neg, neg_inp)
                if grad_neg is None: grad_neg = tf.zeros_like(inputs_tf)
                
                # Central Difference
                hess_col = (grad_pos - grad_neg) / (2.0 * epsilon)
                hess_cols.append(hess_col)
            
            # Stack to (N, 3, 3)
            hess_mat = tf.stack(hess_cols, axis=2).numpy()
            eigs = np.linalg.eigvalsh(hess_mat)
            return eigs[:, 0] # Min Eigenvalue

        # --- 1. HISTOGRAM ---
        n_samples = 5000
        vecs = np.random.randn(n_samples, 3)
        vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
        min_eigs = get_robust_hessian(vecs[:,0], vecs[:,1], vecs[:,2])

        plt.figure(figsize=(8, 5))
        # Use log scale but handle negatives cleanly
        plt.hist(min_eigs, bins=50, color='teal', edgecolor='black')
        plt.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero')
        plt.title(f"Hessian Min Eigenvalues (Min: {min_eigs.min():.2e})")
        plt.xlabel("Eigenvalue"); plt.ylabel("Count"); plt.legend(); plt.grid(True, alpha=0.3)
        plt.yscale('log')
        plt.savefig(os.path.join(self.plot_dir, "convexity_histogram.png")); plt.close()

        # --- 2. STABILITY SLICE (Equator) ---
        theta = np.linspace(0, 2*np.pi, 200)
        s11=np.cos(theta); s22=np.sin(theta); s12=np.zeros_like(theta)
        slice_eigs = get_robust_hessian(s11, s22, s12)
        
        plt.figure(figsize=(8, 4))
        plt.plot(theta/np.pi, slice_eigs, 'k-', linewidth=1)
        plt.fill_between(theta/np.pi, slice_eigs, 0, where=(slice_eigs < -1e-5), color='red', alpha=0.5, label='Unstable')
        plt.fill_between(theta/np.pi, slice_eigs, 0, where=(slice_eigs >= -1e-5), color='green', alpha=0.3, label='Stable')
        plt.axhline(0, color='k', linestyle='--')
        plt.title("Stability Slice (Equator)"); plt.xlabel(r"Theta ($\times \pi$)"); plt.ylabel("Min Eig")
        plt.legend(); plt.grid(True)
        plt.savefig(os.path.join(self.plot_dir, "convexity_slice_1d.png")); plt.close()

        # --- 3. BINARY MAP ---
        res = 60 # Lower res for finite diff speed
        T = np.linspace(0, 2*np.pi, res); P = np.linspace(0, np.pi, res)
        TT, PP = np.meshgrid(T, P)
        r=1.0; S12=r*np.cos(PP); Rp=r*np.sin(PP); S11=Rp*np.cos(TT); S22=Rp*np.sin(TT)
        
        grid_eigs = get_robust_hessian(S11.flatten(), S22.flatten(), S12.flatten())
        grid_eigs = grid_eigs.reshape(TT.shape)
        
        # Binary: 1=Green, 0=Red
        binary_map = np.where(grid_eigs >= -1e-5, 1.0, 0.0)
        
        plt.figure(figsize=(7, 6))
        cmap = mcolors.ListedColormap(['red', 'green'])
        plt.contourf(TT/np.pi, PP/np.pi, binary_map, levels=[-0.1, 0.5, 1.1], cmap=cmap)
        cbar = plt.colorbar(ticks=[0, 1])
        cbar.ax.set_yticklabels(['Unstable', 'Stable'])
        plt.title("Binary Stability Map"); plt.xlabel(r"Theta ($\times \pi$)"); plt.ylabel(r"Phi ($\times \pi$)")
        plt.savefig(os.path.join(self.plot_dir, "convexity_binary_map.png")); plt.close()
        
    # =========================================================================
    #  CHECK 6: GLOBAL STATISTICS
    # =========================================================================
    def check_global_statistics(self):
        print("Running Global Statistics (Homogeneity & Error Analysis)...")
        original_samples = self.config['data']['samples']
        # Use sufficient samples for good statistics
        self.config['data']['samples'] = {'loci': 2000, 'uniaxial': 1000}
        
        loader = YieldDataLoader(self.config)
        
        # FIX: Use _generate_raw_data to get the separated tuples (Shape, Phys)
        # instead of get_numpy_data() which now returns flattened arrays.
        (data_shape, data_phys) = loader._generate_raw_data(needs_physics=True)
        
        self.config['data']['samples'] = original_samples 
        
        # --- 1. STRESS STATISTICS (Using Shape Data) ---
        inputs_s, _ = data_shape
        
        # APPLY RANDOM SCALING (Homogeneity Check)
        n_samples = len(inputs_s)
        scales = np.random.uniform(0.5, 2.0, size=(n_samples, 1)).astype(np.float32)
        inputs_scaled = inputs_s * scales
        
        s11_s, s22_s, s12_s = inputs_scaled[:,0], inputs_scaled[:,1], inputs_scaled[:,2]
        (val_nn, _, _), (val_vm, _) = self._get_predictions(s11_s, s22_s, s12_s)
        
        # Plot A: Homogeneity Parity
        plt.figure(figsize=(6, 6))
        plt.scatter(val_vm, val_nn, alpha=0.3, s=10, c='blue', label='Scaled Data')
        
        dmin, dmax = min(val_vm.min(), val_nn.min()), max(val_vm.max(), val_nn.max())
        plt.plot([dmin, dmax], [dmin, dmax], 'k--', label='Perfect Fit')
        
        plt.title(f"Stress Parity (Homogeneity Check)\nRange: [{dmin:.1f}, {dmax:.1f}] MPa")
        plt.xlabel("True Equivalent Stress (Von Mises)")
        plt.ylabel("Predicted Equivalent Stress (NN)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(self.plot_dir, "stats_parity_stress.png"))
        plt.close()

        # Plot B: Relative Error Histogram
        rel_error = (val_nn - val_vm) / (val_vm + 1e-8)
        
        plt.figure(figsize=(8, 5))
        plt.hist(rel_error, bins=50, color='skyblue', edgecolor='black')
        plt.axvline(0, color='red', linestyle='--', label='Zero Error')
        
        mean_err = np.mean(rel_error)
        std_err = np.std(rel_error)
        plt.title(f"Stress Relative Error Distribution\nMean: {mean_err:.2e}, Std: {std_err:.2e}")
        plt.xlabel("Relative Error ((NN - True) / True)")
        plt.ylabel("Count")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(self.plot_dir, "stats_histogram_error.png"))
        plt.close()
        
        # --- 2. R-VALUE STATISTICS (Using Physics Data) ---
        inputs_p, _, r_target_p, geo_p, mask_p = data_phys
        s11_p, s22_p, s12_p = inputs_p[:,0], inputs_p[:,1], inputs_p[:,2]
        
        # Reconstruct alpha
        sin_sq, cos_sq = geo_p[:, 0], geo_p[:, 1]
        alpha_rec = np.arctan2(np.sqrt(sin_sq), np.sqrt(cos_sq))
        
        (_, grad_nn_p, _), _ = self._get_predictions(s11_p, s22_p, s12_p)
        r_nn_p = self._calc_r_values(grad_nn_p, alpha_rec)
        r_vm_p = r_target_p.flatten()
        
        # Filter valid uniaxial points
        valid_idx = (mask_p.flatten() == 1.0) & (np.abs(r_vm_p) < 20)
        
        if np.sum(valid_idx) > 0:
            plt.figure(figsize=(6, 6))
            plt.scatter(r_vm_p[valid_idx], r_nn_p[valid_idx], alpha=0.5, s=20, c='green')
            
            vals = np.concatenate([r_vm_p[valid_idx], r_nn_p[valid_idx]])
            dmin, dmax = vals.min(), vals.max()
            margin = (dmax - dmin) * 0.1 if dmax != dmin else 0.1
            lims = [max(0, dmin - margin), dmax + margin]
            
            plt.plot(lims, lims, 'k--', label='Perfect')
            plt.xlim(lims); plt.ylim(lims)
            plt.title(f"R-value Parity (N={np.sum(valid_idx)})")
            plt.xlabel("True R"); plt.ylabel("Pred R")
            plt.grid(True, alpha=0.3); plt.legend()
            plt.savefig(os.path.join(self.plot_dir, "stats_parity_r.png"))
            plt.close()
        else:
            print("   [Warn] No valid uniaxial points for R-parity check.")
    
    # =========================================================================
    #  CHECK 7: DETAILED LOSS CURVES
    # =========================================================================
    def check_loss_curve(self):
        print("Checking Loss History...")
        hist_path = os.path.join(self.config['training']['save_dir'], self.config['experiment_name'], "loss_history.csv")
        if not os.path.exists(hist_path): return
        df = pd.read_csv(hist_path)
        
        # Create 3x2 subplot to fit everything cleanly
        fig, axes = plt.subplots(3, 2, figsize=(12, 15))
        
        def safe_plot(ax, x, y, c, l, t, yl):
            ax.plot(x, y, c=c, label=l)
            if y.max() > 1e-9: ax.set_yscale('log')
            ax.set_title(t); ax.set_ylabel(yl); ax.grid(True, alpha=0.3); return ax

        # Row 1
        ax=axes[0,0]; ax.plot(df['epoch'], df['train_loss'], 'b-', label='Train')
        if 'val_loss' in df.columns: ax.plot(df['epoch'], df['val_loss'], 'r--', label='Val')
        ax.set_yscale('log'); ax.set_title("Total Loss"); ax.legend(); ax.grid(True, alpha=0.3)
        
        safe_plot(axes[0,1], df['epoch'], df['train_l_se'], 'g', 'SE', "Stress Error", "Loss")
        
        # Row 2
        safe_plot(axes[1,0], df['epoch'], df['train_l_r'], 'purple', 'R', "R-value Error", "Loss")
        
        # Symmetry (New)
        if 'train_l_sym' in df.columns:
            safe_plot(axes[1,1], df['epoch'], df['train_l_sym'], 'cyan', 'Sym', "Symmetry Error (dSE/ds12 @ Equator)", "Loss")
        else:
            axes[1,1].text(0.5, 0.5, "Symmetry Loss Disabled", ha='center')
        
        # Row 3: Stability
        ax=axes[2,0]
        ax.plot(df['epoch'], df['train_l_conv'], 'orange', label='Static Conv')
        if 'train_l_dyn' in df.columns: ax.plot(df['epoch'], df['train_l_dyn'], 'red', linestyle='-.', label='Dynamic Conv')
        if df['train_l_conv'].max() > 1e-9: ax.set_yscale('log')
        ax.set_title("Convexity"); ax.legend(); ax.grid(True, alpha=0.3)

        ax=axes[2,1]
        ax.plot(df['epoch'], df['train_gnorm'], 'gray', label='Grad Norm')
        if df['train_gnorm'].max() > 1e-9: ax.set_yscale('log')
        ax.set_title("Gradient Norm"); ax.grid(True, alpha=0.3)
        
        plt.tight_layout(); plt.savefig(os.path.join(self.plot_dir, "loss_history_detailed.png")); plt.close()

        

    def check_r_calculation_logic(self):
        """Explicitly verifies the R-value math for a known case (45 deg)."""
        print("Running R-value Logic Audit...")
        
        # 1. Define Test Case: Uniaxial 45 deg
        alpha_deg = 45.0
        alpha_rad = np.radians(alpha_deg)
        
        # 2. Calculate Stress Input
        u11 = np.cos(alpha_rad)**2
        u22 = np.sin(alpha_rad)**2
        u12 = np.sin(alpha_rad)*np.cos(alpha_rad)
        
        # Scale to surface (Hill48)
        ref = self.config['model']['ref_stress']
        phys = self.config.get('physics', {})
        F, G, H, N = phys.get('F', 0.5), phys.get('G', 0.5), phys.get('H', 0.5), phys.get('N', 1.5)
        C11, C22, C12, C66 = G+H, F+H, -2*H, 2*N
        
        hill_val = C11*u11**2 + C22*u22**2 + C12*u11*u22 + C66*u12**2
        scale = ref / np.sqrt(hill_val)
        
        # 3. Get Analytical Gradients & R-value
        # (Replicating data_loader logic exactly)
        scale_g = 1.0 / (2.0 * ref)
        g11 = scale_g * (2*C11*(u11*scale) + C12*(u22*scale))
        g22 = scale_g * (2*C22*(u22*scale) + C12*(u11*scale))
        g12 = scale_g * (2*C66*(u12*scale))
        
        dt_a = -(g11 + g22)
        dw_a = g11*np.sin(alpha_rad)**2 + g22*np.cos(alpha_rad)**2 - 2*g12*np.sin(alpha_rad)*np.cos(alpha_rad)
        r_analytical = dw_a / dt_a
        
        # 4. Get Network Prediction R-value
        (_, grad_nn, _), _ = self._get_predictions(np.array([u11*scale]), np.array([u22*scale]), np.array([u12*scale]))
        r_nn = self._calc_r_values(grad_nn, np.array([alpha_rad]))[0]
        
        print(f"   [Audit 45deg] Analytical R: {r_analytical:.4f}")
        print(f"   [Audit 45deg] Network R:    {r_nn:.4f}")
        print(f"   [Audit 45deg] Error:        {abs(r_nn - r_analytical):.4f}")
    
    def _plot_gradient_components(self):
        print("Running Gradient Component Analysis...")
        # Grid setup (Equator slice for clarity, or flattened sphere)
        # Using flattened sphere (Theta-Phi) to see global behavior
        res_theta, res_phi = 60, 30
        theta = np.linspace(0, 2*np.pi, res_theta)
        phi = np.linspace(0, np.pi, res_phi)
        TT, PP = np.meshgrid(theta, phi)
        
        # Map to stress
        r=1.0
        s12 = r * np.cos(PP)
        r_p = r * np.sin(PP)
        s11 = r_p * np.cos(TT)
        s22 = r_p * np.sin(TT)
        
        flat_s11, flat_s22, flat_s12 = s11.flatten(), s22.flatten(), s12.flatten()
        
        # Get Gradients
        (_, grad_nn, _), (_, grad_vm) = self._get_predictions(flat_s11, flat_s22, flat_s12)
        
        # Normalize for direction comparison (optional, but good for shape)
        # Or compare raw magnitudes. Let's compare Raw Magnitudes for rigorous check.
        # If magnitudes differ significantly, normalization helps debug direction vs scale.
        # Let's stick to raw values as requested for "derivatives".
        
        comps = ['dPhi/ds11', 'dPhi/ds22', 'dPhi/ds12']
        
        fig, axes = plt.subplots(3, 3, figsize=(14, 12))
        plt.suptitle("Gradient Component Analysis (Model vs Theory)", fontsize=16)
        
        # X/Y for plotting
        X_plot, Y_plot = TT / np.pi, PP / np.pi
        
        for i in range(3): # Rows: s11, s22, s12
            # Theory
            g_vm = grad_vm[:, i].reshape(TT.shape)
            ax = axes[i, 0]
            cp1 = ax.contourf(X_plot, Y_plot, g_vm, levels=30, cmap='bwr')
            plt.colorbar(cp1, ax=ax)
            ax.set_title(f"Theory {comps[i]}")
            ax.set_ylabel(r"Phi ($\times \pi$)")
            
            # Model
            g_nn = grad_nn[:, i].reshape(TT.shape)
            ax = axes[i, 1]
            cp2 = ax.contourf(X_plot, Y_plot, g_nn, levels=30, cmap='bwr')
            plt.colorbar(cp2, ax=ax)
            ax.set_title(f"Model {comps[i]}")
            
            # Error (Abs Diff)
            err = np.abs(g_nn - g_vm)
            ax = axes[i, 2]
            cp3 = ax.contourf(X_plot, Y_plot, err, levels=30, cmap='viridis')
            plt.colorbar(cp3, ax=ax)
            ax.set_title(f"Abs Error")
            
        axes[2, 0].set_xlabel(r"Theta ($\times \pi$)")
        axes[2, 1].set_xlabel(r"Theta ($\times \pi$)")
        axes[2, 2].set_xlabel(r"Theta ($\times \pi$)")
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Make room for suptitle
        plt.savefig(os.path.join(self.plot_dir, "gradient_components.png"))
        plt.close()
    
    def check_benchmark_derivatives(self):
        """
        Verifies that the Analytical Benchmark Gradients match Finite Difference approximations.
        """
        print("Running Benchmark Derivative Audit...")
        
        # 1. Generate random stress states (Float64 for test)
        s = np.random.uniform(-2, 2, (10, 3)).astype(np.float64)
        s11, s22, s12 = s[:,0], s[:,1], s[:,2]
        
        # 2. Get Analytical Gradients
        _, (_, grad_analytical) = self._get_predictions(s11, s22, s12)
        
        # 3. Compute Finite Difference
        phys = self.config.get('physics', {})
        F, G, H, N = phys.get('F', 0.5), phys.get('G', 0.5), phys.get('H', 0.5), phys.get('N', 1.5)
        
        def hill48_func(s1, s2, s3):
            val_sq = F*s2**2 + G*s1**2 + H*(s1-s2)**2 + 2*N*s3**2
            return np.sqrt(np.maximum(val_sq, 1e-16))

        epsilon = 1e-4 # Tuned for stability
        grad_fd = np.zeros_like(grad_analytical)
        
        for i in range(len(s11)):
            # d/ds11
            v_p = hill48_func(s11[i]+epsilon, s22[i], s12[i])
            v_m = hill48_func(s11[i]-epsilon, s22[i], s12[i])
            grad_fd[i, 0] = (v_p - v_m) / (2*epsilon)
            
            # d/ds22
            v_p = hill48_func(s11[i], s22[i]+epsilon, s12[i])
            v_m = hill48_func(s11[i], s22[i]-epsilon, s12[i])
            grad_fd[i, 1] = (v_p - v_m) / (2*epsilon)
            
            # d/ds12
            v_p = hill48_func(s11[i], s22[i], s12[i]+epsilon)
            v_m = hill48_func(s11[i], s22[i], s12[i]-epsilon)
            grad_fd[i, 2] = (v_p - v_m) / (2*epsilon)

        # 4. Compare
        error = np.abs(grad_analytical - grad_fd)
        max_error = np.max(error)
        
        print(f"   Max discrepancy (Analytical vs FD): {max_error:.2e}")
        
        if max_error < 1e-4: # slightly relaxed tolerance for numerical noise
            print("   ✅ Benchmark Derivatives are consistent.")
        else:
            print("   ❌ Benchmark Derivatives have a BUG.")
            # Print first failure for debugging
            idx = np.unravel_index(np.argmax(error), error.shape)
            print(f"      Fail at sample {idx[0]}, component {idx[1]}")
            print(f"      Analytical: {grad_analytical[idx]:.5f}, FD: {grad_fd[idx]:.5f}")
    
    def run_all(self):
        print("--- Starting Sanity Checks ---")
        # self.check_r_calculation_logic()
        # self.check_2d_loci_slices()
        # self.check_radius_vs_theta()
        # self.check_r_values()
        # self.check_full_domain_benchmark()
        # self.check_convexity_detailed()
        # self.check_global_statistics()
        # self.check_loss_curve()
        # self._plot_gradient_components()
        self.check_benchmark_derivatives()
        print(f"Done. Plots in '{self.plot_dir}'")