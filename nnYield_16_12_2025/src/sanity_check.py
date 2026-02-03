import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import qmc
from .model import HomogeneousYieldModel
from .data_loader import YieldDataLoader

class SanityChecker:
    def __init__(self, model, config, output_dir):
        # FIX: Ensure config is a dictionary so subscription [ ] works
        # This handles both cases: if 'config' is passed as an Object or a Dict.
        if hasattr(config, 'to_dict'):
            self.config = config.to_dict()
        else:
            self.config = config
        
        # Use the passed output_dir, not just the config path (safer)
        self.plot_dir = os.path.join(output_dir, "plots")
        self.csv_dir = os.path.join(output_dir, "csv_data")
        
        os.makedirs(self.plot_dir, exist_ok=True)
        os.makedirs(self.csv_dir, exist_ok=True)
        
        # Now 'model' exists because we passed it in above
        self.model = model
        self.output_dir = output_dir
        
        # IMPORTANT: Since you are passing a pre-trained model, 
        # you likely DO NOT need to call self._load_weights() here anymore,
        # unless you specifically want to reload from disk instead of using the object in memory.
        # If the model passed in is already trained, you can comment this out:
        # self._load_weights() 
        
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
        # Now valid because self.config is a dict
        path = self.config['data'].get('experimental_csv', None)
        if path and os.path.exists(path):
            return pd.read_csv(path)
        return None

    def _save_to_csv(self, dataframe, filename):
        save_path = os.path.join(self.csv_dir, filename)
        dataframe.to_csv(save_path, index=False)
        print(f"   -> Data exported to: {save_path}")

    # =========================================================================
    #  HELPER: TF GRAPH FUNCTION (Prevents Retracing)
    # =========================================================================
    @tf.function(reduce_retracing=True)
    def _predict_graph(self, inputs):
        """
        Optimized graph execution for NN predictions + Hessian.
        Compiles once, runs fast, avoids 'retracing' warnings.
        """
        with tf.GradientTape(persistent=True) as tape2:
            tape2.watch(inputs)
            with tf.GradientTape() as tape1:
                tape1.watch(inputs)
                val_nn = self.model(inputs)
            grads_nn = tape1.gradient(val_nn, inputs)
            
        # Handle cases where gradients might be None (start of training)
        if grads_nn is None:
            grads_nn = tf.zeros_like(inputs)
            hess_nn = tf.zeros((tf.shape(inputs)[0], 3, 3))
        else:
            hess_nn = tape2.batch_jacobian(grads_nn, inputs)
            
        del tape2
        return val_nn, grads_nn, hess_nn

    # =========================================================================
    #  CORE CALCULATION ENGINE (Updated)
    # =========================================================================
    def _get_predictions(self, s11, s22, s12):
        # 1. Prepare Inputs
        inputs = np.stack([s11, s22, s12], axis=1).astype(np.float32)
        inputs_tf = tf.constant(inputs)
        
        # 2. Call the Cached Graph Function (Fixes Warning)
        val_nn_tf, grads_nn_tf, hess_nn_tf = self._predict_graph(inputs_tf)
        
        # Convert to Numpy
        val_nn = val_nn_tf.numpy().flatten()
        grads_nn = grads_nn_tf.numpy()
        hess_nn = hess_nn_tf.numpy()

        # 3. Benchmark (Analytical)
        s11_64 = s11.astype(np.float64)
        s22_64 = s22.astype(np.float64)
        s12_64 = s12.astype(np.float64)
        
        phys = self.config.get('physics', {})
        F, G, H, N = phys.get('F', 0.5), phys.get('G', 0.5), phys.get('H', 0.5), phys.get('N', 1.5)
        
        # Hill48 Equivalent Stress
        term = F*s22_64**2 + G*s11_64**2 + H*(s11_64-s22_64)**2 + 2*N*s12_64**2
        val_vm = np.sqrt(np.maximum(term, 1e-16)) 
        
        # Analytical Derivatives
        denom = val_vm 
        denom = np.where(denom < 1e-12, 1e-12, denom)
        
        dg_d11 = (G*s11_64 + H*(s11_64-s22_64)) / denom
        dg_d22 = (F*s22_64 - H*(s11_64-s22_64)) / denom
        dg_d12 = (2*N*s12_64) / denom
        
        grads_vm = np.stack([dg_d11, dg_d22, dg_d12], axis=1)
        
        return (val_nn, grads_nn, hess_nn), (val_vm, grads_vm)

    # =========================================================================
    #  AUXILIARY CALCULATION: R-VALUES
    def _calc_r_values(self, grads, alpha_rad):
        G11, G22, G12 = grads[:, 0], grads[:, 1], grads[:, 2]
        sin_a, cos_a = np.sin(alpha_rad), np.cos(alpha_rad)
        d_eps_t = -(G11 + G22)
        d_eps_w = G11*(sin_a**2) + G22*(cos_a**2) - 2*G12*sin_a*cos_a
        return np.divide(d_eps_w, d_eps_t, out=np.zeros_like(d_eps_w), where=np.abs(d_eps_t)>1e-6)
    
    # =========================================================================
    #  SAMPLING HELPER (Matches Trainer Logic)
    def _sample_points_on_surface(self, n_samples):
        """
        Generates stress points exactly on the yield surface using Sobol sampling.
        """
        # 1. Sobol Sampling (Uniform Direction)
        # d=2 mapped to Sphere surface area
        sampler = qmc.Sobol(d=2, scramble=True)
        
        # Next power of 2 for balance
        m = int(np.ceil(np.log2(n_samples)))
        sample = sampler.random(n=2**m)
        sample = sample[:n_samples]
        
        # Map to Upper Hemisphere (Symmetry: S12 >= 0)
        # Theta ~ Uniform[0, 2pi]
        theta = sample[:, 0] * 2 * np.pi
        
        # Z ~ Uniform[0, 1] -> Phi = arccos(z)
        z = sample[:, 1]
        phi = np.arccos(z)
        
        # 2. Unit Vectors
        s12_u = np.cos(phi)
        r_plane_u = np.sin(phi)
        s11_u = r_plane_u * np.cos(theta)
        s22_u = r_plane_u * np.sin(theta)
        
        unit_inputs = np.stack([s11_u, s22_u, s12_u], axis=1).astype(np.float32)
        
        # 3. Scale to Yield Surface
        # Query model for yield radius along these directions
        inputs_tf = tf.constant(unit_inputs)
        pred_se = self.model(inputs_tf).numpy().flatten()
        
        ref_stress = self.config['model']['ref_stress']
        radii = ref_stress / (pred_se + 1e-8)
        
        surface_points = unit_inputs * radii[:, None]
        return surface_points

        inputs_tf = tf.constant(unit_inputs)
        pred_se = self.model(inputs_tf).numpy().flatten()
        
        ref_stress = self.config['model']['ref_stress']
        radii = ref_stress / (pred_se + 1e-8)
        
        surface_points = unit_inputs * radii[:, None]
        return surface_points

    # =========================================================================
    #  HELPER: PRINCIPAL MINORS (Sylvester's Criterion)
    # =========================================================================
    def _get_principal_minors_numpy(self, hess_matrix):
        """
        Calculates the three leading principal minors for a batch of 3x3 matrices.
        Input: hess_matrix (N, 3, 3)
        Output: min_minor (N,) - The minimum of the three minors for each point.
        """
        # 1st Principal Minor (Top-left element)
        m1 = hess_matrix[:, 0, 0]
        
        # 2nd Principal Minor (Top-left 2x2 determinant)
        # | H11 H12 |
        # | H21 H22 |  -> H11*H22 - H12*H21
        m2 = (hess_matrix[:, 0, 0] * hess_matrix[:, 1, 1]) - \
             (hess_matrix[:, 0, 1] * hess_matrix[:, 1, 0])
        
        # 3rd Principal Minor (Full 3x3 determinant)
        m3 = np.linalg.det(hess_matrix)
        
        # For convexity, ALL minors must be >= 0.
        # We track the minimum one to see if the condition fails.
        min_minor = np.minimum(np.minimum(m1, m2), m3)
        
        return min_minor
    # =========================================================================
    #  HELPER: HESSIAN VIA AUTODIFF 
    def _get_hessian_autodiff(self, points):
        """Method 1: Exact Hessian via Automatic Differentiation."""
        inputs = tf.convert_to_tensor(points, dtype=tf.float32)
        
        with tf.GradientTape(persistent=True) as tape2:
            tape2.watch(inputs)
            with tf.GradientTape() as tape1:
                tape1.watch(inputs)
                val = self.model(inputs)
            grads = tape1.gradient(val, inputs)
            
        hess_matrix = tape2.batch_jacobian(grads, inputs)
        del tape2
        return hess_matrix.numpy()

        inputs_tf = tf.constant(unit_inputs)
        pred_se = self.model(inputs_tf).numpy().flatten()
        
        ref_stress = self.config['model']['ref_stress']
        radii = ref_stress / (pred_se + 1e-8)
        
        surface_points = unit_inputs * radii[:, None]
        return surface_points

    # def _get_hessian_fdm(self, points):
    #     """
    #     Approximate Hessian via Finite Differences with Symmetry Handling.
    #     """
    #     inputs = tf.convert_to_tensor(points, dtype=tf.float32)
    #     epsilon = 1e-1  # Large epsilon for stability at high stress (~100MPa)
    #     hess_cols = []
        
    #     # Check if symmetry is active (S12 >= 0)
    #     enforce_symmetry = self.config['data'].get('symmetry', False)

    #     for i in range(3):
    #         vec = np.zeros((1, 3), dtype=np.float32); vec[0, i] = epsilon
    #         vec_tf = tf.constant(vec)
            
    #         with tf.GradientTape() as t1:
    #             t1.watch(inputs)
    #             pos = inputs + vec_tf
                
    #             # FIX: Handle Symmetry Boundary (e.g. S12 = 0 - eps)
    #             if enforce_symmetry and i == 2: # If perturbing S12
    #                 # Use tf.abs to map negative shear back to positive domain
    #                 # We reconstruct the tensor to apply abs only to S12
    #                 s11, s22, s12 = tf.unstack(pos, axis=1)
    #                 pos_wrapped = tf.stack([s11, s22, tf.abs(s12)], axis=1)
    #                 val_pos = self.model(pos_wrapped)
    #             else:
    #                 val_pos = self.model(pos)
            
    #         # Gradient is w.r.t 'pos' (the perturbed input), effectively capturing the 
    #         # slope at the mirrored point if wrapped.
    #         grad_pos = t1.gradient(val_pos, pos)
            
    #         with tf.GradientTape() as t2:
    #             t2.watch(inputs)
    #             neg = inputs - vec_tf
                
    #             # FIX: Handle Symmetry Boundary
    #             if enforce_symmetry and i == 2:
    #                 s11, s22, s12 = tf.unstack(neg, axis=1)
    #                 neg_wrapped = tf.stack([s11, s22, tf.abs(s12)], axis=1)
    #                 val_neg = self.model(neg_wrapped)
    #             else:
    #                 val_neg = self.model(neg)

    #         grad_neg = t2.gradient(val_neg, neg)
            
    #         hess_col = (grad_pos - grad_neg) / (2.0 * epsilon)
    #         hess_cols.append(hess_col)
            
    #     hess_matrix = tf.stack(hess_cols, axis=2).numpy()
    #     return hess_matrix

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
        alpha_deg = np.linspace(0, 90, 91).astype(np.float32)
        alpha_rad = np.radians(alpha_deg)
        
        # 1. Base Uniaxial Vector
        u11 = np.cos(alpha_rad)**2
        u22 = np.sin(alpha_rad)**2
        u12 = np.sin(alpha_rad)*np.cos(alpha_rad)
        
        unit_inputs = np.stack([u11, u22, u12], axis=1)

        # 2. Scale to NN Yield Surface (STRICT FIX)
        # Previously scaled to Hill48 surface. Now we scale to the NN surface.
        inputs_tf = tf.constant(unit_inputs)
        pred_se = self.model(inputs_tf).numpy().flatten()
        
        ref_stress = self.config['model']['ref_stress']
        radii = ref_stress / (pred_se + 1e-8)
        
        s11 = u11 * radii
        s22 = u22 * radii
        s12 = u12 * radii
        
        # 3. Calculate Gradients
        (_, grad_nn, _), (_, grad_vm) = self._get_predictions(s11, s22, s12)
        rv_nn = self._calc_r_values(grad_nn, alpha_rad)
        rv_vm = self._calc_r_values(grad_vm, alpha_rad)
        
        # ... (Plotting remains the same) ...
        plt.figure(figsize=(7, 5))
        plt.plot(alpha_deg, rv_vm, 'k--', label='Benchmark')
        plt.plot(alpha_deg, rv_nn, 'r-', label='NN Prediction')
        
        all_vals = np.concatenate([rv_nn, rv_vm])
        valid = all_vals[np.abs(all_vals) < 10]
        if len(valid) > 0:
            dmin, dmax = valid.min(), valid.max()
            margin = (dmax - dmin) * 0.2 if dmax!=dmin else 0.5
            plt.ylim(max(0, dmin-margin), dmax+margin)
            
        plt.title("R-values"); plt.xlabel("Angle (deg)"); plt.ylabel("R-value")
        plt.legend(); plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(self.plot_dir, "r_values.png")); plt.close()

    # =========================================================================
    #  CHECK 4: FULL DOMAIN HEATMAPS
    # =========================================================================
    def check_full_domain_benchmark(self):
        print("Running Full Domain Benchmark...")
        res_theta, res_phi = 100, 50
        theta = np.linspace(0, 2*np.pi, res_theta).astype(np.float32)
        # Symmetry: S12 >= 0 [0, pi/2]
        phi = np.linspace(0, np.pi/2.0, res_phi).astype(np.float32)
        TT, PP = np.meshgrid(theta, phi)
        
        # 1. Generate Unit Directions (Spherical -> Cartesian)
        u12 = np.cos(PP)
        r_plane = np.sin(PP)
        u11 = r_plane * np.cos(TT)
        u22 = r_plane * np.sin(TT)
        
        flat_u11, flat_u22, flat_u12 = u11.flatten(), u22.flatten(), u12.flatten()
        unit_inputs = np.stack([flat_u11, flat_u22, flat_u12], axis=1)
        
        # 2. Scale to Yield Surface (CRITICAL FIX)
        # Query model with unit vectors to find the yield radius
        inputs_tf = tf.constant(unit_inputs)
        pred_se = self.model(inputs_tf).numpy().flatten()
        
        # Radius = Ref / Pred_SE
        ref_stress = self.config['model']['ref_stress']
        radii = ref_stress / (pred_se + 1e-8)
        
        # Scale inputs to get real surface points
        s11 = flat_u11 * radii
        s22 = flat_u22 * radii
        s12 = flat_u12 * radii
        
        # 3. Calculate Predictions on Surface Points
        (val_nn, grad_nn, hess_nn), (val_vm, grad_vm) = self._get_predictions(s11, s22, s12)
        
        # ... (Metrics Calculation remains the same) ...
        err_se = np.abs(val_nn - val_vm) / (val_vm + 1e-8)
        norm_nn = np.linalg.norm(grad_nn, axis=1)
        norm_vm = np.linalg.norm(grad_vm, axis=1)
        cosine = np.clip(np.sum(grad_nn*grad_vm, axis=1)/(norm_nn*norm_vm+1e-8), -1, 1)
        err_angle_rad = np.arccos(cosine)
        eigs = np.linalg.eigvalsh(hess_nn); min_eigs = eigs[:, 0]

        err_se = np.nan_to_num(err_se).reshape(TT.shape)
        err_angle_rad = np.nan_to_num(err_angle_rad).reshape(TT.shape)
        min_eigs = np.nan_to_num(min_eigs).reshape(TT.shape)

        # Save CSV
        df = pd.DataFrame({
            'theta_rad': TT.flatten(), 'phi_rad': PP.flatten(), 'min_eig': min_eigs.flatten()
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
            plt.title(title); plt.xlabel(r"Theta ($\times \pi$)"); plt.ylabel(r"Phi ($\times \pi$)")
            plt.gca().invert_yaxis()
            plt.savefig(os.path.join(self.plot_dir, f"{fname}.png")); plt.close()

    # =========================================================================
    #  CHECK 5: CONVEXITY COMPARISON (Eigenvalues vs Minors)
    # =========================================================================
    def check_convexity_detailed(self):
        # 1. Get Threshold from Config
        threshold = self.config['training']['convexity_threshold']
        threshold = -1.0 * abs(threshold)
        print(f"Running Convexity Analysis (Threshold: {threshold:.1e})...")
        
        # 2. Sample Points (Sobol, Scaled to Surface)
        n_samples = 4096
        points = self._sample_points_on_surface(n_samples)
        
        # 3. Compute Hessians (AutoDiff)
        H_auto = self._get_hessian_autodiff(points)
        
        # 4. Compute Metrics
        # Method A: Eigenvalues
        eigs = np.linalg.eigvalsh(H_auto)
        min_eig = eigs[:, 0]
        
        # Method B: Principal Minors
        min_minor = self._get_principal_minors_numpy(H_auto)
        
        # 5. Plot Comparison
        plt.figure(figsize=(12, 5))
        
        # Plot A: Eigenvalues
        plt.subplot(1, 2, 1)
        plt.hist(min_eig, bins=50, color='teal', alpha=0.7)
        plt.axvline(threshold, color='red', linestyle='--', linewidth=2, label=f'Thresh={threshold:.1e}')
        plt.yscale('log')
        fail_eig = np.sum(min_eig < threshold)
        plt.title(f"Method 1: Min Eigenvalue\nMin: {min_eig.min():.2e} | Fail (<{threshold:.1e}): {fail_eig}")
        plt.xlabel("Value")
        plt.legend(loc='upper left')
        plt.grid(True, alpha=0.3)
        
        # Plot B: Principal Minors
        plt.subplot(1, 2, 2)
        plt.hist(min_minor, bins=50, color='purple', alpha=0.7)
        plt.axvline(threshold, color='red', linestyle='--', linewidth=2, label=f'Thresh={threshold:.1e}')
        plt.yscale('log')
        fail_minor = np.sum(min_minor < threshold)
        plt.title(f"Method 2: Min Principal Minor\nMin: {min_minor.min():.2e} | Fail (<{threshold:.1e}): {fail_minor}")
        plt.xlabel("Value")
        plt.legend(loc='upper left')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plot_dir, "convexity_method_comparison.png"))
        plt.close()

        # --- PART B: BINARY MAP (Using Principal Minors & Threshold) ---
        res = 60 
        T = np.linspace(0, 2*np.pi, res)
        P = np.linspace(0, np.pi/2.0, res)
        TT, PP = np.meshgrid(T, P)
        
        # Generate Grid Points
        U12 = np.cos(PP); R_plane = np.sin(PP)
        U11 = R_plane * np.cos(TT); U22 = R_plane * np.sin(TT)
        
        flat_u11, flat_u22, flat_u12 = U11.flatten(), U22.flatten(), U12.flatten()
        unit_inputs = np.stack([flat_u11, flat_u22, flat_u12], axis=1).astype(np.float32)
        
        # Scale to Surface
        inputs_tf = tf.constant(unit_inputs)
        pred_se = self.model(inputs_tf).numpy().flatten()
        radii = self.config['model']['ref_stress'] / (pred_se + 1e-8)
        surface_points = unit_inputs * radii[:, None]
        
        # Compute Minors Map
        H_grid = self._get_hessian_autodiff(surface_points)
        grid_minors = self._get_principal_minors_numpy(H_grid)
        grid_minors = grid_minors.reshape(TT.shape)
        
        # Apply Configured Threshold
        binary_map = np.where(grid_minors >= threshold, 1.0, 0.0)
        
        plt.figure(figsize=(7, 6))
        cmap = mcolors.ListedColormap(['red', 'green'])
        plt.contourf(TT/np.pi, PP/np.pi, binary_map, levels=[-0.1, 0.5, 1.1], cmap=cmap)
        cbar = plt.colorbar(ticks=[0, 1])
        cbar.ax.set_yticklabels(['Unstable', 'Stable'])
        
        plt.title(f"Binary Stability Map (Minors >= {threshold:.1e})")
        plt.xlabel(r"Theta ($\times \pi$)"); plt.ylabel(r"Phi ($\times \pi$)")
        plt.gca().invert_yaxis()
        
        plt.savefig(os.path.join(self.plot_dir, "convexity_binary_map.png"))
        plt.close()

    # =========================================================================
    #  CHECK 5B: CONVEXITY 1D SLICE (Updated to Principal Minors)
    # =========================================================================
    def check_convexity_1d_slice(self):
        print("Running Convexity 1D Slice...")
        theta = np.linspace(0, 2*np.pi, 360).astype(np.float32)
        
        # 1. Generate Unit Directions
        u11, u22, u12 = np.cos(theta), np.sin(theta), np.zeros_like(theta)
        unit_inputs = np.stack([u11, u22, u12], axis=1)
        
        # 2. Scale to Yield Surface
        inputs_tf = tf.constant(unit_inputs)
        pred_se = self.model(inputs_tf).numpy().flatten()
        
        ref_stress = self.config['model']['ref_stress']
        radii = ref_stress / (pred_se + 1e-8)
        
        # Calculate actual stress points
        points = unit_inputs * radii[:, None]
        
        # 3. Calculate Hessian (AutoDiff)
        H = self._get_hessian_autodiff(points)
        
        # 4. Compute Principal Minors
        min_minor = self._get_principal_minors_numpy(H)
        
        # 5. Plot
        plt.figure(figsize=(10, 5))
        plt.plot(theta/np.pi, min_minor, 'k-', linewidth=1.5)
        
        plt.fill_between(theta/np.pi, min_minor, 0, where=(min_minor < -1e-5), 
                         color='red', alpha=0.5, label='Unstable')
        plt.fill_between(theta/np.pi, min_minor, 0, where=(min_minor >= -1e-5), 
                         color='green', alpha=0.3, label='Stable')
        
        plt.axhline(0, color='k', linestyle='--')
        plt.title("Equator Stability Slice (Principal Minors)")
        plt.xlabel(r"Theta ($\times \pi$)"); plt.ylabel("Min Principal Minor")
        plt.legend(loc='lower right')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.plot_dir, "convexity_slice_1d.png"))
        plt.close()
        
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
        inputs_p, _, r_target_p, geo_p, mask_p, _ = data_phys
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
    # def check_loss_curve(self):
    #     csv_path = os.path.join(self.output_dir, "loss_history.csv")
    #     if not os.path.exists(csv_path):
    #         print("No loss history found.")
    #         return

    #     df = pd.read_csv(csv_path)
        
    #     # We now plot 3 things: Total Loss, Components, and Minimum Eigenvalues
    #     plt.figure(figsize=(15, 5))

    #     # Plot 1: Main Loss (Log Scale)
    #     plt.subplot(1, 3, 1)
    #     plt.plot(df['epoch'], df['train_loss'], label='Total Loss', color='black')
    #     plt.yscale('log')
    #     plt.title("Total Loss")
    #     plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.grid(True, which="both", alpha=0.3)
    #     plt.legend()

    #     # Plot 2: Components (Stress vs R vs Convexity Loss)
    #     plt.subplot(1, 3, 2)
    #     if 'train_l_se' in df.columns: plt.plot(df['epoch'], df['train_l_se'], label='Stress')
    #     if 'train_l_r' in df.columns: plt.plot(df['epoch'], df['train_l_r'], label='R-val')
    #     # Note: 'train_l_conv' is the scalar loss, not the physical min
    #     if 'train_l_conv' in df.columns: plt.plot(df['epoch'], df['train_l_conv'], label='Conv Loss')
    #     plt.yscale('log')
    #     plt.title("Loss Components")
    #     plt.xlabel("Epoch"); plt.legend(); plt.grid(True, which="both", alpha=0.3)

    #     # Plot 3: PHYSICAL EIGENVALUES (The Convexity Proof)
    #     plt.subplot(1, 3, 3)
    #     # We want to see these lines cross Zero and stay there
    #     if 'train_min_stat' in df.columns: 
    #         plt.plot(df['epoch'], df['train_min_stat'], label='Min Eig (Static)', color='red', alpha=0.7)
    #     if 'train_min_dyn' in df.columns: 
    #         plt.plot(df['epoch'], df['train_min_dyn'], label='Min Eig (Dynamic)', color='orange', alpha=0.7)
        
    #     plt.axhline(0, color='green', linestyle='--', linewidth=2, label='Stability Limit')
        
    #     # Use dot notation safely or dict access?
    #     # Since we forced self.config to be a dict in __init__, use dictionary syntax safely
    #     threshold = self.config['training'].get('convexity_threshold', None)
        
    #     if threshold:
    #         plt.axhline(-abs(threshold), color='gray', linestyle=':', label='Threshold')
        
    #     plt.title("Physical Stability (Must be > 0)")
    #     plt.xlabel("Epoch"); plt.ylabel("Min Eigenvalue")
    #     plt.legend(loc='lower right')
    #     plt.grid(True, alpha=0.3)

    #     plt.tight_layout()
    #     plt.savefig(os.path.join(self.plot_dir, "training_history.png"))
    #     plt.close()
    
    # =========================================================================
    #  CHECK 7B: DETAILED LOSS CURVES (SEPARATE PLOTS)  
    def check_loss_history_detailed(self):
        """
        Creates a 2x3 grid showing detailed breakdown of all loss components.
        """
        csv_path = os.path.join(self.output_dir, "loss_history.csv")
        if not os.path.exists(csv_path):
            print("No loss history found.")
            return

        df = pd.read_csv(csv_path)
        
        # Define the 6 metrics we want to track
        metrics = [
            ('train_loss', 'Total Loss', 'black'),
            ('train_l_se', 'Stress Loss (MSE)', 'blue'),
            ('train_l_r', 'R-value Loss', 'green'),
            ('train_l_conv', 'Static Convexity Loss', 'purple'),
            ('train_l_dyn', 'Dynamic Convexity Loss', 'magenta'),
            ('train_gnorm', 'Gradient Norm', 'orange')
        ]

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle(f"Training History: {self.config['experiment_name']}", fontsize=16)
        
        for i, (col, title, color) in enumerate(metrics):
            ax = axes.flat[i]
            if col in df.columns:
                ax.plot(df['epoch'], df[col], label=title, color=color, linewidth=1.5)
                ax.set_title(title)
                ax.set_xlabel("Epoch")
                ax.grid(True, which="both", alpha=0.3)
                
                # Use Log scale for losses, linear for Gnorm if preferred (or log for all)
                # Usually losses are best in Log, Gnorm can vary.
                ax.set_yscale('log')
                ax.legend()
            else:
                ax.text(0.5, 0.5, "Not Tracked", ha='center', va='center')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(os.path.join(self.plot_dir, "loss_history_detailed.png"))
        plt.close()

    # =========================================================================
    #  CHECK 7C: LOSS STABILITY MONITORING
    def check_loss_stability(self):
        """
        Plots ONLY the physical minimum eigenvalues to track stability.
        """
        csv_path = os.path.join(self.output_dir, "loss_history.csv")
        if not os.path.exists(csv_path):
            return

        df = pd.read_csv(csv_path)
        
        plt.figure(figsize=(10, 6))
        
        # Plot Static and Dynamic Min Eigs
        if 'train_min_stat' in df.columns: 
            plt.plot(df['epoch'], df['train_min_stat'], label='Min Eig (Static)', color='red', alpha=0.7)
        if 'train_min_dyn' in df.columns: 
            plt.plot(df['epoch'], df['train_min_dyn'], label='Min Eig (Dynamic)', color='orange', alpha=0.7)
        
        # Zero Line (Goal)
        plt.axhline(0, color='green', linestyle='--', linewidth=2, label='Stability Limit (0.0)')
        
        # Threshold Line (Stop Condition)
        threshold = self.config['training'].get('convexity_threshold', None)
        if threshold:
            target = -abs(threshold)
            plt.axhline(target, color='gray', linestyle=':', label=f'Stop Threshold ({target})')
            
        plt.title("Physical Stability Monitoring (Eigenvalues)")
        plt.xlabel("Epoch")
        plt.ylabel("Minimum Eigenvalue (Must be > 0)")
        plt.legend(loc='lower right')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plot_dir, "loss_history_stability.png"))
        plt.close()

    # =========================================================================
    #  CHECK 8: R-VALUE CALCULATION LOGIC AUDIT
    # =========================================================================
    def check_r_calculation_logic(self):
        print("Running R-value Logic Audit...")
        
        # 1. Define Test Case: Uniaxial 45 deg
        alpha_deg = 45.0
        alpha_rad = np.radians(alpha_deg)
        
        u11 = np.cos(alpha_rad)**2
        u22 = np.sin(alpha_rad)**2
        u12 = np.sin(alpha_rad)*np.cos(alpha_rad)
        unit_input = np.array([[u11, u22, u12]], dtype=np.float32)

        # 2. Scale to NN Yield Surface (STRICT FIX)
        pred_se = self.model(tf.constant(unit_input)).numpy()[0,0]
        ref = self.config['model']['ref_stress']
        scale = ref / (pred_se + 1e-8)
        
        s_input = unit_input * scale

        # 3. Analytical Reference (Hill48 params)
        phys = self.config.get('physics', {})
        F, G, H, N = phys.get('F', 0.5), phys.get('G', 0.5), phys.get('H', 0.5), phys.get('N', 1.5)
        C11, C22, C12, C66 = G+H, F+H, -2*H, 2*N
        
        # Analytical Scaling for Benchmark calculation
        hill_val = C11*u11**2 + C22*u22**2 + C12*u11*u22 + C66*u12**2
        scale_analytical = ref / np.sqrt(hill_val)
        
        # Analytical R calculation
        scale_g = 1.0 / (2.0 * ref)
        g11 = scale_g * (2*C11*(u11*scale_analytical) + C12*(u22*scale_analytical))
        g22 = scale_g * (2*C22*(u22*scale_analytical) + C12*(u11*scale_analytical))
        g12 = scale_g * (2*C66*(u12*scale_analytical))
        
        dt_a = -(g11 + g22)
        dw_a = g11*np.sin(alpha_rad)**2 + g22*np.cos(alpha_rad)**2 - 2*g12*np.sin(alpha_rad)*np.cos(alpha_rad)
        r_analytical = dw_a / dt_a
        
        # 4. Network R-value
        (_, grad_nn, _), _ = self._get_predictions(s_input[:,0], s_input[:,1], s_input[:,2])
        r_nn = self._calc_r_values(grad_nn, np.array([alpha_rad]))[0]
        
        print(f"   [Audit 45deg] Analytical R: {r_analytical:.4f}")
        print(f"   [Audit 45deg] Network R:    {r_nn:.4f}")
        print(f"   [Audit 45deg] Error:        {abs(r_nn - r_analytical):.4f}")
    
    # =========================================================================
    #  CHECK 9: GRADIENT COMPONENT ANALYSIS
    # =========================================================================
    def _plot_gradient_components(self):
        print("Running Gradient Component Analysis...")
        res_theta, res_phi = 60, 30
        theta = np.linspace(0, 2*np.pi, res_theta).astype(np.float32)
        phi = np.linspace(0, np.pi/2.0, res_phi).astype(np.float32)
        TT, PP = np.meshgrid(theta, phi)
        
        # 1. Generate Unit Vectors
        u12 = np.cos(PP)
        r_plane = np.sin(PP)
        u11 = r_plane * np.cos(TT)
        u22 = r_plane * np.sin(TT)
        
        flat_u11, flat_u22, flat_u12 = u11.flatten(), u22.flatten(), u12.flatten()
        unit_inputs = np.stack([flat_u11, flat_u22, flat_u12], axis=1)

        # 2. Scale to Yield Surface (CRITICAL FIX)
        inputs_tf = tf.constant(unit_inputs)
        pred_se = self.model(inputs_tf).numpy().flatten()
        radii = self.config['model']['ref_stress'] / (pred_se + 1e-8)
        
        s11 = flat_u11 * radii
        s22 = flat_u22 * radii
        s12 = flat_u12 * radii
        
        # 3. Get Predictions on Scaled Points
        (_, grad_nn, _), (_, grad_vm) = self._get_predictions(s11, s22, s12)
        
        comps = ['dPhi/ds11', 'dPhi/ds22', 'dPhi/ds12']
        fig, axes = plt.subplots(3, 3, figsize=(14, 12))
        fig.suptitle("Gradient Component Analysis (Model vs Theory)", fontsize=16)
        
        X_plot, Y_plot = TT / np.pi, PP / np.pi
        
        for i in range(3): 
            # Theory
            g_vm = grad_vm[:, i].reshape(TT.shape)
            ax = axes[i, 0]
            cp1 = ax.contourf(X_plot, Y_plot, g_vm, levels=30, cmap='bwr')
            plt.colorbar(cp1, ax=ax); ax.set_title(f"Theory {comps[i]}")
            ax.set_ylabel(r"Phi ($\times \pi$)"); ax.invert_yaxis()
            
            # Model
            g_nn = grad_nn[:, i].reshape(TT.shape)
            ax = axes[i, 1]
            cp2 = ax.contourf(X_plot, Y_plot, g_nn, levels=30, cmap='bwr')
            plt.colorbar(cp2, ax=ax); ax.set_title(f"Model {comps[i]}")
            ax.invert_yaxis()
            
            # Error (Abs Diff)
            err = (g_nn - g_vm)
            ax = axes[i, 2]
            cp3 = ax.contourf(X_plot, Y_plot, err, levels=30, cmap='viridis')
            plt.colorbar(cp3, ax=ax); ax.set_title(f"Difference")
            ax.invert_yaxis()
            
        axes[2, 0].set_xlabel(r"Theta ($\times \pi$)")
        axes[2, 1].set_xlabel(r"Theta ($\times \pi$)")
        axes[2, 2].set_xlabel(r"Theta ($\times \pi$)")
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95]) 
        plt.savefig(os.path.join(self.plot_dir, "gradient_components.png")); plt.close()
    
    # =========================================================================
    #  CHECK 10: BENCHMARK DERIVATIVE AUDIT
    # =========================================================================
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
        self.check_r_calculation_logic()
        self.check_2d_loci_slices()
        self.check_radius_vs_theta()
        self.check_r_values()
        self.check_full_domain_benchmark()
        self.check_convexity_detailed()
        self.check_convexity_1d_slice()
        self.check_global_statistics()
        self.check_loss_history_detailed()
        self.check_loss_stability()
        self._plot_gradient_components()
        # self.check_benchmark_derivatives()
        print(f"Done. Plots in '{self.plot_dir}'")