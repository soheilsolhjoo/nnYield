import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os
import pandas as pd

class PhysicsChecks:
    """
    Physics Validation Module.
    
    This module verifies that the learned Yield Surface obeys fundamental physical laws
    and material properties. It compares the Neural Network's predictions against
    the analytical Hill48 benchmark.
    
    Key Checks:
    1.  **Yield Loci (2D Slices)**: Visualizes the shape of the yield surface at different 
        shear stress levels. Ensures the shape looks "ellipse-like" and smooth.
    2.  **R-values (Anisotropy)**: Verifies the model captures the material's directional 
        deformation behavior (Lankford coefficients).
    3.  **Convexity**: A critical thermodynamic requirement. The yield surface must be 
        convex everywhere. We verify this using Hessian eigenvalues and principal minors.
    4.  **Full Domain Error**: Maps the error distribution across the entire stress space 
        (Theta-Phi) to find localized "problem zones".
    """

    # =========================================================================
    #  HELPERS (Hessian & Minors)
    # =========================================================================
    def _get_hessian_autodiff(self, points):
        """
        Computes the exact Hessian matrix (2nd derivatives) for a batch of points 
        using TensorFlow's Automatic Differentiation.
        
        Args:
            points (np.array): (N, 3) stress points [S11, S22, S12].
            
        Returns:
            np.array: (N, 3, 3) Hessian matrices.
        """
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

    def _get_principal_minors_numpy(self, hess_matrix):
        """
        Calculates the three leading Principal Minors of the Hessian matrix.
        According to Sylvester's Criterion, a matrix is positive-definite (convex)
        if and only if all leading principal minors are positive.
        
        Args:
            hess_matrix (np.array): (N, 3, 3) batch of matrices.
            
        Returns:
            np.array: (N,) The minimum of the three minors for each point.
                      If this value is < 0, the point violates convexity.
        """
        # 1st Principal Minor (Top-left element)
        m1 = hess_matrix[:, 0, 0]
        
        # 2nd Principal Minor (Determinant of top-left 2x2 submatrix)
        # | H11 H12 |
        # | H21 H22 |  -> Det = H11*H22 - H12*H21
        m2 = (hess_matrix[:, 0, 0] * hess_matrix[:, 1, 1]) - \
             (hess_matrix[:, 0, 1] * hess_matrix[:, 1, 0])
        
        # 3rd Principal Minor (Determinant of full 3x3 matrix)
        m3 = np.linalg.det(hess_matrix)
        
        # We track the 'worst' minor. If min(m1, m2, m3) < 0, convexity is broken.
        min_minor = np.minimum(np.minimum(m1, m2), m3)
        return min_minor

    # =========================================================================
    #  CHECK 1: 2D YIELD LOCI SLICES
    # =========================================================================
    def check_2d_loci_slices(self):
        """
        Plots 2D cross-sections of the yield surface at increasing shear stress levels.
        
        Physics:
        - At Shear=0 (S12=0), we see the standard Plane Stress ellipse.
        - As Shear increases, the elastic domain should shrink (Von Mises / Hill criterion).
        - At Max Shear, the domain should collapse to a point or small region.
        
        Output: plots/yield_loci_slices.png
        """
        print("Running 2D Loci Slices Check...")
        ref_stress = self.config['model']['ref_stress']
        phys = self.config.get('physics', {})
        N = phys.get('N', 1.5)
        
        # Calculate the theoretical max shear stress for Hill48
        max_shear = ref_stress / np.sqrt(2*N)
        
        shear_ratios = [0.0, 0.4, 0.8, 0.95]
        colors = plt.cm.viridis(np.linspace(0, 1, len(shear_ratios)))
        theta = np.linspace(0, 2*np.pi, 360)
        
        plt.figure(figsize=(7, 7))
        # Dummy plot for legend ordering
        plt.plot([], [], 'k:', linewidth=1.5, label='Benchmark')

        max_val_plotted = 0.0

        for i, ratio in enumerate(shear_ratios):
            current_s12_val = ratio * max_shear
            
            # Generate input circle in S11-S22 plane
            s11_in = np.cos(theta)
            s22_in = np.sin(theta)
            s12_in = np.full_like(theta, current_s12_val)
            
            # Get predictions
            (val_nn, _, _), (val_vm, _) = self._get_predictions(s11_in, s22_in, s12_in)
            
            # Convert Model Output (Equivalent Stress) to Yield Radius
            # Radius = Ref_Stress / Pred_Equiv_Stress
            rad_nn = ref_stress / (val_nn + 1e-8)
            rad_vm = ref_stress / (val_vm + 1e-8)
            
            # Track plotting limits
            current_max = max(rad_nn.max(), rad_vm.max())
            if current_max > max_val_plotted: max_val_plotted = current_max
            
            # Plot NN (Solid Color) vs Benchmark (Dotted Black)
            plt.plot(rad_nn*s11_in, rad_nn*s22_in, color=colors[i], linewidth=2, label=f"Shear={ratio:.2f}")
            plt.plot(rad_vm*s11_in, rad_vm*s22_in, color='k', linestyle=':', linewidth=1.5, alpha=0.6)

        # Mark the theoretical max shear point
        plt.scatter([0], [0], color='red', marker='x', s=100, label=f'Max Shear ({max_shear:.2f})', zorder=10)
        
        limit = max_val_plotted * 1.1
        plt.xlim(-limit, limit); plt.ylim(-limit, limit)
        plt.axis('equal'); plt.xlabel("S11"); plt.ylabel("S22")
        plt.grid(True, alpha=0.3)
        plt.title(f"Yield Loci Slices (Ref={ref_stress})")
        plt.legend(loc='best', fontsize='small', framealpha=0.9)
        plt.tight_layout()
        plt.savefig(os.path.join(self.plot_dir, "yield_loci_slices.png")); plt.close()

    # =========================================================================
    #  CHECK 2: RADIUS VS THETA
    # =========================================================================
    def check_radius_vs_theta(self):
        """
        Plots the yield radius as a function of angle for the Plane Stress case (S12=0).
        This "unrolls" the 2D locus into a linear plot, making it easier to see 
        small deviations in shape.
        
        Output: plots/radius_vs_theta.png
        """
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
        plt.legend(); plt.grid(True, alpha=0.3)
        
        dmin, dmax = min(r_nn.min(), r_vm.min()), max(r_nn.max(), r_vm.max())
        margin = (dmax - dmin) * 0.2 if dmax != dmin else 0.1
        plt.ylim(dmin - margin, dmax + margin)
        plt.savefig(os.path.join(self.plot_dir, "radius_vs_theta.png")); plt.close()

    # =========================================================================
    #  CHECK 3: R-VALUES & STRESS TRACE
    # =========================================================================
    def check_r_values(self):
        """
        Verifies the model's Anisotropy predictions.
        
        1. **R-value (Lankford Coeff)**: Ratio of width strain to thickness strain during 
           uniaxial tension. Calculated via the gradients (normal vector) of the yield surface.
           R = d_eps_width / d_eps_thick
           
        2. **Yield Stress**: The stress at which yielding occurs for a given angle.
        
        Output: plots/r_values.png
        """
        print("Running R-value & Stress Trace Analysis...")
        n_steps = 91
        angles = np.linspace(0, 90, n_steps).astype(np.float32)
        rads = np.radians(angles)
        
        # 1. Define Uniaxial Stress Directions
        # S11 = cos^2(a), S22 = sin^2(a), S12 = sin(a)cos(a)
        sin_a, cos_a = np.sin(rads), np.cos(rads)
        u11 = cos_a**2; u22 = sin_a**2; u12 = sin_a * cos_a
        inputs_unit = np.stack([u11, u22, u12], axis=1).astype(np.float32)

        phys = self.config['physics']
        F, G, H, N = phys['F'], phys['G'], phys['H'], phys['N']
        ref_stress = self.config['model']['ref_stress']

        # --- A. BENCHMARK CALCULATIONS (Analytical) ---
        term = F*u22**2 + G*u11**2 + H*(u11-u22)**2 + 2*N*u12**2
        sigma_bench = ref_stress / np.sqrt(term + 1e-8)
        
        # Analytical Gradients at the yield point
        s11_b = u11 * sigma_bench; s22_b = u22 * sigma_bench; s12_b = u12 * sigma_bench
        denom = ref_stress
        dg11 = (G*s11_b + H*(s11_b-s22_b)) / denom
        dg22 = (F*s22_b - H*(s11_b-s22_b)) / denom
        dg12 = (2*N*s12_b) / denom
        
        # R-value formula using gradients
        d_eps_t = -(dg11 + dg22)
        d_eps_w = dg11*sin_a**2 + dg22*cos_a**2 - 2*dg12*sin_a*cos_a
        r_bench = (dg11*sin_a**2 + dg22*cos_a**2 - 2*dg12*sin_a*cos_a) / (-(dg11 + dg22) + 1e-8)

        # --- B. NN CALCULATIONS ---
        # 1. Find Yield Stress (Scale unit vector until Model(x) = Ref)
        inputs_tf = tf.convert_to_tensor(inputs_unit, dtype=tf.float32)
        pred_unit = self.model(inputs_tf).numpy().flatten()
        sigma_nn = ref_stress / (pred_unit + 1e-8)
        
        # 2. Find Gradients at that Yield Point
        inputs_nn_surf = inputs_unit * sigma_nn[:, None]
        inputs_surf_tf = tf.convert_to_tensor(inputs_nn_surf, dtype=tf.float32)

        with tf.GradientTape() as tape:
            tape.watch(inputs_surf_tf)
            pred_val = self.model(inputs_surf_tf)
        grads = tape.gradient(pred_val, inputs_surf_tf).numpy()
        
        # Normalize to get Normal Vector
        grads_n = grads / (np.linalg.norm(grads, axis=1, keepdims=True) + 1e-8)
        ds_11, ds_22, ds_12 = grads_n[:,0], grads_n[:,1], grads_n[:,2]
        
        # Calculate R-value
        r_nn = (ds_11*sin_a**2 + ds_22*cos_a**2 - 2*ds_12*sin_a*cos_a) / (-(ds_11 + ds_22) + 1e-8)

        # --- PLOTTING ---
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10), sharex=True)
        ax1.plot(angles, r_bench, 'k--', linewidth=2, label='Benchmark (Hill48)')
        ax1.plot(angles, r_nn, 'r-', linewidth=2, alpha=0.8, label='NN Prediction')
        ax1.set_ylabel("R-value"); ax1.set_title("Anisotropy (R-value) vs. Angle")
        ax1.grid(True, alpha=0.3); ax1.legend()
        
        ax2.plot(angles, sigma_bench, 'k--', linewidth=2, label='Benchmark')
        ax2.plot(angles, sigma_nn, 'b-', linewidth=2, alpha=0.8, label='NN Prediction')
        ax2.set_ylabel("Yield Stress (MPa)"); ax2.set_xlabel("Angle (deg)")
        ax2.set_title("Directional Yield Strength"); ax2.grid(True, alpha=0.3); ax2.legend()
        plt.tight_layout(); plt.savefig(os.path.join(self.plot_dir, "r_values.png")); plt.close()

    # =========================================================================
    #  CHECK 4: FULL DOMAIN HEATMAPS
    # =========================================================================
    def check_full_domain_benchmark(self):
        """
        Generates heatmaps of error metrics over the entire Theta-Phi stress space.
        This helps identify if the model fails in specific regions (e.g., high shear).
        
        Maps generated:
        1. SE Rel Error: Relative error in yield stress.
        2. Grad Angle Dev: Deviation between predicted and analytical normal vectors (in degrees).
        3. Convexity (Min Eig): Map of the minimum Hessian eigenvalue (should be > 0).
        """
        print("Running Full Domain Benchmark...")
        res_theta, res_phi = 100, 50
        theta = np.linspace(0, 2*np.pi, res_theta).astype(np.float32)
        phi = np.linspace(0, np.pi/2.0, res_phi).astype(np.float32)
        TT, PP = np.meshgrid(theta, phi)
        
        # Spherical -> Cartesian Unit Vectors
        u12 = np.cos(PP); r_plane = np.sin(PP)
        u11 = r_plane * np.cos(TT); u22 = r_plane * np.sin(TT)
        flat_u = np.stack([u11.flatten(), u22.flatten(), u12.flatten()], axis=1)
        
        # Scale to Yield Surface
        pred_se = self.model(tf.constant(flat_u)).numpy().flatten()
        radii = self.config['model']['ref_stress'] / (pred_se + 1e-8)
        s_surf = flat_u * radii[:, None]
        
        # Compute NN and Benchmark
        (val_nn, grad_nn, hess_nn), (val_vm, grad_vm) = self._get_predictions(s_surf[:,0], s_surf[:,1], s_surf[:,2])
        
        # Calculate Metrics
        # 1. Stress Error
        err_se = np.abs(val_nn - val_vm) / (val_vm + 1e-8)
        
        # 2. Gradient Angle Error (Dot Product)
        norm_nn = np.linalg.norm(grad_nn, axis=1)
        norm_vm = np.linalg.norm(grad_vm, axis=1)
        cosine = np.clip(np.sum(grad_nn*grad_vm, axis=1)/(norm_nn*norm_vm+1e-8), -1, 1)
        err_angle_rad = np.arccos(cosine)
        
        # 3. Convexity (Eigenvalues)
        eigs = np.linalg.eigvalsh(hess_nn); min_eigs = eigs[:, 0]

        # Reshape for contour plots
        err_se = np.nan_to_num(err_se).reshape(TT.shape)
        err_angle_rad = np.nan_to_num(err_angle_rad).reshape(TT.shape)
        min_eigs = np.nan_to_num(min_eigs).reshape(TT.shape)

        # Save Raw Data
        pd.DataFrame({'theta_rad': TT.flatten(), 'phi_rad': PP.flatten(), 'min_eig': min_eigs.flatten()}).to_csv(
            os.path.join(self.csv_dir, "full_domain_metrics.csv"), index=False)

        # Generate Plots
        metrics = [
            (err_se, "SE Rel. Error", "se_error_map", 'viridis', "Rel Error"),
            (err_angle_rad, "Grad Angle Deviation", "grad_angle_map", 'magma', "Dev (Rad)"),
            (min_eigs, "Convexity (Min Eig)", "convexity_map", 'RdBu', "Min Eig")
        ]
        
        X_plot, Y_plot = TT / np.pi, PP / np.pi
        for data, title, fname, cmap, cbar_label in metrics:
            plt.figure(figsize=(8, 6))
            norm = None
            # Center the colormap at 0 for Convexity to clearly show +/- regions
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
    #  CHECK 5: CONVEXITY DETAILED (Eigenvalues vs Minors)
    # =========================================================================
    def check_convexity_detailed(self):
        """
        Performs a deep-dive statistical analysis of convexity.
        
        Plots:
        1. **Method Comparison Histogram**: Compares distribution of Minimum Eigenvalues 
           vs Minimum Principal Minors. Both should be positive.
        2. **Binary Stability Map**: A black/white map of the full domain showing exactly 
           where convexity is violated (0=Unstable, 1=Stable).
        """
        threshold = self.config['training']['convexity_threshold']
        threshold = -1.0 * abs(threshold) # Ensure it's negative
        print(f"Running Convexity Analysis (Threshold: {threshold:.1e})...")
        
        # 1. Sample Points
        n_samples = 4096
        points = self._sample_points_on_surface(n_samples)
        
        # 2. Compute Hessians
        H_auto = self._get_hessian_autodiff(points)
        
        # 3. Compute Metrics
        min_eig = np.linalg.eigvalsh(H_auto)[:, 0]
        min_minor = self._get_principal_minors_numpy(H_auto)
        
        # 4. Plot Comparison Histograms
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.hist(min_eig, bins=50, color='teal', alpha=0.7)
        plt.axvline(threshold, color='red', linestyle='--', label='Threshold')
        plt.yscale('log'); plt.title("Method 1: Min Eigenvalue"); plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.hist(min_minor, bins=50, color='purple', alpha=0.7)
        plt.axvline(threshold, color='red', linestyle='--', label='Threshold')
        plt.yscale('log'); plt.title("Method 2: Min Principal Minor"); plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plot_dir, "convexity_method_comparison.png")); plt.close()

        # 5. Generate Binary Stability Map
        res = 60 
        T = np.linspace(0, 2*np.pi, res); P = np.linspace(0, np.pi/2.0, res)
        TT, PP = np.meshgrid(T, P)
        
        U12 = np.cos(PP); R_plane = np.sin(PP)
        U11 = R_plane * np.cos(TT); U22 = R_plane * np.sin(TT)
        flat_u = np.stack([U11.flatten(), U22.flatten(), U12.flatten()], axis=1).astype(np.float32)
        
        inputs_tf = tf.constant(flat_u)
        pred_se = self.model(inputs_tf).numpy().flatten()
        surface_points = flat_u * (self.config['model']['ref_stress'] / (pred_se + 1e-8))[:, None]
        
        H_grid = self._get_hessian_autodiff(surface_points)
        grid_minors = self._get_principal_minors_numpy(H_grid).reshape(TT.shape)
        
        # Binary Classification: 1 if Stable, 0 if Unstable
        binary_map = np.where(grid_minors >= threshold, 1.0, 0.0)
        
        plt.figure(figsize=(7, 6))
        cmap = mcolors.ListedColormap(['red', 'green'])
        plt.contourf(TT/np.pi, PP/np.pi, binary_map, levels=[-0.1, 0.5, 1.1], cmap=cmap)
        cbar = plt.colorbar(ticks=[0, 1]); cbar.ax.set_yticklabels(['Unstable', 'Stable'])
        plt.title(f"Binary Stability Map (Minors >= {threshold:.1e})")
        plt.xlabel(r"Theta ($\times \pi$)"); plt.ylabel(r"Phi ($\times \pi$)")
        plt.gca().invert_yaxis()
        plt.savefig(os.path.join(self.plot_dir, "convexity_binary_map.png")); plt.close()

    # =========================================================================
    #  CHECK 5B: CONVEXITY 1D SLICE
    # =========================================================================
    def check_convexity_slice_1d(self):
        """
        Plots the convexity metric (Principal Minor) along the Equator (Plane Stress).
        
        Why this is useful:
        - The equator is the most critical region for sheet metal forming.
        - This simple 1D plot lets us pinpoint exactly WHICH angles are unstable.
        
        Output: plots/convexity_slice_1d.png
        """
        print("Running Convexity 1D Slice...")
        theta = np.linspace(0, 2*np.pi, 360).astype(np.float32)
        unit_inputs = np.stack([np.cos(theta), np.sin(theta), np.zeros_like(theta)], axis=1)
        
        inputs_tf = tf.constant(unit_inputs)
        pred_se = self.model(inputs_tf).numpy().flatten()
        points = unit_inputs * (self.config['model']['ref_stress'] / (pred_se + 1e-8))[:, None]
        
        H = self._get_hessian_autodiff(points)
        min_minor = self._get_principal_minors_numpy(H)
        
        plt.figure(figsize=(10, 5))
        plt.plot(theta/np.pi, min_minor, 'k-', linewidth=1.5, label='Min Principal Minor')
        
        # Color-code regions
        plt.fill_between(theta/np.pi, min_minor, 0, where=(min_minor < -1e-5), 
                         color='red', alpha=0.5, label='Unstable')
        plt.fill_between(theta/np.pi, min_minor, 0, where=(min_minor >= -1e-5), 
                         color='green', alpha=0.3, label='Stable')
        
        plt.axhline(0, color='k', linestyle='--')
        plt.title("Equator Stability Slice (Principal Minors)")
        plt.xlabel(r"Theta ($\times \pi$)"); plt.ylabel("Min Principal Minor")
        plt.legend(loc='lower right'); plt.grid(True, alpha=0.3); plt.tight_layout()
        plt.savefig(os.path.join(self.plot_dir, "convexity_slice_1d.png")); plt.close()