"""
Physics Validation Module for nnYield.

This module verifies that the learned Yield Surface obeys fundamental physical laws
and material properties. It compares the Neural Network's predictions against
the modular analytical benchmark.
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os
import pandas as pd
from .core import BaseChecker

class PhysicsChecks(BaseChecker):
    """
    Validation of Physical Material Behavior.
    
    This class implements visualizations and audits that ensure the model 
    behaves like a real material, including directional strength and stability.
    """

    # =========================================================================
    #  HELPERS (Hessian & Minors)
    # =========================================================================
    def _get_hessian_autodiff(self, points):
        """
        Computes the exact Hessian matrix (2nd derivatives) for a batch of points.
        This is used to measure the 'Curvature' of the yield surface.
        """
        inputs = tf.convert_to_tensor(points, dtype=tf.float32)
        with tf.GradientTape(persistent=True) as tape2:
            tape2.watch(inputs)
            with tf.GradientTape() as tape1:
                tape1.watch(inputs)
                val = self.model(inputs)
            # First derivative (Normality)
            grads = tape1.gradient(val, inputs)
        
        # Second derivative (Jacobian of the Gradient)
        hess_matrix = tape2.batch_jacobian(grads, inputs)
        del tape2
        return hess_matrix.numpy()

    def _get_principal_minors_numpy(self, hess_matrix):
        """
        Calculates the three leading Principal Minors of the Hessian matrix.
        Used for Sylvester's Criterion: A surface is convex if and only if 
        all leading principal minors are positive.
        """
        # Minor 1: Top-left element
        m1 = hess_matrix[:, 0, 0]
        # Minor 2: Determinant of top-left 2x2 submatrix
        m2 = (hess_matrix[:, 0, 0] * hess_matrix[:, 1, 1]) - \
             (hess_matrix[:, 0, 1] * hess_matrix[:, 1, 0])
        # Minor 3: Determinant of full 3x3 matrix
        m3 = np.linalg.det(hess_matrix)
        
        # Track the 'worst' violation across all minors
        min_minor = np.minimum(np.minimum(m1, m2), m3)
        return min_minor

    # =========================================================================
    #  CHECK 1: 2D YIELD LOCI SLICES
    # =========================================================================
    def check_2d_loci_slices(self):
        """
        Plots 2D cross-sections of the yield surface at increasing shear stress levels.
        
        Purpose:
        - Verifies that the surface remains closed and smooth under shear.
        - Visually compares the NN shape (Solid) against the Benchmark (Dotted).
        """
        print("Running 2D Loci Slices Check...")
        ref = self.config.model.ref_stress
        shear_ratios = [0.0, 0.4, 0.8, 0.95]
        colors = ['purple', 'blue', 'green', 'orange']
        
        theta = np.linspace(0, 2*np.pi, 360).astype(np.float32)
        c, s = np.cos(theta), np.sin(theta)
        
        plt.figure(figsize=(7, 7))
        for i, ratio in enumerate(shear_ratios):
            phi_val = np.arccos(ratio)
            # 1. Benchmark Ground Truth (Generic Factory Model)
            total_rad_bench = self.physics_bench.solve_radius(theta, phi_val)
            r_plane_bench = total_rad_bench * np.sin(phi_val)
            s11_vm, s22_vm = r_plane_bench * c, r_plane_bench * s
            
            # 2. Neural Network Prediction (Numerical Bisection Solver)
            # We solve for the radius 'r' at a fixed shear height.
            r_lo, r_hi = np.zeros_like(theta), np.full_like(theta, ref * 3.0)
            target_s12 = total_rad_bench[0] * ratio
            for _ in range(12):
                r_mid = (r_lo + r_hi) / 2.0
                inputs = np.stack([r_mid*c, r_mid*s, np.full_like(theta, target_s12)], axis=1)
                pred = self.model(tf.constant(inputs)).numpy().flatten()
                mask_high = pred > ref
                r_hi = np.where(mask_high, r_mid, r_hi)
                r_lo = np.where(mask_high, r_lo, r_mid)
            rad_nn = (r_lo + r_hi) / 2.0
            
            # 3. Visualization
            lbl_bench = 'Benchmark' if i == 0 else None
            plt.plot(s11_vm, s22_vm, 'k:', linewidth=1.5, alpha=0.8, label=lbl_bench, zorder=5)
            plt.plot(rad_nn * c, rad_nn * s, color=colors[i], linewidth=2, label=f"Shear={ratio:.2f}", zorder=5)

        limit = ref * 1.5
        plt.xlim(-limit, limit); plt.ylim(-limit, limit); plt.gca().set_aspect('equal')
        plt.xlabel("S11"); plt.ylabel("S22"); plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(self.plot_dir, "yield_loci_slices.png")); plt.close()

    # =========================================================================
    #  CHECK 2: RADIUS VS THETA (Linearized view)
    # =========================================================================
    def check_radius_vs_theta(self):
        """ 
        Linearized yield radius vs direction (Plane Stress case, S12=0). 
        This captures the 'shape accuracy' specifically at the equator.
        
        Output: plots/radius_vs_theta.png
        """
        print("Running Radius vs Theta Check...")
        ref_stress = self.config.model.ref_stress
        theta = np.linspace(0, 2*np.pi, 360)
        s11_in, s22_in, s12_in = np.cos(theta), np.sin(theta), np.zeros_like(theta)
        
        (val_nn, _, _), (val_bench, _) = self._get_predictions(s11_in, s22_in, s12_in)
        r_nn = ref_stress / (val_nn + 1e-8)
        r_bench = ref_stress / (val_bench + 1e-8)
        
        plt.figure(figsize=(8, 5))
        plt.plot(theta/np.pi, r_bench, 'k--', label='Benchmark')
        plt.plot(theta/np.pi, r_nn, 'r-', label='NN Prediction')
        plt.xlabel(r"Theta ($\times \pi$ rad)"); plt.ylabel("Yield Radius (MPa)"); plt.legend(); plt.grid(True, alpha=0.3)
        plt.title("Yield Radius vs Direction (Plane Stress)")
        plt.savefig(os.path.join(self.plot_dir, "radius_vs_theta.png")); plt.close()

    # =========================================================================
    #  CHECK 3: R-VALUES & STRESS TRACE
    # =========================================================================
    def check_r_values(self):
        """ 
        Verifies directional Anisotropy (Lankford coefficients).
        Ensures the surface 'slopes' (normal vectors) are physically correct.
        """
        print("Running R-value & Stress Trace Analysis...")
        angles = np.linspace(0, 90, 91).astype(np.float32)
        ref = self.config.model.ref_stress
        rads = np.radians(angles); sin_a, cos_a = np.sin(rads), np.cos(rads)
        
        u11, u22, u12 = cos_a**2, sin_a**2, sin_a*cos_a
        sigma_bench = ref / (self.physics_bench.equivalent_stress(u11, u22, u12) + 1e-12)
        r_bench = np.array([self.physics_bench.predict_r_value(a) for a in angles])
        
        pred_unit = self.model(tf.constant(np.stack([u11, u22, u12], axis=1))).numpy().flatten()
        sigma_nn = ref / (pred_unit + 1e-8)
        
        (_, grads, _), _ = self._get_predictions(u11*sigma_nn, u22*sigma_nn, u12*sigma_nn)
        grads_n = grads / (np.linalg.norm(grads, axis=1, keepdims=True) + 1e-8)
        ds_11, ds_22, ds_12 = grads_n[:,0], grads_n[:,1], grads_n[:,2]
        r_nn = (ds_11*sin_a**2 + ds_22*cos_a**2 - ds_12*sin_a*cos_a) / (-(ds_11 + ds_22) + 1e-8)

        # Print detailed comparisons for key directions.
        for ang in [0, 45, 90]:
            idx = np.argmin(np.abs(angles - ang))
            print(f"   [R-value] Angle {ang:>2} deg: Pred={r_nn[idx]:.4f}, Bench={r_bench[idx]:.4f}")

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10), sharex=True)
        ax1.plot(angles, r_bench, 'k--', label='Benchmark'); ax1.plot(angles, r_nn, 'r-', label='NN'); ax1.set_ylabel("R-value"); ax1.legend()
        ax2.plot(angles, sigma_bench, 'k--', label='Benchmark'); ax2.plot(angles, sigma_nn, 'b-', label='NN'); ax2.set_ylabel("Yield Stress"); ax2.legend()
        plt.tight_layout(); plt.savefig(os.path.join(self.plot_dir, "r_values.png")); plt.close()

    # =========================================================================
    #  CHECK 4: FULL DOMAIN HEATMAPS (SE Error & Convexity)
    # =========================================================================
    def check_full_domain_benchmark(self):
        """ 
        Maps error and stability distributions over the entire 3D stress space.
        
        Outputs:
        - plots/se_error_map.png
        - plots/convexity_map.png
        - plots/grad_angle_map.png
        """
        print("Running Full Domain Benchmark Heatmaps...")
        res_t, res_p = 100, 50
        T, P = np.meshgrid(np.linspace(0, 2*np.pi, res_t), np.linspace(0, np.pi/2.0, res_p))
        u12, r_p = np.cos(P), np.sin(P)
        u11, u22 = r_p * np.cos(T), r_p * np.sin(T)
        flat_u = np.stack([u11.flatten(), u22.flatten(), u12.flatten()], axis=1).astype(np.float32)
        
        # Scale unit directions to the learned boundary
        pred_se = self.model(tf.constant(flat_u)).numpy().flatten()
        s_surf = flat_u * (self.config.model.ref_stress / (pred_se + 1e-8))[:, None]
        
        # Get dual predictions (NN vs Bench)
        (val_nn, grad_nn, hess_nn), (val_bench, grad_bench) = self._get_predictions(s_surf[:,0], s_surf[:,1], s_surf[:,2])
        
        # 1. Stress Magnitude Error (SE Rel Error)
        err_se = (np.abs(val_nn - val_bench) / (val_bench + 1e-8)).reshape(T.shape)
        
        # 2. Curvature Quality (Hessian Min Eigenvalue)
        min_eigs = np.linalg.eigvalsh(hess_nn)[:, 0].reshape(T.shape)
        
        # 3. Surface Normality Deviation (Degrees)
        norm_nn, norm_bench = np.linalg.norm(grad_nn, axis=1), np.linalg.norm(grad_bench, axis=1)
        cosine = np.clip(np.sum(grad_nn*grad_bench, axis=1)/(norm_nn*norm_bench+1e-8), -1, 1)
        err_angle = np.degrees(np.arccos(cosine)).reshape(T.shape)

        metrics = [
            (err_se, "Equivalent Stress Rel. Error", "se_error_map", 'viridis'), 
            (err_angle, "Grad Angle Deviation (Deg)", "grad_angle_map", 'magma'), 
            (min_eigs, "Surface Convexity (Min Eig)", "convexity_map", 'RdBu')
        ]
        
        for data, title, fname, cmap in metrics:
            plt.figure(figsize=(8, 6))
            # Center RdBu colormap at 0 to clearly show stable vs unstable regions.
            norm = mcolors.TwoSlopeNorm(vmin=min(data.min(), -1e-4), vcenter=0., vmax=max(data.max(), 1e-4)) if cmap == 'RdBu' else None
            cp = plt.contourf(T/np.pi, P/np.pi, data, levels=50, cmap=cmap, norm=norm)
            plt.colorbar(cp); plt.title(title); plt.xlabel(r"Theta ($\times \pi$)"); plt.ylabel(r"Phi ($\times \pi$)")
            plt.gca().invert_yaxis(); plt.tight_layout()
            plt.savefig(os.path.join(self.plot_dir, f"{fname}.png")); plt.close()

    # =========================================================================
    #  CHECK 5: CONVEXITY DETAILED (Stability Map & 1D Slice)
    # =========================================================================
    def check_convexity_detailed(self):
        """ Deep-dive analysis of curvature violations across the domain. """
        threshold = -abs(self.config.training.stopping_criteria.convexity_threshold)
        print(f"Running Convexity Detailed Analysis...")
        
        # 1. Comparison Histogram (Eigenvalues vs Minors)
        pts = self._sample_points_on_surface(4096)
        H = self._get_hessian_autodiff(pts)
        min_eig, min_minor = np.linalg.eigvalsh(H)[:, 0], self._get_principal_minors_numpy(H)
        
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1); plt.hist(min_eig, bins=50, color='teal'); plt.axvline(threshold, color='r', ls='--'); plt.yscale('log'); plt.title("Min Eigenvalue Distribution")
        plt.subplot(1, 2, 2); plt.hist(min_minor, bins=50, color='purple'); plt.axvline(threshold, color='r', ls='--'); plt.yscale('log'); plt.title("Min Principal Minor Distribution")
        plt.tight_layout(); plt.savefig(os.path.join(self.plot_dir, "convexity_method_comparison.png")); plt.close()

        # 2. Binary Stability Map (Full 3D Domain)
        res = 60 
        T, P = np.meshgrid(np.linspace(0, 2*np.pi, res), np.linspace(0, np.pi/2.0, res))
        u12, r_p = np.cos(P), np.sin(P); u11, u22 = r_p * np.cos(T), r_p * np.sin(T)
        flat_u = np.stack([u11.flatten(), u22.flatten(), u12.flatten()], axis=1).astype(np.float32)
        pred = self.model(tf.constant(flat_u)).numpy().flatten()
        points = flat_u * (self.config.model.ref_stress / (pred + 1e-8))[:, None]
        H_grid = self._get_hessian_autodiff(points)
        minors = self._get_principal_minors_numpy(H_grid).reshape(T.shape)
        
        plt.figure(figsize=(7, 6))
        plt.contourf(T/np.pi, P/np.pi, np.where(minors >= threshold, 1.0, 0.0), levels=[-0.1, 0.5, 1.1], cmap=mcolors.ListedColormap(['red', 'green']))
        cbar = plt.colorbar(ticks=[0, 1]); cbar.ax.set_yticklabels(['Unstable', 'Stable'])
        plt.title("Binary Stability Map (Green=Stable)"); plt.gca().invert_yaxis()
        plt.savefig(os.path.join(self.plot_dir, "convexity_binary_map.png")); plt.close()

    def check_convexity_slice_1d(self):
        """ 
        Detailed stability analysis specifically along the Equator (S12=0). 
        Provides high-precision angular resolution of unstable regions.
        
        Output: plots/convexity_slice_1d.png
        """
        print("Running Convexity 1D Slice Analysis...")
        theta = np.linspace(0, 2*np.pi, 360).astype(np.float32)
        unit = np.stack([np.cos(theta), np.sin(theta), np.zeros_like(theta)], axis=1)
        
        # Scale to the learned yield surface
        pred = self.model(tf.constant(unit)).numpy().flatten()
        pts = unit * (self.config.model.ref_stress / (pred + 1e-8))[:, None]
        
        # Calculate curvatures
        H = self._get_hessian_autodiff(pts)
        min_minor = self._get_principal_minors_numpy(H)
        
        plt.figure(figsize=(10, 5)); plt.plot(theta/np.pi, min_minor, 'k-', linewidth=1.5)
        # Highlight regions using color fill
        plt.fill_between(theta/np.pi, min_minor, 0, where=(min_minor < -1e-5), color='red', alpha=0.5, label='Unstable')
        plt.fill_between(theta/np.pi, min_minor, 0, where=(min_minor >= -1e-5), color='green', alpha=0.3, label='Stable')
        plt.axhline(0, color='k', ls='--'); plt.title("Equator Stability Slice (Principal Minors)")
        plt.xlabel(r"Theta ($\times \pi$)"); plt.ylabel("Curvature Metric"); plt.legend()
        plt.savefig(os.path.join(self.plot_dir, "convexity_slice_1d.png")); plt.close()

    def check_benchmark_derivatives(self):
        """ Verifies mathematical consistency of the selected analytical model. """
        print("Running Benchmark Derivative Audit...")
        s = np.random.uniform(-2, 2, (10, 3)).astype(np.float64)
        _, (_, grad_analytical) = self._get_predictions(s[:,0], s[:,1], s[:,2])
        
        eps = 1e-4
        grad_fd = np.zeros_like(grad_analytical)
        for i in range(len(s)):
            for j in range(3):
                s_p, s_m = s[i].copy(), s[i].copy()
                s_p[j] += eps; s_m[j] -= eps
                v_p = self.physics_bench.equivalent_stress(s_p[0], s_p[1], s_p[2])
                v_m = self.physics_bench.equivalent_stress(s_m[0], s_m[1], s_m[2])
                grad_fd[i, j] = (v_p - v_m) / (2*eps)
        
        err = np.max(np.abs(grad_analytical - grad_fd))
        print(f"   Max discrepancy (Analytical vs FD): {err:.2e}")
        if err < 1e-4: print("   ✅ Benchmark logic is consistent.")
        else: print("   ❌ Warning: Benchmark math inconsistency detected.")
