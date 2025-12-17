import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os

class ReportingChecks:
    """
    Reporting Validation Module.
    
    This module is responsible for quantifying the overall performance of the trained model.
    It moves beyond visual checks (like in PhysicsChecks) to provide concrete numbers 
    (MAE, MSE) and statistical distributions of errors.
    
    Key Functions:
    1.  **Dense Sampling**: Generates a large dataset of synthetic uniaxial stress points 
        (2000 angles) to ensure our statistics are robust, not just based on the 
        limited experimental data points.
    2.  **Global Metrics**: Calculates Mean Absolute Error (MAE) for both Yield Stress 
        and R-values across this dense dataset.
    3.  **Error Distributions**: Visualizes the spread of errors via histograms to check 
        for bias (non-zero mean error) or outliers.
    """

    def check_global_statistics(self):
        """
        Calculates aggregate error metrics (MAE, MSE) over a dense set of synthetic 
        uniaxial stress states and generates summary plots.
        
        Outputs:
            plots/stats_histogram_error.png
            plots/global_stats.txt
        """
        print("Running Global Statistics (Dense Sampling)...")
        
        # =========================================================
        # 1. GENERATE DENSE UNIAXIAL DATA (Synthetic)
        # =========================================================
        # We generate 2000 random angles to densely cover the R-value domain (0 to 90 deg)
        n_dense = 2000
        angles = np.random.uniform(0, 90, n_dense).astype(np.float32)
        rads = np.radians(angles)
        
        # Calculate Unit Vectors for Uniaxial Tension at these angles
        # s11 = cos^2(a), s22 = sin^2(a), s12 = sin(a)cos(a)
        sin_a, cos_a = np.sin(rads), np.cos(rads)
        u11 = cos_a**2
        u22 = sin_a**2
        u12 = sin_a * cos_a
        
        # Stack into input batch (N, 3)
        inputs_dense = np.stack([u11, u22, u12], axis=1).astype(np.float32)
        
        # =========================================================
        # 2. BENCHMARK CALCULATIONS (Hill48)
        # =========================================================
        # Load Physics Parameters
        phys = self.config.get('physics', {})
        F, G, H, N = phys.get('F', 0.5), phys.get('G', 0.5), phys.get('H', 0.5), phys.get('N', 1.5)
        ref_stress = self.config['model']['ref_stress']
        
        # A. Benchmark Yield Stress
        # Hill48 Criterion: F*s22^2 + G*s11^2 + H(s11-s22)^2 + 2N*s12^2 = (Sigma_ref / Sigma)^2
        term = F*u22**2 + G*u11**2 + H*(u11-u22)**2 + 2*N*u12**2
        sigma_bench = ref_stress / np.sqrt(term + 1e-8)
        
        # B. Benchmark R-values
        # 1. Calculate stress components at the yield surface
        s11_b, s22_b, s12_b = u11 * sigma_bench, u22 * sigma_bench, u12 * sigma_bench
        
        # 2. Analytical Gradients (Normal Vector)
        denom = ref_stress
        dg11 = (G*s11_b + H*(s11_b-s22_b)) / denom
        dg22 = (F*s22_b - H*(s11_b-s22_b)) / denom
        dg12 = (2*N*s12_b) / denom
        
        # 3. Calculate Strains & R-value
        d_eps_t = -(dg11 + dg22) # Thickness strain
        d_eps_w = dg11*sin_a**2 + dg22*cos_a**2 - 2*dg12*sin_a*cos_a # Width strain
        
        # Avoid division by zero for R-value
        r_bench = np.divide(d_eps_w, d_eps_t, out=np.zeros_like(d_eps_w), where=np.abs(d_eps_t)>1e-8)

        # =========================================================
        # 3. NEURAL NETWORK CALCULATIONS
        # =========================================================
        inputs_tf = tf.constant(inputs_dense)
        
        # A. NN Yield Stress
        # Model predicts 'Equivalent Stress Ratio'. Yield Stress = Ref / Prediction
        pred_unit = self.model(inputs_tf).numpy().flatten()
        sigma_nn = ref_stress / (pred_unit + 1e-8)
        
        # B. NN R-values
        # 1. Project unit vectors to the predicted yield surface
        inputs_nn_surf = inputs_dense * sigma_nn[:, None]
        inputs_surf_tf = tf.constant(inputs_nn_surf)
        
        # 2. Compute Gradients via AutoDiff
        with tf.GradientTape() as tape:
            tape.watch(inputs_surf_tf)
            pred_val = self.model(inputs_surf_tf)
        
        grads = tape.gradient(pred_val, inputs_surf_tf).numpy()
        
        # 3. Normalize Gradients
        gnorms = np.linalg.norm(grads, axis=1, keepdims=True) + 1e-8
        grads_n = grads / gnorms
        ds_11, ds_22, ds_12 = grads_n[:,0], grads_n[:,1], grads_n[:,2]
        
        # 4. Calculate R-value from Gradients
        d_eps_t_nn = -(ds_11 + ds_22)
        d_eps_w_nn = ds_11*sin_a**2 + ds_22*cos_a**2 - 2*ds_12*sin_a*cos_a
        r_nn = np.divide(d_eps_w_nn, d_eps_t_nn, out=np.zeros_like(d_eps_w_nn), where=np.abs(d_eps_t_nn)>1e-8)

        # =========================================================
        # 4. METRICS & REPORTS
        # =========================================================
        mae_stress = np.mean(np.abs(sigma_nn - sigma_bench))
        mae_r = np.mean(np.abs(r_nn - r_bench))
        
        print(f"   -> Stress MAE (Uniaxial): {mae_stress:.2e}")
        print(f"   -> R-value MAE (Uniaxial): {mae_r:.2e}")

        # Save metrics to text file
        with open(os.path.join(self.plot_dir, "global_stats.txt"), "w") as f:
            f.write("Global Statistics Report (Dense Uniaxial)\n")
            f.write("=======================================\n")
            f.write(f"Stress MAE: {mae_stress:.6e}\n")
            f.write(f"R-value MAE: {mae_r:.6e}\n")

        # =========================================================
        #  PLOT: STATS HISTOGRAM ERROR
        # =========================================================
        plt.figure(figsize=(12, 5))
        
        # Subplot 1: Stress Error Distribution
        plt.subplot(1, 2, 1)
        error_s = sigma_nn - sigma_bench
        plt.hist(error_s, bins=50, color='blue', alpha=0.7)
        plt.axvline(0, color='k', linestyle='--')
        plt.title(f"Stress Error Distribution\nMAE: {mae_stress:.2e}")
        plt.xlabel("Error (Pred - Target)")
        plt.grid(True, alpha=0.3)

        # Subplot 2: R-value Error Distribution
        plt.subplot(1, 2, 2)
        error_r = r_nn - r_bench
        
        # Clip extreme outliers (top 2%) for a cleaner histogram visualization
        limit = np.percentile(np.abs(error_r), 98)
        error_r_clipped = error_r[np.abs(error_r) <= limit]
        
        plt.hist(error_r_clipped, bins=50, color='green', alpha=0.7)
        plt.axvline(0, color='k', linestyle='--')
        plt.title(f"R-value Error Distribution\nMAE: {mae_r:.2e}")
        plt.xlabel("Error (Pred - Target)")
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plot_dir, "stats_histogram_error.png"))
        plt.close()