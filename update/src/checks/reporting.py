"""
Reporting and Quantitative Validation Module for nnYield.

This module calculates the numerical performance metrics (MAE, MSE) 
required for academic publication. It provides concrete proof of how 
closely the Neural Network approximates the analytical benchmarks.
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
from .core import BaseChecker

class ReportingChecks(BaseChecker):
    """
    Quantitative Statistics and Publication Plots.
    
    Implements the final reporting layer that aggregates error distributions 
    over a statistically robust number of stress states.
    """

    def check_global_statistics(self):
        """
        TRICK: DENSE UNIAXIAL SAMPLING.
        
        Experimental data is usually sparse (e.g., only 3-7 angles). 
        To truly quantify the model's accuracy, we generate a synthetic dataset 
        of 2000 random angles covering the full 0-90 degree range. 
        
        This provides a robust Mean Absolute Error (MAE) that is not biased 
        by specific directions and ensures the model is smooth everywhere.
        
        Output: 
        - plots/stats_histogram_error.png
        - plots/global_stats.txt
        """
        print("Running Global Statistics (Dense Sampling)...")
        
        # 1. Generate 2000 random tensile directions.
        n_dense = 2000
        angles = np.random.uniform(0, 90, n_dense).astype(np.float32)
        ref = self.config.model.ref_stress
        
        rads = np.radians(angles); sin_a, cos_a = np.sin(rads), np.cos(rads)
        # S11 = cos^2, S22 = sin^2, S12 = sin*cos
        u11, u22, u12 = cos_a**2, sin_a**2, sin_a*cos_a
        inputs_dense = np.stack([u11, u22, u12], axis=1).astype(np.float32)
        
        # 2. Benchmark Ground Truth (Modular Physics Factory).
        sigma_bench = ref / (self.physics_bench.equivalent_stress(u11, u22, u12) + 1e-12)
        r_bench = np.array([self.physics_bench.predict_r_value(a) for a in angles])

        # 3. Neural Network Performance.
        pred_unit = self.model(tf.constant(inputs_dense)).numpy().flatten()
        sigma_nn = ref / (pred_unit + 1e-8)
        
        # Calculate R-values via normality (Gradients).
        s11_surf, s22_surf, s12_surf = u11*sigma_nn, u22*sigma_nn, u12*sigma_nn
        (_, grads, _), _ = self._get_predictions(s11_surf, s22_surf, s12_surf)
        
        # Normalize normality vector to unit length.
        grads_n = grads / (np.linalg.norm(grads, axis=1, keepdims=True) + 1e-8)
        ds_11, ds_22, ds_12 = grads_n[:,0], grads_n[:,1], grads_n[:,2]
        
        # Associated Flow Rule: Rotation of strains into the material frame.
        d_eps_t_nn = -(ds_11 + ds_22)
        d_eps_w_nn = ds_11*sin_a**2 + ds_22*cos_a**2 - ds_12*sin_a*cos_a
        r_nn = np.divide(d_eps_w_nn, d_eps_t_nn, out=np.zeros_like(d_eps_w_nn), where=np.abs(d_eps_t_nn)>1e-8)

        # 4. Numerical Metrics.
        mae_stress = np.mean(np.abs(sigma_nn - sigma_bench))
        mae_r = np.mean(np.abs(r_nn - r_bench))
        
        print(f"   -> Stress MAE (Uniaxial): {mae_stress:.2e}")
        print(f"   -> R-value MAE (Uniaxial): {mae_r:.2e}")

        # Save result to a text summary for academic documentation.
        with open(os.path.join(self.plot_dir, "global_stats.txt"), "w") as f:
            f.write("Global Statistics Report (Dense Uniaxial)\n")
            f.write("=======================================\n")
            f.write(f"Stress MAE: {mae_stress:.6e}\n")
            f.write(f"R-value MAE: {mae_r:.6e}\n")

        # 5. ERROR DISTRIBUTION HISTOGRAMS.
        plt.figure(figsize=(12, 5))
        
        # Subplot 1: Yield Stress error spread.
        plt.subplot(1, 2, 1)
        plt.hist(sigma_nn - sigma_bench, bins=50, color='blue', alpha=0.7)
        plt.axvline(0, color='k', linestyle='--'); plt.title(f"Stress Error (MAE: {mae_stress:.2e})")
        plt.xlabel("Delta Stress (MPa)"); plt.grid(True, alpha=0.3)

        # Subplot 2: R-value error spread.
        plt.subplot(1, 2, 2)
        error_r = r_nn - r_bench
        # TRICK: ROBUST HISTOGRAM VISUALIZATION.
        # R-values can have extreme outliers near zero-thickness strain regions. 
        # We clip the top 2% of errors to keep the histogram interpretable.
        limit = np.percentile(np.abs(error_r), 98)
        plt.hist(error_r[np.abs(error_r) <= limit], bins=50, color='green', alpha=0.7)
        plt.axvline(0, color='k', linestyle='--'); plt.title(f"R-value Error (MAE: {mae_r:.2e})")
        plt.xlabel("Delta R-value"); plt.grid(True, alpha=0.3)
        
        plt.tight_layout(); plt.savefig(os.path.join(self.plot_dir, "stats_histogram_error.png")); plt.close()
