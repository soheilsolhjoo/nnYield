"""
ML Diagnostic Validation Module for nnYield.

This module focuses on the 'Health' of the Neural Network training process.
It tracks loss stability, gradient flow, and optimizer performance to detect 
overfitting, plateaus, or numerical instabilities.
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import pandas as pd
from .core import BaseChecker

class DiagnosticsChecks(BaseChecker):
    """
    ML Health and Training Stability Monitor.
    """

    # =========================================================================
    #  CHECK 6: DETAILED LOSS BREAKDOWN
    # =========================================================================
    def check_loss_history_detailed(self):
        """
        Generates a 2x3 grid of grouped training metrics for deep-dive analysis.
        
        Purpose:
        - Detects objective conflicts (e.g., Stress vs R-value).
        - Monitors constraint enforcement (Convexity, Orthotropy).
        - Tracks physical stability (MinEig) convergence.
        
        Output: plots/loss_history_detailed.png
        """
        print("Running Grouped Loss History Analysis...")
        log_path = os.path.join(self.output_dir, "loss_history.csv")
        if not os.path.exists(log_path): return

        df = pd.read_csv(log_path)
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle(f"Training History Breakdown: {self.config.experiment_name}", fontsize=16)

        def plot_group(ax, title, metrics, use_log=True):
            """ Helper to plot related metric groups. """
            ax.set_title(title)
            has_data = False
            for label, col, color in metrics:
                if col in df.columns:
                    # Avoid log(0) issues by filtering small values.
                    plot_mask = df[col] > 1e-15 if use_log else np.ones(len(df), dtype=bool)
                    valid_df = df[plot_mask]
                    if not valid_df.empty:
                        ax.plot(valid_df['epoch'], valid_df[col], color=color, label=label, linewidth=1.5)
                        has_data = True
            if has_data:
                if use_log: ax.set_yscale('log')
                ax.legend(fontsize='small', framealpha=0.8); ax.grid(True, which="both", alpha=0.3)
            ax.set_xlabel("Epoch")

        # Column 1: Objectives & Constraints
        plot_group(axes[0, 0], "Total Objective", [("Total Loss", 'train_loss_total', 'black')])
        plot_group(axes[1, 0], "Physics Constraints", [("Batch Conv", 'train_loss_batch_conv', 'purple'), ("Dyn Conv", 'train_loss_dyn_conv', 'orange'), ("Ortho", 'train_loss_ortho', 'magenta')])
        
        # Column 2: Accuracy & Stability
        plot_group(axes[0, 1], "Stress Accuracy", [("Train Stress", 'train_loss_stress', 'blue'), ("Val Stress", 'val_loss_stress', 'cyan')])
        plot_group(axes[1, 1], "Physical Stability (MinEig)", [("Batch MinEig", 'train_min_eig_batch', 'red'), ("Dyn MinEig", 'train_min_eig_dyn', 'brown')], use_log=False)
        axes[1, 1].axhline(0, color='black', ls='-', alpha=0.6)
        
        # Column 3: Anisotropy & Optimizer
        plot_group(axes[0, 2], "Anisotropy Accuracy", [("Train R", 'train_loss_r', 'green'), ("Val R", 'val_loss_r', 'lime')])
        plot_group(axes[1, 2], "Optimizer Health", [("Learning Rate", 'learning_rate', 'gray'), ("Grad Penalty", 'train_gnorm_penalty', 'gold')])

        plt.tight_layout(rect=[0, 0.03, 1, 0.95]); plt.savefig(os.path.join(self.plot_dir, "loss_history_detailed.png")); plt.close()
    
    # =========================================================================
    #  CHECK 7: CONVERGENCE STABILITY
    # =========================================================================
    def check_loss_stability(self):
        """ 
        Analyzes the volatility of training in the final stages.
        Output: plots/stability_report.txt
        """
        print("Running Stability Check...")
        log_path = os.path.join(self.output_dir, "loss_history.csv")
        if not os.path.exists(log_path): return
        df = pd.read_csv(log_path)
        
        # We look at the final 10% of epochs to determine steady-state stability.
        n_tail = max(5, int(len(df) * 0.1)); tail = df.tail(n_tail)
        
        with open(os.path.join(self.plot_dir, "stability_report.txt"), "w") as f:
            f.write("Stability Analysis (Last 10% Epochs)\n------------------------------------\n")
            cols = [c for c in df.columns if 'train_' in c]
            for col in cols:
                # CoV > 0.05 indicates high noise or non-convergence.
                cov = tail[col].std() / (tail[col].mean() + 1e-8)
                f.write(f"{col}: CoV={cov:.4f} [{'Stable' if cov < 0.05 else 'Noisy'}]\n")

    # =========================================================================
    #  CHECK 8: GRADIENT COMPONENT MAPS
    # =========================================================================
    def _plot_gradient_components(self):
        """ 
        Visualizes the individual components of the Surface Normals (dPhi/dS).
        Essential for detectingOverfitting or 'jagged' normality vectors.
        
        Output: plots/gradient_components.png
        """
        print("Running Gradient Component Analysis (3x3 Map)...")
        res_t, res_p = 60, 30
        T, P = np.meshgrid(np.linspace(0, 2*np.pi, res_t), np.linspace(0, np.pi/2.0, res_p))
        u12, r_p = np.cos(P), np.sin(P); u11, u22 = r_p * np.cos(T), r_p * np.sin(T)
        flat_u = np.stack([u11.flatten(), u22.flatten(), u12.flatten()], axis=1)
        
        # Project unit directions onto the boundary.
        pred = self.model(tf.constant(flat_u)).numpy().flatten()
        radii = self.config.model.ref_stress / (pred + 1e-8)
        s11, s22, s12 = u11.flatten()*radii, u22.flatten()*radii, u12.flatten()*radii
        
        # Compare NN normality vs Benchmark ground truth.
        (_, g_nn, _), (_, g_bench) = self._get_predictions(s11, s22, s12)
        comps = ['dPhi/ds11', 'dPhi/ds22', 'dPhi/ds12']
        fig, axes = plt.subplots(3, 3, figsize=(14, 12))
        fig.suptitle("Gradient Component Analysis (Benchmark vs NN)", fontsize=16)
        
        for i in range(3): 
            # Row = Stress Component | Col = (Bench, NN, Difference)
            for j, (data, title, cmap) in enumerate([(g_bench[:,i], f"Bench {comps[i]}", 'bwr'), (g_nn[:,i], f"NN {comps[i]}", 'bwr'), (g_nn[:,i]-g_bench[:,i], "Difference", 'viridis')]):
                cp = axes[i, j].contourf(T/np.pi, P/np.pi, data.reshape(T.shape), levels=30, cmap=cmap)
                plt.colorbar(cp, ax=axes[i, j]); axes[i, j].set_title(title); axes[i, j].invert_yaxis()
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95]); plt.savefig(os.path.join(self.plot_dir, "gradient_components.png")); plt.close()

    def check_r_calculation_logic(self):
        """ Dry-run of the gradient math to catch NaNs before they hit the plots. """
        print("Running R-Calc Dry Run...")
        s11, s22, s12 = np.array([1.0]), np.array([0.0]), np.array([0.0])
        (_, grads, _), _ = self._get_predictions(s11, s22, s12)
        print(f"   [Dry Run] Gradients at Uniaxial X: {grads[0]}")
        if np.any(np.isnan(grads)): print("   [CRITICAL] NaNs detected in gradient flow!")
