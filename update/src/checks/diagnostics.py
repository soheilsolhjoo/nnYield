import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import pandas as pd

class DiagnosticsChecks:
    """
    Diagnostic Validation Module.
    
    This module focuses on the 'Health' of the training process and the numerical 
    stability of the model. Unlike PhysicsChecks (which checks material science validity),
    this module checks for Machine Learning issues.
    
    Key Checks:
    1.  **Loss History**: Visualizes how different loss components (Stress, R-value, Convexity)
        evolve over time. Crucial for detecting if the model is ignoring one objective 
        to satisfy another.
    2.  **Stability**: Analyzes the variance (noise) in the loss toward the end of training.
        High instability suggests the learning rate is too high.
    3.  **Gradient Smoothness**: Visualizes the raw gradients (normal vectors) of the yield 
        surface. "Jagged" gradients indicate overfitting or numerical noise, even if the 
        yield surface shape looks correct.
    4.  **Math Logic Audit**: A single-point 'dry run' to verify the R-value calculation 
        pipeline isn't producing NaNs or logical errors.
    """

    # =========================================================================
    #  CHECK 6: LOSS HISTORY
    # =========================================================================
    def check_loss_history_detailed(self):
        """
        Generates a 2x3 grid of grouped training metrics for deep-dive analysis.
        """
        print("Running Grouped Loss History Analysis...")
        
        log_path = os.path.join(self.output_dir, "loss_history.csv")
        if not os.path.exists(log_path):
            return

        df = pd.read_csv(log_path)
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        exp_name = self.config.get('experiment_name', 'Experiment')
        fig.suptitle(f"Training History Breakdown: {exp_name}", fontsize=16)

        def plot_group(ax, title, metrics, use_log=True):
            ax.set_title(title)
            has_data = False
            for label, col, color in metrics:
                if col in df.columns:
                    # Filter for valid values to avoid log(0)
                    if use_log:
                        # Find values that are positive and not too small
                        plot_mask = df[col] > 1e-15
                        valid_df = df[plot_mask]
                    else:
                        valid_df = df
                    
                    if not valid_df.empty:
                        ax.plot(valid_df['epoch'], valid_df[col], color=color, label=label, linewidth=1.5)
                        has_data = True
            
            if has_data:
                if use_log: 
                    ax.set_yscale('log')
                ax.legend(fontsize='small', framealpha=0.8)
                ax.grid(True, which="both", alpha=0.3)
            else:
                ax.text(0.5, 0.5, "No Data Recorded", ha='center', va='center', color='gray', alpha=0.5)
            ax.set_xlabel("Epoch")

        # 1. THE OBJECTIVE (Top-Left)
        plot_group(axes[0, 0], "Total Objective", [
            ("Total Loss", 'train_loss_total', 'black')
        ])

        # 2. STRESS ACCURACY (Top-Center)
        plot_group(axes[0, 1], "Stress Accuracy (Stress)", [
            ("Train Stress", 'train_loss_stress', 'blue'),
            ("Val Stress", 'val_loss_stress', 'cyan')
        ])

        # 3. ANISOTROPY ACCURACY (Top-Right)
        plot_group(axes[0, 2], "Anisotropy Accuracy (R)", [
            ("Train R", 'train_loss_r', 'green'),
            ("Val R", 'val_loss_r', 'lime')
        ])

        # 4. SHAPE CONSTRAINTS (Bottom-Left)
        plot_group(axes[1, 0], "Physics Constraints (Penalties)", [
            ("Batch Conv", 'train_loss_batch_conv', 'purple'),
            ("Dyn Conv", 'train_loss_dyn_conv', 'orange'),
            ("Symmetry", 'train_loss_sym', 'magenta')
        ])

        # 5. PHYSICAL STABILITY (Bottom-Center) - LINEAR SCALE
        plot_group(axes[1, 1], "Physical Stability (Min Eig)", [
            ("Batch MinEig", 'train_min_eig_batch', 'red'),
            ("Dyn MinEig", 'train_min_eig_dyn', 'brown')
        ], use_log=False)
        # Add a clear zero-line for stability reference
        axes[1, 1].axhline(0, color='black', linestyle='-', linewidth=1.0, alpha=0.6)

        # 6. OPTIMIZER HEALTH (Bottom-Right)
        plot_group(axes[1, 2], "Optimizer Health", [
            ("Learning Rate", 'learning_rate', 'gray'),
            ("Grad Penalty", 'train_gnorm_penalty', 'gold')
        ])

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(os.path.join(self.plot_dir, "loss_history_detailed.png"))
        plt.close()
        print(f"   -> Detailed loss plot saved to {self.plot_dir}")
    
    # =========================================================================
    #  CHECK 7: STABILITY (Noise Analysis)
    # =========================================================================
    def check_loss_stability(self):
        """
        Analyzes the last 10% of training epochs to quantify stability.
        
        Metric: Coefficient of Variation (CoV = StdDev / Mean).
        - Low CoV (< 0.05): Stable convergence.
        - High CoV (> 0.20): Training is oscillating or unstable (LR might be too high).
        
        Outputs:
            plots/stability_report.txt
        """
        print("Running Stability Check...")
        
        # 1. Load Data
        log_path = os.path.join(self.output_dir, "loss_history.csv")
        if not os.path.exists(log_path): 
            log_path = os.path.join(self.output_dir, "training_log.csv")
        if not os.path.exists(log_path): return

        df = pd.read_csv(log_path)
        
        # 2. Extract the "Tail" (Last 10% of epochs)
        n_tail = max(5, int(len(df) * 0.1))
        tail = df.tail(n_tail)
        
        # 3. Calculate CoV for relevant columns and save report
        with open(os.path.join(self.plot_dir, "stability_report.txt"), "w") as f:
            f.write("Stability Analysis (Last 10% Epochs)\n")
            f.write("------------------------------------\n")
            
            # Scan for any column that looks like a metric
            # cols = [c for c in df.columns if 'loss' in c or 'l_' in c or c in ['gnorm', 'G', 'SE', 'R']]
            cols = [c for c in df.columns if 'train_' in c or c in ['w_conv', 'w_r']]
            
            for col in cols:
                mean = tail[col].mean()
                std = tail[col].std()
                # Avoid division by zero
                cov = std / (mean + 1e-8)
                
                status = "Stable" if cov < 0.05 else "Noisy"
                f.write(f"{col}: CoV={cov:.4f} [{status}]\n")

    # =========================================================================
    #  CHECK 8: GRADIENT COMPONENTS
    # =========================================================================
    def _plot_gradient_components(self):
        """
        Plots a detailed comparison of Surface Normals (Gradients).
        
        Why this is critical:
        - The Yield Surface shape (Value) is easy to learn.
        - The Normal Vectors (Gradients) determine plastic flow (R-values).
        - A model can get the Shape right but have "wobbly" gradients, leading to 
          terrible R-value predictions. This plot reveals that "wobble".
        
        Layout:
        - Rows: Component (dPhi/dS11, dPhi/dS22, dPhi/dS12)
        - Cols: Theory (Hill48), Model (NN), Difference
        
        Outputs:
            plots/gradient_components.png
        """
        print("Running Gradient Component Analysis (3x3 Map)...")
        
        # 1. Setup Grid (Theta vs Phi)
        res_theta, res_phi = 60, 30
        theta = np.linspace(0, 2*np.pi, res_theta).astype(np.float32)
        phi = np.linspace(0, np.pi/2.0, res_phi).astype(np.float32)
        TT, PP = np.meshgrid(theta, phi)
        
        # 2. Generate Unit Directions
        u12 = np.cos(PP)
        r_plane = np.sin(PP)
        u11 = r_plane * np.cos(TT)
        u22 = r_plane * np.sin(TT)
        
        flat_u11, flat_u22, flat_u12 = u11.flatten(), u22.flatten(), u12.flatten()
        unit_inputs = np.stack([flat_u11, flat_u22, flat_u12], axis=1)

        # 3. Scale to Yield Surface
        # We need the gradient AT the yield surface, not on the unit sphere.
        inputs_tf = tf.constant(unit_inputs)
        pred_se = self.model(inputs_tf).numpy().flatten()
        radii = self.config['model']['ref_stress'] / (pred_se + 1e-8)
        
        s11 = flat_u11 * radii
        s22 = flat_u22 * radii
        s12 = flat_u12 * radii
        
        # 4. Get Predictions (Model vs Benchmark)
        # Helper returns: ((val_nn, grad_nn, hess_nn), (val_vm, grad_vm))
        (_, grad_nn, _), (_, grad_vm) = self._get_predictions(s11, s22, s12)
        
        comps = ['dPhi/ds11', 'dPhi/ds22', 'dPhi/ds12']
        fig, axes = plt.subplots(3, 3, figsize=(14, 12))
        fig.suptitle("Gradient Component Analysis (Theory vs Model)", fontsize=16)
        
        X_plot, Y_plot = TT / np.pi, PP / np.pi
        
        # 5. Plot Loops
        for i in range(3): 
            # Column 1: Theory (Benchmark)
            g_vm = grad_vm[:, i].reshape(TT.shape)
            ax = axes[i, 0]
            cp1 = ax.contourf(X_plot, Y_plot, g_vm, levels=30, cmap='bwr')
            plt.colorbar(cp1, ax=ax)
            ax.set_title(f"Theory {comps[i]}")
            ax.set_ylabel(r"Phi ($\times \pi$)")
            ax.invert_yaxis()
            
            # Column 2: Model (Neural Network)
            g_nn = grad_nn[:, i].reshape(TT.shape)
            ax = axes[i, 1]
            cp2 = ax.contourf(X_plot, Y_plot, g_nn, levels=30, cmap='bwr')
            plt.colorbar(cp2, ax=ax)
            ax.set_title(f"Model {comps[i]}")
            ax.invert_yaxis()
            
            # Column 3: Error (Difference)
            err = (g_nn - g_vm)
            ax = axes[i, 2]
            cp3 = ax.contourf(X_plot, Y_plot, err, levels=30, cmap='viridis')
            plt.colorbar(cp3, ax=ax)
            ax.set_title(f"Difference")
            ax.invert_yaxis()
            
        axes[2, 0].set_xlabel(r"Theta ($\times \pi$)")
        axes[2, 1].set_xlabel(r"Theta ($\times \pi$)")
        axes[2, 2].set_xlabel(r"Theta ($\times \pi$)")
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95]) 
        plt.savefig(os.path.join(self.plot_dir, "gradient_components.png"))
        plt.close()

    # =========================================================================
    #  CHECK 9: CALCULATION LOGIC (Math Audit)
    # =========================================================================
    def check_r_calculation_logic(self):
        """
        Performs a single-point dry run of the Gradient Calculation.
        
        Purpose:
        - Quickly verify that the pipeline doesn't crash on simple inputs.
        - Check if gradients are producing NaNs or Zeros (which would indicate 
          the model has "died" or gradients aren't flowing).
        """
        print("Running R-Calc Dry Run...")
        
        # Test Point: Pure Uniaxial Stress in X-direction
        s11 = np.array([1.0], dtype=np.float32)
        s22 = np.array([0.0], dtype=np.float32)
        s12 = np.array([0.0], dtype=np.float32)
        
        # Unpack the nested return structure correctly
        (_, grads, _), _ = self._get_predictions(s11, s22, s12)
        
        print(f"   [Dry Run] Gradients at Uniaxial X: {grads[0]}")
        
        if np.any(np.isnan(grads)):
            print("   [CRITICAL WARNING] Gradients contain NaNs!")