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
        (2) Generates a 2x3 grid. 
        Top Row: total | stress loss | r value loss
        Second Row: convexity loss (static) | convexity loss (dynamic) | symmetry loss
        """
        print("Running Loss History Analysis...")
        
        log_path = os.path.join(self.output_dir, "loss_history.csv")
        if not os.path.exists(log_path):
            return

        df = pd.read_csv(log_path)
        
        # Mapping to the exact keys recorded in loss_history.csv
        plots_config = [
            # First Row
            ("Total Loss", 'total_loss', 'black'),
            ("Stress Loss (eqS)", 'se_total_loss', 'blue'),
            ("R-value Loss", 'r_loss', 'green'),
            # Second Row
            ("Convexity Loss (Static)", 'conv_static_loss', 'purple'),
            ("Convexity Loss (Dynamic)", 'conv_dynamic_loss', 'orange'),
            ("Symmetry Loss", 'sym_loss', 'magenta'),
        ]

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        exp_name = self.config.get('experiment_name', 'Experiment')
        fig.suptitle(f"Training History: {exp_name}", fontsize=16)
        
        for i, (title, col_name, color) in enumerate(plots_config):
            ax = axes.flat[i]
            
            if col_name in df.columns:
                # Filter for values > 1e-12 for log-scale stability
                plot_df = df[df[col_name] > 1e-12]
                if not plot_df.empty:
                    ax.plot(plot_df['epoch'], plot_df[col_name], color=color, linewidth=1.5)
                    ax.set_yscale('log')
                else:
                    ax.text(0.5, 0.5, "Value is 0.0", ha='center', va='center', color='gray')
            else:
                ax.text(0.5, 0.5, "Column Not Found", ha='center', va='center', color='red')

            ax.set_title(title)
            ax.set_xlabel("Epoch")
            ax.grid(True, which="both", alpha=0.3)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(os.path.join(self.plot_dir, "loss_history_detailed.png"))
        plt.close()
    
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