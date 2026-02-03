import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import yaml

# Add project root to path
sys.path.append(os.getcwd())

from update.src.model import HomogeneousYieldModel
from update.src.data_loader import YieldDataLoader
from update.src.config import Config

def main():
    # 1. Load Config & Model
    config_path = "update/outputs/von_mises_benchmark/config.yaml"
    weights_path = "update/outputs/von_mises_benchmark/best_model.weights.h5"
    
    config = Config.from_yaml(config_path)
    model = HomogeneousYieldModel(config.to_dict())
    
    # Build model
    dummy = tf.zeros((1, 3))
    model(dummy)
    model.load_weights(weights_path)
    
    # 2. Generate Training Data (Same Logic as Training)
    # We generate a large batch to see the distribution the model saw
    config.data.samples['loci'] = 500 # Generate 500 points
    loader = YieldDataLoader(config.to_dict())
    
    # Get raw points (Inputs, Targets)
    (inputs_s, targets_s), _ = loader._generate_raw_data(needs_physics=False)
    
    # inputs_s is (N, 3) [s11, s22, s12]
    # targets_s is (N, 1) [ref_stress]
    
    # 3. Convert Points to Spherical Coords (Radius, Theta, Phi)
    # r = norm(s)
    # theta = atan2(s22, s11)
    # phi = acos(s12 / r)
    
    r_points = np.linalg.norm(inputs_s, axis=1)
    theta_points = np.arctan2(inputs_s[:, 1], inputs_s[:, 0])
    # Map theta to [0, 2pi]
    theta_points = np.where(theta_points < 0, theta_points + 2*np.pi, theta_points)
    
    phi_points = np.arccos(inputs_s[:, 2] / (r_points + 1e-8))
    
    # 4. Generate Model Prediction Curve (Equator: Phi=pi/2)
    theta_line = np.linspace(0, 2*np.pi, 360).astype(np.float32)
    phi_line = np.ones_like(theta_line) * (np.pi / 2.0) # Equator
    
    r_pred = model.predict_radius(theta_line, phi_line).numpy().flatten()
    
    # --- 4b. Generate Analytical Benchmark Curve ---
    # Hill48 parameters from config
    phys = config.physics
    F, G, H, N = phys.F, phys.G, phys.H, phys.N
    ref_stress = config.model.ref_stress
    
    # At equator (Phi=pi/2), S12=0.
    # Radius r satisfies: (G+H)(r*c)^2 + (F+H)(r*s)^2 - 2H(r*c)(r*s) = ref^2
    # r^2 [ (G+H)c^2 + (F+H)s^2 - 2Hcs ] = ref^2
    
    c = np.cos(theta_line)
    s = np.sin(theta_line)
    denom = (G+H)*c**2 + (F+H)*s**2 - 2*H*c*s
    r_bench = ref_stress / np.sqrt(denom + 1e-8)

    # --- 4c. Calculate Errors ---
    mse_equator = np.mean(np.square(r_pred - r_bench))
    mae_equator = np.mean(np.abs(r_pred - r_bench))
    max_err_equator = np.max(np.abs(r_pred - r_bench))
    
    print("\n" + "="*40)
    print("EQUATOR ERROR ANALYSIS")
    print("="*40)
    print(f"MSE (Prediction vs Benchmark): {mse_equator:.2e}")
    print(f"MAE (Prediction vs Benchmark): {mae_equator:.2e}")
    print(f"MAX (Prediction vs Benchmark): {max_err_equator:.2e}")
    print(f"Config Loss Threshold:        {config.training.loss_threshold:.2e}")
    print("="*40 + "\n")

    # 5. Plotting
    plt.figure(figsize=(10, 6))
    
    # A. Plot Model Equator
    plt.plot(theta_line, r_pred, 'r-', linewidth=2, label='Model Prediction (Equator)')
    
    # B. Plot Training Points
    plt.scatter(theta_points, r_points, c=phi_points, cmap='coolwarm', alpha=0.6, s=20, label='Training Data (All Phi)')
    plt.colorbar(label='Phi (rad)')
    
    # C. Plot Analytical Benchmark
    plt.plot(theta_line, r_bench, 'k--', linewidth=2, label='Benchmark (Analytical)')
    
    plt.xlabel('Theta (rad)')
    plt.ylabel('Yield Radius')
    plt.title('Model Fit vs Training Data Distribution')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    
    out_path = 'debug_files/training_fit_check.png'
    plt.savefig(out_path)
    print(f"Plot saved to: {out_path}")

if __name__ == "__main__":
    main()
