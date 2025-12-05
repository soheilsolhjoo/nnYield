import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from src.model import HomogeneousYieldModel
from src.utils import load_config

def audit_dynamic_sampling(output_image_name):
    print("=== AUDITING DYNAMIC SAMPLING LOGIC ===")
    
    # 1. Setup
    config = load_config("configs/von_mises_synthetic.yaml")
    model = HomogeneousYieldModel(config)
    
    # Initialize structure
    model(tf.zeros((1, 3))) 
    
    # --- NEW: Load Trained Weights (if they exist) ---
    weights_path = os.path.join(config['training']['save_dir'], config['experiment_name'], "model.weights.h5")
    if os.path.exists(weights_path):
        print(f"   [INFO] Loading trained weights from: {weights_path}")
        model.load_weights(weights_path)
    else:
        print("   [WARN] No trained weights found. Using RANDOM initialization.")
    # -------------------------------------------------
    
    n_samples = 500
    use_symmetry = config['data'].get('symmetry', True)
    
    print(f"Generating {n_samples} samples (Symmetry={use_symmetry})...")
    
    # A. Generate Angles
    theta_gen = tf.random.uniform((n_samples,), 0, 2*np.pi)
    max_phi = np.pi / 2.0 if use_symmetry else np.pi
    phi_gen = tf.random.uniform((n_samples,), 0, max_phi)
    
    # B. Query Model
    pred_se = model.predict_radius(theta_gen, phi_gen)
    radius_gen = config['model']['ref_stress'] / (pred_se + 1e-8)
    radius_gen = tf.reshape(radius_gen, [-1]) # Flatten
    
    # C. Convert
    s12 = radius_gen * tf.cos(phi_gen)
    r_plane = radius_gen * tf.sin(phi_gen)
    s11 = r_plane * tf.cos(theta_gen)
    s22 = r_plane * tf.sin(theta_gen)
    
    # --- 3. VISUALIZATION ---
    s11_np = s11.numpy(); s22_np = s22.numpy(); s12_np = s12.numpy()
    
    fig = plt.figure(figsize=(12, 5))
    
    # Plot 1: The 3D Cloud (Should look like your trained surface)
    from mpl_toolkits.mplot3d import Axes3D
    ax1 = fig.add_subplot(121, projection='3d')
    sc = ax1.scatter(s11_np, s22_np, s12_np, c=s12_np, cmap='viridis', s=2, alpha=0.6)
    ax1.set_title(f"Dynamic Samples (N={n_samples})")
    ax1.set_xlabel('S11'); ax1.set_ylabel('S22'); ax1.set_zlabel('S12')
    
    # Plot 2: Angular Coverage
    ax2 = fig.add_subplot(122)
    ax2.scatter(theta_gen, phi_gen, s=2, alpha=0.5, c='blue')
    ax2.set_title("Angular Coverage")
    ax2.set_xlabel("Theta"); ax2.set_ylabel("Phi")
    ax2.set_xlim(0, 2*np.pi); ax2.set_ylim(0, np.pi/2 + 0.1)
    
    plt.savefig(output_image_name)
    plt.close()
    print(f"Saved plot to '{output_image_name}'")

if __name__ == "__main__":
    audit_dynamic_sampling("debug_dynamic_audit_1.png")
    audit_dynamic_sampling("debug_dynamic_audit_2.png")