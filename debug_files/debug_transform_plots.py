import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D

# --- Import Data Loader ---
from src.data_loader import YieldDataLoader

# --- Transformation Logic (The Code Under Test) ---
def cartesian_to_spherical(inputs):
    inputs = tf.cast(inputs, tf.float32)
    s11 = inputs[:, 0]
    s22 = inputs[:, 1]
    s12 = inputs[:, 2]
    
    # 1. Magnitude
    r = tf.sqrt(tf.square(s11) + tf.square(s22) + tf.square(s12) + 1e-8)
    
    # 2. Theta (0 to 360 deg)
    theta = tf.math.atan2(s22, s11)
    theta = tf.where(theta < 0, theta + 2 * np.pi, theta)
    
    # 3. Phi (Angle from Shear Axis s12)
    # 0 deg = Pure Positive Shear
    # 90 deg = Zero Shear (Plane Stress)
    # 180 deg = Pure Negative Shear
    ratio = s12 / r
    ratio = tf.clip_by_value(ratio, -1.0, 1.0)
    phi = tf.math.acos(ratio)
    
    return r.numpy(), theta.numpy(), phi.numpy()

def run_visualization():
    output_dir = "debug_plots"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Generating data-driven transform plots in '{output_dir}'...")

    # 1. Generate Data
    print("Initializing Data Loader...")
    mock_config = {
        'data': {'type': 'synthetic', 'samples': 2000},
        'model': {'ref_stress': 1.0},
        'training': {'batch_size': 32}
    }
    loader = YieldDataLoader(mock_config)
    inputs, _, _ = loader.get_numpy_data()
    
    # 2. Transform
    r, theta, phi = cartesian_to_spherical(inputs)
    
    # 3. PRINT CALIBRATION TABLE (CRITICAL DEBUG STEP)
    # We define specific masks to find "extreme" points in the dataset
    print("\n" + "="*80)
    print(f"{'POINT TYPE':<20} | {'S11':<6} {'S22':<6} {'S12':<6} | {'R':<6} | {'Theta':<6} | {'Phi':<6}")
    print("-" * 80)
    
    def print_row(name, idx):
        row_in = inputs[idx]
        deg_t = np.degrees(theta[idx])
        deg_p = np.degrees(phi[idx])
        print(f"{name:<20} | {row_in[0]:<6.2f} {row_in[1]:<6.2f} {row_in[2]:<6.2f} | {r[idx]:<6.2f} | {deg_t:<6.1f} | {deg_p:<6.1f}")

    # Find point with min shear (closest to zero)
    idx_zero_shear = np.argmin(np.abs(inputs[:, 2]))
    print_row("Zero Shear (Equator)", idx_zero_shear)
    
    # Find point with max positive shear
    idx_max_shear = np.argmax(inputs[:, 2])
    print_row("Max Shear (+)", idx_max_shear)
    
    # Find point with pure S11 tension (approx)
    # Minimize dist to [1, 0, 0] ignoring magnitude
    dirs = inputs / np.linalg.norm(inputs, axis=1, keepdims=True)
    dist = np.linalg.norm(dirs - np.array([1,0,0]), axis=1)
    idx_s11 = np.argmin(dist)
    print_row("Pure Tension S11", idx_s11)
    
    print("="*80 + "\n")

    # 4. Visualize
    fig = plt.figure(figsize=(14, 6))

    # Subplot 1: 3D Physical Space
    ax1 = fig.add_subplot(121, projection='3d')
    sc1 = ax1.scatter(inputs[:,0], inputs[:,1], inputs[:,2], c=np.degrees(phi), cmap='twilight', s=2)
    ax1.set_xlabel('S11'); ax1.set_ylabel('S22'); ax1.set_zlabel('S12')
    ax1.set_title('Physical Space (Color=Phi)')
    plt.colorbar(sc1, ax=ax1, label='Phi (deg)')

    # Subplot 2: Network Space
    ax2 = fig.add_subplot(122)
    sc2 = ax2.scatter(np.degrees(theta), np.degrees(phi), c=inputs[:,2], cmap='viridis', s=2)
    ax2.set_xlabel('Theta (deg) [0-360]')
    ax2.set_ylabel('Phi (deg) [0-180]')
    ax2.set_title('Network Space (Color=Shear)')
    ax2.axhline(90, color='k', linestyle='--', alpha=0.5, label='Equator (s12=0)')
    plt.colorbar(sc2, ax=ax2, label='Shear Stress (S12)')
    ax2.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "transform_check_loci.png"))
    plt.close()
    
    # 5. Save Debug CSV
    df = pd.DataFrame({
        's11': inputs[:,0], 's22': inputs[:,1], 's12': inputs[:,2],
        'r': r, 'theta_deg': np.degrees(theta), 'phi_deg': np.degrees(phi)
    })
    df.to_csv(os.path.join(output_dir, "transform_loci_data.csv"), index=False)
    print(f"Full data saved to {os.path.join(output_dir, 'transform_loci_data.csv')}")

if __name__ == "__main__":
    run_visualization()