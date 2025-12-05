import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from src.data_loader import YieldDataLoader

def visualize_mask():
    print("Initializing Data Loader...")
    
    # Use the same config parameters as your training
    mock_config = {
        'data': {'type': 'synthetic', 'samples': 5000},
        'model': {'ref_stress': 1.0},
        'training': {'batch_size': 32}
    }
    
    loader = YieldDataLoader(mock_config)
    
    # Get raw data: inputs, se, r_vals, geo, mask
    inputs, _, r_vals, _, r_mask = loader.get_numpy_data()
    
    s11 = inputs[:, 0]
    s22 = inputs[:, 1]
    s12 = inputs[:, 2]
    
    # Flatten mask for boolean indexing
    mask_bool = r_mask.flatten().astype(bool)
    
    # Split data
    active_s11 = s11[mask_bool]
    active_s22 = s22[mask_bool]
    active_s12 = s12[mask_bool]
    
    ignored_s11 = s11[~mask_bool]
    ignored_s22 = s22[~mask_bool]
    ignored_s12 = s12[~mask_bool]
    
    print(f"\nTotal Points: {len(s11)}")
    print(f"Active Points (Mask=1): {len(active_s11)} ({len(active_s11)/len(s11):.1%} of data)")
    print(f"Ignored Points (Mask=0): {len(ignored_s11)}")

    # --- Plotting ---
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot Ignored points (Faint Grey)
    ax.scatter(ignored_s11, ignored_s22, ignored_s12, 
               c='lightgrey', s=5, alpha=0.1, label='Ignored (Stress Only)')
    
    # Plot Active points (Red)
    ax.scatter(active_s11, active_s22, active_s12, 
               c='red', s=15, alpha=1.0, label='Active (Stress + R-value)')
    
    ax.set_xlabel('Sigma 11')
    ax.set_ylabel('Sigma 22')
    ax.set_zlabel('Sigma 12')
    ax.set_title(f"R-value Training Mask Visualization\n(Red points affect R-value Loss)")
    ax.legend()
    
    # Axis limits for equal aspect ratio feel
    max_range = np.array([s11.max()-s11.min(), s22.max()-s22.min(), s12.max()-s12.min()]).max() / 2.0
    mid_x = (s11.max()+s11.min()) * 0.5
    mid_y = (s22.max()+s22.min()) * 0.5
    mid_z = (s12.max()+s12.min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    plt.tight_layout()
    plt.savefig("debug_mask_check.png")
    print("Plot saved to 'debug_mask_check.png'")

if __name__ == "__main__":
    visualize_mask()