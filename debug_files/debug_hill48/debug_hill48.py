import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import qmc

# =============================================================================
#  USER CONFIGURATION
# =============================================================================
# Hill48 Parameters
# OBVIOUS ANISOTROPY
# R0 = H/G = 1.0/2.0 = 0.5
# R90 = H/F = 1.0/0.5 = 2.0
F = 0.5
G = 2.0
H = 1.0
N = 3.125
REF_STRESS = 1.0

# Sampling Settings
N_POINTS = 500  # Number of points for scatter plot (Data Loader Style)
SHEAR_RATIOS = [0.0, 0.4, 0.8, 0.95] # For the lines (Physics Plot Style)

# =============================================================================
#  HILL48 HELPER
# =============================================================================
def get_hill_coeffs(F, G, H, N):
    C11 = G + H
    C22 = F + H
    C12 = -H 
    C66 = 2 * N
    return C11, C22, C12, C66

# =============================================================================
#  METHOD 1: DATA LOADER STYLE (Cylindrical / Constant Shear)
# =============================================================================
def generate_data_loader_points(n_points, f, g, h, n, ref):
    """
    Simulates the 'Loci' generation from data_loader.py.
    1. Sample random S12 (Shear).
    2. Sample random Theta.
    3. Solve for Radius in S11-S22 plane.
    """
    print("Generating Data Loader Points (Cylindrical)...")
    
    sampler = qmc.Sobol(d=2, scramble=True)
    m = int(np.ceil(np.log2(n_points)))
    sample = sampler.random(2**m)[:n_points]
    
    max_shear = ref / np.sqrt(2*n)
    s12 = sample[:, 0] * max_shear
    
    theta = sample[:, 1] * 2 * np.pi
    c = np.cos(theta)
    s = np.sin(theta)
    
    # Solve for In-Plane Radius
    A = (g+h)*c**2 + (f+h)*s**2 - 2*h*c*s
    B = 2*n
    
    rhs = np.maximum(ref**2 - B*s12**2, 0)
    r = np.sqrt(rhs / (A + 1e-8))
    
    s11 = r * c
    s22 = r * s
    
    return np.stack([s11, s22, s12], axis=1)

# =============================================================================
#  METHOD 2: PHYSICS PLOT STYLE (Spherical / Conical Projection)
# =============================================================================
def generate_physics_plot_lines(shear_ratios, f, g, h, n, ref):
    """
    Simulates the 'check_2d_loci_slices' from physics.py.
    1. Define 'Target Shear' based on ratio of max shear.
    2. Construct direction vector [cos(t), sin(t), Target_Shear].
    3. Scale vector by Yield Radius (Homogeneous Scaling).
    """
    print("Generating Physics Plot Lines (Spherical)...")
    
    lines = []
    max_shear = ref / np.sqrt(2*n)
    theta = np.linspace(0, 2*np.pi, 360) # Higher res for smooth curves
    
    for ratio in shear_ratios:
        target_s12 = ratio * max_shear
        
        u11 = np.cos(theta)
        u22 = np.sin(theta)
        u12 = np.full_like(theta, target_s12)
        
        # Hill Value
        term = f*u22**2 + g*u11**2 + h*(u11-u22)**2 + 2*n*u12**2
        hill_val = np.sqrt(np.maximum(term, 1e-16))
        
        # Scale
        scale = ref / (hill_val + 1e-8)
        
        s11 = u11 * scale
        s22 = u22 * scale
        s12 = u12 * scale 
        
        lines.append(np.stack([s11, s22, s12], axis=1))
        
    return lines

# =============================================================================
#  MAIN VISUALIZATION
# =============================================================================
# =============================================================================
#  METHOD 3: CORRECTED PHYSICS PLOT (Cylindrical / Constant Shear)
# =============================================================================
def generate_corrected_physics_lines(shear_ratios, f, g, h, n, ref):
    """
    CORRECTED Benchmark Logic.
    Instead of projecting a ray, we solve for the exact radius at fixed shear.
    This matches the Data Loader formulation.
    """
    print("Generating Corrected Physics Lines (Cylindrical)...")
    
    lines = []
    max_shear = ref / np.sqrt(2*n)
    theta = np.linspace(0, 2*np.pi, 360) 
    
    c = np.cos(theta)
    s = np.sin(theta)
    
    # Pre-calculate angular term A (Stiffness in S11-S22 plane)
    A = (g+h)*c**2 + (f+h)*s**2 - 2*h*c*s
    B = 2*n
    
    for ratio in shear_ratios:
        target_s12 = ratio * max_shear
        
        # 1. Fix S12 exactly
        s12 = np.full_like(theta, target_s12)
        
        # 2. Solve for In-Plane Radius 'r'
        # Hill48: A*r^2 + B*s12^2 = ref^2
        # r = sqrt( (ref^2 - B*s12^2) / A )
        
        rhs = np.maximum(ref**2 - B*target_s12**2, 0)
        r = np.sqrt(rhs / (A + 1e-8))
        
        s11 = r * c
        s22 = r * s
        
        lines.append(np.stack([s11, s22, s12], axis=1))
        
    return lines

# =============================================================================
#  MAIN VISUALIZATION
# =============================================================================
def main():
    # 1. Generate Data
    data_loader_points = generate_data_loader_points(N_POINTS, F, G, H, N, REF_STRESS)
    physics_lines = generate_physics_plot_lines(SHEAR_RATIOS, F, G, H, N, REF_STRESS)
    corrected_lines = generate_corrected_physics_lines(SHEAR_RATIOS, F, G, H, N, REF_STRESS)
    
    # 2. Plotting Setup
    fig = plt.figure(figsize=(18, 6))
    
    # --- SUBPLOT 1: 2D View (S11 vs S22) ---
    ax = fig.add_subplot(131)
    
    # Plot Scatter (Data Loader)
    ax.scatter(data_loader_points[:,0], data_loader_points[:,1],
               c=data_loader_points[:,2], cmap='viridis', alpha=0.3, s=15, label='Data Loader')
    
    # Plot Wavy Physics Lines (Dotted)
    for i, line in enumerate(physics_lines):
        line_loop = np.concatenate([line, line[0:1]], axis=0)
        ax.plot(line_loop[:,0], line_loop[:,1], 
                color='red', linestyle=':', linewidth=1, alpha=0.7)

    # Plot Corrected Lines (Solid Green)
    for i, line in enumerate(corrected_lines):
        ratio = SHEAR_RATIOS[i]
        line_loop = np.concatenate([line, line[0:1]], axis=0)
        ax.plot(line_loop[:,0], line_loop[:,1], 
                color='green', linewidth=2, label=f'Corrected (R={ratio})')

    ax.set_xlabel('S11')
    ax.set_ylabel('S22')
    ax.set_title(f'Yield Loci (Top View)\nF={F}, G={G}, H={H}, N={N}')
    ax.legend(loc='best', fontsize='x-small')
    ax.grid(True, alpha=0.3)
    ax.axis('equal')

    # --- SUBPLOT 2: 2D Projection (S12 vs S11) ---
    ax2 = fig.add_subplot(132)
    ax2.scatter(data_loader_points[:,0], data_loader_points[:,2], 
                c='blue', alpha=0.1, s=10)
    
    # Wavy Lines
    for line in physics_lines:
        line_loop = np.concatenate([line, line[0:1]], axis=0)
        ax2.plot(line_loop[:,0], line_loop[:,2], color='red', linestyle=':', alpha=0.5)
        
    # Corrected Lines
    for line in corrected_lines:
        line_loop = np.concatenate([line, line[0:1]], axis=0)
        ax2.plot(line_loop[:,0], line_loop[:,2], color='green', linewidth=2)
        
    ax2.set_xlabel('S11')
    ax2.set_ylabel('S12 (Shear)')
    ax2.set_title('Projection: S12 vs S11')
    ax2.grid(True, alpha=0.3)

    # --- SUBPLOT 3: 2D Projection (S12 vs S22) ---
    ax3 = fig.add_subplot(133)
    ax3.scatter(data_loader_points[:,1], data_loader_points[:,2], 
                c='blue', alpha=0.1, s=10)
    
    # Wavy Lines
    for line in physics_lines:
        line_loop = np.concatenate([line, line[0:1]], axis=0)
        ax3.plot(line_loop[:,1], line_loop[:,2], color='red', linestyle=':', alpha=0.5)
        
    # Corrected Lines
    for line in corrected_lines:
        line_loop = np.concatenate([line, line[0:1]], axis=0)
        ax3.plot(line_loop[:,1], line_loop[:,2], color='green', linewidth=2)
        
    ax3.set_xlabel('S22')
    ax3.set_ylabel('S12 (Shear)')
    ax3.set_title('Projection: S12 vs S22')
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = 'debug_files/debug_hill48/comparison_3d.png'
    plt.savefig(output_path)
    print(f"Plot saved to: {output_path}")

if __name__ == "__main__":
    main()