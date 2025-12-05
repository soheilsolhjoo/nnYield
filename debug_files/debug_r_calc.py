import numpy as np

def debug_r_calculations():
    print("\n=== R-VALUE CALCULATION AUDIT ===")
    
    # 1. Define Physics Constants (Hill48 / Von Mises)
    # Using Von Mises (F=G=H=0.5, N=1.5) for simplicity, but math holds for Hill
    F, G, H, N = 0.5, 0.5, 0.5, 1.5
    C11, C22, C12, C66 = G+H, F+H, -2*H, 2*N
    ref_stress = 1.0
    
    # 2. Define a Test Case: Uniaxial Tension at 45 degrees
    alpha_deg = 45.0
    alpha_rad = np.radians(alpha_deg)
    
    print(f"Test Case: Uniaxial Tension at {alpha_deg} deg")
    
    # --- STEP A: STRESS STATE (Data Loader Logic) ---
    sin_a = np.sin(alpha_rad)
    cos_a = np.cos(alpha_rad)
    
    # Uniaxial Vector
    u11 = cos_a**2
    u22 = sin_a**2
    u12 = sin_a * cos_a
    
    # Scale to Yield Surface
    hill_val = C11*u11**2 + C22*u22**2 + C12*u11*u22 + C66*u12**2
    scale = ref_stress / np.sqrt(hill_val)
    
    s11 = u11 * scale
    s22 = u22 * scale
    s12 = u12 * scale
    
    print(f"\n[A] Stress State (Normalized):")
    print(f"    S11: {s11:.4f}")
    print(f"    S22: {s22:.4f}")
    print(f"    S12: {s12:.4f}")

    # --- STEP B: ANALYTICAL GRADIENTS (Data Loader Logic) ---
    # Gradients of Hill48 Yield Function
    scale_grad = 1.0 / (2.0 * ref_stress)
    
    g11 = scale_grad * (2*C11*s11 + C12*s22)
    g22 = scale_grad * (2*C22*s22 + C12*s11)
    g12 = scale_grad * (2*C66*s12)
    
    print(f"\n[B] Analytical Gradients (d_SE / d_Sigma):")
    print(f"    G11: {g11:.4f}")
    print(f"    G22: {g22:.4f}")
    print(f"    G12: {g12:.4f}")

    # --- STEP C: GEOMETRY FACTORS (Data Loader -> Trainer) ---
    # These are passed to the trainer to resolve width/thick direction
    geo_sin2 = sin_a**2
    geo_cos2 = cos_a**2
    geo_sc   = sin_a * cos_a
    
    print(f"\n[C] Geometry Factors (Passed to Trainer):")
    print(f"    Sin^2: {geo_sin2:.4f}")
    print(f"    Cos^2: {geo_cos2:.4f}")
    print(f"    Sin*Cos: {geo_sc:.4f}")

    # --- STEP D: STRAIN & R-VALUE CALCULATION (Shared Logic) ---
    # This logic appears in both Data Loader (generation) and Trainer (loss)
    
    # 1. Thickness Strain (Volume Conservation)
    # d_eps_t = -(d_eps_11 + d_eps_22)
    # Validity: Associative flow rule d_eps_ij = lambda * dF/d_sigma_ij
    d_eps_t = -(g11 + g22)
    
    # 2. Width Strain (Tensor Rotation)
    # d_eps_w = eps_xx * sin^2 + eps_yy * cos^2 - 2 * eps_xy * sin * cos
    # Note: g12 corresponds to tensor shear epsilon_12
    d_eps_w = g11 * geo_sin2 + g22 * geo_cos2 - 2 * g12 * geo_sc
    
    # 3. R-value
    r_val = d_eps_w / d_eps_t
    
    print(f"\n[D] Strain & R-value Results:")
    print(f"    Thick Strain: {d_eps_t:.4f}")
    print(f"    Width Strain: {d_eps_w:.4f}")
    print(f"    Calculated R: {r_val:.4f}")
    print(f"    Expected R (Von Mises): 1.0000")

    # --- STEP E: IMPLICIT LOSS CHECK (Trainer Logic) ---
    # Loss = (Width - R_target * Thick)^2
    # Let's verify if the equation holds true with the calculated values
    implicit_residual = d_eps_w - (r_val * d_eps_t)
    
    print(f"\n[E] Implicit Loss Check:")
    print(f"    Residual (should be 0): {implicit_residual:.2e}")
    
    if abs(implicit_residual) < 1e-6:
        print("\n>>> STATUS: CALCULATION CONSISTENT <<<")
    else:
        print("\n>>> STATUS: CALCULATION ERROR DETECTED <<<")

if __name__ == "__main__":
    debug_r_calculations()