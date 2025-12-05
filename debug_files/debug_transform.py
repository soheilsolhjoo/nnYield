import tensorflow as tf
import numpy as np

def cartesian_to_spherical_prototype(inputs):
    """
    Prototype of the logic to be moved into model.py.
    """
    # inputs: [batch, 3] -> s11, s22, s12
    s11 = inputs[:, 0:1]
    s22 = inputs[:, 1:2]
    s12 = inputs[:, 2:3]
    
    # 1. Magnitude (r)
    r = tf.sqrt(tf.square(s11) + tf.square(s22) + tf.square(s12) + 1e-8)
    
    # 2. Theta (Azimuth): Angle in s11-s22 plane
    # atan2 returns [-pi, pi].
    theta = tf.math.atan2(s22, s11)
    
    # SHIFT LOGIC: If theta < 0, add 2*pi. Result is [0, 2*pi].
    theta = tf.where(theta < 0, theta + 2 * np.pi, theta)
    
    # 3. Phi (Polar): Angle from s12 axis
    # acos returns [0, pi]. This fits the 0-360 constraint naturally.
    # We clip the ratio to [-1, 1] to avoid NaNs from floating point errors (e.g. 1.0000001)
    # ratio = s12 / r
    ratio = tf.clip_by_value(s12 / r, -1.0, 1.0)
    phi = tf.math.acos(ratio)
    
    return r, theta, phi

def test_transformation():
    print(f"{'Input [S11, S22, S12]':<25} | {'R':<6} | {'Theta (deg)':<12} | {'Phi (deg)':<12} | Expected Behavior")
    print("-" * 100)

    test_cases = [
        # --- PLANE STRESS (Phi=90) ---
        # Quadrant 1
        ([1.0, 0.0, 0.0],   "0 deg (Ref)"),
        ([1.0, 1.0, 0.0],   "45 deg"),
        ([0.0, 1.0, 0.0],   "90 deg"),
        
        # Quadrant 2
        ([-1.0, 1.0, 0.0],  "135 deg"),
        ([-1.0, 0.0, 0.0],  "180 deg"),
        
        # Quadrant 3 (Critical Check: Should NOT be negative)
        ([-1.0, -1.0, 0.0], "225 deg (NOT -135)"),
        ([0.0, -1.0, 0.0],  "270 deg (NOT -90)"),
        
        # Quadrant 4 (Critical Check)
        ([1.0, -1.0, 0.0],  "315 deg (NOT -45)"),

        # --- SHEAR (Phi changes) ---
        ([0.0, 0.0, 1.0],   "Phi=0 (North Pole)"),
        ([0.0, 0.0, -1.0],  "Phi=180 (South Pole)"),
        ([1.0, 0.0, 1.0],   "Phi=45 (Mixed)"),
    ]

    for vec, desc in test_cases:
        inputs = tf.constant([vec], dtype=tf.float32)
        
        # Test the prototype logic
        r, theta, phi = cartesian_to_spherical_prototype(inputs)
        
        # Convert to degrees for human readability
        r_val = r.numpy()[0, 0]
        theta_deg = np.degrees(theta.numpy()[0, 0])
        phi_deg = np.degrees(phi.numpy()[0, 0])
        
        print(f"{str(vec):<25} | {r_val:<6.3f} | {theta_deg:<12.1f} | {phi_deg:<12.1f} | {desc}")

if __name__ == "__main__":
    test_transformation()