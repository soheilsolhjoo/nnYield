import tensorflow as tf
from scipy.stats import qmc

class PhysicsLoss:
    """
    Computes physics-informed losses for the Yield Surface Model.
    
    Loss Components:
    1. Shape Stress Loss: MSE between predicted and target stress for random loci.
    2. Physics Stress Loss: MSE for uniaxial data points.
    3. R-value Loss: Error between predicted R-value (via gradients) and target.
    4. Convexity Loss: Penalty for negative Hessian eigenvalues.
    5. Symmetry Loss: Penalty for non-zero shear gradient at equator.
    """
    def __init__(self, config):
        self.config = config

    def calculate_losses(self, model, inputs, targets, weights, run_convexity=False, run_symmetry=False):
        """
        Args:
            model: The neural network model.
            inputs: Tuple (inputs_shape, inputs_physics).
            targets: Tuple (target_stress_shape, target_stress_phys, target_r_phys, geometry_phys).
            weights: WeightsConfig object.
            run_convexity: Bool, whether to compute dynamic convexity loss.
            run_symmetry: Bool, whether to compute symmetry loss.
        """
        # Unpack Inputs
        (inputs_shape, inputs_physics) = inputs
        (target_stress_shape, target_stress_physics, target_r_physics, geometry_physics) = targets
        
        has_shape_data = tf.greater(tf.shape(inputs_shape)[0], 0)
        has_phys_data = tf.greater(tf.shape(inputs_physics)[0], 0)

        # Initialize Loss Components
        loss_stress_shape = tf.constant(0.0)
        loss_stress_physics = tf.constant(0.0)
        loss_r_value = tf.constant(0.0)
        loss_convexity_dynamic = tf.constant(0.0)
        loss_symmetry = tf.constant(0.0)
        
        # Metric for logging
        min_eigenvalue = tf.constant(0.0)

        # ---------------------------------------------------------------------
        # A. SHAPE STREAM (General Stress States)
        # ---------------------------------------------------------------------
        if (weights.stress > 0) and has_shape_data:
            pred_stress_shape = model(inputs_shape)
            loss_stress_shape = tf.reduce_mean(tf.square(pred_stress_shape - target_stress_shape))

        # ---------------------------------------------------------------------
        # B. PHYSICS STREAM (Uniaxial/Experimental Paths)
        # ---------------------------------------------------------------------
        if (weights.r_value > 0) and has_phys_data:
            with tf.GradientTape() as tape:
                tape.watch(inputs_physics)
                pred_stress_physics = model(inputs_physics)
            
            # 1. Stress Consistency on Physics Path
            # Even if R-value weight is low, we want the stress at these points to be correct.
            if weights.stress > 0:
                loss_stress_physics = tf.reduce_mean(tf.square(pred_stress_physics - target_stress_physics))
            
            # 2. R-Value Consistency (Anisotropy)
            # Calculate gradients (Normal to the yield surface)
            gradients = tape.gradient(pred_stress_physics, inputs_physics)
            
            # Unpack Gradients: dF/dS11, dF/dS22, dF/dS12
            # Note: For R-value, we don't strictly need to normalize, as it's a ratio.
            # But normalizing stabilizes the division.
            gnorms = tf.norm(gradients, axis=1, keepdims=True) + 1e-8
            grads_norm = gradients / gnorms
            
            df_ds11 = grads_norm[:, 0:1]
            df_ds22 = grads_norm[:, 1:2]
            df_ds12 = grads_norm[:, 2:3]
            
            # Unpack Geometry: sin^2(a), cos^2(a), sin(a)cos(a)
            sin_sq = geometry_physics[:, 0:1]
            cos_sq = geometry_physics[:, 1:2]
            sin_cos = geometry_physics[:, 2:3]
            
            # Calculate Strains (Associated Flow Rule)
            # Thickness Strain: -(d_eps_11 + d_eps_22)
            d_eps_thickness = -(df_ds11 + df_ds22)
            
            # Width Strain (Rotated to specimen axis)
            # eps_w = eps_11*sin^2 + eps_22*cos^2 - gamma_12*sin*cos
            # gamma_12 = df_ds12 (Engineering Shear Strain = Derivative w.r.t S12)
            d_eps_width = df_ds11 * sin_sq + df_ds22 * cos_sq - df_ds12 * sin_cos
            
            # R-value = Width Strain / Thickness Strain
            pred_r_value = d_eps_width / (d_eps_thickness + 1e-8)
            
            loss_r_value = tf.reduce_mean(tf.abs(pred_r_value - target_r_physics))

        # Combine Stress Losses (Weighted Average)
        # This prevents the "sum" bug where adding more physics data changes the stress loss scale.
        r_fraction = self.config.anisotropy_ratio.batch_r_fraction
        loss_stress_total = (loss_stress_shape * (1.0 - r_fraction)) + (loss_stress_physics * r_fraction)

        # ---------------------------------------------------------------------
        # C. DYNAMIC CONVEXITY (Self-Consistency)
        # ---------------------------------------------------------------------
        if run_convexity and (weights.dynamic_convexity > 0):
            num_samples = self.config.dynamic_convexity.samples
            
            # 1. Sample Directions (Sobol)
            sampler = qmc.Sobol(d=3, scramble=True)
            raw_np = sampler.random(n=num_samples)
            raw_tensor = tf.convert_to_tensor(raw_np, dtype=tf.float32)
            
            # Normalize to Unit Sphere
            directions = raw_tensor / (tf.norm(raw_tensor, axis=1, keepdims=True) + 1e-8)
            
            # 2. Project to Yield Surface
            radius_pred = model.predict_radius(directions) # Requires model.predict_radius
            surface_points = directions * radius_pred
            
            # 3. Compute Hessian
            with tf.GradientTape(persistent=True) as tape2:
                tape2.watch(surface_points)
                with tf.GradientTape() as tape1:
                    tape1.watch(surface_points)
                    val = model(surface_points)
                grad = tape1.gradient(val, surface_points)
            hessian = tape2.batch_jacobian(grad, surface_points)
            del tape2
            
            # 4. Compute Eigenvalues
            eigenvalues = tf.linalg.eigvalsh(hessian)
            min_eig = eigenvalues[:, 0] # Smallest eigenvalue
            min_eigenvalue = tf.reduce_min(min_eig)
            
            # Loss = Penalty for negative eigenvalues
            loss_convexity_dynamic = tf.reduce_mean(tf.nn.relu(-min_eig))

        # ---------------------------------------------------------------------
        # D. SYMMETRY (Orthotropy Check)
        # ---------------------------------------------------------------------
        if run_symmetry and (weights.symmetry > 0):
            num_samples = self.config.symmetry.samples
            
            # 1. Sample Equator Directions (Theta uniform, Phi=pi/2)
            # theta ~ U[0, 2pi]
            theta = tf.random.uniform([num_samples], minval=0.0, maxval=6.28318, dtype=tf.float32)
            phi = tf.ones([num_samples], dtype=tf.float32) * 1.57079 # pi/2
            
            # 2. Project to Surface
            radius_pred = model.predict_radius(theta, phi)
            
            # Convert to Cartesian (S12 = r * cos(pi/2) = 0)
            # s11 = r * sin(pi/2) * cos(theta) = r * cos(theta)
            # s22 = r * sin(pi/2) * sin(theta) = r * sin(theta)
            r = tf.reshape(radius_pred, [-1])
            s11 = r * tf.cos(theta)
            s22 = r * tf.sin(theta)
            s12 = tf.zeros_like(r)
            
            inputs_symmetry = tf.stack([s11, s22, s12], axis=1)
            
            # 3. Compute Gradient w.r.t Shear
            with tf.GradientTape() as tape:
                tape.watch(inputs_symmetry)
                pred_sym = model(inputs_symmetry)
            
            grads_sym = tape.gradient(pred_sym, inputs_symmetry)
            
            # For Orthotropy, dF/dS12 should be 0 at S12=0
            shear_gradient = grads_sym[:, 2]
            loss_symmetry = tf.reduce_mean(tf.square(shear_gradient))

        # ---------------------------------------------------------------------
        # TOTAL LOSS
        # ---------------------------------------------------------------------
        # Add L2 Regularization (automatically tracked by Keras layers)
        loss_l2 = tf.reduce_sum(model.losses)
        
        total_loss = (weights.stress * loss_stress_total) + \
                     (weights.r_value * loss_r_value) + \
                     (weights.dynamic_convexity * loss_convexity_dynamic) + \
                     (weights.symmetry * loss_symmetry) + \
                     loss_l2

        return {
            'total_loss': total_loss, 
            'loss_stress_total': loss_stress_total,
            'loss_r_value': loss_r_value,
            'loss_convexity': loss_convexity_dynamic,
            'loss_symmetry': loss_symmetry, 
            'loss_l2': loss_l2,
            'min_eigenvalue': min_eigenvalue,
            'loss_stress_shape': loss_stress_shape,
            'loss_stress_physics': loss_stress_physics
        }

        