import tensorflow as tf
import numpy as np

class PhysicsLoss:
    """
    Physics-Informed Loss Module.
    
    This class encapsulates all the mathematical logic for training the Yield Surface Model.
    
    Architectural Role:
    - Receives: Model predictions and Ground Truth targets (from DataLoader).
    - Computes: Error metrics (Stress, R-value, Convexity).
    - Returns: A dictionary of losses to be logged/optimized by the Trainer.
    
    Physical Constraints Implemented:
    1. Yield Stress Accuracy (Global Shape + Experimental Points).
    2. Anisotropy (R-values via Gradient Direction).
    3. Convexity (Hessian Matrix Positive Definiteness).
    4. Symmetry (Shear Symmetry).
    """
    
    def __init__(self, config):
        """
        Initializes the physics engine with material parameters.
        """
        self.config = config
        
        # Load Hill48 Anisotropy Parameters (The "Ground Truth" Physics)
        # These are used to calculate benchmark targets if needed, 
        # though targets are primarily passed in via the training step.
        phys = config.physics
        self.F = tf.constant(phys.F, dtype=tf.float32)
        self.G = tf.constant(phys.G, dtype=tf.float32)
        self.H = tf.constant(phys.H, dtype=tf.float32)
        self.N = tf.constant(phys.N, dtype=tf.float32)
        
        # Reference Stress (Scaling Factor)
        self.ref_stress = tf.constant(config.model.ref_stress, dtype=tf.float32)

    # =========================================================================
    #  CORE MATH: AUTO-DIFFERENTIATION
    # =========================================================================
    @tf.function
    def compute_gradients_and_hessians(self, model, inputs):
        """
        Computes Model Predictions, Gradients (1st Deriv), and Hessians (2nd Deriv)
        in a single optimized graph execution.
        
        Args:
            model: The Keras model being trained.
            inputs: Batch of stress unit vectors (N, 3).
            
        Returns:
            val: Predicted Yield Potential.
            grads: Gradient vector (Normal to the surface).
            hess: Hessian matrix (Curvature of the surface).
        """
        with tf.GradientTape(persistent=True) as tape2:
            tape2.watch(inputs)
            with tf.GradientTape() as tape1:
                tape1.watch(inputs)
                # Forward Pass
                val = model(inputs)
            
            # First Derivative (Gradient)
            # Necessary for R-value calculation (Plastic Flow Rule)
            grads = tape1.gradient(val, inputs)
            
        # Second Derivative (Hessian)
        # Necessary for Convexity Check (Sylvester's Criterion)
        hess = tape2.batch_jacobian(grads, inputs)
        
        del tape2 # Clean up resources
        return val, grads, hess

    # =========================================================================
    #  BENCHMARK UTILITIES
    # =========================================================================
    def get_benchmark_stress(self, inputs):
        """
        Calculates the Analytical Hill48 Equivalent Stress for a batch of inputs.
        Used by the Data Loader (or Trainer) to generate synthetic targets.
        """
        s11, s22, s12 = inputs[:, 0], inputs[:, 1], inputs[:, 2]
        
        # Hill48 Criterion
        term = (self.F * s22**2 + 
                self.G * s11**2 + 
                self.H * (s11 - s22)**2 + 
                2.0 * self.N * s12**2)
        
        # Sigma = Ref / sqrt(Pot)
        sigma_vm = self.ref_stress / tf.sqrt(tf.maximum(term, 1e-8))
        return sigma_vm

    # =========================================================================
    #  LOSS CALCULATION (The Main Logic)
    # =========================================================================
    def calculate_losses(self, model, inputs, targets, weights):
        """
        Calculates and aggregates all loss components.
        
        Args:
            model: The Neural Network.
            inputs: Tuple (inputs_random, inputs_physics)
                    - random: Large batch covering the whole sphere (Shape learning).
                    - physics: Specific experimental angles (Accuracy & R-values).
            targets: Tuple (target_random, target_phys_stress, target_r, geo_p)
            weights: Dynamic weights object from Config.
            
        Returns:
            dict: Individual losses + 'total_loss'.
        """
        # Unpack the Hybrid Batch
        inputs_s, inputs_p = inputs
        target_s, target_p_stress, target_r, geo_p = targets
        
        losses = {}
        total_loss = 0.0

        # --- 1. STRESS LOSS (Accuracy) ---
        
        # A. Random Points (General Shape)
        # Ensures the model learns the overall Hill48 ellipsoid shape.
        pred_s = model(inputs_s)
        l_se_rand = tf.reduce_mean(tf.square(pred_s - target_s))
        losses['l_se_rand'] = l_se_rand
        
        # B. Physics Points (Experimental Accuracy)
        # Ensures the model is perfectly accurate at the specific angles where we have data.
        l_se_uni = 0.0
        if inputs_p.shape[0] > 0:
            pred_p_stress = model(inputs_p)
            l_se_uni = tf.reduce_mean(tf.square(pred_p_stress - target_p_stress))
            
        losses['l_se_uni'] = l_se_uni
        
        # Combined Stress Loss (Equal Weighting)
        l_se_total = 0.5 * l_se_rand + 0.5 * l_se_uni
        total_loss += weights.stress * l_se_total
        losses['l_se'] = l_se_total

        # --- 2. PHYSICS LOSSES (R-values & Convexity) ---
        # Initialize defaults for logging (avoids crashes if batch is empty)
        losses['l_r'] = 0.0
        losses['l_conv'] = 0.0
        losses['l_sym'] = 0.0
        losses['min_eig'] = 0.0 # Diagnostic metric
        
        if inputs_p.shape[0] > 0:
            
            # Compute Derivatives (Expensive, so only done on physics batch)
            val_p, grads_p, hess_p = self.compute_gradients_and_hessians(model, inputs_p)

            # A. R-value Loss (Anisotropy)
            if weights.r_value > 0:
                # 1. Normalize Gradients -> Unit Direction Vectors
                # Physics: R-value is determined by the DIRECTION of the normal vector.
                # We normalize magnitude to 1.0 so the calculation is purely directional.
                local_norms = tf.norm(grads_p, axis=1, keepdims=True) + 1e-8
                grads_norm = grads_p / local_norms
                
                ds_11, ds_22, ds_12 = grads_norm[:, 0], grads_norm[:, 1], grads_norm[:, 2]
                
                # 2. Calculate Plastic Strain Increments
                sin2, cos2, sc = geo_p[:, 0], geo_p[:, 1], geo_p[:, 2]
                d_eps_thick = -(ds_11 + ds_22)
                d_eps_width = ds_11*sin2 + ds_22*cos2 - 2.0*ds_12*sc
                
                # 3. Geometric Error Metric
                # Measures distance between Predicted Strain Vector and Target Strain Vector.
                # Robust against R -> Infinity.
                num = d_eps_width - (target_r * d_eps_thick)
                den = tf.sqrt(1.0 + tf.square(target_r))
                geo_error = tf.abs(num / den)
                
                l_r = tf.reduce_mean(geo_error)
                
                total_loss += weights.r_value * l_r
                losses['l_r'] = l_r

            # B. Convexity Loss (Thermodynamics)
            if weights.convexity > 0 or weights.dynamic_convexity > 0:
                # Calculate Principal Minors of the Hessian Matrix
                # Condition: All minors must be > 0 for convexity.
                m1 = hess_p[:, 0, 0]
                m2 = (hess_p[:, 0, 0] * hess_p[:, 1, 1]) - (hess_p[:, 0, 1] * hess_p[:, 1, 0])
                m3 = tf.linalg.det(hess_p)
                
                # Find the most negative (violating) minor
                min_minor = tf.minimum(tf.minimum(m1, m2), m3)
                
                # Penalty: ReLU ensures we only punish if min_minor < 0
                violation = tf.nn.relu(-min_minor)
                l_conv = tf.reduce_mean(tf.square(violation))
                
                # Combine Static and Dynamic Weights
                w_conv_total = weights.convexity + weights.dynamic_convexity
                total_loss += w_conv_total * l_conv
                
                losses['l_conv'] = l_conv
                losses['min_eig'] = tf.reduce_min(min_minor) # Log worst violation

            # C. Symmetry Loss (Constraint)
            # Enforces Model(S11, S22, S12) == Model(S11, S22, -S12)
            if weights.symmetry > 0:
                inputs_flip = tf.stack([inputs_p[:,0], inputs_p[:,1], -inputs_p[:,2]], axis=1)
                val_flip = model(inputs_flip)
                
                l_sym = tf.reduce_mean(tf.square(val_p - val_flip))
                
                total_loss += weights.symmetry * l_sym
                losses['l_sym'] = l_sym

        losses['total_loss'] = total_loss
        return losses