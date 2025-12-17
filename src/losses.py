import tensorflow as tf

class PhysicsLoss:
    def __init__(self, config):
        self.config = config

    def calculate_losses(self, model, inputs, targets, weights, run_convexity=False, run_symmetry=False):
        """
        Calculates loss components.
        Optimized: Runs forward pass on physics data only once for both Stress and R-value calculations.
        """
        # 1. Unpack Data
        (inputs_s, inputs_p) = inputs
        (target_s, target_p_stress, target_r, geo_p) = targets
        
        has_shape_data = tf.shape(inputs_s)[0] > 0
        has_phys_data = tf.shape(inputs_p)[0] > 0

        # 2. Initialize Loss Components
        l_se_shape = tf.constant(0.0, dtype=tf.float32)
        l_se_path = tf.constant(0.0, dtype=tf.float32)
        l_r = tf.constant(0.0, dtype=tf.float32)
        l_conv = tf.constant(0.0, dtype=tf.float32)
        l_sym = tf.constant(0.0, dtype=tf.float32)
        min_eig_val = tf.constant(0.0, dtype=tf.float32)

        # ---------------------------------------------------------------------
        # A. Shape Stress Loss (Random Data)
        # ---------------------------------------------------------------------
        if (weights.stress > 0) and has_shape_data:
            pred_s = model(inputs_s)
            l_se_shape = tf.reduce_mean(tf.square(pred_s - target_s))

        # ---------------------------------------------------------------------
        # B. Physics Path Loss (Stress + R-value)
        # ---------------------------------------------------------------------
        if (weights.r_value > 0) and has_phys_data:
            with tf.GradientTape() as tape:
                tape.watch(inputs_p)
                pred_pot = model(inputs_p)
            
            # 1. Path Stress Loss (reuse the forward pass result)
            if weights.stress > 0:
                l_se_path = tf.reduce_mean(tf.square(pred_pot - target_p_stress))

            # 2. R-Value Loss (use gradients)
            grads = tape.gradient(pred_pot, inputs_p)
            
            df_ds11 = grads[:, 0:1]
            df_ds22 = grads[:, 1:2]
            df_ds12 = grads[:, 2:3]
            
            sin2 = geo_p[:, 0:1]
            cos2 = geo_p[:, 1:2]
            sincos = geo_p[:, 2:3]
            
            d_eps_t = -(df_ds11 + df_ds22)
            d_eps_w = df_ds11 * sin2 + df_ds22 * cos2 - 2 * df_ds12 * sincos
            
            r_pred = d_eps_w / (d_eps_t + 1e-8)
            l_r = tf.reduce_mean(tf.abs(r_pred - target_r))

        # Combine Stress Losses
        l_se_total = l_se_shape + l_se_path

        # ---------------------------------------------------------------------
        # C. Convexity Loss
        # ---------------------------------------------------------------------
        if run_convexity and (weights.convexity > 0 or weights.dynamic_convexity > 0) and has_shape_data:
            with tf.GradientTape() as tape2:
                tape2.watch(inputs_s)
                with tf.GradientTape() as tape1:
                    tape1.watch(inputs_s)
                    y = model(inputs_s)
                grads = tape1.gradient(y, inputs_s)
            
            hessian = tape2.batch_jacobian(grads, inputs_s)
            eigs = tf.linalg.eigvalsh(hessian)
            min_eig = eigs[:, 0]
            min_eig_val = tf.reduce_mean(min_eig)
            
            l_conv = tf.reduce_mean(tf.nn.relu(-min_eig))

        # ---------------------------------------------------------------------
        # D. Symmetry Loss
        # ---------------------------------------------------------------------
        if run_symmetry and (weights.symmetry > 0) and has_shape_data:
            inputs_sym = tf.stack([inputs_s[:, 0], inputs_s[:, 1], -inputs_s[:, 2]], axis=1)
            pred_orig = model(inputs_s)
            pred_sym = model(inputs_sym)
            l_sym = tf.reduce_mean(tf.square(pred_orig - pred_sym))

        # ---------------------------------------------------------------------
        # E. Total Loss
        # ---------------------------------------------------------------------
        w_conv_total = weights.convexity + weights.dynamic_convexity
        
        total_loss = (
            (weights.stress * l_se_total) + 
            (weights.r_value * l_r) + 
            (w_conv_total * l_conv) + 
            (weights.symmetry * l_sym)
        )

        return {
            'total_loss': total_loss,
            'l_se_total': l_se_total,
            'l_se_shape': l_se_shape, 
            'l_se_path': l_se_path,   
            'l_r': l_r,
            'l_conv': l_conv,
            'l_sym': l_sym,
            'min_eig': min_eig_val
        }