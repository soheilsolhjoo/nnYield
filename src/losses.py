import tensorflow as tf

class PhysicsLoss:
    def __init__(self, config):
        self.config = config

    def calculate_losses(self, model, inputs, targets, weights, run_convexity=False, run_symmetry=False):
        (inputs_s, inputs_p) = inputs
        (target_s, target_p_stress, target_r, geo_p) = targets
        
        has_shape_data = tf.greater(tf.shape(inputs_s)[0], 0)
        has_phys_data = tf.greater(tf.shape(inputs_p)[0], 0)

        l_se_shape = l_se_path = l_r = l_conv_static = l_conv_dynamic = l_sym = tf.constant(0.0)
        min_eig_val = tf.constant(0.0)

        # Initialize pred_s to prevent "None" errors in graph branches
        pred_s = tf.zeros((0, 1))
        if has_shape_data:
            pred_s = model(inputs_s)

        # A. Shape Stress
        if (weights.stress > 0) and has_shape_data:
            l_se_shape = tf.reduce_mean(tf.square(pred_s - target_s))

        # B. Physics Path (Stress + R-value)
        if (weights.r_value > 0) and has_phys_data:
            with tf.GradientTape() as tape:
                tape.watch(inputs_p)
                pred_pot = model(inputs_p)
            if weights.stress > 0:
                l_se_path = tf.reduce_mean(tf.square(pred_pot - target_p_stress))
            grads = tape.gradient(pred_pot, inputs_p)
            
            df_ds11, df_ds22, df_ds12 = grads[:, 0:1], grads[:, 1:2], grads[:, 2:3]
            sin2, cos2, sincos = geo_p[:, 0:1], geo_p[:, 1:2], geo_p[:, 2:3]
            
            d_eps_t = -(df_ds11 + df_ds22)
            d_eps_w = df_ds11 * sin2 + df_ds22 * cos2 - 2 * df_ds12 * sincos
            
            with tf.GradientTape() as tape:
                tape.watch(inputs_p)
                pred_pot = model(inputs_p)
            if weights.stress > 0:
                l_se_path = tf.reduce_mean(tf.square(pred_pot - target_p_stress))
            grads = tape.gradient(pred_pot, inputs_p)
            # ... (Flow rule math remains same) ...
            r_pred = d_eps_w / (d_eps_t + 1e-8)
            l_r = tf.reduce_mean(tf.abs(r_pred - target_r))

        l_se_total = l_se_shape + l_se_path

        # C. Convexity (Split for Detailed Logging)
        if run_convexity and has_shape_data:
            with tf.GradientTape() as tape2:
                tape2.watch(inputs_s)
                with tf.GradientTape() as tape1:
                    tape1.watch(inputs_s)
                    y = model(inputs_s)
                grads = tape1.gradient(y, inputs_s)
            hessian = tape2.batch_jacobian(grads, inputs_s)
            min_eig = tf.linalg.eigvalsh(hessian)[:, 0]
            min_eig_val = tf.reduce_mean(min_eig)
            raw_penalty = tf.reduce_mean(tf.nn.relu(-min_eig))
            
            if weights.convexity > 0: l_conv_static = raw_penalty
            if weights.dynamic_convexity > 0: l_conv_dynamic = raw_penalty

        # D. Symmetry (Reuses pred_s)
        if run_symmetry and (weights.symmetry > 0) and has_shape_data:
            inputs_sym = tf.stack([inputs_s[:, 0], inputs_s[:, 1], -inputs_s[:, 2]], axis=1)
            pred_sym = model(inputs_sym)
            l_sym = tf.reduce_mean(tf.square(pred_s - pred_sym))

        total_loss = (weights.stress * l_se_total) + (weights.r_value * l_r) + \
                     (weights.convexity * l_conv_static) + \
                     (weights.dynamic_convexity * l_conv_dynamic) + \
                     (weights.symmetry * l_sym)

        return {
            'total_loss': total_loss, 'l_se_total': l_se_total, 'l_r': l_r,
            'l_conv_static': l_conv_static, 'l_conv_dynamic': l_conv_dynamic,
            'l_sym': l_sym, 'min_eig': min_eig_val
        }