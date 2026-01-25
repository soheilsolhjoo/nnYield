import tensorflow as tf
from scipy.stats import qmc

class PhysicsLoss:
    def __init__(self, config):
        self.config = config

    def calculate_losses(self, model, inputs, targets, weights, run_convexity=False, run_symmetry=False):
        (inputs_s, inputs_p) = inputs
        (target_s, target_p, target_r, geo_p) = targets
        
        has_shape_data = tf.greater(tf.shape(inputs_s)[0], 0)
        has_phys_data = tf.greater(tf.shape(inputs_p)[0], 0)

        l_se_shape = l_se_path = l_r = l_conv_dynamic = l_sym = tf.constant(0.0)
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
                pred_p = model(inputs_p)
            
            # Stress Comparison on Path
            if weights.stress > 0:
                l_se_path = tf.reduce_mean(tf.square(pred_p - target_p))
            
            # R-Value Comparison on Path
            grads = tape.gradient(pred_p, inputs_p)
            df_ds11, df_ds22, df_ds12 = grads[:, 0:1], grads[:, 1:2], grads[:, 2:3]
            sin2, cos2, sincos = geo_p[:, 0:1], geo_p[:, 1:2], geo_p[:, 2:3]
            d_eps_t = -(df_ds11 + df_ds22)
            d_eps_w = df_ds11 * sin2 + df_ds22 * cos2 - 2 * df_ds12 * sincos
            pred_r = d_eps_w / (d_eps_t + 1e-8)
            l_r = tf.reduce_mean(tf.abs(pred_r - target_r))

        l_se_total = l_se_shape + l_se_path

        # # C1. Static Convexity (Calculated on Training Data)
        # # This acts as a regularizer on the actual data points being fitted.
        # if (weights.convexity > 0) and has_shape_data:
        #     with tf.GradientTape() as tape2:
        #         tape2.watch(inputs_s)
        #         with tf.GradientTape() as tape1:
        #             tape1.watch(inputs_s)
        #             y = model(inputs_s)
        #         grads = tape1.gradient(y, inputs_s)
        #     hessian = tape2.batch_jacobian(grads, inputs_s)
        #     min_eig = tf.linalg.eigvalsh(hessian)[:, 0]
        #     # Record min_eig_val from training data for logging
        #     min_eig_val = tf.reduce_mean(min_eig)
        #     l_conv_static = tf.reduce_mean(tf.nn.relu(-min_eig))

        # C. Dynamic Convexity (Calculated on Generated Data)
        # This periodically probes the domain to ensure convexity in "unseen" regions.
        if run_convexity and (weights.dynamic_convexity > 0):
            num_samples = self.config.dynamic_convexity.samples
            
            # 1. Generate Directions
            sampler = qmc.Sobol(d=3, scramble=True)
            raw_np = sampler.random(n=num_samples)
            raw = tf.convert_to_tensor(raw_np, dtype=tf.float32)
            # raw = tf.random.uniform(shape=(num_samples, 3), 
            #                         minval=self.config.data.input_range[0], 
            #                         maxval=self.config.data.input_range[1])
            directions = raw / (tf.norm(raw, axis=1, keepdims=True) + 1e-8)
            
            # 2. Project to Surface using Model Radius
            r_yield = model.predict_radius(directions)
            surface_points = directions * r_yield
            
            # 3. Hessian Logic on Surface Points
            with tf.GradientTape() as t2:
                t2.watch(surface_points)
                with tf.GradientTape() as t1:
                    t1.watch(surface_points)
                    y = model(surface_points)
                g = t1.gradient(y, surface_points)
            h = t2.batch_jacobian(g, surface_points)
            min_eig = tf.linalg.eigvalsh(h)[:, 0]
            min_eig_val = tf.reduce_mean(min_eig)
            l_conv_dynamic = tf.reduce_mean(tf.nn.relu(-min_eig))

        # D. Symmetry (Orthotropy: Enforce zero shear gradient at s12=0)
        if run_symmetry and (weights.symmetry > 0) and has_shape_data:
            with tf.GradientTape() as tape:
                tape.watch(inputs_s)
                pred_sym = model(inputs_s)
            
            grads = tape.gradient(pred_sym, inputs_s)
            df_ds12 = grads[:, 2]
            l_sym = tf.reduce_mean(tf.square(df_ds12))

        total_loss = (weights.stress * l_se_total) + (weights.r_value * l_r) + \
                     (weights.dynamic_convexity * l_conv_dynamic) + \
                     (weights.symmetry * l_sym)

        return {
            'total_loss': total_loss, 
            'l_se_total': l_se_total,
            'l_r': l_r,
            'l_conv_dynamic': l_conv_dynamic,
            'l_sym': l_sym, 
            'min_eig': min_eig_val,
            'l_se_shape': l_se_shape,
            'l_se_path': l_se_path
        }