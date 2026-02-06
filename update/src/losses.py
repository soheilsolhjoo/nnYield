import tensorflow as tf
import numpy as np
from scipy.stats import qmc

class PhysicsLoss:
    """
    Handles the mathematical formulation of physics-informed losses.
    Extracted from the stable nnYield_16_12_2025 version.
    """
    def __init__(self, config):
        # config can be a dict or a Config object
        self.config = config
        
    def _get_config_attr(self, path):
        """ Safe accessor for nested config values. """
        parts = path.split('.')
        val = self.config
        for p in parts:
            if isinstance(val, dict):
                val = val.get(p)
            else:
                val = getattr(val, p)
        return val

    def sample_dynamic_surface(self, model, num_samples, force_equator=False):
        """
        Generates random directions on unit sphere and projects them to the yield surface.
        Logic restored from the approved nnYield_16_12_2025 baseline.
        """
        # 1. Sample Theta uniformly [0, 2pi]
        theta = tf.random.uniform([num_samples], minval=0.0, maxval=2*np.pi, dtype=tf.float32)

        # 2. Sample Phi using Inverse Transform Sampling (prevents clumping at poles)
        if force_equator:
            phi = tf.ones([num_samples], dtype=tf.float32) * (np.pi / 2.0)
        else:
            use_positive_shear = self._get_config_attr('data.positive_shear')
            max_phi = np.pi / 2.0 if use_positive_shear else np.pi
            
            # Inverse Transform Sampling: z ~ Uniform[cos(max_phi), 1.0]
            min_z = tf.cos(max_phi)
            max_z = 1.0
            z = tf.random.uniform([num_samples], minval=min_z, maxval=max_z, dtype=tf.float32)
            phi = tf.math.acos(z)

        # 3. Project to Surface using Model's internal polar logic
        radius = model.predict_radius(theta, phi)
        radius = tf.reshape(radius, [-1])

        # 4. Convert to Cartesian Stress components
        s12 = radius * tf.cos(phi)
        r_plane = radius * tf.sin(phi)
        s11 = r_plane * tf.cos(theta)
        s22 = r_plane * tf.sin(theta)
        
        surface_points = tf.stack([s11, s22, s12], axis=1)
        return surface_points

    def get_principal_minors(self, matrix):
        """
        Calculates the three leading principal minors for a 3x3 matrix batch.
        input: matrix (batch_size, 3, 3)
        """
        m1 = matrix[:, 0, 0]
        m2 = (matrix[:, 0, 0] * matrix[:, 1, 1]) - (matrix[:, 0, 1] * matrix[:, 1, 0])
        m3 = tf.linalg.det(matrix)
        return [m1, m2, m3]

    def compute_convexity_loss(self, model, inputs):
        """
        Calculates Convexity Loss and tracks the Physical Minimum Eigenvalue.
        Method: Rectified Minimax
        """
        with tf.GradientTape(persistent=True) as tape2:
            tape2.watch(inputs)
            with tf.GradientTape() as tape1:
                tape1.watch(inputs)
                val = model(inputs)
            grads = tape1.gradient(val, inputs, unconnected_gradients=tf.UnconnectedGradients.ZERO)

        hess_matrix = tape2.batch_jacobian(grads, inputs)
        del tape2 
        
        eigs = tf.linalg.eigvalsh(hess_matrix)
        min_eig_val = tf.reduce_min(eigs)

        # Loss = Mean Violation + Worst-Case Violation
        mean_violation = tf.reduce_mean(tf.reduce_sum(tf.nn.relu(-eigs), axis=1))
        min_violation = tf.nn.relu(-min_eig_val)
        
        total_loss = mean_violation + min_violation
        return total_loss, min_eig_val

    def validate_r_values(self, model, inputs, targets, geometry):
        """
        Validates R-values on the full physics dataset.
        Returns: (mae_stress, mae_r)
        """
        # Unpack
        inp_p = inputs
        tar_se_p = targets[0]
        tar_r = targets[1]
        sin2, cos2, sc = geometry[:,0], geometry[:,1], geometry[:,2]
        
        # 1. Forward Pass & Gradients
        with tf.GradientTape() as tape:
            tape.watch(inp_p)
            pred_se_p = model(inp_p)
        
        grads_p = tape.gradient(pred_se_p, inp_p)
        gnorms = tf.norm(grads_p, axis=1, keepdims=True) + 1e-8
        grads_norm = tf.math.divide_no_nan(grads_p, gnorms)
        
        ds_11, ds_22, ds_12 = grads_norm[:,0], grads_norm[:,1], grads_norm[:,2]
        
        # 2. Calculate Predicted R-values
        d_eps_thick = -(ds_11 + ds_22)
        d_eps_width = ds_11*sin2 + ds_22*cos2 - ds_12*sc
        
        pred_r = tf.math.divide_no_nan(d_eps_width, d_eps_thick + 1e-8)
        
        # 3. Metrics (MAE)
        mae_stress = tf.reduce_mean(tf.abs(pred_se_p - tar_se_p))
        mae_r = tf.reduce_mean(tf.abs(pred_r - tar_r))
        
        return float(mae_stress), float(mae_r)

    def calculate_losses(self, model, batch_shape, batch_phys, do_dyn_conv, do_orthotropy, mode):
        """
        Consolidated loss calculation logic.
        """
        inp_s, tar_se = batch_shape
        # Get WeightsConfig object
        w = self._get_config_attr('training.weights')
        
        loss_batch_conv = tf.constant(0.0)
        loss_dyn_conv = tf.constant(0.0)
        loss_ortho = tf.constant(0.0)
        loss_r = tf.constant(0.0)
        loss_stress_uni = tf.constant(0.0)
        
        min_eig_batch = tf.constant(0.0)
        min_eig_dyn = tf.constant(0.0)

        # 1. Shape Stream
        pred_se = model(inp_s)
        loss_stress_loci = tf.reduce_mean(tf.square(pred_se - tar_se))

        # 2. Physics Stream (Dual Mode)
        if mode == 'dual':
            inp_sp, tar_se_p, tar_r, geo_p, r_mask = batch_phys
            
            if getattr(w, 'r_value', 0) > 0:
                with tf.GradientTape() as tape_r:
                    tape_r.watch(inp_sp)
                    pred_se_p = model(inp_sp)
                grads_p = tape_r.gradient(pred_se_p, inp_sp, unconnected_gradients=tf.UnconnectedGradients.ZERO)
                gnorms = tf.norm(grads_p, axis=1, keepdims=True) + 1e-8
                grads_norm = tf.math.divide_no_nan(grads_p, gnorms)
                
                ds_11, ds_22, ds_12 = grads_norm[:,0], grads_norm[:,1], grads_norm[:,2]
                sin2, cos2, sc = geo_p[:,0], geo_p[:,1], geo_p[:,2]
                
                d_eps_thick = -(ds_11 + ds_22)
                # REMOVED factor of 2 from ds_12 term
                d_eps_width = ds_11*sin2 + ds_22*cos2 - ds_12*sc
                numerator = d_eps_width - (tar_r * d_eps_thick)
                denominator = tf.sqrt(1.0 + tf.square(tar_r))
                geo_error = tf.math.divide_no_nan(numerator, denominator)
                
                masked_sq_error = tf.square(geo_error) * r_mask
                loss_r = tf.reduce_sum(masked_sq_error) / (tf.reduce_sum(r_mask) + 1e-8)
                loss_stress_uni = tf.reduce_mean(tf.square(pred_se_p - tar_se_p))
            else:
                pred_se_p = model(inp_sp)
                loss_stress_uni = tf.reduce_mean(tf.square(pred_se_p - tar_se_p))

        # 3. Regularization & Consistency
        if getattr(w, 'batch_convexity', 0) > 0:
            loss_batch_conv, min_eig_batch = self.compute_convexity_loss(model, inp_s)
        
        if do_dyn_conv and getattr(w, 'dynamic_convexity', 0) > 0:
            n_dyn = self._get_config_attr('dynamic_convexity.samples')
            inp_dyn = self.sample_dynamic_surface(model, n_dyn, force_equator=False)
            loss_dyn_conv, min_eig_dyn = self.compute_convexity_loss(model, inp_dyn)

        if do_orthotropy and getattr(w, 'orthotropy', 0) > 0:
            n_sym = self._get_config_attr('orthotropy.samples')
            inp_sym = self.sample_dynamic_surface(model, n_sym, force_equator=True)
            with tf.GradientTape() as tape_sym:
                tape_sym.watch(inp_sym)
                pred_se_sym = model(inp_sym)
            grads_sym = tape_sym.gradient(pred_se_sym, inp_sym)
            if grads_sym is None: grads_sym = tf.zeros_like(inp_sym)
            loss_ortho = tf.reduce_mean(tf.square(grads_sym[:, 2]))

        # 4. Stress Combination
        r_frac = self._get_config_attr('anisotropy.batch_r_fraction') if mode == 'dual' else 0.0
        loss_stress = (loss_stress_loci * (1.0 - r_frac)) + (loss_stress_uni * r_frac)
        
        primary_loss = (getattr(w, 'stress', 1.0) * loss_stress) + \
                       (getattr(w, 'r_value', 0.0) * loss_r) + \
                       (getattr(w, 'batch_convexity', 0.0) * loss_batch_conv) + \
                       (getattr(w, 'dynamic_convexity', 0.0) * loss_dyn_conv) + \
                       (getattr(w, 'orthotropy', 0.0) * loss_ortho)

        return {
            'primary_loss': primary_loss,
            'loss_stress': loss_stress,
            'loss_r': loss_r,
            'loss_batch_conv': loss_batch_conv,
            'loss_dyn_conv': loss_dyn_conv,
            'loss_ortho': loss_ortho,
            'min_eig_batch': min_eig_batch,
            'min_eig_dyn': min_eig_dyn
        }