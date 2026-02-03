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

    def calculate_losses(self, model, batch_shape, batch_phys, do_dyn_conv, do_symmetry, mode):
        """
        Consolidated loss calculation logic.
        """
        inp_s, tar_se_s = batch_shape
        # Get WeightsConfig object
        w = self._get_config_attr('training.weights')
        
        loss_conv = tf.constant(0.0)
        loss_dyn = tf.constant(0.0)
        loss_sym = tf.constant(0.0)
        loss_r = tf.constant(0.0)
        loss_stress_uni = tf.constant(0.0)
        
        min_eig_stat = tf.constant(0.0)
        min_eig_dyn = tf.constant(0.0)

        # 1. Shape Stream
        pred_se_s = model(inp_s)
        loss_stress_s = tf.reduce_mean(tf.square(pred_se_s - tar_se_s))

        # 2. Physics Stream (Dual Mode)
        if mode == 'dual':
            inp_p, tar_se_p, tar_r, geo_p, r_mask, tar_se_p_stress = batch_phys
            
            if getattr(w, 'r_value', 0) > 0:
                with tf.GradientTape() as tape_r:
                    tape_r.watch(inp_p)
                    pred_se_p = model(inp_p)
                grads_p = tape_r.gradient(pred_se_p, inp_p, unconnected_gradients=tf.UnconnectedGradients.ZERO)
                gnorms = tf.norm(grads_p, axis=1, keepdims=True) + 1e-8
                grads_norm = tf.math.divide_no_nan(grads_p, gnorms)
                
                ds_11, ds_22, ds_12 = grads_norm[:,0], grads_norm[:,1], grads_norm[:,2]
                sin2, cos2, sc = geo_p[:,0], geo_p[:,1], geo_p[:,2]
                
                d_eps_thick = -(ds_11 + ds_22)
                d_eps_width = ds_11*sin2 + ds_22*cos2 - 2*ds_12*sc
                numerator = d_eps_width - (tar_r * d_eps_thick)
                denominator = tf.sqrt(1.0 + tf.square(tar_r))
                geo_error = tf.math.divide_no_nan(numerator, denominator)
                
                masked_sq_error = tf.square(geo_error) * r_mask
                loss_r = tf.reduce_sum(masked_sq_error) / (tf.reduce_sum(r_mask) + 1e-8)
                loss_stress_uni = tf.reduce_mean(tf.square(pred_se_p - tar_se_p_stress))
            else:
                pred_se_p = model(inp_p)
                loss_stress_uni = tf.reduce_mean(tf.square(pred_se_p - tar_se_p_stress))

        # 3. Regularization & Consistency
        if getattr(w, 'batch_convexity', 0) > 0:
            loss_conv, min_eig_stat = self.compute_convexity_loss(model, inp_s)
        
        if do_dyn_conv and getattr(w, 'dynamic_convexity', 0) > 0:
            n_dyn = self._get_config_attr('dynamic_convexity.samples')
            inp_dyn = self.sample_dynamic_surface(model, n_dyn, force_equator=False)
            loss_dyn, min_eig_dyn = self.compute_convexity_loss(model, inp_dyn)

        if do_symmetry and getattr(w, 'symmetry', 0) > 0:
            n_sym = self._get_config_attr('symmetry.samples')
            inp_sym = self.sample_dynamic_surface(model, n_sym, force_equator=True)
            with tf.GradientTape() as tape_sym:
                tape_sym.watch(inp_sym)
                pred_se_sym = model(inp_sym)
            grads_sym = tape_sym.gradient(pred_se_sym, inp_sym)
            if grads_sym is None: grads_sym = tf.zeros_like(inp_sym)
            loss_sym = tf.reduce_mean(tf.square(grads_sym[:, 2]))

        # 4. Stress Combination
        r_frac = self._get_config_attr('anisotropy_ratio.batch_r_fraction') if mode == 'dual' else 0.0
        loss_stress = (loss_stress_s * (1.0 - r_frac)) + (loss_stress_uni * r_frac)
        
        primary_loss = (getattr(w, 'stress', 1.0) * loss_stress) + \
                       (getattr(w, 'r_value', 0.0) * loss_r) + \
                       (getattr(w, 'batch_convexity', 0.0) * loss_conv) + \
                       (getattr(w, 'dynamic_convexity', 0.0) * loss_dyn) + \
                       (getattr(w, 'symmetry', 0.0) * loss_sym)

        return {
            'primary_loss': primary_loss,
            'l_se': loss_stress,
            'l_r': loss_r,
            'l_conv': loss_conv,
            'l_dyn': loss_dyn,
            'l_sym': loss_sym,
            'min_stat': min_eig_stat,
            'min_dyn': min_eig_dyn
        }
