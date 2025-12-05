import tensorflow as tf
# import tensorflow_probability as tfp
from scipy.stats import qmc
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from .model import HomogeneousYieldModel
from .data_loader import YieldDataLoader
from .utils import save_config
# import sys5

class Trainer:
    def __init__(self, config, fold_idx=None):
        self.config = config
        base_dir = os.path.join(config['training']['save_dir'], config['experiment_name'])
        if fold_idx is not None:
            self.output_dir = os.path.join(base_dir, f"fold_{fold_idx}")
        else:
            self.output_dir = base_dir
        os.makedirs(self.output_dir, exist_ok=True)
        if fold_idx is None or fold_idx == 1:
            save_config(config, base_dir)

        self.model = HomogeneousYieldModel(config)
        print("Initializing weights...")
        self.model(tf.constant(np.zeros((1, 3), dtype=np.float32)))
        # Option: Switch to SGD if needed
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=config['training']['learning_rate'])

        # Store symmetry for dynamic sampler
        self.use_symmetry = config['data'].get('symmetry', True)

        w = config['training']['weights']
        print(f"Active Losses -> Stress: {w['stress']>0}, R-value: {w['r_value']>0}, Convexity: {w['convexity']>0}, Symmetry: {w.get('symmetry', 0.0)>0}")

    def _sample_dynamic_surface(self, n_samples, force_equator=False):
        """Generates random points ON the yield surface."""
        # theta = tf.random.uniform((n_samples,), 0, 2*np.pi)
        # if force_equator:
        #     phi = tf.ones((n_samples,), dtype=tf.float32) * (np.pi / 2.0)
        # else:
        #     max_phi = np.pi / 2.0 if self.use_symmetry else np.pi
        #     phi = tf.random.uniform((n_samples,), 0, max_phi)
        
        # # with TFP
        # dim = 1 if force_equator else 2
        # # This ensures every call to this function returns different points.
        # skip_val = tf.random.uniform(shape=[], minval=0, maxval=2**10, dtype=tf.int32)
        # # Returns values in [0, 1]
        # sobol_samples = tfp.math.qmc.sobol_sample(dim=dim, num_results=n_samples, skip=skip_val)
        # theta = sobol_samples[:, 0] * 2 * np.pi
        # if force_equator:
        #     phi = tf.ones((n_samples,), dtype=tf.float32) * (np.pi / 2.0)
        # else:
        #     # Determine the range for phi
        #     max_phi = np.pi / 2.0 if self.use_symmetry else np.pi
        #     z_values = np.cos(max_phi) + (1.0 - np.cos(max_phi)) * sobol_samples[:, 1]
        #     phi = tf.math.acos(z_values)

        # with QMC
        dim = 1 if force_equator else 2
        sampler = qmc.Sobol(d=dim, scramble=True)
        skip_val = np.random.randint(0, 100000) 
        sampler.fast_forward(skip_val)
        sample = sampler.random(n=n_samples)
        theta = tf.convert_to_tensor(sample[:, 0] * 2 * np.pi, dtype=tf.float32)
        if force_equator:
            phi_np = np.ones((n_samples,)) * (np.pi / 2.0)
        else:
            max_phi = np.pi / 2.0 if self.use_symmetry else np.pi
            z_values = np.cos(max_phi) + (1.0 - np.cos(max_phi)) * sample[:, 1]
            phi_np = np.arccos(z_values)
        phi = tf.convert_to_tensor(phi_np, dtype=tf.float32)

        radius = self.model.predict_radius(theta, phi)
        radius = tf.reshape(radius, [-1])

        s12 = radius * tf.cos(phi)
        r_plane = radius * tf.sin(phi)
        s11 = r_plane * tf.cos(theta)
        s22 = r_plane * tf.sin(theta)
        
        sigma_dynamic = tf.stack([s11, s22, s12], axis=1)
        return tf.stop_gradient(sigma_dynamic)

    # def _compute_convexity_loss(self, inputs):
    #     """Robust Finite Difference Hessian Calculation"""
    #     epsilon = 1e-3
    #     hess_list = []
    #     for i in range(3):
    #         vec = tf.one_hot(i, 3, dtype=tf.float32) * epsilon
            
    #         # x + h
    #         with tf.GradientTape() as t1:
    #             t1.watch(inputs); inp_pos = inputs + vec
    #             pred_pos = self.model(inp_pos)
    #         grad_pos = t1.gradient(pred_pos, inp_pos)
    #         if grad_pos is None: grad_pos = tf.zeros_like(inputs)

    #         # x - h
    #         with tf.GradientTape() as t2:
    #             t2.watch(inputs); inp_neg = inputs - vec
    #             pred_neg = self.model(inp_neg)
    #         grad_neg = t2.gradient(pred_neg, inp_neg)
    #         if grad_neg is None: grad_neg = tf.zeros_like(inputs)
            
    #         hess_col = (grad_pos - grad_neg) / (2.0 * epsilon)
    #         hess_list.append(hess_col)
        
    #     hess_matrix = tf.stack(hess_list, axis=2)
    #     eigs = tf.linalg.eigvalsh(hess_matrix)
    #     return tf.reduce_sum(tf.square(tf.nn.relu(-eigs)))

    def _compute_convexity_loss(self, inputs):
        """
        Calculates Convexity Loss using Automatic Differentiation (Hessian).
        Faster than Finite Differences, requires C2-continuous model graph.
        """
        # Ensure inputs are tracked
        with tf.GradientTape(persistent=True) as tape2:
            tape2.watch(inputs)
            
            with tf.GradientTape() as tape1:
                tape1.watch(inputs)
                val = self.model(inputs)
            
            # First Derivative (Gradient)
            grads = tape1.gradient(val, inputs, unconnected_gradients=tf.UnconnectedGradients.ZERO)

        # Second Derivative (Hessian)
        # batch_jacobian computes partial derivatives of vector 'grads' w.r.t vector 'inputs'
        # Result shape: (Batch, 3, 3)
        hess_matrix = tape2.batch_jacobian(grads, inputs)
        del tape2 # Clean up persistent tape

        # Stability Check (Eigenvalues)
        # A matrix is convex (Positive Definite) if all Eigenvalues > 0
        eigs = tf.linalg.eigvalsh(hess_matrix)
        
        # Penalize negative eigenvalues
        # Square the error to create a strong gradient pushing it back to positive
        return tf.reduce_sum(tf.square(tf.nn.relu(-eigs)))

    @tf.function
    def train_step_dual(self, batch_shape, batch_phys, do_dyn_conv, do_symmetry):
        inp_s, tar_se_s = batch_shape
        inp_p, tar_se_p, tar_r, geo_p, r_mask = batch_phys

        w_stress = self.config['training']['weights']['stress']
        w_r = self.config['training']['weights']['r_value']
        w_conv = self.config['training']['weights']['convexity']
        w_sym = self.config['training']['weights'].get('symmetry', 0.0)
        w_dyn = self.config['training']['weights'].get('dynamic_convexity', 0.0)
        
        n_dyn = self.config['training'].get('dynamic_convexity', {}).get('samples', 1000)
        n_sym = self.config['training'].get('symmetry', {}).get('samples', 1000)

        loss_conv = tf.constant(0.0)
        loss_dyn = tf.constant(0.0)
        loss_sym = tf.constant(0.0)
        
        # Generate Extra Points (Outside Tape)
        if do_dyn_conv and w_dyn > 0:
            inp_dyn = self._sample_dynamic_surface(n_dyn, force_equator=False)
        if do_symmetry and w_sym > 0:
            inp_sym = self._sample_dynamic_surface(n_sym, force_equator=True)

        with tf.GradientTape() as model_tape:
            # 1. Static Convexity
            if w_conv > 0:
                loss_conv = self._compute_convexity_loss(inp_s)
            
            # 2. Dynamic Convexity
            if do_dyn_conv and w_dyn > 0:
                loss_dyn = self._compute_convexity_loss(inp_dyn)

            # 3. Symmetry Loss
            if do_symmetry and w_sym > 0:
                with tf.GradientTape() as tape_sym:
                    tape_sym.watch(inp_sym)
                    pred_se_sym = self.model(inp_sym)
                grads_sym = tape_sym.gradient(pred_se_sym, inp_sym)
                if grads_sym is None: grads_sym = tf.zeros_like(inp_sym)
                loss_sym = tf.reduce_mean(tf.square(grads_sym[:, 2]))

            # 4. Stress Loss
            pred_se_s = self.model(inp_s)
            loss_stress_s = tf.reduce_mean(tf.square(pred_se_s - tar_se_s))

            # 5. R-value Loss
            if w_r > 0:
                with tf.GradientTape() as tape_r:
                    tape_r.watch(inp_p)
                    pred_se_p = self.model(inp_p)
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
                loss_stress_p = tf.reduce_mean(tf.square(pred_se_p - tar_se_p))
            else:
                pred_se_p = self.model(inp_p)
                loss_stress_p = tf.reduce_mean(tf.square(pred_se_p - tar_se_p))

            r_frac = self.config['training'].get('batch_r_fraction', 0.5)
            loss_stress = (loss_stress_s * (1.0 - r_frac)) + (loss_stress_p * r_frac)
            
            total_loss = (w_stress * loss_stress) + (w_r * loss_r) + \
                         (w_conv * loss_conv) + (w_dyn * loss_dyn) + (w_sym * loss_sym)

        model_grads = model_tape.gradient(total_loss, self.model.trainable_variables)
        gnorm_val = tf.linalg.global_norm(model_grads)
        self.optimizer.apply_gradients(zip(model_grads, self.model.trainable_variables))
        
        # Consistent Return Keys
        return {
            'loss': total_loss, 'l_se': loss_stress, 'l_r': loss_r, 
            'l_conv': loss_conv, 'l_dyn': loss_dyn, 'l_sym': loss_sym, 
            'gnorm': gnorm_val
        }

    @tf.function
    def train_step_shape(self, batch_shape, do_dyn_conv, do_symmetry):
        inp_s, tar_se_s = batch_shape
        
        w_stress = self.config['training']['weights']['stress']
        w_conv = self.config['training']['weights']['convexity']
        w_sym = self.config['training']['weights'].get('symmetry', 0.0)
        w_dyn = self.config['training']['weights'].get('dynamic_convexity', 0.0)
        
        n_dyn = self.config['training'].get('dynamic_convexity', {}).get('samples', 1000)
        n_sym = self.config['training'].get('symmetry', {}).get('samples', 1000)

        loss_conv = tf.constant(0.0)
        loss_dyn = tf.constant(0.0)
        loss_sym = tf.constant(0.0)
        
        if do_dyn_conv and w_dyn > 0:
            inp_dyn = self._sample_dynamic_surface(n_dyn, force_equator=False)
        if do_symmetry and w_sym > 0:
            inp_sym = self._sample_dynamic_surface(n_sym, force_equator=True)

        with tf.GradientTape() as model_tape:
            if w_conv > 0:
                loss_conv = self._compute_convexity_loss(inp_s)
            
            if do_dyn_conv and w_dyn > 0:
                loss_dyn = self._compute_convexity_loss(inp_dyn)

            if do_symmetry and w_sym > 0:
                with tf.GradientTape() as tape_sym:
                    tape_sym.watch(inp_sym)
                    pred_se_sym = self.model(inp_sym)
                grads_sym = tape_sym.gradient(pred_se_sym, inp_sym)
                if grads_sym is None: grads_sym = tf.zeros_like(inp_sym)
                loss_sym = tf.reduce_mean(tf.square(grads_sym[:, 2]))

            pred_se_s = self.model(inp_s)
            loss_stress = tf.reduce_mean(tf.square(pred_se_s - tar_se_s))
            
            total_loss = (w_stress * loss_stress) + (w_conv * loss_conv) + (w_dyn * loss_dyn) + (w_sym * loss_sym)

        model_grads = model_tape.gradient(total_loss, self.model.trainable_variables)
        gnorm_val = tf.linalg.global_norm(model_grads)
        self.optimizer.apply_gradients(zip(model_grads, self.model.trainable_variables))
        
        # Consistent Return Keys (l_r is 0)
        return {
            'loss': total_loss, 'l_se': loss_stress, 'l_r': 0.0, 
            'l_conv': loss_conv, 'l_dyn': loss_dyn, 'l_sym': loss_sym, 
            'gnorm': gnorm_val
        }

    @tf.function
    def val_step(self, batch_shape, batch_phys):
        inp_s, tar_se_s = batch_shape
        inp_p, tar_se_p, tar_r, geo_p, r_mask = batch_phys
        
        w_stress = self.config['training']['weights']['stress']
        w_r = self.config['training']['weights']['r_value']
        w_conv = self.config['training']['weights']['convexity']
        
        loss_conv = tf.constant(0.0)
        if w_conv > 0:
             loss_conv = self._compute_convexity_loss(inp_s)
        
        pred_se_s = self.model(inp_s)
        loss_stress_s = tf.reduce_mean(tf.square(pred_se_s - tar_se_s))
        
        loss_r = tf.constant(0.0)
        loss_stress_p = tf.constant(0.0)
        if w_r > 0:
            with tf.GradientTape() as tape:
                tape.watch(inp_p)
                pred_se_p = self.model(inp_p)
            grads = tape.gradient(pred_se_p, inp_p)
            gnorm = tf.math.divide_no_nan(grads, tf.norm(grads, axis=1, keepdims=True) + 1e-8)
            ds_11, ds_22, ds_12 = gnorm[:,0], gnorm[:,1], gnorm[:,2]
            sin2, cos2, sc = geo_p[:,0], geo_p[:,1], geo_p[:,2]
            d_eps_thick = -(ds_11 + ds_22); d_eps_w = ds_11*sin2 + ds_22*cos2 - 2*ds_12*sc
            num = d_eps_width - (tar_r * d_eps_thick); den = tf.sqrt(1.0 + tf.square(tar_r))
            loss_r = tf.reduce_sum(tf.square(tf.math.divide_no_nan(num, den)) * r_mask) / (tf.reduce_sum(r_mask) + 1e-8)
            loss_stress_p = tf.reduce_mean(tf.square(pred_se_p - tar_se_p))
        else:
            pred_se_p = self.model(inp_p)
            loss_stress_p = tf.reduce_mean(tf.square(pred_se_p - tar_se_p))

        r_frac = self.config['training'].get('batch_r_fraction', 0.5)
        loss_stress = (loss_stress_s * (1.0 - r_frac)) + (loss_stress_p * r_frac)
        
        # Validation returns subset of keys
        return {'loss': w_stress*loss_stress + w_r*loss_r, 'l_se': loss_stress, 'l_r': loss_r, 'l_conv': loss_conv}

    def run(self, train_dataset=None, val_dataset=None):
        if train_dataset is None:
            loader = YieldDataLoader(self.config)
            ds_shape, ds_phys, steps = loader.get_dataset()
        else:
            ds_shape, ds_phys, steps = train_dataset 
            
        print(f"Training in: {self.output_dir}")
        
        if ds_phys is not None:
            dataset = tf.data.Dataset.zip((ds_shape, ds_phys)).take(steps)
            mode = 'dual'
        else:
            dataset = ds_shape.take(steps)
            mode = 'shape'

        # Decoupled Config
        conf_dyn = self.config.get('dynamic_convexity', {})
        conf_sym = self.config.get('symmetry', {})
        
        w_dyn = self.config['training']['weights'].get('dynamic_convexity', 0.0)
        w_sym = self.config['training']['weights'].get('symmetry', 0.0)
        
        dyn_interval = conf_dyn.get('interval', 0)
        sym_interval = conf_sym.get('interval', 0)
        dyn_en = conf_dyn.get('enabled', False)
        sym_en = conf_sym.get('enabled', False)
        n_dyn = conf_dyn.get('samples', 0)
        n_sym = conf_sym.get('samples', 0)

        r_interval = self.config['anisotropy_ratio'].get('interval', 0)
        stop_loss = self.config['training'].get('loss_threshold', None)
        
        global_step = 0
        history = []
        best_metric = float('inf')
        
        for epoch in range(1, self.config['training']['epochs'] + 1):
            # Keys MUST match return dict of train_step
            train_metrics = {k: tf.keras.metrics.Mean() for k in ['loss', 'l_se', 'l_r', 'l_conv', 'l_dyn', 'l_sym', 'gnorm']}
            val_metrics = {k: tf.keras.metrics.Mean() for k in ['loss', 'l_se', 'l_r', 'l_conv']}
            
            for batch_data in dataset:
                # Check Intervals
                do_dyn_conv = (dyn_en and w_dyn > 0 and n_dyn > 0) and \
                              (dyn_interval == 0 or global_step % dyn_interval == 0)
                do_symmetry = (sym_en and w_sym > 0 and n_sym > 0) and \
                              (sym_interval == 0 or global_step % sym_interval == 0)
                do_anisotropy = (mode == 'dual') and (r_interval == 0 or global_step % r_interval == 0)

                # if mode == 'dual':
                if do_anisotropy:
                    step_res = self.train_step_dual(batch_data[0], batch_data[1], do_dyn_conv, do_symmetry)
                else:
                    step_res = self.train_step_shape(batch_data, do_dyn_conv, do_symmetry)
                    
                for k, v in step_res.items(): train_metrics[k].update_state(v)
                global_step += 1
            
            if val_dataset:
                for bs, bp in val_dataset:
                    res = self.val_step(bs, bp)
                    for k, v in res.items(): val_metrics[k].update_state(v)

            row = {'epoch': epoch, 'lr': self.optimizer.learning_rate.numpy()}
            for k, v in train_metrics.items(): row[f"train_{k}"] = v.result().numpy()
            if val_dataset:
                for k, v in val_metrics.items(): row[f"val_{k}"] = v.result().numpy()
            
            history.append(row)
            
            if epoch % 5 == 0 or epoch == 1:
                log_str = (f"Epoch {epoch}: Loss {row['train_loss']:.5f} "
                           f"(SE: {row['train_l_se']:.5f}, S-Conv: {row['train_l_conv']:.5f}, "
                           f"D-Sym: {row['train_l_sym']:.5f}, D-Conv: {row['train_l_dyn']:.5f})")
                if mode == 'dual': log_str += f", R: {row['train_l_r']:.5f}"
                # if do_anisotropy: log_str += f", R: {row['train_l_r']:.5f}"
                print(log_str)

            if row['train_loss'] < best_metric:
                best_metric = row['train_loss']
                self.model.save_weights(os.path.join(self.output_dir, "best_model.weights.h5"))
            
            # --- CHECK STOP THRESHOLD ---
            if stop_loss is not None and row['train_loss'] <= stop_loss:
                print(f"\n[Stop] Target training loss {stop_loss} reached at epoch {epoch}.")
                # Save model before exiting
                self.model.save_weights(os.path.join(self.output_dir, "model.weights.h5"))
                break

        pd.DataFrame(history).to_csv(os.path.join(self.output_dir, "loss_history.csv"), index=False)
        return best_metric