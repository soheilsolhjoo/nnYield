import tensorflow as tf
from scipy.stats import qmc
import os
import pandas as pd
import numpy as np
import pickle
import glob
from .model import HomogeneousYieldModel
from .data_loader import YieldDataLoader
from .config import Config  # Import the strictly typed Config class

class Trainer:
    def __init__(self, config: Config, resume_path=None, transfer_path=None, fold_idx=None):
        self.config = config
        self.start_epoch = 0
        self.history = []
        self.rng_state = None
        
        # --- 1. SETUP DIRECTORIES ---
        base_dir = os.path.join(config.training.save_dir, config.experiment_name)
        if fold_idx is not None:
            self.output_dir = os.path.join(base_dir, f"fold_{fold_idx}")
        else:
            self.output_dir = base_dir
        
        # Define Checkpoint Subfolder
        self.ckpt_dir = os.path.join(self.output_dir, "checkpoints")

        # --- SAFETY CHECK & CREATION ---
        if not resume_path:
            # If we are NOT resuming, check if folder exists and is populated
            if os.path.exists(self.output_dir):
                has_history = os.path.exists(os.path.join(self.output_dir, "loss_history.csv"))
                has_weights = os.path.exists(os.path.join(self.output_dir, "best_model.weights.h5"))
                
                if has_history or has_weights:
                    raise FileExistsError(
                        f"⛔ Output directory '{self.output_dir}' already contains training data.\n"
                        "   Action required: Change 'experiment_name' in config, delete the folder, or use --resume."
                    )

            os.makedirs(self.output_dir, exist_ok=True)
            os.makedirs(self.ckpt_dir, exist_ok=True) 
            
            # Save config for fresh run (only if main run or fold 1)
            if fold_idx is None or fold_idx == 1:
                with open(os.path.join(base_dir, "config.yaml"), 'w') as f:
                    import yaml
                    yaml.dump(config.to_dict(), f)

        # --- 2. INITIALIZE OPTIMIZER ---
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=config.training.learning_rate)

        # --- 3. HANDLE CHECKPOINTS ---
        if resume_path:
            print(f"🔄 Resuming from: {resume_path}", flush=True)
            self._load_checkpoint(resume_path, mode='resume')
            self.output_dir = resume_path
            self.ckpt_dir = os.path.join(self.output_dir, "checkpoints") 
        
        elif transfer_path:
            print(f"🚀 Transfer Learning from: {transfer_path}", flush=True)
            self._load_checkpoint(transfer_path, mode='transfer')
        
        else:
            print("✨ Starting fresh training...", flush=True)
            self.model = HomogeneousYieldModel(self.config.to_dict())
            self.model(tf.constant(np.zeros((1, 3), dtype=np.float32)))

        # --- 4. SETUP UTILS ---
        self.use_symmetry = config.data.symmetry
        w = config.training.weights
        print(f"Active Losses -> Stress: {w.stress>0}, R-value: {w.r_value>0}, "
              f"Convexity: {w.convexity>0}, Symmetry: {w.symmetry>0}", flush=True)
        
    def _save_checkpoint(self, epoch, is_best=False):
        if is_best:
            name = "best_model"
            target_dir = self.output_dir
        else:
            name = f"ckpt_epoch_{epoch}"
            target_dir = self.ckpt_dir

        weights_path = os.path.join(target_dir, f"{name}.weights.h5")
        self.model.save_weights(weights_path)

        try:
            opt_weights = self.optimizer.get_weights()
        except AttributeError:
            opt_weights = [v.numpy() for v in self.optimizer.variables]

        state_path = os.path.join(target_dir, f"{name}.state.pkl")
        state_dict = {
            'epoch': epoch,
            'optimizer_weights': opt_weights, 
            'config': self.config.to_dict(),
            'rng_numpy': np.random.get_state(),
            'history': self.history
        }
        
        with open(state_path, 'wb') as f:
            pickle.dump(state_dict, f)
        
        if not is_best:
            print(f"Saved checkpoint to {weights_path}", flush=True)
        
    def _load_checkpoint(self, path, mode):
        if mode == 'resume':
            if os.path.isdir(path):
                # 1. Look in checkpoints/ subfolder first (Preferred for Resume)
                ckpt_dir = os.path.join(path, "checkpoints")
                states = glob.glob(os.path.join(ckpt_dir, "ckpt_epoch_*.state.pkl"))
                
                # 2. Look in root if subfolder empty (Legacy or Best Model only)
                if not states:
                    root_states = glob.glob(os.path.join(path, "*.state.pkl"))
                    # Filter to avoid accidentally picking up best_model if we want latest epoch
                    # But if only best_model exists, we use it.
                    if not root_states:
                        raise FileNotFoundError(f"No checkpoint states found in {path} or {ckpt_dir}")
                    states = root_states

                latest_state = max(states, key=os.path.getctime)
                state_path = latest_state
                weights_path = latest_state.replace(".state.pkl", ".weights.h5")
            else:
                raise ValueError("For --resume, provide the FOLDER path.")

        elif mode == 'transfer':
            if path.endswith(".weights.h5"):
                weights_path = path
                state_path = path.replace(".weights.h5", ".state.pkl")
            elif path.endswith(".state.pkl"):
                state_path = path
                weights_path = path.replace(".state.pkl", ".weights.h5")
            else:
                weights_path = path
                state_path = path + ".state.pkl"
            
            if not os.path.exists(state_path):
                print("⚠️ Warning: No state file found. Assuming architecture matches config.", flush=True)
                self.model = HomogeneousYieldModel(self.config.to_dict())
                self.model(tf.constant(np.zeros((1, 3), dtype=np.float32)))
                self.model.load_weights(weights_path)
                return

        with open(state_path, 'rb') as f:
            saved_state = pickle.load(f)

        saved_config_dict = saved_state['config']
        self.config.model.hidden_layers = saved_config_dict['model']['hidden_layers']
        self.config.model.activation = saved_config_dict['model']['activation']
        
        print(f"🏗️ Rebuilding model: {self.config.model.hidden_layers}", flush=True)
        self.model = HomogeneousYieldModel(self.config.to_dict())
        self.model(tf.constant(np.zeros((1, 3), dtype=np.float32))) 

        print(f"📥 Loading weights from {os.path.basename(weights_path)}...", flush=True)
        self.model.load_weights(weights_path)

        if mode == 'resume':
            self.start_epoch = saved_state['epoch']
            print(f"⏱️ Resuming from Epoch {self.start_epoch}", flush=True)

            zero_grad = [tf.zeros_like(w) for w in self.model.trainable_variables]
            self.optimizer.apply_gradients(zip(zero_grad, self.model.trainable_variables))
            
            try:
                self.optimizer.set_weights(saved_state['optimizer_weights'])
            except (AttributeError, ValueError):
                print("⚠️ Warning: Manual optimizer restore triggered.", flush=True)
                opt_vars = self.optimizer.variables
                saved_vars = saved_state['optimizer_weights']
                if len(opt_vars) == len(saved_vars):
                    for v, val in zip(opt_vars, saved_vars):
                        v.assign(val)

            self.history = saved_state.get('history', [])
            if 'rng_numpy' in saved_state:
                np.random.set_state(saved_state['rng_numpy'])

        elif mode == 'transfer':
            print("✅ Transfer complete. Fresh optimizer.", flush=True)
    
    def _sample_dynamic_surface(self, n_samples, force_equator=False):
        """Generates random points ON the yield surface."""
        dim = 1 if force_equator else 2
        sampler = qmc.Sobol(d=dim, scramble=True)
        # Skip logic to ensure randomness across steps
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

    def _compute_convexity_loss(self, inputs):
        """Calculates Convexity Loss using Automatic Differentiation (Hessian)."""
        with tf.GradientTape(persistent=True) as tape2:
            tape2.watch(inputs)
            with tf.GradientTape() as tape1:
                tape1.watch(inputs)
                val = self.model(inputs)
            grads = tape1.gradient(val, inputs, unconnected_gradients=tf.UnconnectedGradients.ZERO)

        hess_matrix = tape2.batch_jacobian(grads, inputs)
        del tape2 
        eigs = tf.linalg.eigvalsh(hess_matrix)
        return tf.reduce_sum(tf.square(tf.nn.relu(-eigs)))

    @tf.function
    def train_step_dual(self, batch_shape, batch_phys, do_dyn_conv, do_symmetry):
        inp_s, tar_se_s = batch_shape
        inp_p, tar_se_p, tar_r, geo_p, r_mask = batch_phys
        w = self.config.training.weights
        
        # Setup nested tape for second-order derivative (Gradient Penalty)
        with tf.GradientTape() as tape_outer:
            with tf.GradientTape() as tape_inner:
                
                loss_conv = tf.constant(0.0)
                loss_dyn = tf.constant(0.0)
                loss_sym = tf.constant(0.0)
                
                # Dynamic Sampling
                if do_dyn_conv and w.dynamic_convexity > 0:
                    n_dyn = self.config.dynamic_convexity.samples
                    inp_dyn = self._sample_dynamic_surface(n_dyn, force_equator=False)
                
                if do_symmetry and w.symmetry > 0:
                    n_sym = self.config.symmetry.samples
                    inp_sym = self._sample_dynamic_surface(n_sym, force_equator=True)

                # 1. Static Convexity
                if w.convexity > 0:
                    loss_conv = self._compute_convexity_loss(inp_s)
                
                # 2. Dynamic Convexity
                if do_dyn_conv and w.dynamic_convexity > 0:
                    loss_dyn = self._compute_convexity_loss(inp_dyn)

                # 3. Symmetry Loss
                if do_symmetry and w.symmetry > 0:
                    with tf.GradientTape() as tape_sym:
                        tape_sym.watch(inp_sym)
                        pred_se_sym = self.model(inp_sym)
                    grads_sym = tape_sym.gradient(pred_se_sym, inp_sym)
                    if grads_sym is None: grads_sym = tf.zeros_like(inp_sym)
                    loss_sym = tf.reduce_mean(tf.square(grads_sym[:, 2]))

                # 4. Stress Loss (Shape)
                pred_se_s = self.model(inp_s)
                loss_stress_s = tf.reduce_mean(tf.square(pred_se_s - tar_se_s))

                # 5. R-value & Stress (Phys)
                if w.r_value > 0:
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
                    loss_r = tf.constant(0.0)

                # Combine Primary Losses
                r_frac = self.config.anisotropy_ratio.batch_r_fraction
                loss_stress = (loss_stress_s * (1.0 - r_frac)) + (loss_stress_p * r_frac)
                
                primary_loss = (w.stress * loss_stress) + (w.r_value * loss_r) + \
                               (w.convexity * loss_conv) + (w.dynamic_convexity * loss_dyn) + \
                               (w.symmetry * loss_sym)
            
            # --- GRADIENT PENALTY LOGIC ---
            # 1. Calculate gradients of Primary Loss w.r.t weights
            grads_primary = tape_inner.gradient(primary_loss, self.model.trainable_variables)
            
            # Handle None gradients safely
            grads_primary_safe = [g if g is not None else tf.zeros_like(v) 
                                  for g, v in zip(grads_primary, self.model.trainable_variables)]
            
            # 2. Compute Norm
            gnorm_val = tf.linalg.global_norm(grads_primary_safe)
            
            # 3. Add to Total Loss (Weighted)
            total_loss = primary_loss
            if w.gradient_norm > 0:
                total_loss += (w.gradient_norm * gnorm_val)

        # 4. Compute Final Gradients (Second Derivative)
        final_grads = tape_outer.gradient(total_loss, self.model.trainable_variables)
        final_grads_safe = [g if g is not None else tf.zeros_like(v) 
                            for g, v in zip(final_grads, self.model.trainable_variables)]
        
        self.optimizer.apply_gradients(zip(final_grads_safe, self.model.trainable_variables))
        
        return {
            'loss': primary_loss, # Log the physics loss, not the penalized one
            'l_se': loss_stress, 'l_r': loss_r, 
            'l_conv': loss_conv, 'l_dyn': loss_dyn, 'l_sym': loss_sym, 
            'gnorm': gnorm_val # Log the norm we calculated
        }

    @tf.function
    def train_step_shape(self, batch_shape, do_dyn_conv, do_symmetry):
        inp_s, tar_se_s = batch_shape
        w = self.config.training.weights

        with tf.GradientTape() as tape_outer:
            with tf.GradientTape() as tape_inner:
                loss_conv = tf.constant(0.0)
                loss_dyn = tf.constant(0.0)
                loss_sym = tf.constant(0.0)
                
                if do_dyn_conv and w.dynamic_convexity > 0:
                    n_dyn = self.config.dynamic_convexity.samples
                    inp_dyn = self._sample_dynamic_surface(n_dyn, force_equator=False)
                
                if do_symmetry and w.symmetry > 0:
                    n_sym = self.config.symmetry.samples
                    inp_sym = self._sample_dynamic_surface(n_sym, force_equator=True)

                if w.convexity > 0:
                    loss_conv = self._compute_convexity_loss(inp_s)
                
                if do_dyn_conv and w.dynamic_convexity > 0:
                    loss_dyn = self._compute_convexity_loss(inp_dyn)

                if do_symmetry and w.symmetry > 0:
                    with tf.GradientTape() as tape_sym:
                        tape_sym.watch(inp_sym)
                        pred_se_sym = self.model(inp_sym)
                    grads_sym = tape_sym.gradient(pred_se_sym, inp_sym)
                    if grads_sym is None: grads_sym = tf.zeros_like(inp_sym)
                    loss_sym = tf.reduce_mean(tf.square(grads_sym[:, 2]))

                pred_se_s = self.model(inp_s)
                loss_stress = tf.reduce_mean(tf.square(pred_se_s - tar_se_s))
                
                primary_loss = (w.stress * loss_stress) + (w.convexity * loss_conv) + \
                               (w.dynamic_convexity * loss_dyn) + (w.symmetry * loss_sym)

            # Gradient Penalty Logic
            grads_primary = tape_inner.gradient(primary_loss, self.model.trainable_variables)
            grads_primary_safe = [g if g is not None else tf.zeros_like(v) 
                                  for g, v in zip(grads_primary, self.model.trainable_variables)]
            
            gnorm_val = tf.linalg.global_norm(grads_primary_safe)
            
            total_loss = primary_loss
            if w.gradient_norm > 0:
                total_loss += (w.gradient_norm * gnorm_val)

        final_grads = tape_outer.gradient(total_loss, self.model.trainable_variables)
        final_grads_safe = [g if g is not None else tf.zeros_like(v) 
                            for g, v in zip(final_grads, self.model.trainable_variables)]
        
        self.optimizer.apply_gradients(zip(final_grads_safe, self.model.trainable_variables))
        
        return {
            'loss': primary_loss, 'l_se': loss_stress, 'l_r': 0.0, 
            'l_conv': loss_conv, 'l_dyn': loss_dyn, 'l_sym': loss_sym, 
            'gnorm': gnorm_val
        }

    @tf.function
    def val_step(self, batch_shape, batch_phys):
        inp_s, tar_se_s = batch_shape
        inp_p, tar_se_p, tar_r, geo_p, r_mask = batch_phys
        
        # Access weights via dot notation
        w = self.config.training.weights
        
        # 1. Static Convexity Check
        loss_conv = tf.constant(0.0)
        if w.convexity > 0:
             loss_conv = self._compute_convexity_loss(inp_s)
        
        # 2. Shape Stress Loss
        pred_se_s = self.model(inp_s)
        loss_stress_s = tf.reduce_mean(tf.square(pred_se_s - tar_se_s))
        
        loss_r = tf.constant(0.0)
        loss_stress_p = tf.constant(0.0)
        
        # 3. Anisotropy / Physical Loss
        if w.r_value > 0:
            with tf.GradientTape() as tape:
                tape.watch(inp_p)
                pred_se_p = self.model(inp_p)
            
            # Calculate gradients for R-value (same logic as training)
            grads = tape.gradient(pred_se_p, inp_p)
            gnorm = tf.math.divide_no_nan(grads, tf.norm(grads, axis=1, keepdims=True) + 1e-8)
            
            ds_11, ds_22, ds_12 = gnorm[:,0], gnorm[:,1], gnorm[:,2]
            sin2, cos2, sc = geo_p[:,0], geo_p[:,1], geo_p[:,2]
            
            d_eps_thick = -(ds_11 + ds_22)
            d_eps_width = ds_11*sin2 + ds_22*cos2 - 2*ds_12*sc
            
            numerator = d_eps_width - (tar_r * d_eps_thick)
            denominator = tf.sqrt(1.0 + tf.square(tar_r))
            geo_error = tf.math.divide_no_nan(numerator, denominator)
            
            # Masked Mean Squared Error for R-values
            loss_r = tf.reduce_sum(tf.square(geo_error) * r_mask) / (tf.reduce_sum(r_mask) + 1e-8)
            loss_stress_p = tf.reduce_mean(tf.square(pred_se_p - tar_se_p))
        else:
            # If R-value weight is 0, just compute stress loss on physical batch
            pred_se_p = self.model(inp_p)
            loss_stress_p = tf.reduce_mean(tf.square(pred_se_p - tar_se_p))

        # 4. Combine Stress Losses
        r_frac = self.config.anisotropy_ratio.batch_r_fraction
        loss_stress = (loss_stress_s * (1.0 - r_frac)) + (loss_stress_p * r_frac)
        
        # Total Validation Loss
        # Note: We typically don't run dynamic checks (convexity/symmetry) in validation for speed
        total_loss = (w.stress * loss_stress) + (w.r_value * loss_r) + (w.convexity * loss_conv)
        
        return {
            'loss': total_loss, 
            'l_se': loss_stress, 
            'l_r': loss_r, 
            'l_conv': loss_conv
        }
        
    def run(self, train_dataset=None, val_dataset=None):
        if train_dataset is None:
            loader = YieldDataLoader(self.config.to_dict())
            ds_shape, ds_phys, steps = loader.get_dataset()
        else:
            ds_shape, ds_phys, steps = train_dataset 
            
        print(f"Training Output Directory: {self.output_dir}", flush=True)
        
        # --- DETERMINE MODE ---
        n_uni = self.config.data.samples.get('uniaxial', 0)
        w_r = self.config.training.weights.r_value
        ani_config = self.config.anisotropy_ratio
        
        use_dual_stream = (n_uni > 0) and (w_r > 0) and (ani_config.batch_r_fraction > 0) and ani_config.enabled

        if ds_phys is not None and use_dual_stream:
            dataset = tf.data.Dataset.zip((ds_shape, ds_phys)).take(steps)
            mode = 'dual'
            print("🚀 Mode: DUAL STREAM (Shape + Anisotropy)", flush=True)
        else:
            dataset = ds_shape.take(steps)
            mode = 'shape'
            print("🚀 Mode: SHAPE ONLY", flush=True)

        conf_dyn = self.config.dynamic_convexity
        conf_sym = self.config.symmetry
        conf_ani = self.config.anisotropy_ratio
        w = self.config.training.weights
        
        # --- STOPPING THRESHOLDS ---
        stop_loss = self.config.training.loss_threshold
        stop_conv = self.config.training.convexity_threshold
        stop_gnorm = self.config.training.gnorm_threshold
        stop_r = self.config.training.r_threshold
        
        global_step = self.start_epoch * steps 
        best_metric = float('inf')
        
        # --- TRAINING LOOP ---
        for epoch in range(self.start_epoch + 1, self.config.training.epochs + 1):
            
            train_metrics = {k: tf.keras.metrics.Mean() for k in ['loss', 'l_se', 'l_r', 'l_conv', 'l_dyn', 'l_sym', 'gnorm']}
            
            for batch_data in dataset:
                do_dyn_conv = (conf_dyn.enabled and w.dynamic_convexity > 0) and \
                              (conf_dyn.interval == 0 or global_step % conf_dyn.interval == 0)
                
                do_symmetry = (conf_sym.enabled and w.symmetry > 0) and \
                              (conf_sym.interval == 0 or global_step % conf_sym.interval == 0)
                
                run_r_step_now = (conf_ani.interval == 0) or (global_step % conf_ani.interval == 0)

                if mode == 'dual':
                    if run_r_step_now:
                        step_res = self.train_step_dual(batch_data[0], batch_data[1], do_dyn_conv, do_symmetry)
                    else:
                        step_res = self.train_step_shape(batch_data[0], do_dyn_conv, do_symmetry)
                else:
                    step_res = self.train_step_shape(batch_data, do_dyn_conv, do_symmetry)
                    
                for k, v in step_res.items(): train_metrics[k].update_state(v)
                global_step += 1
            
            row = {'epoch': epoch, 'lr': self.optimizer.learning_rate.numpy()}
            for k, v in train_metrics.items(): row[f"train_{k}"] = v.result().numpy()
            
            self.history.append(row)
            
            if epoch % 5 == 0 or epoch == 1:
                log_str = (f"Ep {epoch}: Loss {row['train_loss']:.5f} | "
                           f"SE: {row['train_l_se']:.5f} | R: {row['train_l_r']:.5f} | "
                           f"Cv: {row['train_l_conv']:.1e} | D-Cv: {row['train_l_dyn']:.1e} | "
                           f"Sym: {row['train_l_sym']:.1e} | G: {row['train_gnorm']:.5f}")
                print(log_str, flush=True)

            pd.DataFrame(self.history).to_csv(os.path.join(self.output_dir, "loss_history.csv"), index=False)

            if row['train_loss'] < best_metric:
                best_metric = row['train_loss']
                self._save_checkpoint(epoch, is_best=True)
            
            ckpt_interval = self.config.training.checkpoint_interval
            if ckpt_interval > 0 and epoch % ckpt_interval == 0:
                self._save_checkpoint(epoch, is_best=False)

            # --- 4-WAY STOPPING CONDITION ---
            # 1. Total Loss
            pass_loss = (stop_loss is None) or (row['train_loss'] <= stop_loss)
            
            # 2. Convexity (Safety)
            pass_conv = (stop_conv is None) or \
                        ((row['train_l_conv'] <= stop_conv) and (row['train_l_dyn'] <= stop_conv))
            
            # 3. Gradient Norm (Stability)
            pass_gnorm = (stop_gnorm is None) or (row['train_gnorm'] <= stop_gnorm)
            
            # 4. R-value (Accuracy - Only if weight > 0)
            pass_r = True
            if stop_r is not None and w.r_value > 0:
                pass_r = (row['train_l_r'] <= stop_r)

            # Check if ANY threshold is actually set to control the stop
            any_limit_set = (stop_loss or stop_conv or stop_gnorm or stop_r)

            if any_limit_set and pass_loss and pass_conv and pass_gnorm and pass_r:
                print(f"\n[Stop] All targets reached at epoch {epoch}.", flush=True)
                print(f"       Loss: {row['train_loss']:.5f} (Limit: {stop_loss})")
                if stop_conv: print(f"       Conv (Stat/Dyn): {row['train_l_conv']:.2e} / {row['train_l_dyn']:.2e} (Limit: {stop_conv})")
                if stop_gnorm: print(f"       Gnorm: {row['train_gnorm']:.2f} (Limit: {stop_gnorm})")
                if stop_r and w.r_value > 0: print(f"       R-val: {row['train_l_r']:.5f} (Limit: {stop_r})")
                
                self._save_checkpoint(epoch, is_best=True)
                break

        return best_metric