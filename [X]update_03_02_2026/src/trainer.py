import tensorflow as tf
from scipy.stats import qmc
import os
import sys
import shutil
import yaml
import pandas as pd
import numpy as np
import pickle
import glob
import time
from .model import HomogeneousYieldModel
from .data_loader import YieldDataLoader
from .config import Config
from .losses import PhysicsLoss

class Trainer:
    def __init__(self, config: Config, config_path=None, resume_path=None, transfer_path=None, fold_idx=None):
        self.config = config
        self.start_epoch = 0
        self.history = []
        self.rng_state = None
        
        # Initialize Loss Calculator
        self.loss_fn = PhysicsLoss(config)
        
        # --- 1. SETUP DIRECTORIES ---
        base_dir = os.path.join(config.training.save_dir, config.experiment_name)
        if fold_idx is not None:
            self.output_dir = os.path.join(base_dir, f"fold_{fold_idx}")
        else:
            self.output_dir = base_dir
        
        self.ckpt_dir = os.path.join(self.output_dir, "checkpoints")

        # --- SAFETY CHECKS (INTERACTIVE) ---
        if not resume_path:
            if os.path.exists(self.output_dir):
                has_history = os.path.exists(os.path.join(self.output_dir, "loss_history.csv"))
                has_weights = os.path.exists(os.path.join(self.output_dir, "best_model.weights.h5"))
                
                if has_history or has_weights:
                    print(f"\n⚠️  WARNING: Output directory '{self.output_dir}' already contains training artifacts.")
                    
                    try:
                        # Attempt interactive input
                        user_choice = input("👉 Overwrite existing data? (y/n): ").lower().strip()
                        do_overwrite = (user_choice == 'y')
                    except (EOFError, RuntimeError):
                        # Fallback to config if input() is not supported by the environment
                        print("📝 Non-interactive environment detected. Using config 'overwrite' setting.")
                        do_overwrite = self.config.training.overwrite

                    if do_overwrite:
                        print(f"🧹 Overwriting existing directory: {self.output_dir}")
                        shutil.rmtree(self.output_dir)
                    else:
                        print("🚫 Run cancelled.")
                        sys.exit()

            os.makedirs(self.output_dir, exist_ok=True)
            os.makedirs(self.ckpt_dir, exist_ok=True)

        # --- 2. INITIALIZE OPTIMIZER ---
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=config.training.learning_rate)

        # --- 3. LOAD CHECKPOINTS / TRANSFER ---
        config_was_modified = False
        
        if resume_path:
            print(f"🔄 Resuming from: {resume_path}", flush=True)
            self._load_checkpoint(resume_path, mode='resume')
            self.output_dir = resume_path
            self.ckpt_dir = os.path.join(self.output_dir, "checkpoints") 
        
        elif transfer_path:
            print(f"🚀 Transfer Learning from: {transfer_path}", flush=True)
            self._load_checkpoint(transfer_path, mode='transfer')
            config_was_modified = True
        
        else:
            print("✨ Starting fresh training...", flush=True)
            self.model = HomogeneousYieldModel(self.config.to_dict())
            self.model(tf.constant(np.zeros((1, 3), dtype=np.float32)))

        # --- 4. SAVE CONFIG FILE ---
        if fold_idx is None or fold_idx == 1:
            target_cfg = os.path.join(base_dir, "config.yaml")
            if resume_path:
                pass 
            elif config_was_modified:
                with open(target_cfg, 'w') as f:
                    yaml.dump(self.config.to_dict(), f, sort_keys=False)
            elif config_path and os.path.exists(config_path):
                shutil.copy2(config_path, target_cfg)
            else:
                with open(target_cfg, 'w') as f:
                    yaml.dump(self.config.to_dict(), f, sort_keys=False)

        # --- 5. LOGGING ---
        self.use_symmetry = config.data.symmetry
        weights = config.training.weights

        # Define active status based on both switches and weights
        stress_active = weights.stress > 0
        r_active = (weights.r_value > 0) and config.anisotropy_ratio.enabled
        dyn_conv_active = (weights.dynamic_convexity > 0) and config.dynamic_convexity.enabled
        sym_active = (weights.symmetry > 0) and config.symmetry.enabled

        print(f"Active Losses -> Stress: {stress_active}, "
              f"R-value: {r_active}, "
              f"Dynamic Convexity: {dyn_conv_active}, "
              f"Symmetry: {sym_active}", flush=True)

    def _save_checkpoint(self, epoch, is_best=False):
        checkpoint_name = "best_model" if is_best else f"ckpt_epoch_{epoch}"
        target_directory = self.output_dir if is_best else self.ckpt_dir

        weights_path = os.path.join(target_directory, f"{checkpoint_name}.weights.h5")
        self.model.save_weights(weights_path)

        try:
            optimizer_weights = self.optimizer.get_weights()
        except AttributeError:
            optimizer_weights = [v.numpy() for v in self.optimizer.variables]

        state_path = os.path.join(target_directory, f"{checkpoint_name}.state.pkl")
        state_dict = {
            'epoch': epoch,
            'optimizer_weights': optimizer_weights, 
            'config': self.config.to_dict(),
            'rng_numpy': np.random.get_state(),
            'history': self.history
        }
        
        with open(state_path, 'wb') as f:
            pickle.dump(state_dict, f)

    def _load_checkpoint(self, path, mode):
        if mode == 'resume':
            if os.path.isdir(path):
                checkpoint_dir = os.path.join(path, "checkpoints")
                states = glob.glob(os.path.join(checkpoint_dir, "ckpt_epoch_*.state.pkl"))
                if not states:
                    root_states = glob.glob(os.path.join(path, "*.state.pkl"))
                    if not root_states: raise FileNotFoundError(f"No checkpoint found in {path}")
                    states = root_states
                latest_state = max(states, key=os.path.getctime)
                state_path = latest_state
                weights_path = latest_state.replace(".state.pkl", ".weights.h5")
            else:
                raise ValueError("Provide FOLDER path for resume.")
        elif mode == 'transfer':
            weights_path = path if path.endswith(".h5") else path + ".weights.h5"
            state_path = weights_path.replace(".weights.h5", ".state.pkl")

        with open(state_path, 'rb') as f:
            saved_state = pickle.load(f)

        saved_model_config = saved_state['config']['model']
        self.config.model.hidden_layers = saved_model_config['hidden_layers']
        self.config.model.activation = saved_model_config['activation']
        if 'ref_stress' in saved_model_config:
             self.config.model.ref_stress = saved_model_config['ref_stress']
        
        self.model = HomogeneousYieldModel(self.config.to_dict())
        self.model(tf.constant(np.zeros((1, 3), dtype=np.float32))) 
        self.model.load_weights(weights_path)

        if mode == 'resume':
            self.start_epoch = saved_state['epoch']
            self.history = saved_state.get('history', [])
            np.random.set_state(saved_state['rng_numpy'])

    def validate_on_path(self):
        _, physics_data = self.loader._generate_raw_data(needs_physics=True)
        inputs_physics, target_stress_physics, target_r_values, geometry_physics = physics_data
        
        if len(inputs_physics) == 0: 
            return 0.0, 0.0

        inputs_physics = tf.convert_to_tensor(inputs_physics)
        target_stress_physics = tf.convert_to_tensor(target_stress_physics)
        target_r_values = tf.convert_to_tensor(target_r_values)
        geometry_physics = tf.convert_to_tensor(geometry_physics)

        with tf.GradientTape() as tape:
            tape.watch(inputs_physics)
            predicted_potential = self.model(inputs_physics)
        
        stress_error = tf.reduce_mean(tf.abs(predicted_potential - target_stress_physics))
        
        gradients = tape.gradient(predicted_potential, inputs_physics)
        norm_gradients = tf.math.divide_no_nan(
            gradients, 
            tf.norm(gradients, axis=1, keepdims=True) + 1e-8
        )
        
        ds11 = norm_gradients[:, 0]
        ds22 = norm_gradients[:, 1]
        ds12 = norm_gradients[:, 2]
        
        sin2 = geometry_physics[:, 0]
        cos2 = geometry_physics[:, 1]
        sc_term = geometry_physics[:, 2]
        
        d_thick = -(ds11 + ds22)
        d_width = ds11 * sin2 + ds22 * cos2 - ds12 * sc_term
        
        predicted_r = d_width / (d_thick + 1e-8)
        r_value_error = tf.reduce_mean(tf.abs(predicted_r - target_r_values))
        
        return float(stress_error), float(r_value_error)

    @tf.function
    def train_step_dual(self, batch_shape, batch_physics, run_dynamic_convexity, run_symmetry):
        inputs_shape, target_stress_shape = batch_shape
        # Note: batch_physics tuple structure: 
        # (inputs, target_stress, target_r, geometry, mask, target_stress_uni)
        # PhysicsLoss expects: (inputs_shape, inputs_physics) and targets tuple
        
        inputs_physics = batch_physics[0]
        target_stress_physics = batch_physics[5] # The 6th element is stress target
        target_r_physics = batch_physics[2]
        geometry_physics = batch_physics[3]
        
        # Prepare arguments for PhysicsLoss
        inputs_tuple = (inputs_shape, inputs_physics)
        targets_tuple = (target_stress_shape, target_stress_physics, target_r_physics, geometry_physics)
        weights = self.config.training.weights
        
        with tf.GradientTape() as tape:
            # Delegate all calculation to PhysicsLoss
            losses = self.loss_fn.calculate_losses(
                self.model, 
                inputs_tuple, 
                targets_tuple, 
                weights, 
                run_convexity=run_dynamic_convexity, 
                run_symmetry=run_symmetry
            )
            
            total_loss = losses['total_loss']
            
            # Apply gradient norm penalty if enabled (placeholder for future implementation)
            if hasattr(weights, 'gradient_norm') and weights.gradient_norm > 0:
                pass

        gradients = tape.gradient(total_loss, self.model.trainable_variables)
        # gnorm_val = tf.linalg.global_norm(gradients)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        
        # Map PhysicsLoss outputs to the dictionary structure Trainer expects for logging
        return {
            'loss': losses['total_loss'], 
            'l_se': losses['loss_stress_total'], 
            'l_r': losses['loss_r_value'], 
            'l_dyn': losses['loss_convexity'], 
            'l_sym': losses['loss_symmetry'], 
            'min_dyn': losses['min_eigenvalue']
        }

    @tf.function
    def train_step_shape(self, batch_shape, run_dynamic_convexity, run_symmetry):
        inputs_shape, target_stress_shape = batch_shape
        
        # Create dummy physics tensors to satisfy the function signature
        dummy_phys_in = tf.zeros((0, 3))
        dummy_phys_tar = tf.zeros((0, 1))
        dummy_r = tf.zeros((0, 1))
        dummy_geo = tf.zeros((0, 3))
        
        inputs_tuple = (inputs_shape, dummy_phys_in)
        targets_tuple = (target_stress_shape, dummy_phys_tar, dummy_r, dummy_geo)
        weights = self.config.training.weights
        
        with tf.GradientTape() as tape:
            losses = self.loss_fn.calculate_losses(
                self.model, 
                inputs_tuple, 
                targets_tuple, 
                weights, 
                run_convexity=run_dynamic_convexity, 
                run_symmetry=run_symmetry
            )
            total_loss = losses['total_loss']
                
        gradients = tape.gradient(total_loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        
        return {
            'loss': losses['total_loss'], 
            'l_se': losses['loss_stress_total'], 
            'l_r': 0.0, 
            'l_dyn': losses['loss_convexity'], 
            'l_sym': losses['loss_symmetry'], 
            'min_dyn': losses['min_eigenvalue']
        }

    @tf.function
    def val_step(self, batch_shape, batch_physics):
        inputs_shape, target_stress_shape = batch_shape
        
        inputs_physics = batch_physics[0]
        target_stress_physics = batch_physics[5]
        target_r_physics = batch_physics[2]
        geometry_physics = batch_physics[3]
        
        inputs_tuple = (inputs_shape, inputs_physics)
        targets_tuple = (target_stress_shape, target_stress_physics, target_r_physics, geometry_physics)
        weights = self.config.training.weights
        
        losses = self.loss_fn.calculate_losses(
            self.model,
            inputs_tuple,
            targets_tuple,
            weights,
            run_convexity=False,
            run_symmetry=False
        )
        
        return {
            'loss': losses['total_loss'], 
            'l_se': losses['loss_stress_total'], 
            'l_r': losses['loss_r_value']
        }
    
    def train(self, train_dataset=None, val_dataset=None):
        """ Consolidated Training Loop. """
        if train_dataset is None:
            self.loader = YieldDataLoader(self.config)
            ds_shape, ds_physics, steps = self.loader.get_dataset()
        else:
            ds_shape, ds_physics, steps = train_dataset
            
        anisotropy_config = self.config.anisotropy_ratio
        weights = self.config.training.weights
        training_config = self.config.training
        
        use_dual_stream = (
            (self.config.data.samples.get('uniaxial', 0) > 0) and 
            (weights.r_value > 0) and 
            anisotropy_config.enabled
        )
        
        if use_dual_stream:
            training_dataset = tf.data.Dataset.zip((ds_shape, ds_physics)).take(steps)
            training_mode = 'dual'
        else:
            training_dataset = ds_shape.take(steps)
            training_mode = 'shape'

        global_step = self.start_epoch * steps
        session_start = time.time()
        previous_time = float(self.history[-1].get('time', 0.0)) if self.history else 0.0
        best_metric_value = float('inf')

        for epoch in range(self.start_epoch + 1, training_config.epochs + 1):
            epoch_metrics_list = []
            run_dynamic = (self.config.dynamic_convexity.enabled and 
                          global_step % self.config.dynamic_convexity.interval == 0)
            run_symmetry = (self.config.symmetry.enabled and 
                           global_step % self.config.symmetry.interval == 0)

            for batch in training_dataset:
                if training_mode == 'dual':
                    step_results = self.train_step_dual(batch[0], batch[1], run_dynamic, run_symmetry)
                else:
                    step_results = self.train_step_shape(batch, run_dynamic, run_symmetry)
                
                step_results_numeric = {k: float(v) for k, v in step_results.items()}
                epoch_metrics_list.append(step_results_numeric)
                global_step += 1
            
            average_metrics = pd.DataFrame(epoch_metrics_list).mean().to_dict()
            
            validation_stress_error = 0.0
            validation_r_error = 0.0
            pass_physics_se = True
            pass_physics_r = True

            if anisotropy_config.enabled and weights.r_value > 0:
                validation_stress_error, validation_r_error = self.validate_on_path()
                
                if training_config.loss_threshold is not None:
                    pass_physics_se = (validation_stress_error <= training_config.loss_threshold)
                
                if training_config.r_threshold is not None:
                    pass_physics_r = (validation_r_error <= training_config.r_threshold)

            log_entry = {
                'epoch': epoch,
                'time': previous_time + (time.time() - session_start),
                'loss': average_metrics['loss'],
                'l_se': average_metrics['l_se'],
                'l_se_val': validation_stress_error,
                'l_r_val': validation_r_error,
                'l_r_train': average_metrics.get('l_r', 0.0),
                'l_dyn': average_metrics.get('l_dyn', 0.0),
                'min_dyn': average_metrics.get('min_dyn', 0.0)
            }
            self.history.append(log_entry)
            
            pd.DataFrame(self.history).to_csv(os.path.join(self.output_dir, "loss_history.csv"), index=False)

            if epoch % training_config.print_interval == 0 or epoch == 1:
                print(
                    f"Epoch {epoch:04d} | Loss: {log_entry['loss']:.2e} | "
                    f"SE(Val): {log_entry['l_se_val']:.2e} | "
                    f"R(Val): {log_entry['l_r_val']:.4f} | "
                    f"MinEig: {log_entry['min_dyn']:.2e}", 
                    flush=True
                )

            pass_loss_threshold = (training_config.loss_threshold is None) or \
                                  (average_metrics['loss'] <= training_config.loss_threshold)
            
            pass_convexity = True
            if training_config.convexity_threshold is not None:
                pass_convexity = (average_metrics.get('min_dyn', 0.0) >= training_config.convexity_threshold)

            if pass_loss_threshold and pass_convexity and pass_physics_se and pass_physics_r:
                if training_config.loss_threshold or training_config.r_threshold:
                    self._save_checkpoint(epoch, is_best=True)
                    print(f"✅ Targets reached at epoch {epoch}. Stopping training.")
                    break
                
            if average_metrics['loss'] < best_metric_value:
                best_metric_value = average_metrics['loss']
                self._save_checkpoint(epoch, is_best=True)
            
            if epoch % training_config.checkpoint_interval == 0:
                self._save_checkpoint(epoch)

        return best_metric_value
