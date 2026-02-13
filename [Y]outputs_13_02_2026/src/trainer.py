import tensorflow as tf
from scipy.stats import qmc
import os
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
from .checkpoint import CheckpointManager

class Trainer:
    """
    Core training engine. 
    Refactored to handle Data Snapping for exact continuity on resume.
    """
    def __init__(self, config: Config, config_path=None, resume_path=None, transfer_path=None, fold_idx=None):
        self.config = config
        self.start_epoch = 0
        self.history = []
        
        # 1. SETUP DIRECTORIES
        base_dir = os.path.join(config.training.save_dir, config.experiment_name)
        if fold_idx is not None:
            self.output_dir = os.path.join(base_dir, f"fold_{fold_idx}")
        else:
            self.output_dir = base_dir
        
        self.ckpt_dir = os.path.join(self.output_dir, "checkpoints")
        self.checkpoint_manager = CheckpointManager(self.output_dir, self.ckpt_dir)

        # 2. INITIALIZE COMPONENTS (Delay Data Loading)
        self.loss_fn = PhysicsLoss(config)
        self.loader = YieldDataLoader(config)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=config.training.learning_rate)
        self.model = HomogeneousYieldModel(config)
        self.model(tf.constant(np.zeros((1, 3), dtype=np.float32)))

        # --- SAFETY & OVERWRITE ---
        if not resume_path:
            if os.path.exists(self.output_dir):
                has_history = os.path.exists(os.path.join(self.output_dir, "loss_history.csv"))
                has_weights = os.path.exists(os.path.join(self.output_dir, "best_model.weights.h5"))
                
                if has_history or has_weights:
                    if self.config.training.overwrite:
                        print(f"🧹 Config 'overwrite' is True. Overwriting existing directory: {self.output_dir}")
                        shutil.rmtree(self.output_dir)
                    else:
                        print(f"\n⚠️  WARNING: Output directory '{self.output_dir}' already contains training artifacts.")
                        try:
                            user_choice = input("👉 Overwrite existing data? (y/n): ").lower().strip()
                            do_overwrite = (user_choice == 'y')
                        except (EOFError, RuntimeError):
                            print("📝 Non-interactive environment detected and overwrite is False/Missing.")
                            do_overwrite = False

                        if do_overwrite:
                            print(f"🧹 Overwriting existing directory: {self.output_dir}")
                            shutil.rmtree(self.output_dir)
                        else:
                            print("🚫 Run cancelled.")
                            import sys
                            sys.exit()

            os.makedirs(self.output_dir, exist_ok=True)
            os.makedirs(self.ckpt_dir, exist_ok=True)

        # --- 3. LOAD CHECKPOINTS / TRANSFER ---
        config_was_modified = False
        if resume_path:
            # Resolve directory if file provided
            if os.path.isfile(resume_path):
                parent = os.path.dirname(resume_path)
                self.output_dir = os.path.dirname(parent) if os.path.basename(parent) == 'checkpoints' else parent
            else:
                self.output_dir = resume_path
            
            print(f"🔄 Resuming from: {resume_path}", flush=True)
            self.start_epoch, self.history = self.checkpoint_manager.load(
                resume_path, 'resume', self.model, self.optimizer, config=self.config
            )
            
            # ATTEMPT DATA RESTORE (The Snap)
            if not self.loader.load_data(self.output_dir):
                print("📝 No snapped data found. Generating new set for continuation.")
                # Note: RNG is already restored by checkpoint_manager
                self.loader.get_dataset() # Triggers generation
                self.loader.save_data(self.output_dir) # Save for future
        
        elif transfer_path:
            print(f"🚀 Transfer Learning from: {transfer_path}", flush=True)
            _, _ = self.checkpoint_manager.load(
                transfer_path, 'transfer', self.model, config=self.config
            )
            config_was_modified = True
            # For transfer, we generate fresh data
            self.loader.get_dataset()
            self.loader.save_data(self.output_dir)
        
        else:
            print("✨ Starting fresh training...", flush=True)
            # Fresh Start: Generate and Snap
            self.loader.get_dataset()
            self.loader.save_data(self.output_dir)

        # --- 4. SAVE CONFIG FILE ---
        if fold_idx is None or fold_idx == 1:
            target_cfg = os.path.join(base_dir, "config.yaml")
            if not resume_path:
                if config_was_modified:
                    with open(target_cfg, 'w') as f: yaml.dump(self.config.to_dict(), f, sort_keys=False)
                elif config_path and os.path.exists(config_path):
                    shutil.copy2(config_path, target_cfg)
                else:
                    with open(target_cfg, 'w') as f: yaml.dump(self.config.to_dict(), f, sort_keys=False)

        # --- 5. LOGGING ---
        w = config.training.weights
        print(f"Active Losses -> Stress: {w.stress>0}, R-value: {w.r_value>0}, "
              f"Batch Convexity: {w.batch_convexity>0}, Orthotropy: {w.orthotropy>0}", flush=True)

    def validate_on_path(self):
        """ Runs validation on the full physics path (Uniaxial). """
        # Fetch fresh physics data
        _, physics_data = self.loader._generate_raw_data(needs_physics=True)
        inputs_physics, target_stress_physics, target_r_values, geometry_physics, _ = physics_data
        
        if len(inputs_physics) == 0: return 0.0, 0.0

        inputs_physics = tf.convert_to_tensor(inputs_physics)
        target_stress_physics = tf.convert_to_tensor(target_stress_physics)
        target_r_values = tf.convert_to_tensor(target_r_values)
        geometry_physics = tf.convert_to_tensor(geometry_physics)
        
        mae_stress, mae_r = self.loss_fn.validate_r_values(
            self.model, inputs_physics, (target_stress_physics, target_r_values), geometry_physics
        )
        return mae_stress, mae_r

    @tf.function
    def train_step_dual(self, batch_shape, batch_phys, do_dyn_conv, do_orthotropy):
        with tf.GradientTape() as tape_outer:
            with tf.GradientTape() as tape_inner:
                loss_res = self.loss_fn.calculate_losses(self.model, batch_shape, batch_phys, do_dyn_conv, do_orthotropy, mode='dual')
                primary_loss = loss_res['primary_loss']
            
            grads_primary = tape_inner.gradient(primary_loss, self.model.trainable_variables)
            grads_primary_safe = [g if g is not None else tf.zeros_like(v) for g, v in zip(grads_primary, self.model.trainable_variables)]
            gnorm_val = tf.linalg.global_norm(grads_primary_safe)
            
            total_loss = primary_loss
            if self.config.training.weights.gnorm_penalty > 0:
                total_loss += (self.config.training.weights.gnorm_penalty * gnorm_val)

        final_grads = tape_outer.gradient(total_loss, self.model.trainable_variables)
        final_grads_safe = [g if g is not None else tf.zeros_like(v) for g, v in zip(final_grads, self.model.trainable_variables)]
        self.optimizer.apply_gradients(zip(final_grads_safe, self.model.trainable_variables))
        
        return {
            'loss_total': primary_loss, 'loss_stress': loss_res['loss_stress'], 'loss_r': loss_res['loss_r'], 
            'loss_batch_conv': loss_res['loss_batch_conv'], 'loss_dyn_conv': loss_res['loss_dyn_conv'], 
            'loss_ortho': loss_res['loss_ortho'], 'gnorm_penalty': gnorm_val,
            'min_eig_batch': loss_res['min_eig_batch'], 'min_eig_dyn': loss_res['min_eig_dyn']
        }
    
    @tf.function
    def train_step_shape(self, batch_shape, do_dyn_conv, do_orthotropy):
        with tf.GradientTape() as tape_outer:
            with tf.GradientTape() as tape_inner:
                loss_res = self.loss_fn.calculate_losses(self.model, batch_shape, None, do_dyn_conv, do_orthotropy, mode='shape')
                primary_loss = loss_res['primary_loss']

            grads_primary = tape_inner.gradient(primary_loss, self.model.trainable_variables)
            grads_primary_safe = [g if g is not None else tf.zeros_like(v) for g, v in zip(grads_primary, self.model.trainable_variables)]
            gnorm_val = tf.linalg.global_norm(grads_primary_safe)
            
            total_loss = primary_loss
            if self.config.training.weights.gnorm_penalty > 0:
                total_loss += (self.config.training.weights.gnorm_penalty * gnorm_val)

        final_grads = tape_outer.gradient(total_loss, self.model.trainable_variables)
        final_grads_safe = [g if g is not None else tf.zeros_like(v) for g, v in zip(final_grads, self.model.trainable_variables)]
        self.optimizer.apply_gradients(zip(final_grads_safe, self.model.trainable_variables))
        
        return {
            'loss_total': primary_loss, 'loss_stress': loss_res['loss_stress'], 'loss_r': 0.0, 
            'loss_batch_conv': loss_res['loss_batch_conv'], 'loss_dyn_conv': loss_res['loss_dyn_conv'], 
            'loss_ortho': loss_res['loss_ortho'], 'gnorm_penalty': gnorm_val,
            'min_eig_batch': loss_res['min_eig_batch'], 'min_eig_dyn': loss_res['min_eig_dyn']
        }
    
    def run(self, train_dataset=None, val_dataset=None):
        """ Main training loop. """
        if train_dataset is None:
            ds_shape, ds_phys, steps = self.loader.get_dataset()
        else:
            ds_shape, ds_phys, steps = train_dataset 
            
        print(f"Training Output Directory: {self.output_dir}", flush=True)
        
        n_uni = self.config.data.samples.get('uniaxial', 0)
        w_r = self.config.training.weights.r_value
        ani_config = self.config.physics_constraints.anisotropy
        
        use_dual_stream = (n_uni > 0) and (w_r > 0) and (ani_config.batch_r_fraction > 0) and ani_config.enabled

        if ds_phys is not None and use_dual_stream:
            dataset = tf.data.Dataset.zip((ds_shape, ds_phys)).take(steps)
            mode = 'dual'
            print("🚀 Mode: DUAL STREAM (Shape + Anisotropy)", flush=True)
        else:
            dataset = ds_shape.take(steps)
            mode = 'shape'
            print("🚀 Mode: SHAPE ONLY", flush=True)

        conf_dyn = self.config.physics_constraints.dynamic_convexity
        conf_ortho = self.config.physics_constraints.orthotropy
        conf_ani = self.config.physics_constraints.anisotropy
        w = self.config.training.weights
        
        stop_criteria = self.config.training.stopping_criteria
        stop_loss, stop_conv, stop_gnorm, stop_r = stop_criteria.loss_threshold, stop_criteria.convexity_threshold, stop_criteria.gnorm_threshold, stop_criteria.r_threshold
        
        global_step = self.start_epoch * steps 
        best_metric = float('inf')
        last_val_r = self.history[-1].get('val_loss_r', float('inf')) if self.history else float('inf')
        if last_val_r is None:
            for h in reversed(self.history):
                if h.get('val_loss_r') is not None:
                    last_val_r = h['val_loss_r']; break
            if last_val_r is None: last_val_r = float('inf')

        best_loss_for_lr = float('inf')
        lr_patience_counter = 0
        session_start = time.time()
        previous_time = self.history[-1].get('time', 0.0) if self.history else 0.0

        # Weights for ramping
        original_w_r, original_w_bc, original_w_dc = w.r_value, w.batch_convexity, w.dynamic_convexity
        r_warmup, conv_warmup = self.config.training.curriculum.r_warmup, self.config.training.curriculum.convexity_warmup

        # --- LOOP ---
        for epoch in range(self.start_epoch + 1, self.config.training.epochs + 1):
            # 1. Warmup
            if r_warmup > 0 and epoch <= r_warmup: w.r_value = original_w_r * (epoch / r_warmup)
            else: w.r_value = original_w_r
            
            if conv_warmup > 0 and epoch <= conv_warmup:
                ratio = epoch / conv_warmup
                w.batch_convexity, w.dynamic_convexity = original_w_bc * ratio, original_w_dc * ratio
            else:
                w.batch_convexity, w.dynamic_convexity = original_w_bc, original_w_dc

            metric_keys = ['loss_total', 'loss_stress', 'loss_r', 'loss_batch_conv', 'loss_dyn_conv', 'loss_ortho', 'gnorm_penalty', 'min_eig_batch', 'min_eig_dyn']
            train_metrics = {k: tf.keras.metrics.Mean() for k in metric_keys}
            
            do_dyn_conv_epoch = (conf_dyn.enabled and w.dynamic_convexity > 0) and (conf_dyn.interval == 0 or epoch % conf_dyn.interval == 0 or epoch == 1)
            do_ortho_epoch = (conf_ortho.enabled and w.orthotropy > 0) and (conf_ortho.interval == 0 or epoch % conf_ortho.interval == 0 or epoch == 1)
                            
            for batch_data in dataset:
                if mode == 'dual': step_res = self.train_step_dual(batch_data[0], batch_data[1], do_dyn_conv_epoch, do_ortho_epoch)
                else: step_res = self.train_step_shape(batch_data, do_dyn_conv_epoch, do_ortho_epoch)
                for k, v in step_res.items():
                    if k in train_metrics: train_metrics[k].update_state(v)
                global_step += 1
            
            # 2. Validation
            current_val_r = None
            if conf_ani.enabled and ( (conf_ani.interval > 0 and epoch % conf_ani.interval == 0) or epoch == 1 or epoch == self.config.training.epochs):
                _, current_val_r = self.validate_on_path()
                last_val_r = current_val_r 

            # 3. Log
            row = {'epoch': epoch, 'time': previous_time + (time.time() - session_start), 'learning_rate': float(self.optimizer.learning_rate.numpy()), 'val_loss_r': current_val_r}
            for k in metric_keys:
                m_val = float(train_metrics[k].result().numpy())
                if k == 'loss_r' and mode == 'shape': row[f"train_{k}"] = None
                elif k in ['loss_dyn_conv', 'min_eig_dyn'] and not do_dyn_conv_epoch: row[f"train_{k}"] = None
                elif k == 'loss_ortho' and not do_ortho_epoch: row[f"train_{k}"] = None
                else: row[f"train_{k}"] = m_val
            
            self.history.append(row)
            
            if epoch % self.config.training.print_interval == 0 or epoch == 1:
                beig = f"{row['train_min_eig_batch']:.1e}" if row['train_min_eig_batch'] is not None else "N/A"
                deig = f"{row['train_min_eig_dyn']:.1e}" if row['train_min_eig_dyn'] is not None else "N/A"
                print(f"Ep {epoch}: Loss {row['train_loss_total']:.5f} | Stress: {row['train_loss_stress']:.5f} | R(Val): {last_val_r:.5f} | MinEig(B/D): {beig} / {deig} | G: {row['train_gnorm_penalty']:.5f}", flush=True)

            pd.DataFrame(self.history).to_csv(os.path.join(self.output_dir, "loss_history.csv"), index=False)

            # 4. Scheduler
            lr_conf = self.config.training.lr_scheduler
            if lr_conf.enabled:
                if row['train_loss_total'] < best_loss_for_lr: best_loss_for_lr, lr_patience_counter = row['train_loss_total'], 0
                else: lr_patience_counter += 1
                if lr_patience_counter >= lr_conf.patience:
                    new_lr = max(float(self.optimizer.learning_rate.numpy()) * lr_conf.factor, lr_conf.min_lr)
                    if new_lr < float(self.optimizer.learning_rate.numpy()):
                        self.optimizer.learning_rate.assign(new_lr)
                        print(f"\n📉 [LR Scheduler] Plateau. New LR: {new_lr:.2e}")
                    lr_patience_counter = 0

            if row['train_loss_total'] < best_metric:
                best_metric = row['train_loss_total']
                self.checkpoint_manager.save(epoch, self.model, self.optimizer, self.config, self.history, is_best=True)
            if self.config.training.checkpoint_interval > 0 and epoch % self.config.training.checkpoint_interval == 0:
                self.checkpoint_manager.save(epoch, self.model, self.optimizer, self.config, self.history, is_best=False)

            # 5. Stop
            pass_loss = (stop_loss is None) or (row['train_loss_total'] <= stop_loss)
            pass_conv = (stop_conv is None) or (((row['train_min_eig_batch'] is None) or (row['train_min_eig_batch'] >= -abs(stop_conv))) and ((not w.dynamic_convexity > 0) or (row['train_min_eig_dyn'] is None) or (row['train_min_eig_dyn'] >= -abs(stop_conv))))
            pass_gnorm = (stop_gnorm is None) or (row['train_gnorm_penalty'] <= stop_gnorm)
            pass_r = (not (stop_r is not None and w.r_value > 0 and self.config.physics_constraints.anisotropy.enabled)) or (last_val_r <= stop_r)

            if (stop_loss or stop_conv or stop_gnorm or stop_r) and pass_loss and pass_conv and pass_gnorm and pass_r:
                print(f"\n[Stop] Targets reached at epoch {epoch}."); self.checkpoint_manager.save(epoch, self.model, self.optimizer, self.config, self.history, is_best=True); break

        return best_metric