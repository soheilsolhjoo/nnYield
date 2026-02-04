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
    def __init__(self, config: Config, config_path=None, resume_path=None, transfer_path=None, fold_idx=None):
        self.config = config
        self.start_epoch = 0
        self.history = []
        self.rng_state = None
        
        # Initialize Loss Function
        self.loss_fn = PhysicsLoss(config)
        
        # --- 1. SETUP DIRECTORIES ---
        base_dir = os.path.join(config.training.save_dir, config.experiment_name)
        if fold_idx is not None:
            self.output_dir = os.path.join(base_dir, f"fold_{fold_idx}")
        else:
            self.output_dir = base_dir
        
        self.ckpt_dir = os.path.join(self.output_dir, "checkpoints")
        self.checkpoint_manager = CheckpointManager(self.output_dir, self.ckpt_dir)

        # --- SAFETY CHECKS (INTERACTIVE) ---
        if not resume_path:
            if os.path.exists(self.output_dir):
                has_history = os.path.exists(os.path.join(self.output_dir, "loss_history.csv"))
                has_weights = os.path.exists(os.path.join(self.output_dir, "best_model.weights.h5"))
                
                if has_history or has_weights:
                    print(f"\n⚠️  WARNING: Output directory '{self.output_dir}' already contains training artifacts.")
                    
                    try:
                        user_choice = input("👉 Overwrite existing data? (y/n): ").lower().strip()
                        do_overwrite = (user_choice == 'y')
                    except (EOFError, RuntimeError):
                        print("📝 Non-interactive environment detected. Using config 'overwrite' setting.")
                        do_overwrite = self.config.training.overwrite

                    if do_overwrite:
                        print(f"🧹 Overwriting existing directory: {self.output_dir}")
                        shutil.rmtree(self.output_dir)
                    else:
                        print("🚫 Run cancelled.")
                        import sys
                        sys.exit()

            os.makedirs(self.output_dir, exist_ok=True)
            os.makedirs(self.ckpt_dir, exist_ok=True)

        # --- 2. INITIALIZE OPTIMIZER ---
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=config.training.learning_rate)

        # --- 3. INITIALIZE MODEL ---
        self.model = HomogeneousYieldModel(self.config.to_dict())
        self.model(tf.constant(np.zeros((1, 3), dtype=np.float32)))

        # --- 4. LOAD CHECKPOINTS / TRANSFER ---
        config_was_modified = False
        
        if resume_path:
            if os.path.isfile(resume_path):
                parent = os.path.dirname(resume_path)
                if os.path.basename(parent) == 'checkpoints':
                    self.output_dir = os.path.dirname(parent)
                else:
                    self.output_dir = parent
            else:
                self.output_dir = resume_path
            
            self.ckpt_dir = os.path.join(self.output_dir, "checkpoints")
            
            print(f"🔄 Resuming from: {resume_path}", flush=True)
            self.start_epoch, self.history = self.checkpoint_manager.load(
                resume_path, 'resume', self.model, self.optimizer
            )
            
            self.checkpoint_manager.output_dir = self.output_dir
            self.checkpoint_manager.ckpt_dir = self.ckpt_dir
        
        elif transfer_path:
            print(f"🚀 Transfer Learning from: {transfer_path}", flush=True)
            _, _ = self.checkpoint_manager.load(
                transfer_path, 'transfer', self.model, config=self.config
            )
            config_was_modified = True
        
        else:
            print("✨ Starting fresh training...", flush=True)

        # --- 5. SAVE CONFIG FILE ---
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

        # --- 6. LOGGING ---
        self.use_symmetry = config.data.symmetry
        w = config.training.weights
        print(f"Active Losses -> Stress: {w.stress>0}, R-value: {w.r_value>0}, "
              f"Batch Convexity: {w.batch_convexity>0}, Symmetry: {w.symmetry>0}", flush=True)

    @tf.function
    def train_step_dual(self, batch_shape, batch_phys, do_dyn_conv, do_symmetry):
        with tf.GradientTape() as tape_outer:
            with tf.GradientTape() as tape_inner:
                loss_res = self.loss_fn.calculate_losses(
                    self.model, batch_shape, batch_phys, 
                    do_dyn_conv, do_symmetry, mode='dual'
                )
                primary_loss = loss_res['primary_loss']
            
            grads_primary = tape_inner.gradient(primary_loss, self.model.trainable_variables)
            grads_primary_safe = [g if g is not None else tf.zeros_like(v) 
                                  for g, v in zip(grads_primary, self.model.trainable_variables)]
            gnorm_val = tf.linalg.global_norm(grads_primary_safe)
            
            total_loss = primary_loss
            w = self.config.training.weights
            if w.gnorm_penalty > 0:
                total_loss += (w.gnorm_penalty * gnorm_val)

        final_grads = tape_outer.gradient(total_loss, self.model.trainable_variables)
        final_grads_safe = [g if g is not None else tf.zeros_like(v) 
                            for g, v in zip(final_grads, self.model.trainable_variables)]
        
        self.optimizer.apply_gradients(zip(final_grads_safe, self.model.trainable_variables))
        
        return {
            'loss_total': primary_loss, 
            'loss_es': loss_res['l_se'], 
            'loss_r': loss_res['l_r'], 
            'loss_batch_conv': loss_res['l_conv'], 
            'loss_dyn_conv': loss_res['l_dyn'], 
            'loss_sym': loss_res['l_sym'], 
            'gnorm_penalty': gnorm_val,
            'min_eig_batch': loss_res['min_stat'], 
            'min_eig_dyn': loss_res['min_dyn']
        }
    
    @tf.function
    def train_step_shape(self, batch_shape, do_dyn_conv, do_symmetry):
        with tf.GradientTape() as tape_outer:
            with tf.GradientTape() as tape_inner:
                loss_res = self.loss_fn.calculate_losses(
                    self.model, batch_shape, None, 
                    do_dyn_conv, do_symmetry, mode='shape'
                )
                primary_loss = loss_res['primary_loss']

            grads_primary = tape_inner.gradient(primary_loss, self.model.trainable_variables)
            grads_primary_safe = [g if g is not None else tf.zeros_like(v) 
                                  for g, v in zip(grads_primary, self.model.trainable_variables)]
            gnorm_val = tf.linalg.global_norm(grads_primary_safe)
            
            total_loss = primary_loss
            w = self.config.training.weights
            if w.gnorm_penalty > 0:
                total_loss += (w.gnorm_penalty * gnorm_val)

        final_grads = tape_outer.gradient(total_loss, self.model.trainable_variables)
        final_grads_safe = [g if g is not None else tf.zeros_like(v) 
                            for g, v in zip(final_grads, self.model.trainable_variables)]
        
        self.optimizer.apply_gradients(zip(final_grads_safe, self.model.trainable_variables))
        
        return {
            'loss_total': primary_loss, 
            'loss_es': loss_res['l_se'], 
            'loss_r': 0.0, 
            'loss_batch_conv': loss_res['l_conv'], 
            'loss_dyn_conv': loss_res['l_dyn'], 
            'loss_sym': loss_res['l_sym'], 
            'gnorm_penalty': gnorm_val,
            'min_eig_batch': loss_res['min_stat'], 
            'min_eig_dyn': loss_res['min_dyn']
        }
    
    @tf.function
    def val_step(self, batch_shape, batch_phys):
        loss_res = self.loss_fn.calculate_losses(
            self.model, batch_shape, batch_phys, 
            False, False, mode='dual'
        )
        return {
            'loss_es': loss_res['l_se'], 
            'loss_r': loss_res['l_r']
        }
        
    def run(self, train_dataset=None, val_dataset=None):
        if train_dataset is None:
            loader = YieldDataLoader(self.config.to_dict())
            ds_shape, ds_phys, steps = loader.get_dataset()
        else:
            ds_shape, ds_phys, steps = train_dataset 
            
        print(f"Training Output Directory: {self.output_dir}", flush=True)
        
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
        
        stop_loss = self.config.training.loss_threshold
        stop_conv = self.config.training.convexity_threshold
        stop_gnorm = self.config.training.gnorm_threshold
        stop_r = self.config.training.r_threshold
        
        global_step = self.start_epoch * steps 
        best_metric = float('inf')
        
        session_start = time.time()
        previous_time = self.history[-1].get('time', 0.0) if self.history else 0.0

        for epoch in range(self.start_epoch + 1, self.config.training.epochs + 1):
            metric_keys = [
                'loss_total', 'loss_es', 'loss_r', 'loss_batch_conv', 
                'loss_dyn_conv', 'loss_sym', 'gnorm_penalty', 'min_eig_batch', 'min_eig_dyn'
            ]
            train_metrics = {k: tf.keras.metrics.Mean() for k in metric_keys}
                            
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
                    
                for k, v in step_res.items():
                    if k in train_metrics: train_metrics[k].update_state(v)
                global_step += 1
            
            val_loss_es, val_loss_r = 0.0, 0.0
            if val_dataset is not None:
                val_metrics_es = tf.keras.metrics.Mean()
                val_metrics_r = tf.keras.metrics.Mean()
                for v_batch in val_dataset:
                    v_res = self.val_step(v_batch[0], v_batch[1])
                    val_metrics_es.update_state(v_res['loss_es'])
                    val_metrics_r.update_state(v_res['loss_r'])
                val_loss_es = float(val_metrics_es.result())
                val_loss_r = float(val_metrics_r.result())

            current_elapsed = time.time() - session_start
            row = {
                'epoch': epoch, 
                'time': previous_time + current_elapsed,
                'learning_rate': float(self.optimizer.learning_rate.numpy())
            }
            for k, v in train_metrics.items(): row[f"train_{k}"] = float(v.result().numpy())
            row['val_loss_es'] = val_loss_es
            row['val_loss_r'] = val_loss_r
            
            self.history.append(row)
            
            if epoch % 5 == 0 or epoch == 1:
                log_str = (f"Ep {epoch}: Loss {row['train_loss_total']:.5f} | "
                           f"ES: {row['train_loss_es']:.5f} | R: {row['train_loss_r']:.5f} | "
                           f"MinEig(B/D): {row['train_min_eig_batch']:.1e} / {row['train_min_eig_dyn']:.1e} | "
                           f"G: {row['train_gnorm_penalty']:.5f}")
                print(log_str, flush=True)

            pd.DataFrame(self.history).to_csv(os.path.join(self.output_dir, "loss_history.csv"), index=False)

            if row['train_loss_total'] < best_metric:
                best_metric = row['train_loss_total']
                self.checkpoint_manager.save(epoch, self.model, self.optimizer, self.config, self.history, is_best=True)
            
            ckpt_interval = self.config.training.checkpoint_interval
            if ckpt_interval > 0 and epoch % ckpt_interval == 0:
                self.checkpoint_manager.save(epoch, self.model, self.optimizer, self.config, self.history, is_best=False)

            pass_loss = (stop_loss is None) or (row['train_loss_total'] <= stop_loss)
            pass_conv = True
            if stop_conv is not None:
                target_min = -1.0 * abs(stop_conv) 
                stat_ok = (row['train_min_eig_batch'] >= target_min)
                dyn_ok = (not w.dynamic_convexity > 0) or (row['train_min_eig_dyn'] >= target_min)
                pass_conv = stat_ok and dyn_ok

            pass_gnorm = (stop_gnorm is None) or (row['train_gnorm_penalty'] <= stop_gnorm)
            pass_r = (not (stop_r is not None and w.r_value > 0)) or (row['train_loss_r'] <= stop_r)

            any_limit_set = (stop_loss or stop_conv or stop_gnorm or stop_r)
            if any_limit_set and pass_loss and pass_conv and pass_gnorm and pass_r:
                print(f"\n[Stop] All targets reached at epoch {epoch}.")
                self.checkpoint_manager.save(epoch, self.model, self.optimizer, self.config, self.history, is_best=True)
                break

        return best_metric