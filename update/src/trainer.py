"""
Core Training Engine Module for nnYield.

This module implements the Trainer class, which orchestrates the PINN 
learning process, including state restoration, data synchronization, 
and performance-optimized training steps.
"""

import tensorflow as tf
import os
import shutil
import yaml
import pandas as pd
import numpy as np
import time
from .model import HomogeneousYieldModel
from .data_loader import YieldDataLoader
from .config import Config
from .losses import PhysicsLoss
from .checkpoint import CheckpointManager

class Trainer:
    """
    PINN Training Orchestrator.
    """
    def __init__(self, config: Config, config_path=None, resume_path=None, transfer_path=None, fold_idx=None):
        self.config = config
        self.start_epoch = 0
        self.history = []
        
        # 1. Directory Setup
        base_dir = os.path.join(config.training.save_dir, config.experiment_name)
        self.output_dir = os.path.join(base_dir, f"fold_{fold_idx}") if fold_idx else base_dir
        self.ckpt_dir = os.path.join(self.output_dir, "checkpoints")
        self.checkpoint_manager = CheckpointManager(self.output_dir, self.ckpt_dir)

        # 2. Components
        self.loss_fn = PhysicsLoss(config)
        self.loader = YieldDataLoader(config)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=config.training.learning_rate)
        self.model = HomogeneousYieldModel(config)
        self.model(tf.constant(np.zeros((1, 3), dtype=np.float32)))

        # --- SAFETY ---
        if not resume_path and os.path.exists(self.output_dir):
            if self.config.training.overwrite:
                shutil.rmtree(self.output_dir)
            else:
                print(f"⚠️ Warning: Directory '{self.output_dir}' exists.")

        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.ckpt_dir, exist_ok=True)

        # --- RESTORATION ---
        if resume_path:
            self.start_epoch, self.history = self.checkpoint_manager.load(resume_path, 'resume', self.model, self.optimizer, config=self.config)
            if not self.loader.load_data(self.output_dir):
                self.loader.get_dataset(); self.loader.save_data(self.output_dir)
        elif transfer_path:
            self.checkpoint_manager.load(transfer_path, 'transfer', self.model, config=self.config)
            self.loader.get_dataset(); self.loader.save_data(self.output_dir)
        else:
            self.loader.get_dataset(); self.loader.save_data(self.output_dir)

        # --- ARCHIVING ---
        if fold_idx is None or fold_idx == 1:
            target_cfg = os.path.join(base_dir, "config.yaml")
            if not resume_path and config_path: shutil.copy2(config_path, target_cfg)

    def validate_on_path(self):
        """ Evaluates MAE on the full uniaxial benchmark path. """
        _, physics_data = self.loader._generate_raw_data(needs_physics=True)
        inputs, targets, r_vals, geometry, _ = physics_data
        if len(inputs) == 0: return 0.0, 0.0
        return self.loss_fn.validate_r_values(self.model, tf.convert_to_tensor(inputs), (targets, r_vals), tf.convert_to_tensor(geometry))

    @tf.function
    def train_step_dual(self, batch_shape, batch_phys, do_dyn_conv, do_orthotropy):
        """ Dual-Stream training step with R-values. """
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
        self.optimizer.apply_gradients(zip(final_grads, self.model.trainable_variables))
        
        loss_res.update({'gnorm_penalty': gnorm_val, 'loss_total': primary_loss})
        return loss_res
    
    @tf.function
    def train_step_shape(self, batch_shape, do_dyn_conv, do_orthotropy):
        """ Optimized step for Shape-Only training. """
        with tf.GradientTape() as tape_outer:
            with tf.GradientTape() as tape_inner:
                loss_res = self.loss_fn.calculate_losses(self.model, batch_shape, None, do_dyn_conv, do_orthotropy, mode='shape')
                primary_loss = loss_res['primary_loss']
            grads_primary = tape_inner.gradient(primary_loss, self.model.trainable_variables)
            grads_primary_safe = [g if g is not None else tf.zeros_like(v) for g, v in zip(grads_primary, self.model.trainable_variables)]
            gnorm_val = tf.linalg.global_norm(grads_primary_safe)
            total_loss = primary_loss + (self.config.training.weights.gnorm_penalty * gnorm_val)

        final_grads = tape_outer.gradient(total_loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(final_grads, self.model.trainable_variables))
        loss_res.update({'gnorm_penalty': gnorm_val, 'loss_total': primary_loss, 'loss_r': 0.0})
        return loss_res
    
    def run(self, train_dataset=None, val_dataset=None):
        """ Training Loop. """
        if train_dataset is None: ds_shape, ds_phys, steps = self.loader.get_dataset()
        else: ds_shape, ds_phys, steps = train_dataset 
            
        use_dual = (ds_phys is not None) and (self.config.training.weights.r_value > 0)
        dataset = tf.data.Dataset.zip((ds_shape, ds_phys)).take(steps) if use_dual else ds_shape.take(steps)
        mode = 'dual' if use_dual else 'shape'
        print(f"🚀 Training Engine: Initializing {mode.upper()} mode.", flush=True)

        w = self.config.training.weights
        stop = self.config.training.stopping_criteria
        best_metric = float('inf')
        best_loss_for_lr = float('inf')
        lr_patience_counter = 0
        session_start = time.time()
        prev_time = self.history[-1].get('time', 0.0) if self.history else 0.0
        last_val_r = self.history[-1].get('val_loss_r', 0.0) if self.history else 0.0

        orig_w = {'r': w.r_value, 'bc': w.batch_convexity, 'dc': w.dynamic_convexity}
        ramp = self.config.training.curriculum

        conf_dyn = self.config.physics_constraints.dynamic_convexity
        conf_ortho = self.config.physics_constraints.orthotropy
        conf_ani = self.config.physics_constraints.anisotropy

        for epoch in range(self.start_epoch + 1, self.config.training.epochs + 1):
            if ramp.r_warmup > 0 and epoch <= ramp.r_warmup: w.r_value = orig_w['r'] * (epoch / ramp.r_warmup)
            else: w.r_value = orig_w['r']
            if ramp.convexity_warmup > 0 and epoch <= ramp.convexity_warmup:
                ratio = epoch / ramp.convexity_warmup
                w.batch_convexity, w.dynamic_convexity = orig_w['bc'] * ratio, orig_w['dc'] * ratio
            else: w.batch_convexity, w.dynamic_convexity = orig_w['bc'], orig_w['dc']

            metric_keys = ['loss_total', 'loss_stress', 'loss_r', 'loss_batch_conv', 'loss_dyn_conv', 'loss_ortho', 'gnorm_penalty', 'min_eig_batch', 'min_eig_dyn']
            train_metrics = {k: tf.keras.metrics.Mean() for k in metric_keys}
            
            do_dc = (conf_dyn.enabled and w.dynamic_convexity > 0) and (conf_dyn.interval == 0 or epoch % conf_dyn.interval == 0 or epoch == 1)
            do_or = (conf_ortho.enabled and w.orthotropy > 0) and (conf_ortho.interval == 0 or epoch % conf_ortho.interval == 0 or epoch == 1)
                            
            for batch in dataset:
                if mode == 'dual': res = self.train_step_dual(batch[0], batch[1], do_dc, do_or)
                else: res = self.train_step_shape(batch, do_dc, do_or)
                for k, v in res.items(): 
                    if k in train_metrics: train_metrics[k].update_state(v)
            
            current_val_r = None
            if conf_ani.enabled and ((conf_ani.interval > 0 and epoch % conf_ani.interval == 0) or epoch == 1 or epoch == self.config.training.epochs):
                _, current_val_r = self.validate_on_path()
                last_val_r = current_val_r

            row = {'epoch': epoch, 'time': prev_time + (time.time() - session_start), 'learning_rate': float(self.optimizer.learning_rate.numpy()), 'val_loss_r': current_val_r}
            for k in metric_keys:
                m_val = float(train_metrics[k].result().numpy())
                if k == 'loss_r' and mode == 'shape': row[f"train_{k}"] = None
                elif k in ['loss_dyn_conv', 'min_eig_dyn'] and not do_dc: row[f"train_{k}"] = None
                elif k == 'loss_ortho' and not do_or: row[f"train_{k}"] = None
                else: row[f"train_{k}"] = m_val
            
            self.history.append(row)
            
            if epoch % self.config.training.print_interval == 0 or epoch == 1:
                beig = f"{row['train_min_eig_batch']:.1e}" if row['train_min_eig_batch'] is not None else "N/A"
                deig = f"{row['train_min_eig_dyn']:.1e}" if row['train_min_eig_dyn'] is not None else "N/A"
                print(f"Ep {epoch}: Loss {row['train_loss_total']:.5f} | Stress: {row['train_loss_stress']:.5f} | R(Val): {last_val_r:.5f} | MinEig(B/D): {beig} / {deig} | G: {row['train_gnorm_penalty']:.5f}", flush=True)

            pd.DataFrame(self.history).to_csv(os.path.join(self.output_dir, "loss_history.csv"), index=False)

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

            pass_loss = (stop.loss_threshold is None) or (row['train_loss_total'] <= stop.loss_threshold)
            pass_conv = (stop.convexity_threshold is None) or (((row['train_min_eig_batch'] is None) or (row['train_min_eig_batch'] >= -abs(stop.convexity_threshold))) and ((not w.dynamic_convexity > 0) or (row['train_min_eig_dyn'] is None) or (row['train_min_eig_dyn'] >= -abs(stop.convexity_threshold))))
            pass_gnorm = (stop.gnorm_threshold is None) or (row['train_gnorm_penalty'] <= stop.gnorm_threshold)
            pass_r = (not (stop.r_threshold is not None and w.r_value > 0 and self.config.physics_constraints.anisotropy.enabled)) or (last_val_r <= stop.r_threshold)

            if pass_loss and pass_conv and pass_gnorm and pass_r:
                print(f"\n[Stop] Targets reached at epoch {epoch}."); self.checkpoint_manager.save(epoch, self.model, self.optimizer, self.config, self.history, is_best=True); break

        return best_metric
