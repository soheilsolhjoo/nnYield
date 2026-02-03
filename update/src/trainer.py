import tensorflow as tf
from scipy.stats import qmc
import os
import shutil
import yaml # Needed to write the updated config
import pandas as pd
import numpy as np
import pickle
import glob
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
        
        # Initialize Loss Function
        self.loss_fn = PhysicsLoss(config)
        
        # --- 1. SETUP DIRECTORIES ---
        base_dir = os.path.join(config.training.save_dir, config.experiment_name)
        if fold_idx is not None:
            self.output_dir = os.path.join(base_dir, f"fold_{fold_idx}")
        else:
            self.output_dir = base_dir
        
        self.ckpt_dir = os.path.join(self.output_dir, "checkpoints")

        # --- SAFETY CHECKS ---
        if not resume_path:
            if os.path.exists(self.output_dir):
                has_history = os.path.exists(os.path.join(self.output_dir, "loss_history.csv"))
                has_weights = os.path.exists(os.path.join(self.output_dir, "best_model.weights.h5"))
                if has_history or has_weights:
                    raise FileExistsError(f"⛔ Output directory '{self.output_dir}' exists. Use --resume or change experiment_name.")

            os.makedirs(self.output_dir, exist_ok=True)
            os.makedirs(self.ckpt_dir, exist_ok=True) 

        # --- 2. INITIALIZE OPTIMIZER ---
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=config.training.learning_rate)

        # --- 3. LOAD CHECKPOINTS / TRANSFER (Architecture Update) ---
        # We do this BEFORE saving the config so we know if architecture changed.
        config_was_modified = False
        
        if resume_path:
            print(f"🔄 Resuming from: {resume_path}", flush=True)
            self._load_checkpoint(resume_path, mode='resume')
            self.output_dir = resume_path
            self.ckpt_dir = os.path.join(self.output_dir, "checkpoints") 
        
        elif transfer_path:
            print(f"🚀 Transfer Learning from: {transfer_path}", flush=True)
            # This updates self.config.model with settings from the file
            self._load_checkpoint(transfer_path, mode='transfer')
            config_was_modified = True
        
        else:
            print("✨ Starting fresh training...", flush=True)
            self.model = HomogeneousYieldModel(self.config.to_dict())
            self.model(tf.constant(np.zeros((1, 3), dtype=np.float32)))

        # --- 4. SAVE CONFIG FILE ---
        # Only needed for the main run or first fold
        if fold_idx is None or fold_idx == 1:
            target_cfg = os.path.join(base_dir, "config.yaml")
            
            # Case A: Resume (Do nothing, file exists)
            if resume_path:
                pass 
                
            # Case B: Transfer (Priority: Correctness)
            # The config object now differs from the file on disk. We MUST dump the object
            # to ensure the saved file reflects the ACTUAL architecture being trained.
            elif config_was_modified:
                print(f"📝 Saving UPDATED config (Transfer Architecture) to: {target_cfg}")
                with open(target_cfg, 'w') as f:
                    yaml.dump(self.config.to_dict(), f, sort_keys=False)
            
            # Case C: Standard Run (Priority: Formatting)
            # Copy the original file to preserve comments and structure.
            elif config_path and os.path.exists(config_path):
                shutil.copy2(config_path, target_cfg)
            
            # Case D: Fallback (Dump object if no path provided)
            else:
                with open(target_cfg, 'w') as f:
                    yaml.dump(self.config.to_dict(), f, sort_keys=False)

        # --- 5. LOGGING ---
        self.use_symmetry = config.data.symmetry
        w = config.training.weights
        print(f"Active Losses -> Stress: {w.stress>0}, R-value: {w.r_value>0}, "
              f"Batch Convexity: {w.batch_convexity>0}, Symmetry: {w.symmetry>0}", flush=True)
        
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
            'config': self.config.to_dict(), # Saves the full config state
            'rng_numpy': np.random.get_state(),
            'history': self.history
        }
        
        with open(state_path, 'wb') as f:
            pickle.dump(state_dict, f)
        
        if not is_best:
            print(f"Saved checkpoint to {weights_path}", flush=True)
        
    def _load_checkpoint(self, path, mode):
        # 1. Resolve Paths
        if mode == 'resume':
            if os.path.isdir(path):
                ckpt_dir = os.path.join(path, "checkpoints")
                states = glob.glob(os.path.join(ckpt_dir, "ckpt_epoch_*.state.pkl"))
                if not states:
                    root_states = glob.glob(os.path.join(path, "*.state.pkl"))
                    if not root_states: raise FileNotFoundError(f"No checkpoint states found in {path}")
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
                weights_path = path + ".weights.h5"
                state_path = path + ".state.pkl"
            
            if not os.path.exists(state_path):
                print("⚠️ Warning: No state file found. Cannot transfer architecture. Assuming config matches.", flush=True)
                self.model = HomogeneousYieldModel(self.config.to_dict())
                self.model(tf.constant(np.zeros((1, 3), dtype=np.float32)))
                self.model.load_weights(weights_path)
                return

        # 2. Load State
        with open(state_path, 'rb') as f:
            saved_state = pickle.load(f)

        # 3. Apply Architecture (Crucial for Transfer)
        saved_model_conf = saved_state['config']['model']
        
        print(f"🏗️  Loading Architecture from: {os.path.basename(state_path)}")
        
        # Overwrite current config with saved architecture
        self.config.model.hidden_layers = saved_model_conf['hidden_layers']
        self.config.model.activation = saved_model_conf['activation']
        
        print(f"   -> Layers: {self.config.model.hidden_layers}")
        print(f"   -> Activation: {self.config.model.activation}")
        
        if 'ref_stress' in saved_model_conf:
             self.config.model.ref_stress = saved_model_conf['ref_stress']
             print(f"   -> Ref Stress: {self.config.model.ref_stress}")
        
        if 'input_dim' in saved_model_conf: self.config.model.input_dim = saved_model_conf['input_dim']
        if 'output_dim' in saved_model_conf: self.config.model.output_dim = saved_model_conf['output_dim']

        # 4. Build Model & Load Weights
        self.model = HomogeneousYieldModel(self.config.to_dict())
        self.model(tf.constant(np.zeros((1, 3), dtype=np.float32))) 

        print(f"📥 Loading weights from {os.path.basename(weights_path)}...", flush=True)
        self.model.load_weights(weights_path)

        # 5. Restore Optimizer (Resume Only - NOT Transfer)
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
            print("✅ Transfer complete. Fresh optimizer initialized.", flush=True)
    
    @tf.function
    def train_step_dual(self, batch_shape, batch_phys, do_dyn_conv, do_symmetry):
        with tf.GradientTape() as tape_outer:
            with tf.GradientTape() as tape_inner:
                loss_res = self.loss_fn.calculate_losses(
                    self.model, batch_shape, batch_phys, 
                    do_dyn_conv, do_symmetry, mode='dual'
                )
                primary_loss = loss_res['primary_loss']
            
            # Gradient Penalty (Double Backprop)
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
            'loss': primary_loss, 
            'l_se': loss_res['l_se'], 
            'l_r': loss_res['l_r'], 
            'l_conv': loss_res['l_conv'], 
            'l_dyn': loss_res['l_dyn'], 
            'l_sym': loss_res['l_sym'], 
            'gnorm': gnorm_val,
            'min_stat': loss_res['min_stat'], 
            'min_dyn': loss_res['min_dyn']
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

            # Gradient Penalty (Double Backprop)
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
            'loss': primary_loss, 
            'l_se': loss_res['l_se'], 
            'l_r': 0.0, 
            'l_conv': loss_res['l_conv'], 
            'l_dyn': loss_res['l_dyn'], 
            'l_sym': loss_res['l_sym'], 
            'gnorm': gnorm_val,
            'min_stat': loss_res['min_stat'], 
            'min_dyn': loss_res['min_dyn']
        }
    
    @tf.function
    def val_step(self, batch_shape, batch_phys):
        loss_res = self.loss_fn.calculate_losses(
            self.model, batch_shape, batch_phys, 
            False, False, mode='dual'
        )
        
        return {
            'loss': loss_res['primary_loss'], 
            'l_se': loss_res['l_se'], 
            'l_r': loss_res['l_r'], 
            'l_conv': loss_res['l_conv']
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
            
            train_metrics = {k: tf.keras.metrics.Mean() for k in 
                            ['loss', 'l_se', 'l_r', 'l_conv', 'l_dyn', 'l_sym', 'gnorm', 'min_stat', 'min_dyn']}
                            
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
                           f"MinEig(S/D): {row['train_min_stat']:.1e} / {row['train_min_dyn']:.1e} | "
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
            
            # 2. Batch Convexity (Safety)
            pass_conv = True
            if stop_conv is not None:
                target_min = -1.0 * abs(stop_conv) 
                stat_ok = (row['train_min_stat'] >= target_min)
                dyn_ok = True
                if w.dynamic_convexity > 0:
                    dyn_ok = (row['train_min_dyn'] >= target_min)
                pass_conv = stat_ok and dyn_ok

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
                if stop_conv: 
                    print(f"       Min Eig: {row['train_min_stat']:.2e} (Limit: {-abs(stop_conv):.2e})")
                if stop_gnorm: print(f"       Gnorm: {row['train_gnorm']:.2f} (Limit: {stop_gnorm})")
                if stop_r and w.r_value > 0: print(f"       R-val: {row['train_l_r']:.5f} (Limit: {stop_r})")
                
                self._save_checkpoint(epoch, is_best=True)
                break

        return best_metric