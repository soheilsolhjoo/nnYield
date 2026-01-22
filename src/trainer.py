import tensorflow as tf
import os
import shutil
import pandas as pd
import numpy as np
import time

from .model import HomogeneousYieldModel
from .data_loader import YieldDataLoader
from .config import Config
from .losses import PhysicsLoss

class Trainer:
    """
    Experiment Orchestrator.
    
    This class manages the lifecycle of the training process.
    It integrates Data, Model, and Physics Engine (Losses).
    
    Key Features:
    - **Dual Stream Training**: Mixes Random Shape data with Physics Path data.
    - **Curriculum Learning**: Ramps up 'Convexity' and 'R-value' weights (Warmup).
    - **Interval Checks**: Runs expensive checks (Hessian/Symmetry) only periodically.
    - **Path-Based Validation**: Evaluates R-error on the strict equidistant path 
      (0 to 90 degrees) to ensure physical accuracy before stopping.
    """
    
    def __init__(self, config: Config, config_path=None, resume_path=None, transfer_path=None, fold_idx=None):
        """
        Args:
            config: Strict Config object.
            config_path: Path to yaml for preservation.
            resume_path: Checkpoint to resume.
            transfer_path: Pre-trained weights.
            fold_idx: Index for K-Fold.
        """
        self.config = config
        self.start_epoch = 0
        self.history = []
        
        # --- 1. SETUP DIRECTORIES ---
        base_dir = os.path.join(config.training.save_dir, config.experiment_name)
        if fold_idx is not None:
            self.output_dir = os.path.join(base_dir, f"fold_{fold_idx}")
        else:
            self.output_dir = base_dir
        
        self.ckpt_dir = os.path.join(self.output_dir, "checkpoints")

        # --- SAFETY & PRESERVATION ---
        if not resume_path:
            if os.path.exists(self.output_dir):
                has_history = os.path.exists(os.path.join(self.output_dir, "loss_history.csv"))
                
                if has_history and not config.training.overwrite:
                    raise FileExistsError(f"Output dir {self.output_dir} exists. Set overwrite=True in config.")
                
                if config.training.overwrite and has_history:
                    print(f"[Trainer] Cleaning output dir: {self.output_dir}")
                    shutil.rmtree(self.output_dir)
            
            os.makedirs(self.ckpt_dir, exist_ok=True)
            if config_path:
                shutil.copy(config_path, os.path.join(self.output_dir, "config.yaml"))

        # --- 2. INITIALIZATION ---
        self.loader = YieldDataLoader(config)
        self.model = HomogeneousYieldModel(config)
        self.physics_engine = PhysicsLoss(config)

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=config.training.learning_rate)
        self.ckpt = tf.train.Checkpoint(optimizer=self.optimizer, model=self.model)
        self.manager = tf.train.CheckpointManager(self.ckpt, self.ckpt_dir, max_to_keep=5)

        # Dynamic Weights Pointer (for Curriculum Learning updates)
        self.w = self.config.training.weights

        # --- 3. RESUME / TRANSFER ---
        if resume_path:
            self._restore_checkpoint(resume_path)
        elif transfer_path:
            self._transfer_weights(transfer_path)
        else:
            if self.manager.latest_checkpoint:
                print(f"[Trainer] Auto-resuming from: {self.manager.latest_checkpoint}")
                self._restore_checkpoint(self.manager.latest_checkpoint)

    def _restore_checkpoint(self, path):
        # 1
        # # # print(f"[Trainer] Restoring checkpoint from: {path}")
        # # # self.ckpt.restore(path).expect_partial()
        # # # hist_path = os.path.join(self.output_dir, "loss_history.csv")
        # # # if os.path.exists(hist_path):
        # # #     df = pd.read_csv(hist_path)
        # # #     if not df.empty:
        # # #         self.start_epoch = int(df.iloc[-1]['epoch']) + 1
        # # #         self.history = df.to_dict('records')
        # # # print(f"[Trainer] Resumed at epoch {self.start_epoch}")

        # 2
        # # If the path is a directory, find the latest checkpoint prefix within it
        # if os.path.isdir(path):
        #     latest = tf.train.latest_checkpoint(os.path.join(path, "checkpoints"))
        #     if latest:
        #         path = latest
        #     else:
        #         print(f"[Trainer] No checkpoints found in {path}. Starting from scratch.")
        #         return

        # # Now path is a specific prefix (e.g., .../checkpoints/ckpt-5)
        # self.ckpt.restore(path).expect_partial()
        
        # # Load History
        # # We look for the CSV in the parent directory of the checkpoints
        # search_dir = os.path.dirname(path) if not os.path.isdir(path) else path
        # if "checkpoints" in search_dir:
        #     search_dir = os.path.dirname(search_dir)
            
        # hist_path = os.path.join(search_dir, "loss_history.csv")
        # if os.path.exists(hist_path):
        #     df = pd.read_csv(hist_path)
        #     if not df.empty:
        #         self.start_epoch = int(df.iloc[-1]['epoch']) + 1
        #         self.history = df.to_dict('records')
        
        # print(f"[Trainer] Resumed at epoch {self.start_epoch} from {path}")
        # # 1. Resolve Directory to Checkpoint File
        # if os.path.isdir(path):
        #     latest = tf.train.latest_checkpoint(os.path.join(path, "checkpoints"))
        #     if not latest:
        #         latest = tf.train.latest_checkpoint(path)
        #     path = latest

        # if not path:
        #     print(f"[Trainer] No checkpoints found in {path}. Starting from scratch.")
        #     return

        # # 2. CRITICAL FIX: Build Model Variables Before Restoring
        # # Without this, restore() finds no variables to populate and silently fails.
        # dummy_input = tf.zeros((1, 3))
        # self.model(dummy_input)

        # # 3. Restore State
        # self.ckpt.restore(path).expect_partial()
        
        # # 4. Load History (to set start_epoch)
        # search_dir = os.path.dirname(os.path.dirname(path)) if "checkpoints" in path else os.path.dirname(path)
        # hist_path = os.path.join(search_dir, "loss_history.csv")
        
        # if os.path.exists(hist_path):
        #     df = pd.read_csv(hist_path)
        #     if not df.empty:
        #         self.start_epoch = int(df.iloc[-1]['epoch']) + 1
        #         self.history = df.to_dict('records')
        
        # print(f"[Trainer] Resumed at epoch {self.start_epoch} from {path}")
        
        # 3
        # 1. Resolve Path (Existing logic...)
        if os.path.isdir(path):
            latest = tf.train.latest_checkpoint(os.path.join(path, "checkpoints"))
            if not latest: latest = tf.train.latest_checkpoint(path)
            path = latest

        if not path:
            print("[Trainer] No checkpoint found. Starting from scratch.")
            return

        # 2. FORCE MODEL & OPTIMIZER BUILD [Critical Fix]
        # Run a dummy forward pass (builds Model weights)
        dummy_in = tf.zeros((1, 3))
        _ = self.model(dummy_in)
        
        # Run a dummy backward pass (builds Optimizer variables like 'm' and 'v')
        # We use a zero-gradient update on a dummy variable to trigger build without messing up weights
        with tf.GradientTape() as tape:
            dummy_loss = tf.reduce_sum(self.model(dummy_in))
        grads = tape.gradient(dummy_loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

        # 3. NOW restore. The slots exist, so the optimizer state will load.
        status = self.ckpt.restore(path)
        status.expect_partial() # Safe now because we know variables exist
        
        # 4. Load History (Existing logic...)
        search_dir = os.path.dirname(os.path.dirname(path)) if "checkpoints" in path else os.path.dirname(path)
        hist_path = os.path.join(search_dir, "loss_history.csv")
        if os.path.exists(hist_path):
            df = pd.read_csv(hist_path)
            if not df.empty:
                self.start_epoch = int(df.iloc[-1]['epoch']) + 1
                self.history = df.to_dict('records')
        
        print(f"[Trainer] Resumed at epoch {self.start_epoch} from {path}")

    def _transfer_weights(self, path):
        temp_model = HomogeneousYieldModel(self.config)
        ckpt = tf.train.Checkpoint(model=temp_model)
        ckpt.restore(path).expect_partial()
        # Init weights with dummy data
        dummy = tf.zeros((1, 3))
        temp_model(dummy)
        self.model(dummy)
        self.model.set_weights(temp_model.get_weights())
        print(f"[Trainer] Weights transferred from {path}")

    # =========================================================================
    #  VALIDATION LOGIC (The "Path" Check)
    # =========================================================================
    def validate_on_path(self):
        """
        Generates the strict Equidistant Path (Stress States along R-values)
        and evaluates the Model's R-error.
        
        Why? Training batches are shuffled and noisy. To rely on 'r_threshold',
        we must evaluate the R-value error on the clean, ordered path (0-90 deg).
        """
        # 1. Generate clean physics data (Linspace)
        # We assume _generate_raw_data respects the 'uniaxial' count in config.
        _, data_phys = self.loader._generate_raw_data(needs_physics=True)
        
        inputs_p, _, target_r, geo_p = data_phys
        
        if len(inputs_p) == 0:
            return 0.0 # No physics data to validate against

        # Convert to Tensor for GradientTape
        inputs_p = tf.convert_to_tensor(inputs_p)
        target_r = tf.convert_to_tensor(target_r)
        geo_p = tf.convert_to_tensor(geo_p)

        # 2. Calculate R-values (Forward + Gradient)
        with tf.GradientTape() as tape:
            tape.watch(inputs_p)
            pred_pot = self.model(inputs_p)
        
        grads = tape.gradient(pred_pot, inputs_p)
        
        # Gradients
        df_ds11 = grads[:, 0:1]
        df_ds22 = grads[:, 1:2]
        df_ds12 = grads[:, 2:3]
        
        # Geometry
        sin2 = geo_p[:, 0:1]
        cos2 = geo_p[:, 1:2]
        sincos = geo_p[:, 2:3]
        
        # Flow Rule
        d_eps_t = -(df_ds11 + df_ds22)
        d_eps_w = df_ds11 * sin2 + df_ds22 * cos2 - 2 * df_ds12 * sincos
        
        r_pred = d_eps_w / (d_eps_t + 1e-8)
        
        # 3. Compute Mean Absolute Error
        r_error = tf.reduce_mean(tf.abs(r_pred - target_r))
        return float(r_error)

    # =========================================================================
    #  TRAINING STEP
    # =========================================================================
    @tf.function
    def train_step(self, inputs, targets, w_stress, w_r_value, w_symmetry, w_convexity, w_dynamic_convexity, run_convexity, run_symmetry):
        """
        Executes one gradient update.
        Accepts unpacked weights to satisfy tf.function tracing.
        """
        # Create a lightweight object to mimic WeightsConfig structure for losses.py
        # This avoids changing losses.py while keeping tf.function happy with primitive inputs.
        class WeightsProxy:
            pass
        
        weights = WeightsProxy()
        weights.stress = w_stress
        weights.r_value = w_r_value
        weights.symmetry = w_symmetry
        weights.convexity = w_convexity
        weights.dynamic_convexity = w_dynamic_convexity

        with tf.GradientTape() as tape:
            # Delegate math to losses.py
            loss_dict = self.physics_engine.calculate_losses(
                self.model, inputs, targets, weights,
                run_convexity=run_convexity,
                run_symmetry=run_symmetry
            )
            total_loss = loss_dict['total_loss']

        # Backpropagation
        gradients = tape.gradient(total_loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        
        return loss_dict

    # =========================================================================
    #  MAIN LOOP
    # =========================================================================
    def train(self):
        cfg = self.config.training
        epochs = cfg.epochs

        # Capture the "True" targets from config once, before they get modified
        target_r_weight = float(self.config.training.weights.r_value)
        target_dyn_conv_weight = float(self.config.training.weights.dynamic_convexity)
        
        print(f"\n[Trainer] Start training for {epochs} epochs.")
        
        # 1. Prepare Data Pipeline
        ds_shape, ds_phys, steps_per_epoch = self.loader.get_dataset()
        
        if ds_phys is not None:
            train_ds = tf.data.Dataset.zip((ds_shape, ds_phys))
        else:
            train_ds = ds_shape
            
        # start_time = time.time()
        # --- RESUME TIME LOGIC ---
        previous_time = 0.0
        if self.history:
            # Get the last recorded 'time' from the history list
            previous_time = float(self.history[-1].get('time', 0.0))
        
        # We track "session time" and add it to "previous accumulated time"
        session_start = time.time()
        
        best_loss = float('inf')

        # 2. Epoch Loop
        for epoch in range(self.start_epoch, epochs + 1):
            
            # --- A. INTERVAL CHECKS & ICNN LOGIC ---
            conv_cfg = self.config.dynamic_convexity
            run_convexity = (
                conv_cfg.enabled and 
                (conv_cfg.interval > 0) and 
                (epoch % conv_cfg.interval == 0) and
                (not self.config.model.use_icnn_constraints)
            )

            # Symmetry Check:
            sym_cfg = self.config.symmetry
            run_symmetry = (
                sym_cfg.enabled and
                (sym_cfg.interval > 0) and
                (epoch % sym_cfg.interval == 0)
            )

            # --- B. CURRICULUM LEARNING (WARMUPS) ---
            ## 1. Convexity Warmup
            # 1. Convexity Warmup
            if cfg.convexity_warmup > 0:
                # Linear ramp: 0 -> Target
                prog_c = min(epoch / float(cfg.convexity_warmup), 1.0)
                # Apply to dynamic_convexity weight
                self.w.dynamic_convexity = target_dyn_conv_weight * prog_c
            else:
                self.w.dynamic_convexity = target_dyn_conv_weight
            
            # 2. R-value Warmup (The "Path" Warmup)
            if cfg.r_warmup > 0:
                prog_r = min(epoch / float(cfg.r_warmup), 1.0)
                self.w.r_value = target_r_weight * prog_r
            else:
                self.w.r_value = target_r_weight

            
            epoch_metrics = []
            
            # --- C. BATCH LOOP ---
            for step, batch_data in enumerate(train_ds):
                if step >= steps_per_epoch:
                    break
                
                # Unpack Data
                if ds_phys is not None:
                    (inputs_s, target_s), (inputs_p, target_p_stress, target_r, geo_p) = batch_data
                    
                    # Move Interval Logic to only apply if specific batches should be skipped
                    # But ensure data is passed if weights are active
                    ani_interval = self.config.anisotropy_ratio.interval
                    is_physics_step = (step % ani_interval == 0)
                    
                    if not is_physics_step:
                         inputs_p = tf.zeros((0, 3), dtype=tf.float32)
                         target_p_stress = tf.zeros((0, 1), dtype=tf.float32)
                         target_r = tf.zeros((0, 1), dtype=tf.float32)
                         geo_p = tf.zeros((0, 3), dtype=tf.float32)
                else:
                    inputs_s, target_s = batch_data
                    inputs_p = tf.zeros((0, 3), dtype=tf.float32)
                    target_p_stress = tf.zeros((0, 1), dtype=tf.float32)
                    target_r = tf.zeros((0, 1), dtype=tf.float32)
                    geo_p = tf.zeros((0, 3), dtype=tf.float32)
                
                # Pack
                inputs = (inputs_s, inputs_p)
                targets = (target_s, target_p_stress, target_r, geo_p)
                
                # Execute Step (Updated with unpacked weights)
                metrics = self.train_step(
                    inputs, 
                    targets, 
                    self.w.stress, 
                    self.w.r_value, 
                    self.w.symmetry, 
                    self.w.convexity, 
                    self.w.dynamic_convexity, 
                    run_convexity, 
                    run_symmetry
                )
                
                # Store metrics
                item = {k: float(v) for k, v in metrics.items()}
                epoch_metrics.append(item)

            # --- D. AGGREGATION & LOGGING ---
            avg_metrics = pd.DataFrame(epoch_metrics).mean().to_dict()
            val_r_error = self.validate_on_path() # Required for stopping/logging

            # Consolidated log entry: Each metric has exactly one unique name
            log_entry = {
                'epoch': epoch, 
                'time': previous_time + (time.time() - session_start),
                'total_loss': avg_metrics.get('total_loss', 0),
                'eqS_loss': avg_metrics.get('l_se_total', 0),
                'r_loss_val': val_r_error,
                'r_loss_train': avg_metrics.get('l_r', 0),
                'conv_static_loss': avg_metrics.get('l_conv_static', 0),
                'conv_dynamic_loss': avg_metrics.get('l_conv_dynamic', 0),
                'sym_loss': avg_metrics.get('l_sym', 0),
                'min_eig_train': avg_metrics.get('min_eig', 0)
            }

            # Map for diagnostic CSV and plotter compatibility
            mapping = {
                'l_se_total': 'eqS_loss',
                'l_se_shape': 'train_l_se_shape',
                'l_se_path': 'train_l_se_path',
                'l_r': 'r_loss_train',
                'l_conv_static': 'conv_static_loss',
                'l_conv_dynamic': 'conv_dynamic_loss',
                'l_sym': 'sym_loss',
                'min_eig': 'min_eig_train'
            }

            for internal_key, csv_key in mapping.items():
                if internal_key in avg_metrics:
                    log_entry[csv_key] = avg_metrics[internal_key]

            # Print to console using requested 'eqS' label
            if epoch % cfg.print_interval == 0:
                msg = f"Epoch {epoch:04d} | Total-Loss: {log_entry['total_loss']:.2e}"
                msg += f" | eqS-Loss: {log_entry['eqS_loss']:.2e}"
                msg += f" | R-Loss: {log_entry['r_loss_val']:.2e}"
                if 'train_min_eig' in log_entry: 
                    msg += f" | MinEig: {log_entry['train_min_eig']:.2e}"
                print(msg)

            # --- E. CHECKPOINTING ---
            if epoch % cfg.checkpoint_interval == 0:
                self.manager.save(checkpoint_number=epoch)
                
            current_loss = avg_metrics['total_loss']
            if current_loss < best_loss:
                best_loss = current_loss
                self.model.save_weights(os.path.join(self.output_dir, "best_model.weights.h5"))

            # --- F. HISTORY SAVE ---
            self.history.append(log_entry)
            pd.DataFrame(self.history).to_csv(os.path.join(self.output_dir, "loss_history.csv"), index=False)

            # --- G. STOPPING CRITERIA ---
            # pass_loss = True
            pass_loss = (avg_metrics['total_loss'] <= cfg.loss_threshold)
            if cfg.loss_threshold is not None:
                pass_loss = (current_loss <= cfg.loss_threshold)
            
            pass_conv = True
            if cfg.convexity_threshold is not None:
                if 'min_eig' in avg_metrics:
                    pass_conv = (avg_metrics['min_eig'] >= cfg.convexity_threshold)
                elif run_convexity: 
                     pass_conv = False
                else:
                     pass_conv = True
            
            # pass_r = True
            pass_r    = (val_r_error <= cfg.r_threshold)
            # if cfg.r_threshold is not None:
            #     pass_r = (avg_metrics['l_r'] <= cfg.r_threshold)

            should_stop = (pass_loss and pass_conv and pass_r)
            any_criteria_set = (cfg.loss_threshold or cfg.convexity_threshold or cfg.r_threshold)
            
            if any_criteria_set and should_stop:
                print(f"\n[Trainer] Targets reached at epoch {epoch}!")
                print(f"   Final Loss: {current_loss:.2e} | Val R-Err: {val_r_error:.4f}")
                self.model.save_weights(os.path.join(self.output_dir, "converged_model.weights.h5"))
                break

        print(f"[Trainer] Training finished. Best Loss: {best_loss:.5f}")