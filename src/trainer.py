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

class Trainer:
    """
    Experiment Orchestrator.
    
    This class manages the lifecycle of the training process (The "Engineering" Layer).
    It is decoupled from the Physics logic (in losses.py) and Data logic (in data_loader.py).
    
    Key Responsibilities:
    1. **Setup**: Initializes Model, Optimizer, Checkpoints, and Directories.
    2. **Looping**: Manages Epochs, Batches, and Weight Scheduling.
    3. **Delegation**: Feeds data from Loader to Model, then results to PhysicsLoss.
    4. **Persistence**: Handles Logging, Checkpointing, and Resuming.
    """
    
    def __init__(self, config: Config, config_path=None, resume_path=None, transfer_path=None, fold_idx=None):
        """
        Args:
            config: Parsed configuration object.
            config_path: Path to original yaml file (for preservation).
            resume_path: Path to a specific checkpoint to resume from.
            transfer_path: Path to a pre-trained model to initialize weights from.
            fold_idx: Integer index for K-Fold cross-validation (appends to output dir).
        """
        self.config = config
        self.start_epoch = 0
        self.history = []
        
        # --- 1. SETUP DIRECTORIES ---
        # Structure: output_dir / experiment_name / fold_X / checkpoints
        base_dir = os.path.join(config.training.save_dir, config.experiment_name)
        if fold_idx is not None:
            self.output_dir = os.path.join(base_dir, f"fold_{fold_idx}")
        else:
            self.output_dir = base_dir
        
        self.ckpt_dir = os.path.join(self.output_dir, "checkpoints")

        # --- SAFETY CHECKS (Overwrite Protection) ---
        if not resume_path:
            # Check if directory exists and has important files
            if os.path.exists(self.output_dir):
                has_history = os.path.exists(os.path.join(self.output_dir, "loss_history.csv"))
                
                # If overwrite is False, crash to prevent accidental data loss
                if has_history and not config.training.overwrite:
                    raise FileExistsError(
                        f"Output directory {self.output_dir} already exists. "
                        "Set 'training.overwrite: True' in config or change experiment_name."
                    )
                
                # If overwrite is True, clean the folder
                if config.training.overwrite and has_history:
                    print(f"[Trainer] Overwriting experiment dir: {self.output_dir}")
                    shutil.rmtree(self.output_dir)
            
            os.makedirs(self.ckpt_dir, exist_ok=True)
            
            # Preserve the configuration file for reproducibility
            if config_path:
                shutil.copy(config_path, os.path.join(self.output_dir, "config.yaml"))
            else:
                with open(os.path.join(self.output_dir, "config.yaml"), 'w') as f:
                    yaml.dump(config.to_dict(), f)

        # --- 2. INITIALIZATION ---
        # Initialize modules
        self.loader = YieldDataLoader(config)
        self.model = HomogeneousYieldModel(config)
        self.physics_engine = PhysicsLoss(config)

        # Optimizer
        lr = config.training.learning_rate
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

        # Checkpoint Manager (Tracks model and optimizer state)
        self.ckpt = tf.train.Checkpoint(optimizer=self.optimizer, model=self.model)
        self.manager = tf.train.CheckpointManager(self.ckpt, self.ckpt_dir, max_to_keep=5)

        # --- 3. RESUME / TRANSFER LOGIC ---
        if resume_path:
            self._restore_checkpoint(resume_path)
        elif transfer_path:
            self._transfer_weights(transfer_path)
        else:
            # Auto-resume: Check if there's a latest checkpoint in the standard folder
            latest = self.manager.latest_checkpoint
            if latest:
                print(f"[Trainer] Auto-resuming from latest: {latest}")
                self._restore_checkpoint(latest)

        # Pointer to weights for dynamic updating (Curriculum Learning)
        self.w = self.config.weights 

    def _restore_checkpoint(self, path):
        """Restores model weights, optimizer state, and epoch counter."""
        print(f"[Trainer] Restoring checkpoint from {path}...")
        try:
            # expect_partial() silences warnings if the checkpoint has extra slots
            self.ckpt.restore(path).expect_partial()
            
            # Recover epoch number from history CSV (more reliable than filename)
            hist_path = os.path.join(self.output_dir, "loss_history.csv")
            if os.path.exists(hist_path):
                df = pd.read_csv(hist_path)
                if not df.empty:
                    self.start_epoch = int(df.iloc[-1]['epoch']) + 1
                    self.history = df.to_dict('records')
            
            print(f"[Trainer] Resumed at epoch {self.start_epoch}")
        except Exception as e:
            print(f"[Trainer] Warning: Failed to restore checkpoint fully: {e}")

    def _transfer_weights(self, path):
        """Transfers ONLY the weights from another model (Warm Start)."""
        print(f"[Trainer] Transfer learning from {path}...")
        # Create a temp model to load weights safely without affecting optimizer state
        temp_model = HomogeneousYieldModel(self.config)
        ckpt = tf.train.Checkpoint(model=temp_model)
        ckpt.restore(path).expect_partial()
        
        # Run dummy input to initialize variables
        dummy = tf.zeros((1, 3))
        temp_model(dummy)
        self.model(dummy)
        
        # Copy weights
        self.model.set_weights(temp_model.get_weights())
        print("[Trainer] Weights transferred successfully.")

    # =========================================================================
    #  TRAINING STEP (Graph Mode)
    # =========================================================================
    @tf.function
    def train_step(self, inputs, targets, weights):
        """
        Performs one gradient update.
        DELEGATES all math to self.physics_engine (src/losses.py).
        
        Args:
            inputs: Tuple (stress_random, stress_physics)
            targets: Tuple (target_random, target_phys, target_r, geometry)
            weights: The current weight configuration
        """
        with tf.GradientTape() as tape:
            # Delegate loss calculation to Physics Engine
            loss_dict = self.physics_engine.calculate_losses(
                self.model, inputs, targets, weights
            )
            total_loss = loss_dict['total_loss']

        # Compute Gradients
        gradients = tape.gradient(total_loss, self.model.trainable_variables)
        
        # Apply Gradients (Backprop)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        
        return loss_dict

    # =========================================================================
    #  MAIN LOOP
    # =========================================================================
    def train(self):
        """
        Main execution loop.
        """
        cfg = self.config.training
        epochs = cfg.epochs
        
        print(f"\n[Trainer] Start training for {epochs} epochs on device: {tf.config.list_physical_devices('GPU')}")
        
        # 1. PREPARE DATA PIPELINE
        # We fetch the datasets once. The datasets themselves are infinite (.repeat()),
        # but 'steps_per_epoch' controls the loop size.
        ds_shape, ds_phys, steps_per_epoch = self.loader.get_dataset()
        
        # Handle Dual Stream vs Single Stream
        if ds_phys is not None:
            # Zip them together so we get (ShapeBatch, PhysBatch) in every iteration
            train_ds = tf.data.Dataset.zip((ds_shape, ds_phys))
        else:
            train_ds = ds_shape
        
        start_time = time.time()
        best_loss = float('inf')

        # 2. EPOCH LOOP
        for epoch in range(self.start_epoch, epochs + 1):
            
            # --- Dynamic Weight Scheduling ---
            # Linearly ramp up convexity weight from 0 to target over 'warmup' epochs
            if self.config.weights.dynamic_convexity > 0:
                warmup = cfg.convexity_warmup
                progress = min(epoch / float(warmup), 1.0)
                self.w.convexity = self.config.weights.dynamic_convexity * progress
            
            epoch_metrics = []
            
            # --- BATCH LOOP ---
            # We explicitly iterate 'steps_per_epoch' times
            # enumerate(train_ds) would run forever because of .repeat(), so we use .take()
            # or just break manual enumeration.
            
            for step, batch_data in enumerate(train_ds):
                if step >= steps_per_epoch:
                    break
                
                # Prepare Data Structure for PhysicsLoss
                if ds_phys is not None:
                    # Unpack Mixed Batch
                    (inputs_s, target_s), (inputs_p, target_p_stress, target_r, geo_p) = batch_data
                else:
                    # Single Stream: Create dummy placeholders for physics data
                    # This ensures PhysicsLoss signature is satisfied without crashing
                    inputs_s, target_s = batch_data
                    inputs_p = tf.zeros((0, 3), dtype=tf.float32)
                    target_p_stress = tf.zeros((0, 1), dtype=tf.float32)
                    target_r = tf.zeros((0, 1), dtype=tf.float32)
                    geo_p = tf.zeros((0, 3), dtype=tf.float32)
                
                # Pack into standard tuples
                inputs = (inputs_s, inputs_p)
                targets = (target_s, target_p_stress, target_r, geo_p)
                
                # Execute Step
                metrics = self.train_step(inputs, targets, self.w)
                
                # Convert tensors to floats for storage
                item = {k: float(v) for k, v in metrics.items()}
                epoch_metrics.append(item)

            # --- 3. AGGREGATE METRICS ---
            avg_metrics = pd.DataFrame(epoch_metrics).mean().to_dict()
            
            log_entry = {'epoch': epoch, 'time': time.time() - start_time}
            # Prefix metrics with 'train_' for clarity in CSV
            for k, v in avg_metrics.items():
                log_entry[f"train_{k}"] = v
            
            log_entry['w_conv'] = self.w.convexity
            
            # --- 4. TERMINAL LOGGING ---
            if epoch % cfg.print_interval == 0:
                msg = f"Epoch {epoch:04d} | Total: {avg_metrics['total_loss']:.2e}"
                if 'l_se' in avg_metrics: msg += f" | Stress: {avg_metrics['l_se']:.2e}"
                if 'l_r' in avg_metrics: msg += f" | R: {avg_metrics['l_r']:.2e}"
                if 'l_conv' in avg_metrics: msg += f" | Conv: {avg_metrics['l_conv']:.2e}"
                print(msg)

            # --- 5. CHECKPOINTING ---
            if epoch % cfg.save_interval == 0:
                self.manager.save(checkpoint_number=epoch)
                
            # Save Best Model
            current_loss = avg_metrics['total_loss']
            if current_loss < best_loss:
                best_loss = current_loss
                self.model.save_weights(os.path.join(self.output_dir, "best_model.weights.h5"))

            # --- 6. HISTORY UPDATE ---
            self.history.append(log_entry)
            pd.DataFrame(self.history).to_csv(os.path.join(self.output_dir, "loss_history.csv"), index=False)

            # --- 7. STOPPING CRITERIA ---
            stop_cfg = cfg.stop_metrics
            stop_loss = stop_cfg.get('total_loss', None)
            stop_conv = stop_cfg.get('min_eig', None)
            stop_r = stop_cfg.get('r_error', None)
            
            pass_loss = (stop_loss is None) or (current_loss <= stop_loss)
            
            pass_conv = True
            if stop_conv is not None and 'min_eig' in avg_metrics:
                # Target is usually negative (e.g. -0.0001). We want actual >= target.
                pass_conv = (avg_metrics['min_eig'] >= stop_conv)

            pass_r = True
            if stop_r is not None and 'l_r' in avg_metrics:
                pass_r = (avg_metrics['l_r'] <= stop_r)

            should_stop = (pass_loss and pass_conv and pass_r)
            any_criteria_set = (stop_loss or stop_conv or stop_r)
            
            if any_criteria_set and should_stop:
                print(f"\n[Trainer] Stopping Early! Targets reached at epoch {epoch}.")
                print(f"   Final Loss: {current_loss:.2e}")
                break

        print(f"[Trainer] Training finished. Best Loss: {best_loss:.5f}")