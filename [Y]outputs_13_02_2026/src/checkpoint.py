"""
Checkpoint Management Module for nnYield.

This module provides the CheckpointManager class, which is responsible for 
saving and restoring the complete training state, including model weights, 
optimizer gradients, iteration counts, random number generator (RNG) states, 
and experiment configurations.
"""

import os
import glob
import pickle
import numpy as np
import tensorflow as tf
from .config import Config

class CheckpointManager:
    """
    Manages the persistence of training artifacts.
    
    This class ensures that training can be paused and resumed with bit-perfect 
    reproducibility by capturing not just weights, but also the mathematical 
    state of the optimizer and the underlying random number generators.
    """
    def __init__(self, output_dir, ckpt_dir):
        """
        Initializes the manager with directory paths.

        Args:
            output_dir (str): Root directory for the current experiment.
            ckpt_dir (str): Subdirectory dedicated to periodic checkpoints.
        """
        self.output_dir = output_dir
        self.ckpt_dir = ckpt_dir

    def save(self, epoch, model, optimizer, config: Config, history, is_best=False):
        """
        Saves model weights and the complete training state to disk.

        Args:
            epoch (int): The current epoch number.
            model (tf.keras.Model): The neural network being trained.
            optimizer (tf.keras.optimizers.Optimizer): The optimizer instance.
            config (Config): The full configuration object.
            history (list): The loss history recorded so far.
            is_best (bool): If True, saves to the root as 'best_model'.
        """
        target_dir = self.output_dir if is_best else self.ckpt_dir
        checkpoint_name = "best_model" if is_best else f"ckpt_epoch_{epoch}"
        
        # 1. SAVE MODEL WEIGHTS (.h5)
        # We use the HDF5 format for weights to ensure compatibility across TF versions.
        weights_path = os.path.join(target_dir, f"{checkpoint_name}.weights.h5")
        model.save_weights(weights_path)

        # 2. CAPTURE OPTIMIZER STATE
        # This is critical for momentum-based optimizers like Adam.
        try:
            optimizer_weights = optimizer.get_weights()
        except AttributeError:
            # Fallback for uninitialized optimizers
            optimizer_weights = [v.numpy() for v in optimizer.variables]

        # 3. CAPTURE GLOBAL RNG STATES
        # Capturing these ensures that the 'randomness' (sampling) continues 
        # exactly where it left off.
        tf_gen = tf.random.get_global_generator()
        
        # 4. BUNDLE TRAINING STATE (.pkl)
        state_dict = {
            'epoch': epoch,
            'optimizer_weights': optimizer_weights, 
            'optimizer_iterations': int(optimizer.iterations.numpy()),
            'config': config.to_dict(),
            'rng_numpy': np.random.get_state(),
            'rng_tf_state': tf_gen.state.numpy(),
            'rng_tf_key': tf_gen.key.numpy(),
            'history': history
        }
        
        state_path = os.path.join(target_dir, f"{checkpoint_name}.state.pkl")
        with open(state_path, 'wb') as f:
            pickle.dump(state_dict, f)
        
        if not is_best:
            print(f"💾 Checkpoint saved: {weights_path}", flush=True)

    def load(self, path, mode, model, optimizer=None, config: Config = None):
        """
        Restores a training state for Resume or Transfer Learning.

        Args:
            path (str): Path to a file or a folder containing checkpoints.
            mode (str): 'resume' (restores everything) or 'transfer' (weights only).
            model (tf.keras.Model): The model to load weights into.
            optimizer (tf.keras.optimizers.Optimizer, optional): Optimizer to restore.
            config (Config, optional): Current config to verify against the saved one.

        Returns:
            tuple: (start_epoch, history)
        """
        # 1. RESOLVE PATHS
        weights_path, state_path = self._resolve_paths(path, mode)
        
        if not os.path.exists(state_path):
            if mode == 'transfer':
                print("⚠️ Warning: No state file found. Loading weights only.", flush=True)
                model.load_weights(weights_path)
                return 0, []
            else:
                raise FileNotFoundError(f"State file not found: {state_path}")

        # 2. READ SAVED STATE
        with open(state_path, 'rb') as f:
            saved_state = pickle.load(f)

        # 3. ARCHITECTURE VERIFICATION (Safety Check)
        # Prevents loading weights from a different model structure.
        if config and mode == 'resume':
            saved_model_conf = saved_state['config']['model']
            curr_model_conf = config.model
            
            mismatches = []
            if saved_model_conf['hidden_layers'] != curr_model_conf.hidden_layers:
                mismatches.append(f"Layers: Saved={saved_model_conf['hidden_layers']} vs Config={curr_model_conf.hidden_layers}")
            if saved_model_conf['activation'] != curr_model_conf.activation:
                mismatches.append(f"Activation: Saved='{saved_model_conf['activation']}' vs Config='{curr_model_conf.activation}'")
            
            if mismatches:
                error_msg = (
                    "\n" + "="*60 +
                    "\n❌ RESUME FAILED: ARCHITECTURE MISMATCH" +
                    "\n" + "="*60 +
                    f"\nCheckpoint: {os.path.basename(state_path)}" +
                    "\n\nTHE CURRENT CONFIG DOES NOT MATCH THE CHECKPOINT:"
                )
                for m in mismatches:
                    error_msg += f"\n  - {m}"
                error_msg += "\n" + "="*60
                raise ValueError(error_msg)

        # 4. LOAD WEIGHTS INTO MODEL
        print(f"📥 Loading weights from {os.path.basename(weights_path)}...", flush=True)
        model.load_weights(weights_path)

        if mode == 'transfer':
            print("✅ Transfer learning initialized (Weights only).", flush=True)
            return 0, []

        # 5. RESTORE REMAINING STATE (Optimizer, History, RNG)
        start_epoch = saved_state['epoch']
        print(f"⏱️ Resuming from Epoch {start_epoch}", flush=True)

        if optimizer:
            try:
                # Ensure optimizer variables exist by doing a dummy update if necessary
                if not optimizer.variables:
                    zero_grads = [tf.zeros_like(w) for w in model.trainable_variables]
                    optimizer.apply_gradients(zip(zero_grads, model.trainable_variables))
                
                # Restore iteration count FIRST (Crucial for Adam bias correction)
                if 'optimizer_iterations' in saved_state:
                    optimizer.iterations.assign(saved_state['optimizer_iterations'])
                
                optimizer.set_weights(saved_state['optimizer_weights'])
                print(f"🧠 Optimizer gradients restored (Step {optimizer.iterations.numpy()}).", flush=True)
            except Exception as e:
                print(f"⚠️ Warning: Could not restore optimizer state ({type(e).__name__}).", flush=True)

        # Restore Loss History
        history = saved_state.get('history', [])
        
        # Restore RNG States (NumPy + TensorFlow)
        if 'rng_numpy' in saved_state:
            np.random.set_state(saved_state['rng_numpy'])
        
        if 'rng_tf_state' in saved_state and 'rng_tf_key' in saved_state:
            try:
                tf.random.get_global_generator().reset_from_state(
                    state=saved_state['rng_tf_state'],
                    key=saved_state['rng_tf_key']
                )
                print("🎲 Global RNG sequence restored.")
            except Exception:
                pass
        
        return start_epoch, history

    def _resolve_paths(self, path, mode):
        """
        Internal helper to find weights and state files from a given input path.

        Supports passing a direct file path or a root experiment directory.
        """
        if mode == 'resume':
            if os.path.isfile(path):
                if path.endswith(".state.pkl"):
                    state_path = path
                    weights_path = path.replace(".state.pkl", ".weights.h5")
                elif path.endswith(".weights.h5"):
                    weights_path = path
                    state_path = path.replace(".weights.h5", ".state.pkl")
                else:
                    raise ValueError("Resume target must be a .state.pkl or .weights.h5 file.")
                return weights_path, state_path

            elif os.path.isdir(path):
                # Search 'checkpoints' subfolder then root folder
                search_dirs = [os.path.join(path, "checkpoints"), path]
                for d in search_dirs:
                    states = glob.glob(os.path.join(d, "ckpt_epoch_*.state.pkl"))
                    if states:
                        # Pick the newest file by creation time
                        latest_state = max(states, key=os.path.getctime)
                        return latest_state.replace(".state.pkl", ".weights.h5"), latest_state
                
                # Try best_model as last resort
                best_state = os.path.join(path, "best_model.state.pkl")
                if os.path.exists(best_state):
                    return best_state.replace(".state.pkl", ".weights.h5"), best_state
                    
                raise FileNotFoundError(f"No valid checkpoints found in directory: {path}")
            else:
                raise ValueError(f"Checkpoint path does not exist: {path}")
        
        elif mode == 'transfer':
            # Simplified resolution for Transfer Learning
            weights_path = path if path.endswith(".h5") else path + ".weights.h5"
            state_path = weights_path.replace(".weights.h5", ".state.pkl")
            return weights_path, state_path
