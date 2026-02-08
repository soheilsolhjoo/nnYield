import os
import glob
import pickle
import numpy as np
import tensorflow as tf
from .config import Config

class CheckpointManager:
    """
    Manages saving and loading of model checkpoints and training state.
    Supports strict object-based architecture verification and full RNG restoration (NP + TF).
    """
    def __init__(self, output_dir, ckpt_dir):
        self.output_dir = output_dir
        self.ckpt_dir = ckpt_dir

    def save(self, epoch, model, optimizer, config: Config, history, is_best=False):
        """
        Saves model weights and training state.
        """
        target_dir = self.output_dir if is_best else self.ckpt_dir
        checkpoint_name = "best_model" if is_best else f"ckpt_epoch_{epoch}"
        
        # 1. Save Weights
        weights_path = os.path.join(target_dir, f"{checkpoint_name}.weights.h5")
        model.save_weights(weights_path)

        # 2. Get Optimizer State
        try:
            optimizer_weights = optimizer.get_weights()
        except AttributeError:
            optimizer_weights = [v.numpy() for v in optimizer.variables]

        # 3. Capture RNG States
        tf_gen = tf.random.get_global_generator()
        
        # 4. Save State Dict
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
            print(f"Saved checkpoint to {weights_path}", flush=True)

    def load(self, path, mode, model, optimizer=None, config: Config = None):
        """
        Loads checkpoint for Resume or Transfer.
        """
        # 1. Resolve Paths
        weights_path, state_path = self._resolve_paths(path, mode)
        
        if not os.path.exists(state_path):
            if mode == 'transfer':
                print("⚠️ Warning: No state file found. Assuming config matches.", flush=True)
                model.load_weights(weights_path)
                return 0, []
            else:
                raise FileNotFoundError(f"State file not found: {state_path}")

        with open(state_path, 'rb') as f:
            saved_state = pickle.load(f)

        # --- 2. ARCHITECTURE VERIFICATION ---
        if config and mode == 'resume':
            saved_model_conf = saved_state['config']['model']
            curr_model_conf = config.model
            
            mismatches = []
            if saved_model_conf['hidden_layers'] != curr_model_conf.hidden_layers:
                mismatches.append(f"Layers: Saved={saved_model_conf['hidden_layers']} vs Current={curr_model_conf.hidden_layers}")
            if saved_model_conf['activation'] != curr_model_conf.activation:
                mismatches.append(f"Activation: Saved='{saved_model_conf['activation']}' vs Current='{curr_model_conf.activation}'")
            
            if mismatches:
                error_msg = (
                    "\n" + "="*60 +
                    "\n❌ RESUME FAILED: ARCHITECTURE MISMATCH" +
                    "\n" + "="*60 +
                    f"\nCheckpoint: {os.path.basename(state_path)}" +
                    "\n\nTHE FOLLOWING SETTINGS DO NOT MATCH:"
                )
                for m in mismatches:
                    error_msg += f"\n  - {m}"
                
                error_msg += (
                    "\n\nHOW TO FIX THIS:" +
                    "\n  1. Update your 'config.yaml' to match the 'Saved' values shown above." +
                    "\n  2. Or, start a fresh training if you intended to change the architecture." +
                    "\n" + "="*60 + "\n"
                )
                raise ValueError(error_msg)

        # --- 3. LOAD WEIGHTS ---
        print(f"📥 Loading weights from {os.path.basename(weights_path)}...", flush=True)
        model.load_weights(weights_path)

        if mode == 'transfer':
            print("✅ Transfer complete. Model weights loaded.", flush=True)
            return 0, []

        # --- 4. RESUME STATE (Optimizer, History, RNG) ---
        start_epoch = saved_state['epoch']
        print(f"⏱️ Resuming from Epoch {start_epoch}", flush=True)

        if optimizer:
            try:
                # Ensure variables exist before setting weights
                if not optimizer.variables:
                    zero_grads = [tf.zeros_like(w) for w in model.trainable_variables]
                    optimizer.apply_gradients(zip(zero_grads, model.trainable_variables))
                
                # Restore iteration count FIRST (Crucial for Adam bias correction)
                if 'optimizer_iterations' in saved_state:
                    optimizer.iterations.assign(saved_state['optimizer_iterations'])
                
                optimizer.set_weights(saved_state['optimizer_weights'])
                print(f"🧠 Optimizer state restored (Step {optimizer.iterations.numpy()}).", flush=True)
            except Exception as e:
                print(f"⚠️ Warning: Could not restore optimizer state ({type(e).__name__}). Continuing with fresh optimizer.", flush=True)

        history = saved_state.get('history', [])
        
        # RESTORE RNG STATE
        if 'rng_numpy' in saved_state:
            np.random.set_state(saved_state['rng_numpy'])
        
        if 'rng_tf_state' in saved_state and 'rng_tf_key' in saved_state:
            try:
                tf.random.get_global_generator().reset_from_state(
                    state=saved_state['rng_tf_state'],
                    key=saved_state['rng_tf_key']
                )
                print("🎲 TensorFlow Global RNG state restored.")
            except Exception as e:
                print(f"⚠️ Warning: TF RNG restore failed ({e})")
        
        return start_epoch, history

    def _resolve_paths(self, path, mode):
        """ Robust path resolution for both file and directory inputs. """
        if mode == 'resume':
            if os.path.isfile(path):
                if path.endswith(".state.pkl"):
                    state_path = path
                    weights_path = path.replace(".state.pkl", ".weights.h5")
                elif path.endswith(".weights.h5"):
                    weights_path = path
                    state_path = path.replace(".weights.h5", ".state.pkl")
                else:
                    raise ValueError("Provide a .state.pkl or .weights.h5 file for resume.")
                return weights_path, state_path

            elif os.path.isdir(path):
                # Search checkpoints subfolder then root
                search_dirs = [os.path.join(path, "checkpoints"), path]
                for d in search_dirs:
                    states = glob.glob(os.path.join(d, "ckpt_epoch_*.state.pkl"))
                    if states:
                        latest_state = max(states, key=os.path.getctime)
                        return latest_state.replace(".state.pkl", ".weights.h5"), latest_state
                
                # Try best_model as last resort
                best_state = os.path.join(path, "best_model.state.pkl")
                if os.path.exists(best_state):
                    return best_state.replace(".state.pkl", ".weights.h5"), best_state
                    
                raise FileNotFoundError(f"No valid checkpoints found in {path}")
            else:
                raise ValueError(f"Path does not exist: {path}")
        
        elif mode == 'transfer':
            weights_path = path if path.endswith(".h5") else path + ".weights.h5"
            state_path = weights_path.replace(".weights.h5", ".state.pkl")
            return weights_path, state_path