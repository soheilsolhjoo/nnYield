import os
import glob
import pickle
import numpy as np
import tensorflow as tf

class CheckpointManager:
    """
    Manages saving and loading of model checkpoints and training state.
    """
    def __init__(self, output_dir, ckpt_dir):
        self.output_dir = output_dir
        self.ckpt_dir = ckpt_dir

    def save(self, epoch, model, optimizer, config, history, is_best=False):
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

        # 3. Save State Dict
        state_path = os.path.join(target_dir, f"{checkpoint_name}.state.pkl")
        state_dict = {
            'epoch': epoch,
            'optimizer_weights': optimizer_weights, 
            'config': config.to_dict(),
            'rng_numpy': np.random.get_state(),
            'history': history
        }
        
        with open(state_path, 'wb') as f:
            pickle.dump(state_dict, f)
        
        if not is_best:
            print(f"Saved checkpoint to {weights_path}", flush=True)

    def load(self, path, mode, model, optimizer=None, config=None):
        """
        Loads checkpoint for Resume or Transfer.
        Returns: start_epoch, history
        """
        # 1. Resolve Paths
        weights_path, state_path = self._resolve_paths(path, mode)
        
        # 2. Transfer: Verify Architecture BEFORE Loading Weights
        if mode == 'transfer':
            if not os.path.exists(state_path):
                print("⚠️ Warning: No state file found. Cannot transfer architecture. Assuming config matches.", flush=True)
                model.load_weights(weights_path)
                return 0, []

            with open(state_path, 'rb') as f:
                saved_state = pickle.load(f)
            
            saved_model_conf = saved_state['config']['model']
            print(f"🏗️  Loading Architecture from: {os.path.basename(state_path)}")
            
            # Update Config Object in-place (if provided)
            if config:
                config.model.hidden_layers = saved_model_conf['hidden_layers']
                config.model.activation = saved_model_conf['activation']
                if 'ref_stress' in saved_model_conf:
                    config.model.ref_stress = saved_model_conf['ref_stress']
                
                print(f"   -> Layers: {config.model.hidden_layers}")
                print(f"   -> Activation: {config.model.activation}")

            # Re-build model with new config is usually done by caller BEFORE calling load()
            # But here we assume the model passed in matches the config.
            # If the architecture changed, the caller needs to rebuild the model.
            # This is tricky. In Trainer, we rebuild.
            # Ideally, CheckpointManager shouldn't rebuild models. It should just load weights.
            # But if weights shape mismatch...
            
            # For simplicity, we assume Trainer handles the rebuilding logic based on Config update.
            # But wait, Trainer calls _load_checkpoint *inside* __init__ to decide architecture.
            # So CheckpointManager needs to return the config updates so Trainer can rebuild.
            
            # Let's stick to what _load_checkpoint did: Update config, then load weights.
            
            print(f"📥 Loading weights from {os.path.basename(weights_path)}...", flush=True)
            model.load_weights(weights_path)
            print("✅ Transfer complete. Fresh optimizer initialized.", flush=True)
            return 0, []

        # 3. Resume: Full Restore
        elif mode == 'resume':
            if not os.path.exists(state_path):
                raise FileNotFoundError(f"State file not found: {state_path}")

            with open(state_path, 'rb') as f:
                saved_state = pickle.load(f)

            print(f"📥 Loading weights from {os.path.basename(weights_path)}...", flush=True)
            model.load_weights(weights_path)

            start_epoch = saved_state['epoch']
            print(f"⏱️ Resuming from Epoch {start_epoch}", flush=True)

            # Restore Optimizer
            if optimizer:
                # Initialize optimizer vars with zero gradients
                zero_grad = [tf.zeros_like(w) for w in model.trainable_variables]
                optimizer.apply_gradients(zip(zero_grad, model.trainable_variables))
                
                try:
                    optimizer.set_weights(saved_state['optimizer_weights'])
                except (AttributeError, ValueError):
                    print("⚠️ Warning: Manual optimizer restore triggered.", flush=True)
                    opt_vars = optimizer.variables
                    saved_vars = saved_state['optimizer_weights']
                    if len(opt_vars) == len(saved_vars):
                        for v, val in zip(opt_vars, saved_vars):
                            v.assign(val)

            history = saved_state.get('history', [])
            if 'rng_numpy' in saved_state:
                np.random.set_state(saved_state['rng_numpy'])
            
            return start_epoch, history

    def _resolve_paths(self, path, mode):
        if mode == 'resume':
            # Case A: User provided a specific checkpoint file (.state.pkl or .weights.h5)
            if os.path.isfile(path):
                if path.endswith(".state.pkl"):
                    state_path = path
                    weights_path = path.replace(".state.pkl", ".weights.h5")
                elif path.endswith(".weights.h5"):
                    weights_path = path
                    state_path = path.replace(".weights.h5", ".state.pkl")
                else:
                    raise ValueError("Resume path must be a directory or a checkpoint file (.state.pkl / .weights.h5)")
                
                if not os.path.exists(weights_path) or not os.path.exists(state_path):
                    raise FileNotFoundError(f"Checkpoint pair missing: {weights_path} / {state_path}")
                return weights_path, state_path

            # Case B: User provided a directory (Find latest)
            elif os.path.isdir(path):
                checkpoint_dir = os.path.join(path, "checkpoints")
                states = glob.glob(os.path.join(checkpoint_dir, "ckpt_epoch_*.state.pkl"))
                if not states:
                    root_states = glob.glob(os.path.join(path, "*.state.pkl"))
                    if not root_states: raise FileNotFoundError(f"No checkpoint found in {path}")
                    states = root_states
                latest_state = max(states, key=os.path.getctime)
                state_path = latest_state
                weights_path = latest_state.replace(".state.pkl", ".weights.h5")
                return weights_path, state_path
            else:
                raise ValueError(f"Path does not exist: {path}")
        
        elif mode == 'transfer':
            weights_path = path if path.endswith(".h5") else path + ".weights.h5"
            state_path = weights_path.replace(".weights.h5", ".state.pkl")
            return weights_path, state_path
