"""
Checkpoint Management Module for nnYield.

This module provides the CheckpointManager class, responsible for 
saving and restoring model weights, optimizer moments, and RNG states.
"""

import os
import glob
import pickle
import numpy as np
import tensorflow as tf
from .config import Config

class CheckpointManager:
    """
    State Persistence Manager.
    """
    def __init__(self, output_dir, ckpt_dir):
        self.output_dir = output_dir
        self.ckpt_dir = ckpt_dir

    def save(self, epoch, model, optimizer, config: Config, history, is_best=False):
        """ Captures full training state. """
        target_dir = self.output_dir if is_best else self.ckpt_dir
        name = "best_model" if is_best else f"ckpt_epoch_{epoch}"
        
        # 1. Weights
        model.save_weights(os.path.join(target_dir, f"{name}.weights.h5"))

        # 2. Optimizer & RNG
        try: opt_weights = optimizer.get_weights()
        except: opt_weights = [v.numpy() for v in optimizer.variables]
        
        tf_gen = tf.random.get_global_generator()
        
        state = {
            'epoch': epoch, 'optimizer_weights': opt_weights, 
            'optimizer_iterations': int(optimizer.iterations.numpy()),
            'config': config.to_dict(), 'history': history,
            'rng_np': np.random.get_state(),
            'rng_tf_state': tf_gen.state.numpy(), 'rng_tf_key': tf_gen.key.numpy()
        }
        
        with open(os.path.join(target_dir, f"{name}.state.pkl"), 'wb') as f:
            pickle.dump(state, f)

    def load(self, path, mode, model, optimizer=None, config: Config = None):
        """ Restores training state for Resume or Transfer. """
        w_path, s_path = self._resolve_paths(path, mode)
        
        if not os.path.exists(s_path):
            if mode == 'transfer': 
                model.load_weights(w_path); return 0, []
            raise FileNotFoundError(f"Missing state: {s_path}")

        with open(s_path, 'rb') as f: saved = pickle.load(f)

        # 1. Architecture Check
        if config and mode == 'resume':
            if saved['config']['model']['hidden_layers'] != config.model.hidden_layers:
                raise ValueError("Architecture mismatch in hidden layers.")

        # 2. Restore
        model.load_weights(w_path)
        if mode == 'transfer': return 0, []

        if optimizer:
            # Ensure optimizer is built before setting weights
            if not optimizer.built:
                optimizer.build(model.trainable_variables)
            optimizer.iterations.assign(saved.get('optimizer_iterations', 0))
            optimizer.set_weights(saved['optimizer_weights'])

        np.random.set_state(saved['rng_np'])
        try:
            tf_gen = tf.random.get_global_generator()
            tf_gen.reset_from_state(state=saved['rng_tf_state'], key=saved['rng_tf_key'])
        except: pass
        
        return saved['epoch'], saved.get('history', [])

    def _resolve_paths(self, path, mode):
        """ Helper to find weight/state pairs. """
        if mode == 'resume':
            if os.path.isfile(path):
                ext = ".state.pkl" if path.endswith(".h5") else ".weights.h5"
                return (path, path.replace(".weights.h5", ".state.pkl")) if path.endswith(".h5") else (path.replace(".state.pkl", ".weights.h5"), path)
            
            # Directory search
            for d in [os.path.join(path, "checkpoints"), path]:
                pkles = glob.glob(os.path.join(d, "*.state.pkl"))
                if pkles:
                    latest = max(pkles, key=os.path.getctime)
                    return latest.replace(".state.pkl", ".weights.h5"), latest
            raise FileNotFoundError(f"No checkpoints in {path}")
        
        w_path = path if path.endswith(".h5") else path + ".weights.h5"
        return w_path, w_path.replace(".weights.h5", ".state.pkl")
