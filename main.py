import os
# Silence TensorFlow Logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import argparse
import sys
import shutil
import pandas as pd
import numpy as np
import tensorflow as tf

from src.trainer import Trainer
from src.export import Exporter
from src.sanity_check import SanityChecker
from src.data_loader import YieldDataLoader
from src.config import Config

def main():
    parser = argparse.ArgumentParser()
    
    # --- Input Groups ---
    # User must provide ONE of: --config, --resume, or --import_dir
    group_in = parser.add_mutually_exclusive_group(required=True)
    group_in.add_argument("--config", type=str, help="Path to config.yaml for fresh training")
    group_in.add_argument("--resume", type=str, help="Path to FOLDER to resume training from")
    group_in.add_argument("--import_dir", type=str, help="Legacy: import config from dir")

    # Optional: Transfer is separate because it usually accompanies a new config
    parser.add_argument("--transfer", type=str, help="Path to .weights.h5 or .state.pkl file for Transfer Learning")

    # Actions
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--export", action="store_true")
    parser.add_argument("--check", action="store_true")

    args = parser.parse_args()

    # --- 1. RESOLVE CONFIGURATION ---
    config = None
    resume_path = None
    transfer_path = args.transfer
    
    if args.resume:
        # CASE: RESUME
        # We load the config FROM the folder to ensure consistency
        resume_path = args.resume
        conf_path = os.path.join(resume_path, "config.yaml")
        if not os.path.exists(conf_path):
            print(f"Error: Cannot resume. No config.yaml found in {resume_path}")
            sys.exit(1)
        config = Config.from_yaml(conf_path)
        print(f"📄 Loaded config from resume folder: {conf_path}")

    elif args.config:
        # CASE: FRESH or TRANSFER
        config = Config.from_yaml(args.config)
        
    elif args.import_dir:
        # CASE: LEGACY IMPORT
        conf_path = os.path.join(args.import_dir, "config.yaml")
        if not os.path.exists(conf_path):
            print(f"Error: No config.yaml found in {args.import_dir}")
            sys.exit(1)
        config = Config.from_yaml(conf_path)
        head, tail = os.path.split(os.path.abspath(args.import_dir))
        config.training.save_dir = head
        config.experiment_name = tail

    # --- 2. TRAINING LOGIC ---
    if args.train:
        # Validate Transfer
        if transfer_path and args.resume:
            print("Error: Cannot use --transfer and --resume together.")
            sys.exit(1)
            
        k_folds = config.training.k_folds
        
        if k_folds is None or k_folds < 2:
            # Standard Mode
            print("\n=== STARTING STANDARD TRAINING ===")
            trainer = Trainer(config, resume_path=resume_path, transfer_path=transfer_path)
            trainer.run()
        
        else:
            # Check for incompatibility
            if args.resume or transfer_path:
                print("Error: K-Fold CV not currently supported with Resume/Transfer mode.")
                sys.exit(1)
                
            # K-Fold Mode
            print(f"\n=== STARTING {k_folds}-FOLD CROSS VALIDATION ===")
            
            # Loader likely still expects a dictionary, so we convert config back
            loader = YieldDataLoader(config.to_dict()) 
            X, y_se, y_r = loader.get_numpy_data()
            
            # Shuffle once
            indices = np.arange(len(X))
            np.random.shuffle(indices)
            fold_size = len(X) // k_folds
            
            fold_results = []
            
            for k in range(k_folds):
                print(f"\n--- Fold {k+1}/{k_folds} ---")
                
                val_start = k * fold_size
                val_end = (k + 1) * fold_size
                val_idx = indices[val_start:val_end]
                train_idx = np.concatenate([indices[:val_start], indices[val_end:]])
                
                bs = config.training.batch_size
                steps = len(train_idx) // bs
                
                # Prepare Datasets (Shape Only for K-Fold default)
                ds_train = tf.data.Dataset.from_tensor_slices((X[train_idx], y_se[train_idx])) \
                                          .shuffle(len(train_idx)).batch(bs)
                
                # Validation Data (Shape Only)
                # Note: If val_step expects dual inputs, we need to adapt or pass dummy physics data.
                # Assuming shape-only validation for K-Fold simplicity:
                ds_val = tf.data.Dataset.from_tensor_slices((X[val_idx], y_se[val_idx], y_r[val_idx])).batch(bs)
                
                # Initialize Trainer
                trainer = Trainer(config, fold_idx=k+1)
                
                # Run Training (passing tuple for shape-only mode)
                final_metric = trainer.run(
                    train_dataset=(ds_train, None, steps), 
                    val_dataset=ds_val
                )
                
                fold_results.append({
                    'fold': k+1, 
                    'val_loss': final_metric,
                    'dir': trainer.output_dir
                })

            # Select Best Fold
            print("\n=== K-FOLD RESULTS ===")
            df_results = pd.DataFrame(fold_results)
            print(df_results)
            
            best_fold = df_results.loc[df_results['val_loss'].idxmin()]
            print(f"\n🏆 Best Fold: {int(best_fold['fold'])} (Loss: {best_fold['val_loss']:.6f})")
            
            # Copy Best Weights to Root
            root_dir = os.path.join(config.training.save_dir, config.experiment_name)
            
            # Source filename is 'best_model.weights.h5' (from Trainer._save_checkpoint)
            src_weights = os.path.join(best_fold['dir'], "best_model.weights.h5")
            dst_weights = os.path.join(root_dir, "best_model.weights.h5")
            
            # Also copy the state file
            src_state = os.path.join(best_fold['dir'], "best_model.state.pkl")
            dst_state = os.path.join(root_dir, "best_model.state.pkl")
            
            if os.path.exists(src_weights):
                shutil.copy(src_weights, dst_weights)
                print(f"   -> Copied best weights to {dst_weights}")
            
            if os.path.exists(src_state):
                shutil.copy(src_state, dst_state)
            
            df_results.to_csv(os.path.join(root_dir, "kfold_summary.csv"), index=False)
            print("   -> Promoted best model to root directory.")

    # --- 3. EXPORT LOGIC ---
    if args.export:
        print("\n=== STARTING EXPORT ===")
        # Updated filename to match new 'best_model' convention
        weights_path = os.path.join(config.training.save_dir, config.experiment_name, "best_model.weights.h5")
        
        # Fallback to old name if best_model doesn't exist
        if not os.path.exists(weights_path):
             weights_path = os.path.join(config.training.save_dir, config.experiment_name, "model.weights.h5")

        if not os.path.exists(weights_path):
            print(f"Error: Weights not found at {weights_path}. Train first.")
            sys.exit(1)
            
        exporter = Exporter(config.to_dict()) # Exporter likely needs dict
        exporter.export_onnx()

    # --- 4. CHECK LOGIC ---
    if args.check:
        print("\n=== STARTING SANITY CHECK ===")
        weights_path = os.path.join(config.training.save_dir, config.experiment_name, "best_model.weights.h5")
        
        if not os.path.exists(weights_path):
             weights_path = os.path.join(config.training.save_dir, config.experiment_name, "model.weights.h5")
             
        if not os.path.exists(weights_path):
            print(f"Error: Weights not found at {weights_path}.")
            sys.exit(1)
            
        checker = SanityChecker(config.to_dict()) # Checker likely needs dict
        checker.run_all()

if __name__ == "__main__":
    main()