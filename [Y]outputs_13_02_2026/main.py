"""
nnYield Entry Point (CLI)

This script serves as the primary interface for the nnYield project. 
It supports four main operations:
1. Fresh Training: Start a new experiment from a config.yaml.
2. Resumed Training: Continue an existing experiment from a checkpoint.
3. Model Export: Convert a trained Keras model to ONNX format.
4. Sanity Check: Run automated physical consistency tests on a trained model.
"""

import os
# Silence TensorFlow info/warning logs to keep the CLI clean
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import argparse
import sys
import shutil
import pandas as pd
import numpy as np
import tensorflow as tf

from src.trainer import Trainer
from src.export import Exporter
from src.checks import SanityChecker
from src.data_loader import YieldDataLoader
from src.config import Config
from src.model import HomogeneousYieldModel 

def main():
    """
    Main execution logic for the nnYield CLI.
    """
    parser = argparse.ArgumentParser(description="nnYield: Physics-Informed Neural Yield Surface Modeling")
    
    # --- INPUT MODE GROUP (Mutually Exclusive) ---
    # The user must provide exactly one source for the experiment configuration.
    group_in = parser.add_mutually_exclusive_group(required=True)
    group_in.add_argument("--config", type=str, help="Path to config.yaml for fresh training.")
    group_in.add_argument("--resume", type=str, help="Path to experimental FOLDER or .state.pkl file to resume.")
    group_in.add_argument("--import_dir", type=str, help="Legacy: Import config from an existing directory.")

    # --- OPTIONAL ARGUMENTS ---
    parser.add_argument("--transfer", type=str, help="Path to .weights.h5 file for transfer learning.")

    # --- ACTION FLAGS ---
    parser.add_argument("--train", action="store_true", help="Execute the training loop.")
    parser.add_argument("--export", action="store_true", help="Export the best model to ONNX.")
    parser.add_argument("--check", action="store_true", help="Perform physical consistency checks.")

    args = parser.parse_args()

    # =============================================================================
    # 1. RESOLVE CONFIGURATION
    # =============================================================================
    config = None
    resume_path = None
    transfer_path = args.transfer
    
    if args.resume:
        # RESUME LOGIC: Locate the config.yaml relative to the provided checkpoint/folder.
        resume_path = args.resume
        if os.path.isfile(resume_path):
            parent = os.path.dirname(resume_path)
            # If user pointed to /checkpoints/ckpt.pkl, the config is one level up.
            base_dir = os.path.dirname(parent) if os.path.basename(parent) == 'checkpoints' else parent
        else:
            base_dir = resume_path

        conf_path = os.path.join(base_dir, "config.yaml")
        if not os.path.exists(conf_path):
            print(f"❌ Error: Cannot resume. No config.yaml found in {base_dir}")
            sys.exit(1)
        config = Config.from_yaml(conf_path)
        print(f"📄 Loaded original configuration: {conf_path}")

    elif args.config:
        # FRESH START LOGIC: Load directly from the provided path.
        config = Config.from_yaml(args.config)
        
    elif args.import_dir:
        # LEGACY IMPORT LOGIC: Similar to resume but allows overriding save paths.
        conf_path = os.path.join(args.import_dir, "config.yaml")
        if not os.path.exists(conf_path):
            print(f"❌ Error: No config.yaml found in {args.import_dir}")
            sys.exit(1)
        config = Config.from_yaml(conf_path)
        head, tail = os.path.split(os.path.abspath(args.import_dir))
        config.training.save_dir = head
        config.experiment_name = tail

    # =============================================================================
    # 2. GLOBAL SEEDING
    # =============================================================================
    if config:
        np.random.seed(config.seed)
        tf.random.set_seed(config.seed)
        print(f"🎲 Global reproducibility seed set to: {config.seed}")

    # =============================================================================
    # 3. ACTION: TRAINING
    # =============================================================================
    if args.train:
        # Validation: Transfer and Resume are incompatible.
        if transfer_path and args.resume:
            print("❌ Error: Cannot use --transfer and --resume simultaneously.")
            sys.exit(1)
            
        k_folds = config.training.k_folds
        
        # --- PATH A: STANDARD TRAINING (Single Fold) ---
        if k_folds is None or k_folds < 2:
            print("\n=== STARTING STANDARD TRAINING ===")
            trainer = Trainer(config, config_path=args.config, resume_path=resume_path, transfer_path=transfer_path)
            trainer.run()
        
        # --- PATH B: CROSS-VALIDATION (Multiple Folds) ---
        else:
            if args.resume or transfer_path:
                print("❌ Error: K-Fold CV is not yet supported in Resume/Transfer modes.")
                sys.exit(1)
                
            print(f"\n=== STARTING {k_folds}-FOLD CROSS VALIDATION ===")
            # Pre-load data to ensure identical splits across folds
            loader = YieldDataLoader(config) 
            X, y_se, y_r = loader.get_numpy_data()
            
            indices = np.arange(len(X))
            np.random.shuffle(indices)
            fold_size = len(X) // k_folds
            
            fold_results = []
            
            for k in range(k_folds):
                print(f"\n--- Fold {k+1}/{k_folds} ---")
                # Split logic: One slice for validation, the rest for training.
                val_start, val_end = k * fold_size, (k + 1) * fold_size
                val_idx = indices[val_start:val_end]
                train_idx = np.concatenate([indices[:val_start], indices[val_end:]])
                
                # Convert subsets to TensorFlow Datasets
                bs = config.training.batch_size
                steps = len(train_idx) // bs
                ds_train = tf.data.Dataset.from_tensor_slices((X[train_idx], y_se[train_idx])) \
                                          .shuffle(len(train_idx)).batch(bs)
                ds_val = tf.data.Dataset.from_tensor_slices((X[val_idx], y_se[val_idx], y_r[val_idx])).batch(bs)
                
                trainer = Trainer(config, config_path=args.config, fold_idx=k+1)
                final_metric = trainer.run(train_dataset=(ds_train, None, steps), val_dataset=ds_val)
                
                fold_results.append({'fold': k+1, 'val_loss': final_metric, 'dir': trainer.output_dir})

            # --- POST-PROCESSING: PROMOTE BEST FOLD ---
            print("\n=== K-FOLD SUMMARY ===")
            df_results = pd.DataFrame(fold_results)
            print(df_results)
            
            best_idx = df_results['val_loss'].idxmin()
            best_fold = df_results.loc[best_idx]
            print(f"\n🏆 Best Fold: {int(best_fold['fold'])} (Loss: {best_fold['val_loss']:.6f})")
            
            # Move the best weights to the root experiment folder
            root_dir = os.path.join(config.training.save_dir, config.experiment_name)
            for ext in [".weights.h5", ".state.pkl"]:
                src = os.path.join(best_fold['dir'], f"best_model{ext}")
                dst = os.path.join(root_dir, f"best_model{ext}")
                if os.path.exists(src):
                    shutil.copy(src, dst)
            
            df_results.to_csv(os.path.join(root_dir, "kfold_summary.csv"), index=False)
            print(f"✅ Promoted best model artifacts to: {root_dir}")

    # =============================================================================
    # 4. ACTION: EXPORT (ONNX)
    # =============================================================================
    if args.export:
        print("\n=== STARTING MODEL EXPORT ===")
        # Attempt to find the 'best' model first, then the last saved model.
        exp_root = os.path.join(config.training.save_dir, config.experiment_name)
        weights_path = os.path.join(exp_root, "best_model.weights.h5")
        if not os.path.exists(weights_path):
             weights_path = os.path.join(exp_root, "model.weights.h5")

        if not os.path.exists(weights_path):
            print(f"❌ Error: Weights not found at {weights_path}. Please train the model first.")
            sys.exit(1)
            
        exporter = Exporter(config) 
        exporter.export_onnx()

    # =============================================================================
    # 5. ACTION: SANITY CHECK (Physics Validation)
    # =============================================================================
    if args.check:
        print("\n=== STARTING SANITY CHECK ===")
        
        # A. Resolve model path
        output_dir = os.path.join(config.training.save_dir, config.experiment_name)
        weights_path = os.path.join(output_dir, "best_model.weights.h5")
        if not os.path.exists(weights_path):
             weights_path = os.path.join(output_dir, "model.weights.h5")
             
        if not os.path.exists(weights_path):
            print(f"❌ Error: Weights not found at {weights_path}.")
            sys.exit(1)
            
        # B. Load Model into Memory
        print(f"   -> Initializing architecture...")
        model = HomogeneousYieldModel(config) 
        _ = model(tf.zeros((1, 3))) # Trigger build
        
        print(f"   -> Loading learned parameters...")
        model.load_weights(weights_path)
        
        # C. Run Diagnostics
        checker = SanityChecker(model, config, output_dir) 
        checker.run_all()

if __name__ == "__main__":
    main()
