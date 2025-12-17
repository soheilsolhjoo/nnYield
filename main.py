import os
# Silence TensorFlow Logs (1=Info, 2=Warning, 3=Error)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import argparse
import sys
import shutil
import pandas as pd
import numpy as np
import tensorflow as tf

from src.checks import SanityChecker
from src.trainer import Trainer
from src.export import Exporter
from src.data_loader import YieldDataLoader
from src.config import Config
from src.model import HomogeneousYieldModel 

def main():
    """
    Main Entry Point for the Physics-Informed Yield Surface Project.
    
    Responsibilities:
    1. Argument Parsing: Handles CLI switches for Train/Check/Export.
    2. Config Management: Resolves Resume paths, Transfers, and Legacy imports.
    3. Orchestration: Manages the K-Fold Cross Validation loop manually.
    4. Model Promotion: Promotes the best model from the best fold to the root dir.
    """
    
    # =========================================================================
    # 1. ARGUMENT PARSING
    # =========================================================================
    parser = argparse.ArgumentParser(description="Yield Surface Training Orchestrator")
    
    # --- Input Configuration Groups (Mutually Exclusive) ---
    # You must provide exactly one way to load the configuration.
    group_in = parser.add_mutually_exclusive_group(required=True)
    group_in.add_argument("--config", type=str, help="Path to config.yaml for fresh training.")
    group_in.add_argument("--resume", type=str, help="Path to a FOLDER to resume training from (must contain config.yaml).")
    group_in.add_argument("--import_dir", type=str, help="Legacy: Import config from a directory structure.")

    # --- Optional Overrides ---
    parser.add_argument("--transfer", type=str, help="Path to .weights.h5 or .state.pkl file for Transfer Learning.")

    # --- Action Flags ---
    parser.add_argument("--train", action="store_true", help="Start the training process.")
    parser.add_argument("--export", action="store_true", help="Export the best model to ONNX.")
    parser.add_argument("--check", action="store_true", help="Run physics sanity checks.")

    args = parser.parse_args()

    # =========================================================================
    # 2. RESOLVE CONFIGURATION
    # =========================================================================
    config = None
    resume_path = None
    transfer_path = args.transfer
    
    # CASE A: Resuming an interrupted run
    if args.resume:
        resume_path = args.resume
        conf_path = os.path.join(resume_path, "config.yaml")
        if not os.path.exists(conf_path):
            print(f"Error: Cannot resume. No config.yaml found in {resume_path}")
            sys.exit(1)
        config = Config.from_yaml(conf_path)
        print(f"📄 Loaded config from resume folder: {conf_path}")

    # CASE B: Fresh Start
    elif args.config:
        config = Config.from_yaml(args.config)
        
    # CASE C: Legacy Import
    elif args.import_dir:
        conf_path = os.path.join(args.import_dir, "config.yaml")
        if not os.path.exists(conf_path):
            print(f"Error: No config.yaml found in {args.import_dir}")
            sys.exit(1)
        config = Config.from_yaml(conf_path)
        # Auto-update save directory to match import location
        head, tail = os.path.split(os.path.abspath(args.import_dir))
        config.training.save_dir = head
        config.experiment_name = tail

    # =========================================================================
    # 3. TRAINING MODE
    # =========================================================================
    if args.train:
        # Safety Check: Transfer and Resume are incompatible
        if transfer_path and args.resume:
            print("Error: Cannot use --transfer and --resume together.")
            sys.exit(1)
            
        k_folds = config.training.k_folds
        
        # --- SUB-MODE 3A: STANDARD TRAINING (No K-Fold) ---
        if k_folds is None or k_folds < 2:
            print("\n=== STARTING STANDARD TRAINING ===")
            # Note: We pass config_path to allow the Trainer to copy it to the output dir for reproducibility
            trainer = Trainer(config, config_path=args.config, resume_path=resume_path, transfer_path=transfer_path)
            trainer.train()
        
        # --- SUB-MODE 3B: K-FOLD CROSS VALIDATION ---
        else:
            if args.resume or transfer_path:
                print("Error: K-Fold CV not currently supported with Resume/Transfer mode.")
                sys.exit(1)
                
            print(f"\n=== STARTING {k_folds}-FOLD CROSS VALIDATION ===")
            
            # 1. Prepare Data for Splitting
            loader = YieldDataLoader(config.to_dict()) 
            X, y_se, y_r = loader.get_numpy_data()
            
            # 2. Create Folds (Shuffle Indices)
            indices = np.arange(len(X))
            np.random.shuffle(indices)
            fold_size = len(X) // k_folds
            
            fold_results = []
            
            # 3. Run Loop
            for k in range(k_folds):
                print(f"\n--- Fold {k+1}/{k_folds} ---")
                
                # Split Indices
                val_start = k * fold_size
                val_end = (k + 1) * fold_size
                val_idx = indices[val_start:val_end]
                train_idx = np.concatenate([indices[:val_start], indices[val_end:]])
                
                # Prepare Datasets
                bs = config.training.batch_size
                steps = len(train_idx) // bs
                
                # Create TF Datasets manually for this fold
                ds_train = tf.data.Dataset.from_tensor_slices((X[train_idx], y_se[train_idx])) \
                                          .shuffle(len(train_idx)).repeat().batch(bs)
                ds_val = tf.data.Dataset.from_tensor_slices((X[val_idx], y_se[val_idx], y_r[val_idx])).batch(bs)
                
                # Initialize Trainer for this specific fold
                trainer = Trainer(config, config_path=args.config, fold_idx=k+1)
                
                # Run Training
                # Note: We pass the pre-sliced datasets directly to run()
                final_metric = trainer.run(train_dataset=(ds_train, None, steps), val_dataset=ds_val)
                
                fold_results.append({'fold': k+1, 'val_loss': final_metric, 'dir': trainer.output_dir})

            # 4. Process Results & Promote Best Model
            print("\n=== K-FOLD RESULTS ===")
            df_results = pd.DataFrame(fold_results)
            print(df_results)
            
            best_fold = df_results.loc[df_results['val_loss'].idxmin()]
            print(f"\n🏆 Best Fold: {int(best_fold['fold'])} (Loss: {best_fold['val_loss']:.6f})")
            
            # Promote the best weights to the root experiment folder
            root_dir = os.path.join(config.training.save_dir, config.experiment_name)
            src_weights = os.path.join(best_fold['dir'], "best_model.weights.h5")
            dst_weights = os.path.join(root_dir, "best_model.weights.h5")
            src_state = os.path.join(best_fold['dir'], "best_model.state.pkl")
            dst_state = os.path.join(root_dir, "best_model.state.pkl")
            
            if os.path.exists(src_weights):
                shutil.copy(src_weights, dst_weights)
                print(f"   -> Copied best weights to {dst_weights}")
            if os.path.exists(src_state):
                shutil.copy(src_state, dst_state)
            
            df_results.to_csv(os.path.join(root_dir, "kfold_summary.csv"), index=False)
            print("   -> Promoted best model to root directory.")

    # =========================================================================
    # 4. EXPORT MODE
    # =========================================================================
    if args.export:
        print("\n=== STARTING EXPORT ===")
        # Fallback logic: check for 'best_model' first, then 'model'
        weights_path = os.path.join(config.training.save_dir, config.experiment_name, "best_model.weights.h5")
        if not os.path.exists(weights_path):
             weights_path = os.path.join(config.training.save_dir, config.experiment_name, "model.weights.h5")

        if not os.path.exists(weights_path):
            print(f"Error: Weights not found at {weights_path}. Train first.")
            sys.exit(1)
            
        exporter = Exporter(config.to_dict()) 
        exporter.export_onnx()

    # =========================================================================
    # 5. SANITY CHECK MODE
    # =========================================================================
    if args.check:
        print("\n=== STARTING SANITY CHECK ===")
        
        # A. Resolve Output Directory and Weights
        output_dir = os.path.join(config.training.save_dir, config.experiment_name)
        weights_path = os.path.join(output_dir, "best_model.weights.h5")
        
        # Fallback to standard model if best_model doesn't exist
        if not os.path.exists(weights_path):
             weights_path = os.path.join(output_dir, "model.weights.h5")
             
        if not os.path.exists(weights_path):
            print(f"Error: Weights not found at {weights_path}.")
            sys.exit(1)
            
        # B. Instantiate and Load Model
        # We must build the model graph with dummy input before loading weights
        print(f"   -> Initializing model from config...")
        model = HomogeneousYieldModel(config)
        
        print("   -> Building model graph...")
        dummy_input = tf.zeros((1, 3)) 
        _ = model(dummy_input)
        
        print(f"   -> Loading weights from: {weights_path}")
        model.load_weights(weights_path)
        
        # C. Run Checker
        # Calling the SanityChecker from src.checks
        checker = SanityChecker(model, config, output_dir) 
        checker.run_all()

if __name__ == "__main__":
    main()