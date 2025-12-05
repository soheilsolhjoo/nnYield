import os
# Silence TensorFlow Logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import argparse
import sys
import shutil
import pandas as pd
import numpy as np
import tensorflow as tf
from src.utils import load_config
from src.trainer import Trainer
from src.export import Exporter
from src.sanity_check import SanityChecker
from src.data_loader import YieldDataLoader

def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--config", type=str)
    group.add_argument("--import_dir", type=str)

    parser.add_argument("--train", action="store_true")
    parser.add_argument("--export", action="store_true")
    parser.add_argument("--check", action="store_true")

    args = parser.parse_args()

    # --- Load Config ---
    if args.config:
        config = load_config(args.config)
    elif args.import_dir:
        config_path = os.path.join(args.import_dir, "config.yaml")
        if not os.path.exists(config_path):
            print(f"Error: No config.yaml found in {args.import_dir}")
            sys.exit(1)
        config = load_config(config_path)
        head, tail = os.path.split(os.path.abspath(args.import_dir))
        config['training']['save_dir'] = head
        config['experiment_name'] = tail

    # --- TRAINING LOGIC ---
    if args.train:
        k_folds = config['training'].get('k_folds', 1)
        
        if k_folds is None or k_folds < 2:
            # Standard Mode
            print("\n=== STARTING STANDARD TRAINING ===")
            trainer = Trainer(config)
            trainer.run()
        
        else:
            # K-Fold Mode
            print(f"\n=== STARTING {k_folds}-FOLD CROSS VALIDATION ===")
            loader = YieldDataLoader(config)
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
                
                bs = config['training']['batch_size']
                ds_train = tf.data.Dataset.from_tensor_slices((X[train_idx], y_se[train_idx], y_r[train_idx])).shuffle(len(train_idx)).batch(bs)
                ds_val = tf.data.Dataset.from_tensor_slices((X[val_idx], y_se[val_idx], y_r[val_idx])).batch(bs)
                
                trainer = Trainer(config, fold_idx=k+1)
                final_metric = trainer.run(train_dataset=ds_train, val_dataset=ds_val)
                
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
            root_dir = os.path.join(config['training']['save_dir'], config['experiment_name'])
            src_weights = os.path.join(best_fold['dir'], "model.weights.h5")
            dst_weights = os.path.join(root_dir, "model.weights.h5")
            shutil.copy(src_weights, dst_weights)
            
            # Copy Config (if not already there)
            src_conf = os.path.join(best_fold['dir'], "config.yaml") # trainer saves it in fold dir too? actually trainer logic says root only for fold 1.
            # Best to ensure root has config. Trainer saves it to root if fold_idx==1.
            
            df_results.to_csv(os.path.join(root_dir, "kfold_summary.csv"), index=False)
            print("   -> Promoted best model to root directory.")

    # --- EXPORT LOGIC ---
    if args.export:
        print("\n=== STARTING EXPORT ===")
        weights_path = os.path.join(config['training']['save_dir'], config['experiment_name'], "model.weights.h5")
        if not os.path.exists(weights_path):
            print(f"Error: Weights not found at {weights_path}. Train first.")
            sys.exit(1)
        exporter = Exporter(config)
        exporter.export_onnx()

    # --- CHECK LOGIC ---
    if args.check:
        print("\n=== STARTING SANITY CHECK ===")
        weights_path = os.path.join(config['training']['save_dir'], config['experiment_name'],
                                    "best_model.weights.h5")
        if not os.path.exists(weights_path):
            print(f"Error: Weights not found at {weights_path}.")
            sys.exit(1)
        checker = SanityChecker(config)
        checker.run_all()

if __name__ == "__main__":
    main()