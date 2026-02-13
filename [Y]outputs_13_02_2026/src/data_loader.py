"""
Data Ingestion and Generation Module for nnYield.

This module provides the YieldDataLoader class, which handles the sampling of 
stress states from the theoretical yield surface (Hill48) and organizes 
them into TensorFlow-ready datasets. It supports both Shape-Only training 
and Dual-Stream training (Shape + Anisotropy).
"""

import tensorflow as tf
import numpy as np
import os
import pandas as pd
from scipy.stats import qmc
from .config import Config

class YieldDataLoader:
    """
    Data Pipeline for Physics-Informed Yield Surface Training.
    
    This loader handles:
    1. Synthetic generation of Hill48 target data.
    2. Data Snapping: Saving generated data to CSV so that resumed training 
       sessions are bit-perfect matches to the original.
    3. Dual-Stream Batching: Combining general stress points with uniaxial 
       points for anisotropy matching.
    """
    def __init__(self, config: Config):
        """
        Initializes the loader with project configuration.

        Args:
            config (Config): The experiment configuration object.
        """
        self.config = config
        # Internal storage for raw data arrays to ensure consistency during one session
        self.data_shape = None # Tuple: (inputs, targets)
        self.data_phys = None  # Tuple: (inputs, targets, r_vals, geo, mask)

    def get_dataset(self):
        """
        Constructs and returns the TensorFlow datasets.

        Returns:
            tuple: (ds_shape, ds_phys, steps_per_epoch)
                - ds_shape: Infinite dataset of general stress states.
                - ds_phys: Infinite dataset of physics-specific states (if enabled).
                - steps_per_epoch: Calculated based on total sample count.
        """
        # 1. EVALUATE DUAL STREAM ELIGIBILITY
        # We only use the dual stream if uniaxial samples exist AND the 
        # anisotropy penalty is active in the config.
        data_cfg = self.config.data
        train_cfg = self.config.training
        ani_cfg = self.config.physics_constraints.anisotropy
        
        n_uni = data_cfg.samples.get('uniaxial', 0)
        w_r = train_cfg.weights.r_value
        batch_r_frac = ani_cfg.batch_r_fraction
        is_enabled = ani_cfg.enabled
        
        use_dual_stream = (n_uni > 0) and (w_r > 0) and (batch_r_frac > 0) and is_enabled

        # 2. ENSURE DATA IS GENERATED (Lazy Loading)
        if self.data_shape is None:
            self.data_shape, self.data_phys = self._generate_raw_data(needs_physics=use_dual_stream)

        # 3. CREATE SHAPE DATASET (The 'Loci' stream)
        # We shuffle and repeat to allow infinite streaming for the Trainer loop.
        ds_shape = tf.data.Dataset.from_tensor_slices(self.data_shape)
        ds_shape = ds_shape.shuffle(len(self.data_shape[0])).repeat()

        # 4. BATCHING AND STREAM COMBINATION
        total_batch = train_cfg.batch_size

        if use_dual_stream:
            # Split the batch into two parts based on the fraction defined in config
            n_phys_batch = int(total_batch * batch_r_frac)
            n_shape_batch = total_batch - n_phys_batch
            
            # Ensure at least one sample per batch for each stream
            n_phys_batch = max(1, n_phys_batch)
            n_shape_batch = max(1, n_shape_batch)
            
            ds_shape = ds_shape.batch(n_shape_batch)
            
            # Create the Physics stream (Uniaxial points)
            ds_phys = tf.data.Dataset.from_tensor_slices(self.data_phys)
            ds_phys = ds_phys.shuffle(len(self.data_phys[0])).repeat()
            ds_phys = ds_phys.batch(n_phys_batch)
            
            # Calculate steps based on the dominant data stream
            steps = len(self.data_shape[0]) // n_shape_batch
            return ds_shape, ds_phys, steps
        else:
            # Single-stream batching
            ds_shape = ds_shape.batch(total_batch)
            steps = len(self.data_shape[0]) // total_batch
            return ds_shape, None, steps

    def save_data(self, output_dir):
        """
        Serializes current training data to CSV files.
        
        This 'Snapping' ensures that if training is stopped and resumed, the 
        model continues to see the exact same random points, preventing 
        loss spikes caused by changing datasets.
        """
        if self.data_shape is None:
            return
            
        # 1. SAVE SHAPE DATA
        inp_s, tar_s = self.data_shape
        df_shape = pd.DataFrame(inp_s, columns=['s11', 's22', 's12'])
        df_shape['target_stress'] = tar_s
        df_shape.to_csv(os.path.join(output_dir, "train_data_shape.csv"), index=False)

        # 2. SAVE PHYSICS DATA
        if self.data_phys is not None and len(self.data_phys[0]) > 0:
            inp_p, tar_p, r_p, geo_p, mask_p = self.data_phys
            df_phys = pd.DataFrame(inp_p, columns=['s11', 's22', 's12'])
            df_phys['target_stress'] = tar_p
            df_phys['target_r'] = r_p
            df_phys[['sin2', 'cos2', 'sc']] = geo_p
            df_phys['mask'] = mask_p
            df_phys.to_csv(os.path.join(output_dir, "train_data_physics.csv"), index=False)
        
        print(f"💾 Training data snapped to CSV in: {output_dir}")

    def load_data(self, output_dir):
        """
        Attempts to restore training data from existing CSV files.

        Args:
            output_dir (str): Folder containing the snapped CSVs.

        Returns:
            bool: True if data was successfully restored, False otherwise.
        """
        path_s = os.path.join(output_dir, "train_data_shape.csv")
        path_p = os.path.join(output_dir, "train_data_physics.csv")

        if not os.path.exists(path_s):
            return False

        try:
            # 1. LOAD SHAPE DATA
            df_s = pd.read_csv(path_s)
            self.data_shape = (
                df_s[['s11', 's22', 's12']].values.astype(np.float32),
                df_s[['target_stress']].values.astype(np.float32)
            )

            # 2. LOAD PHYSICS DATA
            if os.path.exists(path_p):
                df_p = pd.read_csv(path_p)
                self.data_phys = (
                    df_p[['s11', 's22', 's12']].values.astype(np.float32),
                    df_p[['target_stress']].values.astype(np.float32),
                    df_p[['target_r']].values.astype(np.float32),
                    df_p[['sin2', 'cos2', 'sc']].values.astype(np.float32),
                    df_p[['mask']].values.astype(np.float32)
                )
            
            print(f"📖 Restored original training data from: {output_dir}")
            return True
        except Exception as e:
            print(f"⚠️ Warning: Failed to load snapped data ({e}). Regenerating.")
            return False

    def _generate_raw_data(self, needs_physics=True):
        """
        Internal mathematical engine for synthetic data generation.
        Calculates ground-truth stress states and R-values based on Hill48.
        """
        data_cfg = self.config.data
        model_cfg = self.config.model
        phys_cfg = self.config.physics
        
        n_gen = data_cfg.samples.get('loci', 1000)
        n_uni = data_cfg.samples.get('uniaxial', 0) if needs_physics else 0
        
        ref_stress = model_cfg.ref_stress
        F, G, H, N = phys_cfg.F, phys_cfg.G, phys_cfg.H, phys_cfg.N
        
        # Coefficients for the quadratic form of Hill48
        C11, C22, C12, C66 = G+H, F+H, -2*H, 2*N
        use_positive_shear = data_cfg.positive_shear

        # --- 1. SHAPE DATA GENERATION ---
        # Uses Low-Discrepancy Sobol Sequences for superior surface coverage.
        if n_gen > 0:
            sampler = qmc.Sobol(d=2, scramble=True)
            sample = sampler.random(n_gen)
            
            # Map Sobol points to S12 and Theta (Angle in S11-S22 plane)
            max_shear = ref_stress / np.sqrt(C66)
            if use_positive_shear:
                s12_g = (sample[:, 0] * max_shear).astype(np.float32)
            else:
                s12_g = ((sample[:, 0] * 2.0 - 1.0) * max_shear).astype(np.float32)
            
            theta_g = (sample[:, 1] * 2.0 * np.pi).astype(np.float32)
            
            # Solve for radius in the S11-S22 plane given S12
            rhs = np.maximum(ref_stress**2 - C66*s12_g**2, 0)
            c, s = np.cos(theta_g), np.sin(theta_g)
            radius = np.sqrt(rhs / (C11*c**2 + C22*s**2 + C12*c*s + 1e-8))
            
            inputs_gen = np.stack([radius*c, radius*s, s12_g], axis=1)
            se_gen = np.ones((n_gen, 1), dtype=np.float32) * ref_stress
        else:
            inputs_gen = np.zeros((0, 3), dtype=np.float32)
            se_gen = np.zeros((0, 1), dtype=np.float32)

        # --- 1b. ANCHOR POINTS ---
        # Critical points (Pure tension, balanced biaxial, pure shear) 
        # ensuring the surface is pinned correctly at axes.
        anchors_dir = np.array([[1,0,0],[0,1,0],[-1,0,0],[0,-1,0],[1,1,0],[-1,-1,0],[1,-1,0],[-1,1,0],[0,0,1]], dtype=np.float32)
        s11, s22, s12 = anchors_dir[:,0], anchors_dir[:,1], anchors_dir[:,2]
        term = F*s22**2 + G*s11**2 + H*(s11-s22)**2 + 2*N*s12**2
        anchors_inputs = anchors_dir * (ref_stress / np.sqrt(term + 1e-8))[:, None]
        anchors_targets = np.ones((len(anchors_inputs), 1), dtype=np.float32) * ref_stress
        inputs_gen = np.concatenate([inputs_gen, anchors_inputs], axis=0)
        se_gen = np.concatenate([se_gen, anchors_targets], axis=0)
        
        # --- 2. PHYSICS DATA GENERATION ---
        # Generates uniaxial tension points at varying angles 'alpha' from 0 to 90/360 deg.
        if n_uni > 0:
            limit = np.pi / 2.0 if use_positive_shear else 2.0 * np.pi
            alpha_uni = np.linspace(0, limit, n_uni, endpoint=True, dtype=np.float32)
            sin_a, cos_a = np.sin(alpha_uni), np.cos(alpha_uni)
            
            # Map uniaxial tension direction to Cartesian stress components
            u11, u22, u12 = cos_a**2, sin_a**2, sin_a*cos_a
            term_u = F*u22**2 + G*u11**2 + H*(u11-u22)**2 + 2*N*u12**2
            scale_uni = ref_stress / np.sqrt(term_u + 1e-8)
            
            inputs_uni = np.stack([u11*scale_uni, u22*scale_uni, u12*scale_uni], axis=1)
            se_uni = np.ones((n_uni, 1), dtype=np.float32) * ref_stress
            
            # Calculate Theoretical R-values for Hill48
            s11, s22, s12 = inputs_uni[:,0], inputs_uni[:,1], inputs_uni[:,2]
            g11, g22, g12 = (G*s11 + H*(s11-s22))/ref_stress, (F*s22 - H*(s11-s22))/ref_stress, (2*N*s12)/ref_stress
            d_eps_t = -(g11 + g22)
            d_eps_w = g11*sin_a**2 + g22*cos_a**2 - 2.0*g12*sin_a*cos_a
            r_vals = (d_eps_w / (d_eps_t + 1e-8))[:, None]
            
            # Pre-calculate geometry tensors for R-value error math on GPU
            geo_uni = np.stack([sin_a**2, cos_a**2, sin_a*cos_a], axis=1)
            mask_uni = np.ones((len(se_uni), 1), dtype=np.float32)
        else:
            inputs_uni = np.zeros((0, 3), dtype=np.float32)
            se_uni, r_vals, mask_uni = [np.zeros((0, 1), dtype=np.float32) for _ in range(3)]
            geo_uni = np.zeros((0, 3), dtype=np.float32)

        return (inputs_gen, se_gen), (inputs_uni, se_uni, r_vals, geo_uni, mask_uni)
