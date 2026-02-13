"""
Data Ingestion and Generation Module for nnYield.

This module handles the creation of training datasets from analytical models.
It supports both simple shape training and complex anisotropy matching.
"""

import tensorflow as tf
import numpy as np
import os
import pandas as pd
from scipy.stats import qmc
from .config import Config
from .physics_models.factory import get_physics_model

class YieldDataLoader:
    """
    Modular Data Pipeline.
    
    This class handles the interface between the theoretical physics models 
    and the TensorFlow training engine.
    """
    def __init__(self, config: Config):
        """
        Args:
            config (Config): Project configuration object.
        """
        self.config = config
        # Local cache for raw data arrays
        self.data_shape = None 
        self.data_phys = None  
        # Instantiate the selected analytical benchmark for data generation
        self.physics_bench = get_physics_model(self.config)

    def get_dataset(self):
        """
        TRICK: DUAL-STREAM ITERATOR.
        
        This method returns two synchronized TensorFlow datasets. 
        - ds_shape: Infinite stream of random points on the yield surface.
        - ds_phys: Infinite stream of uniaxial points for R-value matching.
        
        By splitting the batch into 'Shape' and 'Physics' streams, the model 
        receives a balanced signal every step, preventing it from overfitting 
        to one specific region of the stress space.
        """
        data_cfg = self.config.data
        train_cfg = self.config.training
        ani_cfg = self.config.physics_constraints.anisotropy
        
        # Check if we should use the dual-stream mode
        n_uni = data_cfg.samples.get('uniaxial', 0)
        w_r = train_cfg.weights.r_value
        batch_r_frac = ani_cfg.batch_r_fraction
        is_enabled = ani_cfg.enabled
        
        use_dual_stream = (n_uni > 0) and (w_r > 0) and (batch_r_frac > 0) and is_enabled

        # lazy loading
        if self.data_shape is None:
            self.data_shape, self.data_phys = self._generate_raw_data(needs_physics=use_dual_stream)

        # 1. SHAPE DATASET
        ds_shape = tf.data.Dataset.from_tensor_slices(self.data_shape)
        ds_shape = ds_shape.shuffle(len(self.data_shape[0])).repeat()

        total_batch = train_cfg.batch_size

        if use_dual_stream:
            # Partition the batch size between the two streams
            n_phys_batch = int(total_batch * batch_r_frac)
            n_shape_batch = total_batch - n_phys_batch
            
            n_phys_batch = max(1, n_phys_batch)
            n_shape_batch = max(1, n_shape_batch)
            
            ds_shape = ds_shape.batch(n_shape_batch)
            
            # 2. PHYSICS DATASET
            ds_phys = tf.data.Dataset.from_tensor_slices(self.data_phys)
            ds_phys = ds_phys.shuffle(len(self.data_phys[0])).repeat()
            ds_phys = ds_phys.batch(n_phys_batch)
            
            # Steps per epoch are based on the main shape stream
            steps = len(self.data_shape[0]) // n_shape_batch
            return ds_shape, ds_phys, steps
        else:
            # Default to single shape stream
            ds_shape = ds_shape.batch(total_batch)
            steps = len(self.data_shape[0]) // total_batch
            return ds_shape, None, steps

    def get_numpy_data(self):
        """
        Returns raw numpy data for K-Fold Cross-Validation.
        """
        if self.data_shape is None:
            self.data_shape, self.data_phys = self._generate_raw_data(needs_physics=True)
            
        inputs_s, targets_s = self.data_shape
        # Return a zero-target for R-values as they are handled in the loss engine
        return inputs_s, targets_s, np.zeros_like(targets_s)

    def save_data(self, output_dir):
        """
        TRICK: DATA SNAPPING.
        
        This method saves the exact generated training samples to CSV files. 
        When training is resumed, the loader reads these files instead of 
        regenerating. This ensures that the model sees the SAME data sequence, 
        preventing loss jumps during a resume session.
        """
        if self.data_shape is None: return
        
        inp_s, tar_s = self.data_shape
        df_shape = pd.DataFrame(inp_s, columns=['s11', 's22', 's12'])
        df_shape['target_stress'] = tar_s
        df_shape.to_csv(os.path.join(output_dir, "train_data_shape.csv"), index=False)

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
        """ Restores data arrays from CSV files. Returns True if successful. """
        path_s = os.path.join(output_dir, "train_data_shape.csv")
        path_p = os.path.join(output_dir, "train_data_physics.csv")
        if not os.path.exists(path_s): return False
        try:
            df_s = pd.read_csv(path_s)
            self.data_shape = (df_s[['s11', 's22', 's12']].values.astype(np.float32), 
                               df_s[['target_stress']].values.astype(np.float32))
            
            if os.path.exists(path_p):
                df_p = pd.read_csv(path_p)
                self.data_phys = (df_p[['s11', 's22', 's12']].values.astype(np.float32), 
                                  df_p[['target_stress']].values.astype(np.float32), 
                                  df_p[['target_r']].values.astype(np.float32), 
                                  df_p[['sin2', 'cos2', 'sc']].values.astype(np.float32), 
                                  df_p[['mask']].values.astype(np.float32))
            print(f"📖 Restored original training data from: {output_dir}")
            return True
        except Exception as e:
            print(f"⚠️ Warning: Failed to load snapped data ({e}). Falling back to fresh generation.")
            return False

    def _generate_raw_data(self, needs_physics=True):
        """
        Internal mathematical generator using Sobol Sequences for uniform coverage.
        """
        data_cfg = self.config.data
        ref_stress = self.config.model.ref_stress
        n_gen = data_cfg.samples.get('loci', 1000)
        n_uni = data_cfg.samples.get('uniaxial', 0) if needs_physics else 0
        use_positive_shear = data_cfg.positive_shear

        # --- 1. SHAPE DATA GENERATION ---
        if n_gen > 0:
            # We use Sobol sequences to sample the spherical domain (theta, phi)
            # This provides better surface coverage than pure random sampling.
            sampler = qmc.Sobol(d=2, scramble=True)
            sample = sampler.random(n_gen)
            
            if use_positive_shear:
                phi_g = np.arccos(sample[:, 0]) # [0, pi/2]
            else:
                phi_g = np.arccos(sample[:, 0] * 2.0 - 1.0) # [0, pi]
            theta_g = sample[:, 1] * 2.0 * np.pi
            
            # Map spherical to the analytical surface radius
            radius = self.physics_bench.solve_radius(theta_g, phi_g)
            
            s12_g = radius * np.cos(phi_g)
            r_plane = radius * np.sin(phi_g)
            s11_g = r_plane * np.cos(theta_g)
            s22_g = r_plane * np.sin(theta_g)
            
            inputs_gen = np.stack([s11_g, s22_g, s12_g], axis=1).astype(np.float32)
            se_gen = np.ones((len(inputs_gen), 1), dtype=np.float32) * ref_stress
        else:
            inputs_gen = np.zeros((0, 3), dtype=np.float32)
            se_gen = np.zeros((0, 1), dtype=np.float32)

        # --- 1b. ANCHOR POINTS ---
        # Add critical physical directions (Tension, Biaxial, Shear)
        anchors_dir = np.array([[1,0,0],[0,1,0],[-1,0,0],[0,-1,0],[1,1,0],[-1,-1,0],[1,-1,0],[-1,1,0],[0,0,1]], dtype=np.float32)
        anchors_inputs = []
        for d in anchors_dir:
            scale = ref_stress / (self.physics_bench.equivalent_stress(d[0], d[1], d[2]) + 1e-12)
            anchors_inputs.append(d * scale)
        
        inputs_gen = np.concatenate([inputs_gen, np.array(anchors_inputs, dtype=np.float32)], axis=0)
        se_gen = np.concatenate([se_gen, np.ones((len(anchors_inputs), 1), dtype=np.float32) * ref_stress], axis=0)
        
        # --- 2. PHYSICS DATA GENERATION (Uniaxial R-values) ---
        if n_uni > 0:
            limit = 90.0 if use_positive_shear else 360.0
            angles = np.linspace(0, limit, n_uni, endpoint=True, dtype=np.float32)
            rads = np.radians(angles)
            sin_a, cos_a = np.sin(rads), np.cos(rads)
            
            # Map angle to unit stress
            u11, u22, u12 = cos_a**2, sin_a**2, sin_a*cos_a
            
            # Normalize to the target surface
            sigma_y = ref_stress / (self.physics_bench.equivalent_stress(u11, u22, u12) + 1e-12)
            inputs_uni = np.stack([u11*sigma_y, u22*sigma_y, u12*sigma_y], axis=1)
            se_uni = np.ones((n_uni, 1), dtype=np.float32) * ref_stress
            
            # Calculate Ground Truth R-values using the Generic physics methods
            r_vals = np.array([self.physics_bench.predict_r_value(a) for a in angles])[:, None]
            
            # Store geometric terms for GPU loss calculation
            geo_uni = np.stack([sin_a**2, cos_a**2, sin_a*cos_a], axis=1)
            mask_uni = np.ones((len(se_uni), 1), dtype=np.float32)
        else:
            inputs_uni = np.zeros((0, 3), dtype=np.float32)
            se_uni, r_vals, mask_uni = [np.zeros((0, 1), dtype=np.float32) for _ in range(3)]
            geo_uni = np.zeros((0, 3), dtype=np.float32)

        return (inputs_gen, se_gen), (inputs_uni, se_uni, r_vals, geo_uni, mask_uni)
