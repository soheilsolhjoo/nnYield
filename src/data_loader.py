import tensorflow as tf
import numpy as np
from scipy.stats import qmc

class YieldDataLoader:
    """
    Data Pipeline for Physics-Informed Yield Surface Training.
    
    This class serves as the 'Ground Truth Generator'. It creates synthetic training data 
    based on the analytical Hill48 yield criterion.
    
    Architectural Strategy: 'Dual Stream'
    -------------------------------------
    To train a physics-informed model effectively, we generate two distinct streams of data:
    
    1. Shape Stream (General Loci):
       - WHAT: Random stress points scattered across the entire 3D stress space (Sobol sampling).
       - WHY: Teaches the model the overall size, convexity, and ellipsoid shape of the yield surface.
       - TARGETS: Yield Stress (Scalar).
       
    2. Physics Stream (Uniaxial/Experimental):
       - WHAT: Specific stress vectors corresponding to uniaxial tension tests at strictly 
               equidistant angles (e.g., 0, 10, 20... 90 degrees).
       - WHY: Teaches the model the precise slope (gradients) needed for accurate R-value prediction.
       - TARGETS: Yield Stress (Scalar) AND R-values (Gradients).
       
    The get_dataset() method combines these into efficient TensorFlow Datasets.
    """
    
    def __init__(self, config):
        """
        Args:
            config: The strict Config object from src/config.py.
        """
        self.config = config

    def get_dataset(self):
        """
        Builds the final TensorFlow Dataset pipeline for training.
        
        Logic:
        - Checks configuration to see if Dual Stream (Physics) training is enabled.
        - Generates raw synthetic data using internal physics engine.
        - Wraps data in tf.data.Dataset objects.
        - Applies batching, shuffling, and repeating strategies.
        
        Returns:
            ds_shape (tf.data.Dataset): The main dataset (random loci).
            ds_phys (tf.data.Dataset): The physics dataset (uniaxial points) or None.
            steps_per_epoch (int): Calculated number of batches to run per epoch.
        """
        # --- 1. CONFIGURATION CHECK ---
        # Accessors updated to match src/config.py Dataclass structure
        
        # 'samples' is a dict inside DataConfig
        n_uni = self.config.data.samples.get('uniaxial', 0)
        
        # 'weights' is a WeightsConfig object inside TrainingConfig
        w_r = self.config.training.weights.r_value
        
        # 'anisotropy_ratio' is an AnisotropyConfig object
        batch_r_frac = self.config.anisotropy_ratio.batch_r_fraction
        is_enabled = self.config.anisotropy_ratio.enabled
        
        # Criteria: We need samples, a non-zero loss weight, a non-zero batch fraction,
        # and the feature explicitly enabled in config.
        use_dual_stream = (n_uni > 0) and (w_r > 0) and (batch_r_frac > 0) and is_enabled

        # --- 2. GENERATE RAW DATA ---
        # Calls the internal engine to create synthetic Hill48 data (Numpy arrays).
        # We get two tuples: one for the Shape stream, one for the Physics stream.
        data_shape, data_phys = self._generate_raw_data(needs_physics=use_dual_stream)

        # --- 3. CREATE SHAPE DATASET (Stream 1) ---
        # Create dataset from tensor slices
        # Structure: (Inputs, Target_Stress)
        ds_shape = tf.data.Dataset.from_tensor_slices(data_shape)
        
        # Shuffle buffer size = total samples to ensure good randomization.
        # Repeat() allows the dataset to flow infinitely (epochs controlled by Trainer loop).
        ds_shape = ds_shape.shuffle(len(data_shape[0])).repeat()

        # --- 4. BATCHING STRATEGY ---
        total_batch = self.config.training.batch_size

        if use_dual_stream:
            # === MODE A: DUAL STREAM (Mixed Batches) ===
            # We split the batch budget: X% Physics points, (100-X)% Shape points.
            # This ensures every training step sees both general shape data
            # AND specific anisotropy data (R-values).
            
            n_phys_batch = int(total_batch * batch_r_frac)
            n_shape_batch = total_batch - n_phys_batch
            
            # Safety: Ensure at least 1 sample from each stream to avoid empty tensors
            n_phys_batch = max(1, n_phys_batch)
            n_shape_batch = max(1, n_shape_batch)
            
            # Batch the Shape stream
            ds_shape = ds_shape.batch(n_shape_batch)
            
            # Create and Batch the Physics stream separately
            # Structure: (Inputs, Target_Stress, Target_R, Geometry)
            ds_phys = tf.data.Dataset.from_tensor_slices(data_phys)
            ds_phys = ds_phys.shuffle(len(data_phys[0])).repeat()
            ds_phys = ds_phys.batch(n_phys_batch)
            
            # Define 1 Epoch = One full pass through the Shape data
            steps = len(data_shape[0]) // n_shape_batch
            
            return ds_shape, ds_phys, steps
        else:
            # === MODE B: SINGLE STREAM (Shape Only) ===
            # Standard batching if physics constraints are disabled.
            ds_shape = ds_shape.batch(total_batch)
            steps = len(data_shape[0]) // total_batch
            
            return ds_shape, None, steps

    def get_numpy_data(self):
        """
        Helper method to retrieve all generated data as flat Numpy arrays.
        Useful for K-Fold Cross Validation or Debugging.
        """
        # Force generation of both streams to utilize all available data
        (X_s, y_s), (X_p, y_p, r_p, _) = self._generate_raw_data(needs_physics=True)
        
        # Concatenate inputs and stress targets from both streams
        X = np.concatenate([X_s, X_p], axis=0)
        y_se = np.concatenate([y_s, y_p], axis=0)
        
        # Handle R-values (Pad shape data with zeros)
        r_s = np.zeros((len(X_s), 1), dtype=np.float32)
        y_r = np.concatenate([r_s, r_p], axis=0)
        
        return X, y_se, y_r

    def _generate_raw_data(self, needs_physics=True):
        """
        Internal engine to generate synthetic data points from Hill48 physics.
        
        Features:
        - **Sobol Sampling** (Shape): Random points covering the 3D stress sphere.
        - **Linspace Sampling** (Physics): Equidistant points (0, 5, ... 90 deg) for R-curves.
        - **Anchor Injection**: Fixed points (Uniaxial X/Y, Equi-Biaxial).
        """
        # Load Sample Counts
        n_gen = self.config.data.samples.get('loci', 1000)
        n_uni = self.config.data.samples.get('uniaxial', 0) if needs_physics else 0
        
        # Load Physics Parameters (Hill48 Coefficients)
        ref_stress = self.config.model.ref_stress
        
        # PhysicsConfig object
        phys = self.config.physics
        F, G, H, N = phys.F, phys.G, phys.H, phys.N
        
        # Derived stiffness-like coefficients
        C11, C22, C12, C66 = G+H, F+H, -2*H, 2*N
        use_symmetry = self.config.data.symmetry

        # =========================================================
        # 1. SHAPE DATA GENERATION (Random Loci - Sobol)
        # =========================================================
        if n_gen > 0:
            sampler = qmc.Sobol(d=2, scramble=True)
            m = int(np.ceil(np.log2(n_gen)))
            sample = sampler.random(2**m)[:n_gen] 
            
            # D1: Shear Stress (S12)
            max_shear = ref_stress / np.sqrt(C66)
            if use_symmetry:
                s12_g = (sample[:, 0] * max_shear).astype(np.float32)
            else:
                s12_g = ((sample[:, 0] * 2.0 - 1.0) * max_shear).astype(np.float32)
            
            # D2: Angle in S11-S22 plane (Theta)
            theta_g = (sample[:, 1] * 2.0 * np.pi).astype(np.float32)
            
            # Analytically solve for Radius
            rhs = np.maximum(ref_stress**2 - C66*s12_g**2, 0)
            c, s = np.cos(theta_g), np.sin(theta_g)
            denom = C11*c**2 + C22*s**2 + C12*c*s
            radius = np.sqrt(rhs / (denom + 1e-8))
            
            inputs_gen = np.stack([radius*c, radius*s, s12_g], axis=1)
            se_gen = np.ones((n_gen, 1), dtype=np.float32) * ref_stress
        else:
            inputs_gen = np.zeros((0, 3), dtype=np.float32)
            se_gen = np.zeros((0, 1), dtype=np.float32)

        # =========================================================
        # 1b. INJECT ANCHOR POINTS
        # =========================================================
        anchors_dir = np.array([
            [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0],
            [1.0, 1.0, 0.0], [-1.0, -1.0, 0.0], [1.0, -1.0, 0.0], [-1.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ], dtype=np.float32)

        s11, s22, s12 = anchors_dir[:,0], anchors_dir[:,1], anchors_dir[:,2]
        term = F*s22**2 + G*s11**2 + H*(s11-s22)**2 + 2*N*s12**2
        scale_factors = ref_stress / np.sqrt(term + 1e-8)
        
        anchors_inputs = anchors_dir * scale_factors[:, None]
        anchors_targets = np.ones((len(anchors_inputs), 1), dtype=np.float32) * ref_stress

        inputs_gen = np.concatenate([inputs_gen, anchors_inputs], axis=0)
        se_gen = np.concatenate([se_gen, anchors_targets], axis=0)
        
        # Shape Stream Tuple: (Inputs, Targets)
        data_shape = (inputs_gen, se_gen)
        
        # =========================================================
        # 2. PHYSICS DATA GENERATION (Uniaxial Tests - Linspace)
        # =========================================================
        if n_uni > 0:
            # [UPDATED] Use Linspace for strict equidistance
            # This ensures we hit 0, 90, and everything in between perfectly.
            limit = np.pi / 2.0 if use_symmetry else 2.0 * np.pi
            
            # Generate n_uni points evenly spaced
            alpha_uni = np.linspace(0, limit, n_uni, endpoint=True, dtype=np.float32)
            
            sin_a, cos_a = np.sin(alpha_uni), np.cos(alpha_uni)
            
            # Unit vector components for Uniaxial Tension at angle alpha
            u11, u22, u12 = cos_a**2, sin_a**2, sin_a*cos_a
            
            # Scale to lie exactly on the Hill48 Yield Surface
            term_u = F*u22**2 + G*u11**2 + H*(u11-u22)**2 + 2*N*u12**2
            scale_uni = ref_stress / np.sqrt(term_u + 1e-8)
            
            inputs_uni = np.stack([u11*scale_uni, u22*scale_uni, u12*scale_uni], axis=1)
            se_uni = np.ones((n_uni, 1), dtype=np.float32) * ref_stress
            
            # Analytical Gradients (for R-value ground truth)
            s11, s22, s12 = inputs_uni[:,0], inputs_uni[:,1], inputs_uni[:,2]
            denom = ref_stress 
            
            g11 = (G*s11 + H*(s11-s22)) / denom
            g22 = (F*s22 - H*(s11-s22)) / denom
            g12 = (2*N*s12) / denom
            
            d_eps_t = -(g11 + g22)
            d_eps_w = g11*sin_a**2 + g22*cos_a**2 - 2*g12*sin_a*cos_a
            
            # R-value Calculation
            r_calc = d_eps_w / (d_eps_t + 1e-8)
            
            # Filtering: Remove points where thickness strain is ~0 (Singularity)
            valid = (np.abs(r_calc) < 20.0) & (np.abs(d_eps_t) > 1e-6)
            
            if np.sum(valid) > 0:
                inputs_uni = inputs_uni[valid]
                se_uni = se_uni[valid]
                r_vals = r_calc[valid][:, None]
                geo_uni = np.stack([sin_a**2, cos_a**2, sin_a*cos_a], axis=1)[valid]
            else:
                inputs_uni, se_uni, r_vals, geo_uni = [np.array([]) for _ in range(4)]
        else:
            inputs_uni = np.zeros((0, 3), dtype=np.float32)
            se_uni, r_vals = [np.zeros((0, 1), dtype=np.float32) for _ in range(2)]
            geo_uni = np.zeros((0, 3), dtype=np.float32)

        # Physics Stream Tuple: (Inputs, Target_Stress, Target_R, Geometry)
        data_phys = (inputs_uni, se_uni, r_vals, geo_uni)
        
        return data_shape, data_phys