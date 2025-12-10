import tensorflow as tf
import numpy as np
from scipy.stats import qmc

class YieldDataLoader:
    def __init__(self, config):
        self.config = config

    def get_dataset(self):
        """
        Returns: (ds_shape, ds_phys, steps_per_epoch)
        """
        # 1. Check Config for Dual Stream Eligibility
        n_uni = self.config['data']['samples'].get('uniaxial', 0)
        
        # FIX: w_r is in training -> weights
        w_r = self.config['training']['weights'].get('r_value', 0.0)
        
        # FIX: batch_r_fraction is now in 'anisotropy_ratio', NOT 'training'
        # We use .get() for safety, but expect the key to exist based on new config
        ani_config = self.config.get('anisotropy_ratio', {})
        batch_r_frac = ani_config.get('batch_r_fraction', 0.0)
        is_enabled = ani_config.get('enabled', False)
        
        # Dual stream requires: Uniaxial samples exist, Weight > 0, Fraction > 0, and Enabled
        use_dual_stream = (n_uni > 0) and (w_r > 0) and (batch_r_frac > 0) and is_enabled

        # 2. Generate Raw Data (Includes Anchors now)
        data_shape, data_phys = self._generate_raw_data(needs_physics=use_dual_stream)

        # 3. Create Shape Dataset (Always exists)
        ds_shape = tf.data.Dataset.from_tensor_slices(data_shape)
        ds_shape = ds_shape.shuffle(len(data_shape[0])).repeat()

        # 4. Batching Logic
        total_batch = self.config['training']['batch_size']

        if use_dual_stream:
            # --- MODE B: DUAL STREAM ---
            n_phys_batch = int(total_batch * batch_r_frac)
            n_shape_batch = total_batch - n_phys_batch
            
            # Safety checks
            n_phys_batch = max(1, n_phys_batch)
            n_shape_batch = max(1, n_shape_batch)
            
            ds_shape = ds_shape.batch(n_shape_batch)
            
            ds_phys = tf.data.Dataset.from_tensor_slices(data_phys)
            ds_phys = ds_phys.shuffle(len(data_phys[0])).repeat()
            ds_phys = ds_phys.batch(n_phys_batch)
            
            steps = len(data_shape[0]) // n_shape_batch
            
            return ds_shape, ds_phys, steps
        else:
            # --- MODE A: SHAPE ONLY ---
            ds_shape = ds_shape.batch(total_batch)
            steps = len(data_shape[0]) // total_batch
            
            return ds_shape, None, steps

    def get_numpy_data(self):
        """
        Flattens the generated streams into single numpy arrays.
        Used for K-Fold Cross Validation and Sanity Checks.
        Returns: X, y_se, y_r
        """
        # Force generation of both streams to get maximum data
        (X_s, y_s), (X_p, y_p, r_p, _, _) = self._generate_raw_data(needs_physics=True)
        
        # Concatenate Shape and Physics inputs
        X = np.concatenate([X_s, X_p], axis=0)
        y_se = np.concatenate([y_s, y_p], axis=0)
        
        # Handle R-values (Shape data has 0 R-value target, but we mask it usually)
        # For K-Fold splitting, we just need the arrays to align
        r_s = np.zeros((len(X_s), 1), dtype=np.float32)
        y_r = np.concatenate([r_s, r_p], axis=0)
        
        return X, y_se, y_r

    def _generate_raw_data(self, needs_physics=True):
        n_gen = self.config['data']['samples'].get('loci', 1000)
        n_uni = self.config['data']['samples'].get('uniaxial', 0) if needs_physics else 0
        
        ref_stress = self.config['model']['ref_stress']
        phys = self.config.get('physics', {})
        F, G, H, N = phys.get('F', 0.5), phys.get('G', 0.5), phys.get('H', 0.5), phys.get('N', 1.5)
        C11, C22, C12, C66 = G+H, F+H, -2*H, 2*N
        
        use_symmetry = self.config['data'].get('symmetry', True)

        # --- 1. SHAPE DATA (Random Loci) ---
        if n_gen > 0:
            sampler = qmc.Sobol(d=2, scramble=True)
            sample = sampler.random(n_gen)
            max_shear = ref_stress / np.sqrt(C66)
            
            if use_symmetry:
                s12_g = (sample[:, 0] * max_shear).astype(np.float32)
            else:
                s12_g = ((sample[:, 0] * 2.0 - 1.0) * max_shear).astype(np.float32)
                
            theta_g = (sample[:, 1] * 2.0 * np.pi).astype(np.float32)
            rhs = np.maximum(ref_stress**2 - C66*s12_g**2, 0)
            c, s = np.cos(theta_g), np.sin(theta_g)
            denom = C11*c**2 + C22*s**2 + C12*c*s
            radius = np.sqrt(rhs / (denom + 1e-8))
            
            inputs_gen = np.stack([radius*c, radius*s, s12_g], axis=1)
            se_gen = np.ones((n_gen, 1), dtype=np.float32) * ref_stress
        else:
            inputs_gen = np.zeros((0, 3), dtype=np.float32)
            se_gen = np.zeros((0, 1), dtype=np.float32)

        # --- 1b. INJECT ANCHOR POINTS ---
        # Critical stress states to enforce physics at boundaries
        anchors_dir = np.array([
            [1.0, 0.0, 0.0],  # Uniaxial X
            [0.0, 1.0, 0.0],  # Uniaxial Y
            [1.0, 1.0, 0.0],  # Equibiaxial
            [0.0, 0.0, 1.0],  # Pure Shear (The s12!=0 case)
            [0.0, 0.0, 0.5]   # Intermediate Shear
        ], dtype=np.float32)

        # Calculate True Hill48 Radius for these directions
        s11, s22, s12 = anchors_dir[:,0], anchors_dir[:,1], anchors_dir[:,2]
        term1 = F * (s22**2)
        term2 = G * (s11**2)
        term3 = H * ((s11 - s22)**2)
        term4 = 2 * N * (s12**2)
        
        # Effective stress for unit vector:
        eff_stress_unit = np.sqrt(term1 + term2 + term3 + term4 + 1e-8)
        
        # Scaling factor to put them on the yield surface
        scale_factors = ref_stress / eff_stress_unit
        anchors_inputs = anchors_dir * scale_factors[:, None]
        anchors_targets = np.ones((len(anchors_inputs), 1), dtype=np.float32) * ref_stress

        # Append Anchors to Shape Data
        inputs_gen = np.concatenate([inputs_gen, anchors_inputs], axis=0)
        se_gen = np.concatenate([se_gen, anchors_targets], axis=0)

        data_shape = (inputs_gen, se_gen)
        
        # --- 2. PHYSICS DATA ---
        if n_uni > 0:
            sampler_uni = qmc.Sobol(d=1, scramble=True)
            sample_uni = sampler_uni.random(n_uni)
            
            if use_symmetry:
                alpha_uni = (sample_uni[:, 0] * np.pi / 2.0).astype(np.float32)
            else:
                alpha_uni = (sample_uni[:, 0] * np.pi).astype(np.float32)
            
            sin_a, cos_a = np.sin(alpha_uni), np.cos(alpha_uni)
            u11, u22, u12 = cos_a**2, sin_a**2, sin_a*cos_a
            
            hill_u = C11*u11**2 + C22*u22**2 + C12*u11*u22 + C66*u12**2
            scale_uni = ref_stress / np.sqrt(hill_u + 1e-8)
            
            inputs_uni = np.stack([u11*scale_uni, u22*scale_uni, u12*scale_uni], axis=1)
            se_uni = np.ones((n_uni, 1), dtype=np.float32) * ref_stress
            
            # Targets
            scale_g = 1.0 / (2.0 * ref_stress)
            g11 = scale_g * (2*C11*inputs_uni[:,0] + C12*inputs_uni[:,1])
            g22 = scale_g * (2*C22*inputs_uni[:,1] + C12*inputs_uni[:,0])
            g12 = scale_g * (2*C66*inputs_uni[:,2])
            
            d_eps_t = -(g11 + g22)
            d_eps_w = g11*sin_a**2 + g22*cos_a**2 - 2*g12*sin_a*cos_a
            r_calc = d_eps_w / (d_eps_t + 1e-8)
            
            valid = (np.abs(r_calc) < 20.0) & (np.abs(d_eps_t) > 1e-6)
            
            if np.sum(valid) > 0:
                inputs_uni = inputs_uni[valid]
                se_uni = se_uni[valid]
                r_vals = r_calc[valid][:, None]
                geo_uni = np.stack([sin_a**2, cos_a**2, sin_a*cos_a], axis=1)[valid]
                mask_uni = np.ones((len(se_uni), 1), dtype=np.float32)
            else:
                inputs_uni = np.zeros((0, 3), dtype=np.float32)
                se_uni = np.zeros((0, 1), dtype=np.float32)
                r_vals = np.zeros((0, 1), dtype=np.float32)
                geo_uni = np.zeros((0, 3), dtype=np.float32)
                mask_uni = np.zeros((0, 1), dtype=np.float32)
        else:
            inputs_uni = np.zeros((0, 3), dtype=np.float32)
            se_uni = np.zeros((0, 1), dtype=np.float32)
            r_vals = np.zeros((0, 1), dtype=np.float32)
            geo_uni = np.zeros((0, 3), dtype=np.float32)
            mask_uni = np.zeros((0, 1), dtype=np.float32)

        data_phys = (inputs_uni, se_uni, r_vals, geo_uni, mask_uni)
        
        return data_shape, data_phys