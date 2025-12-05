import tensorflow as tf
import numpy as np
from scipy.stats import qmc

class YieldDataLoader:
    def __init__(self, config):
        self.config = config

    def get_dataset(self):
        """
        Returns: (ds_shape, ds_phys, steps_per_epoch)
        ds_phys is None if R-value training is disabled.
        """
        # 1. Check Config for Dual Stream Eligibility
        n_uni = self.config['data']['samples'].get('uniaxial', 0)
        w_r = self.config['training']['weights'].get('r_value', 0.0)
        batch_r_frac = self.config['training'].get('batch_r_fraction', 0.0)
        
        use_dual_stream = (n_uni > 0) and (w_r > 0) and (batch_r_frac > 0)

        # 2. Generate Raw Data
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
            
            # Safety: Ensure at least 1 point in each stream
            n_phys_batch = max(1, n_phys_batch)
            n_shape_batch = max(1, n_shape_batch)
            
            ds_shape = ds_shape.batch(n_shape_batch)
            
            # Physics Dataset
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
        # Helper for Sanity Check (Always returns both)
        return self._generate_raw_data(needs_physics=True)

    def _generate_raw_data(self, needs_physics=True):
        n_gen = self.config['data']['samples'].get('loci', 1000)
        n_uni = self.config['data']['samples'].get('uniaxial', 0) if needs_physics else 0
        
        ref_stress = self.config['model']['ref_stress']
        phys = self.config.get('physics', {})
        F, G, H, N = phys.get('F', 0.5), phys.get('G', 0.5), phys.get('H', 0.5), phys.get('N', 1.5)
        C11, C22, C12, C66 = G+H, F+H, -2*H, 2*N
        
        use_symmetry = self.config['data'].get('symmetry', True)

        # --- 1. SHAPE DATA ---
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
            inputs_gen = np.zeros((1, 3), dtype=np.float32)
            se_gen = np.zeros((1, 1), dtype=np.float32)
            
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
                # Fallback
                inputs_uni = np.zeros((1, 3), dtype=np.float32)
                se_uni = np.zeros((1, 1), dtype=np.float32)
                r_vals = np.zeros((1, 1), dtype=np.float32)
                geo_uni = np.zeros((1, 3), dtype=np.float32)
                mask_uni = np.zeros((1, 1), dtype=np.float32)
        else:
            inputs_uni = np.zeros((1, 3), dtype=np.float32)
            se_uni = np.zeros((1, 1), dtype=np.float32)
            r_vals = np.zeros((1, 1), dtype=np.float32)
            geo_uni = np.zeros((1, 3), dtype=np.float32)
            mask_uni = np.zeros((1, 1), dtype=np.float32)

        data_phys = (inputs_uni, se_uni, r_vals, geo_uni, mask_uni)

        # --- DEBUG: VISUALIZE GENERATED DATA ---
        # Place this block just before 'return data_shape, data_phys'
        try:
            import matplotlib.pyplot as plt
            
            # 1. Visualize Loci (Shape Stream)
            if n_gen > 0:
                fig = plt.figure(figsize=(12, 5))
                
                # 3D View
                ax1 = fig.add_subplot(121, projection='3d')
                s11, s22, s12 = inputs_gen[:,0], inputs_gen[:,1], inputs_gen[:,2]
                sc1 = ax1.scatter(s11, s22, s12, c=s12, cmap='viridis', s=1, alpha=0.5)
                ax1.set_title(f'Shape Stream (N={len(s11)})')
                ax1.set_xlabel('S11'); ax1.set_ylabel('S22'); ax1.set_zlabel('S12')
                
                # 2D Projection
                ax2 = fig.add_subplot(122)
                sc2 = ax2.scatter(s11, s22, c=s12, cmap='viridis', s=2, alpha=0.5)
                ax2.set_title('2D Projection (Color=Shear)')
                ax2.set_xlabel('S11'); ax2.set_ylabel('S22'); ax2.axis('equal')
                plt.colorbar(sc2, label='S12')
                
                plt.savefig("debug_loader_loci.png"); plt.close()
                print("   [DEBUG] Saved 'debug_loader_loci.png'")

            # 2. Visualize R-values (Physics Stream)
            # We use the local variables 'alpha_uni' and 'r_calc' from the Physics block
            if n_uni > 0 and 'valid' in locals() and np.sum(valid) > 0:
                # Filter to show only valid points
                alpha_plot = alpha_uni[valid]
                r_plot = r_calc[valid]
                
                plt.figure(figsize=(10, 6))
                plt.scatter(np.degrees(alpha_plot), r_plot, s=10, c='blue', label='Generated Data')
                plt.title(f"Physics Stream: R-values (N={len(r_plot)})")
                plt.xlabel("Loading Angle Alpha (deg)")
                plt.ylabel("R-value")
                plt.grid(True, alpha=0.3); plt.legend()
                plt.savefig("debug_loader_r_values.png"); plt.close()
                print("   [DEBUG] Saved 'debug_loader_r_values.png'")
                
        except Exception as e:
            print(f"   [DEBUG] Plotting failed: {e}")
        # ---------------------------------------
        
        return data_shape, data_phys