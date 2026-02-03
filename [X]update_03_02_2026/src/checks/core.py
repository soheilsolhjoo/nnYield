import numpy as np
import tensorflow as tf
import os
import pandas as pd
from scipy.stats import qmc
from ..model import HomogeneousYieldModel
from ..data_loader import YieldDataLoader

class BaseChecker:
    """
    Core Infrastructure for the Sanity Check Module.
    
    This class serves as the foundation for all validation checks (Physics, Diagnostics, Reporting).
    It centralizes the shared logic to ensure consistency across all tests.
    
    Key Responsibilities:
    1.  **Environment Setup**: Handles configuration parsing, directory creation, and data loading.
    2.  **Optimized Math Engine**: Provides a cached TensorFlow graph (`_predict_graph`) for fast 
        calculation of Neural Network values, gradients, and Hessians.
    3.  **Analytical Benchmark**: Implements the Hill48 yield criterion and its derivatives 
        to serve as the "Ground Truth" for comparison.
    4.  **Geometry & Sampling**: Generates random points uniformly distributed on the yield surface
        using Sobol sequences and spherical coordinate transformations.
    """
    
    def __init__(self, model, config, output_dir):
        """
        Initializes the checking environment.
        
        Args:
            model (tf.keras.Model): The trained Neural Network yield surface model.
            config (dict or object): The configuration containing physics parameters (Hill48) 
                                     and model settings (Ref Stress).
            output_dir (str): Path to the experiment output folder where plots/CSVs will be saved.
        """
        # 1. Handle Config Format (Support both Dictionary and Object access)
        if hasattr(config, 'to_dict'):
            self.config = config.to_dict()
        else:
            self.config = config
        
        self.model = model
        self.output_dir = output_dir
        
        # 2. Setup Output Directories
        # 'plots/' for visual figures, 'csv_data/' for raw metric files
        self.plot_dir = os.path.join(output_dir, "plots")
        self.csv_dir = os.path.join(output_dir, "csv_data")
        
        os.makedirs(self.plot_dir, exist_ok=True)
        os.makedirs(self.csv_dir, exist_ok=True)
        
        # 3. Initialize Data Loader
        # Used to access the same scaling/normalization logic as training
        self.loader = YieldDataLoader(self.config)
        
        # 4. Load Experimental Data (Optional)
        # Only needed if we want to overlay real experimental points on plots
        self.exp_data = self._load_experiments()

    def _load_experiments(self):
        """
        Attempts to load experimental data from the path defined in config.
        Returns:
            pd.DataFrame or None: The loaded data if file exists, else None.
        """
        path = self.config['data'].get('experimental_csv', None)
        if path and os.path.exists(path):
            return pd.read_csv(path)
        return None

    def _save_to_csv(self, dataframe, filename):
        """
        Helper to export metrics for external analysis (e.g. into Excel/Matlab).
        """
        save_path = os.path.join(self.csv_dir, filename)
        dataframe.to_csv(save_path, index=False)
        print(f"   -> Data exported to: {save_path}")

    # =========================================================================
    #  CORE MATH ENGINE: TENSORFLOW GRAPH (Value, Gradient, Hessian)
    # =========================================================================
    @tf.function(reduce_retracing=True)
    def _predict_graph(self, inputs):
        """
        A highly optimized TensorFlow Graph function that computes:
        1. Yield Surface Value (Predicted Equivalent Stress)
        2. First Derivatives (Gradients w.r.t input stress)
        3. Second Derivatives (Hessian Matrix)
        
        Why this is needed:
        - Calculating Hessians in eager execution is very slow.
        - This compiles the logic into a static graph (XLA) for speed.
        
        Args:
            inputs (tf.Tensor): Batch of stress tensors (N, 3) [S11, S22, S12]
            
        Returns:
            val_nn (tf.Tensor): Predicted values (N, 1)
            grads_nn (tf.Tensor): Gradients (N, 3)
            hess_nn (tf.Tensor): Hessian Matrices (N, 3, 3)
        """
        with tf.GradientTape(persistent=True) as tape2:
            tape2.watch(inputs)
            with tf.GradientTape() as tape1:
                tape1.watch(inputs)
                val_nn = self.model(inputs)
            
            # First Derivative (Gradient)
            grads_nn = tape1.gradient(val_nn, inputs)
            
        # Handle edge case: At initialization, gradients might be None
        if grads_nn is None:
            grads_nn = tf.zeros_like(inputs)
            hess_nn = tf.zeros((tf.shape(inputs)[0], 3, 3))
        else:
            # Second Derivative (Jacobian of the Gradient = Hessian)
            hess_nn = tape2.batch_jacobian(grads_nn, inputs)
            
        del tape2 # Clean up the persistent tape
        return val_nn, grads_nn, hess_nn

    def _get_predictions(self, s11, s22, s12):
        """
        The "One-Stop-Shop" for comparing Neural Network vs. Analytical Physics.
        
        Takes raw stress components, computes NN predictions (via the optimized graph),
        computes Analytical Hill48 predictions (via NumPy), and returns both.
        
        Args:
            s11, s22, s12 (np.array): Flattened arrays of stress components.
            
        Returns:
            Tuple: ( (NN_Val, NN_Grad, NN_Hess), (Bench_Val, Bench_Grad) )
        """
        # 1. PREPARE INPUTS FOR NEURAL NETWORK
        inputs = np.stack([s11, s22, s12], axis=1).astype(np.float32)
        inputs_tf = tf.constant(inputs)
        
        # 2. RUN OPTIMIZED NN CALCULATION
        val_nn_tf, grads_nn_tf, hess_nn_tf = self._predict_graph(inputs_tf)
        
        val_nn = val_nn_tf.numpy().flatten()
        grads_nn = grads_nn_tf.numpy()
        hess_nn = hess_nn_tf.numpy()

        # 3. RUN ANALYTICAL BENCHMARK (Hill48)
        # We use float64 here for maximum precision to ensure the "Ground Truth" is accurate.
        s11_64 = s11.astype(np.float64)
        s22_64 = s22.astype(np.float64)
        s12_64 = s12.astype(np.float64)
        
        # Load Hill48 Parameters (F, G, H, N) from config
        phys = self.config.get('physics', {})
        F, G, H, N = phys.get('F', 0.5), phys.get('G', 0.5), phys.get('H', 0.5), phys.get('N', 1.5)
        
        # A. Calculate Hill48 Equivalent Stress
        # Formula: sqrt( F*s22^2 + G*s11^2 + H*(s11-s22)^2 + 2*N*s12^2 )
        term = F*s22_64**2 + G*s11_64**2 + H*(s11_64-s22_64)**2 + 2*N*s12_64**2
        val_vm = np.sqrt(np.maximum(term, 1e-16)) 
        
        # B. Calculate Analytical Gradients
        # The derivative of sqrt(f(x)) is f'(x) / (2 * sqrt(f(x)))
        # Since our quadratic form is effectively 'sigma^2', the 2 cancels out.
        denom = val_vm 
        denom = np.where(denom < 1e-12, 1e-12, denom) # Avoid division by zero
        
        dg_d11 = (G*s11_64 + H*(s11_64-s22_64)) / denom
        dg_d22 = (F*s22_64 - H*(s11_64-s22_64)) / denom
        dg_d12 = (2*N*s12_64) / denom
        
        grads_vm = np.stack([dg_d11, dg_d22, dg_d12], axis=1)
        
        return (val_nn, grads_nn, hess_nn), (val_vm, grads_vm)

    # =========================================================================
    #  SAMPLING UTILITY
    # =========================================================================
    def _sample_points_on_surface(self, n_samples):
        """
        Generates random stress points that lie EXACTLY on the predicted yield surface.
        
        Method:
        1. Generate uniformly distributed directions in 3D space using Sobol sequences.
        2. Query the Neural Network to find the yield stress radius in each direction.
        3. Scale the unit vectors by this radius.
        
        This is crucial for checking Convexity, as we must verify the Hessian matrix
        specifically at the yield surface boundary (f=1), not arbitrary points.
        
        Args:
            n_samples (int): Number of points to generate (e.g., 4096).
            
        Returns:
            np.array: (N, 3) Array of stress points on the yield surface.
        """
        # 1. Sobol Sampling (Quasi-Monte Carlo)
        # Provides better coverage of the sphere than pure random sampling
        sampler = qmc.Sobol(d=2, scramble=True)
        m = int(np.ceil(np.log2(n_samples))) # Next power of 2
        sample = sampler.random(n=2**m)[:n_samples]
        
        # 2. Map Square [0,1]^2 to Spherical Coordinates (Upper Hemisphere)
        # Theta: Azimuthal angle [0, 2pi]
        theta = sample[:, 0] * 2 * np.pi
        
        # Phi: Polar angle. To sample uniformly on a sphere, we sample z ~ U[0,1]
        # and set phi = arccos(z). We only need the upper hemisphere (S12 >= 0) due to symmetry.
        z = sample[:, 1]
        phi = np.arccos(z)
        
        # 3. Convert Spherical to Cartesian Unit Vectors
        # R_plane is the projection radius on the S11-S22 plane
        s12_u = np.cos(phi)            # Vertical axis (Shear)
        r_plane_u = np.sin(phi)        # Horizontal radius
        s11_u = r_plane_u * np.cos(theta)
        s22_u = r_plane_u * np.sin(theta)
        
        unit_inputs = np.stack([s11_u, s22_u, s12_u], axis=1).astype(np.float32)
        
        # 4. Project onto the Neural Network's Yield Surface
        # Pred = Model(Unit_Vector).  Yield_Stress = Ref_Stress / Pred
        inputs_tf = tf.constant(unit_inputs)
        pred_se = self.model(inputs_tf).numpy().flatten()
        
        ref_stress = self.config['model']['ref_stress']
        radii = ref_stress / (pred_se + 1e-8)
        
        # 5. Scale the unit vectors
        surface_points = unit_inputs * radii[:, None]
        return surface_points