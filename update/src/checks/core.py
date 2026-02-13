"""
Core Infrastructure for the nnYield Sanity Check Module.

This module provides the BaseChecker class, which serves as the foundation for all 
specific validation modules (Physics, Diagnostics, Reporting). It centralizes 
shared logic such as environment setup, optimized math engines, and sampling utilities.
"""

import numpy as np
import tensorflow as tf
import os
import pandas as pd
from scipy.stats import qmc
from ..model import HomogeneousYieldModel
from ..data_loader import YieldDataLoader
from ..config import Config
from ..physics_models.factory import get_physics_model

class BaseChecker:
    """
    Foundation class for material model validation and numerical diagnostics.
    
    Key Responsibilities:
    1. **Modular Benchmark**: Instantiates the correct analytical model (Hill48, Barlat, etc.) 
       based on the experiment configuration.
    2. **Performance Math**: Provides a static-graph TensorFlow engine (`_predict_graph`) 
       for high-speed calculation of values, gradients, and Hessians.
    3. **Data Management**: Handles the loading of experimental datasets and the export 
       of diagnostic CSV files.
    4. **Geometric Sampling**: Implements uniform surface projection using Sobol sequences.
    """
    
    def __init__(self, model, config: Config, output_dir):
        """
        Initializes the checking environment.
        
        Args:
            model (HomogeneousYieldModel): The trained neural network to be validated.
            config (Config): The project configuration object.
            output_dir (str): The directory where diagnostic artifacts will be stored.
        """
        self.config = config
        self.model = model
        self.output_dir = output_dir
        
        # Instantiate the generic analytical benchmark via the factory.
        # This allows all diagnostic checks to automatically sync with the selected model.
        self.physics_bench = get_physics_model(self.config)
        
        # Setup standardized directory structure for outputs.
        self.plot_dir = os.path.join(output_dir, "plots")
        self.csv_dir = os.path.join(output_dir, "csv_data")
        os.makedirs(self.plot_dir, exist_ok=True)
        os.makedirs(self.csv_dir, exist_ok=True)
        
        # Initialize helper components for data handling.
        self.loader = YieldDataLoader(self.config)
        self.exp_data = self._load_experiments()

    def _load_experiments(self):
        """
        Attempts to load experimental data from the path specified in the config.
        Returns:
            pd.DataFrame or None: The loaded data if found, else None.
        """
        path = getattr(self.config.data, 'experimental_csv', None)
        if path and os.path.exists(path):
            return pd.read_csv(path)
        return None

    def _save_to_csv(self, dataframe, filename):
        """
        Helper to export diagnostic metrics for external plotting or research.
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
        TRICK: COMPILED DIAGNOSTICS.
        
        Validation often requires thousands of calls to the Hessian (2nd derivative).
        By wrapping the AutoDiff logic in @tf.function, we compile it into a 
        high-performance static graph, making checks like 'Full Domain Heatmaps' 
        run significantly faster than eager execution.
        
        Returns:
            val_nn (tf.Tensor): Equivalent stress predictions.
            grads_nn (tf.Tensor): First derivatives (Surface Normality).
            hess_nn (tf.Tensor): Second derivatives (Surface Curvature/Convexity).
        """
        with tf.GradientTape(persistent=True) as tape2:
            tape2.watch(inputs)
            with tf.GradientTape() as tape1:
                tape1.watch(inputs)
                val_nn = self.model(inputs)
            
            # First Derivative (Gradient)
            grads_nn = tape1.gradient(val_nn, inputs)
            
        if grads_nn is None:
            # Fallback for uninitialized models
            grads_nn = tf.zeros_like(inputs)
            hess_nn = tf.zeros((tf.shape(inputs)[0], 3, 3))
        else:
            # Second Derivative (Jacobian of the Gradient)
            hess_nn = tape2.batch_jacobian(grads_nn, inputs)
            
        del tape2 
        return val_nn, grads_nn, hess_nn

    def _get_predictions(self, s11, s22, s12):
        """
        Comparison engine between the Neural Network and the Analytical Benchmark.
        
        Args:
            s11, s22, s12 (np.array): Flattened arrays of stress components.
            
        Returns:
            nn_data: (Values, Gradients, Hessians)
            bench_data: (Values, Gradients)
        """
        # 1. GPU INFERENCE (Neural Network)
        inputs = np.stack([s11, s22, s12], axis=1).astype(np.float32)
        val_nn_tf, grads_nn_tf, hess_nn_tf = self._predict_graph(tf.constant(inputs))
        
        val_nn = val_nn_tf.numpy().flatten()
        grads_nn = grads_nn_tf.numpy()
        hess_nn = hess_nn_tf.numpy()

        # 2. CPU ANALYTICAL GROUND TRUTH (Modular Model)
        # Use float64 for ground truth to ensure R-value precision.
        s11_64 = s11.astype(np.float64)
        s22_64 = s22.astype(np.float64)
        s12_64 = s12.astype(np.float64)
        
        # Query the modular physics factory instance for values and gradients.
        val_bench = self.physics_bench.equivalent_stress(s11_64, s22_64, s12_64)
        g11, g22, g12 = self.physics_bench.gradients(s11_64, s22_64, s12_64)
        grads_bench = np.stack([g11, g22, g12], axis=1)
        
        return (val_nn, grads_nn, hess_nn), (val_bench, grads_bench)

    # =========================================================================
    #  SAMPLING UTILITY
    # =========================================================================
    def _sample_points_on_surface(self, n_samples):
        """
        TRICK: UNIFORM SURFACE PROJECTION.
        
        Generates random stress points that lie EXACTLY on the LEARNED yield surface.
        Used primarily for convexity checks where curvature must be evaluated 
        specifically at the boundary (f=1).
        
        Method:
        1. Generate low-discrepancy spherical directions using Sobol sequences.
        2. Query the model to find the yield radius in each direction.
        3. Scale unit vectors to the surface boundary.
        """
        # Sobol sequence provides much better domain coverage than pure random noise.
        sampler = qmc.Sobol(d=2, scramble=True)
        m = int(np.ceil(np.log2(n_samples))) 
        sample = sampler.random(n=2**m)[:n_samples]
        
        # Convert 2D square sample to spherical angles.
        theta = sample[:, 0] * 2 * np.pi
        phi = np.arccos(sample[:, 1])
        
        # Map to unit Cartesian directions.
        s12_u = np.cos(phi)            
        r_plane_u = np.sin(phi)        
        s11_u = r_plane_u * np.cos(theta)
        s22_u = r_plane_u * np.sin(theta)
        
        unit_inputs = np.stack([s11_u, s22_u, s12_u], axis=1).astype(np.float32)
        
        # Scale to boundary: R = RefStress / Model(UnitVector).
        pred_se = self.model(tf.constant(unit_inputs)).numpy().flatten()
        ref_stress = self.config.model.ref_stress
        radii = ref_stress / (pred_se + 1e-8)
        
        return unit_inputs * radii[:, None]
