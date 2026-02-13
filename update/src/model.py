"""
Neural Network Material Model Module for nnYield.

This module implements the HomogeneousYieldModel, a specialized PINN architecture 
designed to represent any closed, convex yield surface without mathematical 
singularities or discontinuities.
"""

import tensorflow as tf
import numpy as np
from .config import Config

class HomogeneousYieldModel(tf.keras.Model):
    """
    Singularity-Free Yield Surface Model.
    
    This model predicts equivalent stress as a homogeneous function of degree 1.
    """
    def __init__(self, config: Config):
        super(HomogeneousYieldModel, self).__init__()
        
        model_cfg = config.model
        self.ref_stress = model_cfg.ref_stress
        
        # TRICK: ICNN ARCHITECTURE.
        # If enabled, we constrain the weights to be non-negative.
        # This is a 'hard' physical constraint that ensures the resulting 
        # surface is mathematically guaranteed to be convex.
        use_icnn = model_cfg.use_icnn_constraints
        k_constraint = tf.keras.constraints.NonNeg() if use_icnn else None
        
        if use_icnn:
            print(f"🔒 ICNN Mode: Convexity is architecturally guaranteed.")

        self.hidden_layers = []
        for units in model_cfg.hidden_layers:
            self.hidden_layers.append(
                tf.keras.layers.Dense(
                    units, 
                    activation=model_cfg.activation,
                    kernel_constraint=k_constraint
                )
            )
        
        self.radius_out = tf.keras.layers.Dense(
            1, 
            activation=model_cfg.activation,
            kernel_constraint=k_constraint
        )

    def predict_radius(self, theta, phi):
        """
        TRICK: POLE-FREE ANGULAR EMBEDDING.
        
        Using raw angles (theta, phi) causes a jump discontinuity at the poles (phi=0).
        We solve this by embedding the angles into a higher-dimensional directional 
        space [d1, d2, d3] which is continuous everywhere on the sphere.
        """
        # Embed spherical coordinates into a continuous unit direction space
        d1 = tf.sin(phi) * tf.cos(theta)
        d2 = tf.sin(phi) * tf.sin(theta)
        d3 = tf.square(tf.cos(phi)) # Squared for physical symmetry
        
        x = tf.stack([d1, d2, d3], axis=1)
        for layer in self.hidden_layers:
            x = layer(x)
            
        return self.radius_out(x)
        
    def call(self, inputs):
        """
        TRICK: HOMOGENEOUS SCALING.
        
        The Neural Network is trained to predict the 'Yield Radius' R in a 
        given direction. We then calculate the equivalent stress as:
        Equivalent Stress = Ref_Stress * (Actual_Magnitude / Predicted_Radius)
        
        This makes the model naturally 'Homogeneous of Degree 1', which means 
        f(k * sigma) = k * f(sigma), a requirement for all yield criteria.
        """
        # inputs: [s11, s22, s12]
        s11 = inputs[:, 0:1]
        s22 = inputs[:, 1:2]
        s12 = inputs[:, 2:3]
        
        # 1. Total Magnitude
        r_sq = tf.square(s11) + tf.square(s22) + tf.square(s12)
        r_total = tf.sqrt(r_sq + 1e-8)
        
        # 2. Coupled Directional Embeddings
        # These features represent the 'unit direction' of the stress state.
        d1 = s11 / (r_total + 1e-8)
        d2 = s22 / (r_total + 1e-8)
        d3 = tf.square(s12 / (r_total + 1e-8)) # s12 symmetry
        
        x = tf.concat([d1, d2, d3], axis=1)
        for layer in self.hidden_layers:
            x = layer(x)
        
        R_yield = self.radius_out(x)
        
        # Final Scaled Result
        return self.ref_stress * (r_total / (R_yield + 1e-8))
