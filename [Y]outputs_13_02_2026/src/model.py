"""
Neural Network Material Model Module for nnYield.

This module implements the HomogeneousYieldModel, a neural network designed to 
learn equivalent stress functions (Yield Surfaces). 

The model uses a unique directional embedding strategy to ensure that the 
predicted surface is continuous and free of mathematical singularities, 
even at the shear poles.
"""

import tensorflow as tf
import numpy as np
from .config import Config

class HomogeneousYieldModel(tf.keras.Model):
    """
    Neural Network representation of a material yield surface.
    
    Architecture:
    - Input: Cartesian stress components [S11, S22, S12].
    - Embedding: Maps stresses to a singularity-free unit directional space.
    - Hidden Layers: Standard Dense layers or ICNN-constrained layers.
    - Output: The equivalent yield radius (R_yield) at that direction.
    """
    def __init__(self, config: Config):
        """
        Initializes the model architecture.

        Args:
            config (Config): The experiment configuration object.
        """
        super(HomogeneousYieldModel, self).__init__()
        
        model_cfg = config.model
        self.ref_stress = model_cfg.ref_stress
        
        # ICNN (Input Convex Neural Network) LOGIC:
        # If enabled, weights are constrained to be non-negative.
        # This mathematically guarantees that the learned surface is convex,
        # which is a fundamental requirement of classical plasticity.
        use_icnn = model_cfg.use_icnn_constraints
        k_constraint = tf.keras.constraints.NonNeg() if use_icnn else None
        
        if use_icnn:
            print(f"🔒 ICNN Mode Enabled: Architecture forces convexity via Non-Negative weights.")

        # Construct hidden layers dynamically based on config
        self.hidden_layers = []
        for units in model_cfg.hidden_layers:
            self.hidden_layers.append(
                tf.keras.layers.Dense(
                    units, 
                    activation=model_cfg.activation,
                    kernel_constraint=k_constraint
                )
            )
        
        # Final output layer predicting the yield radius
        self.radius_out = tf.keras.layers.Dense(
            1, 
            activation=model_cfg.activation,
            kernel_constraint=k_constraint
        )

    def predict_radius(self, theta, phi):
        """
        Directly predicts the yield surface radius from spherical angles.
        
        This method is used primarily by the PhysicsLoss for sampling the surface.
        It maps (theta, phi) to directional features [d1, d2, d3] which are 
        equivalent to the Cartesian directional embeddings.

        Args:
            theta (tf.Tensor): Azimuthal angle (S11-S22 plane).
            phi (tf.Tensor): Polar angle (Shear axis).

        Returns:
            tf.Tensor: Predicted yield radius.
        """
        # Directional Feature 1: d1 = sin(phi) * cos(theta)
        # Represents the normalized component pointing towards S11.
        d1 = tf.sin(phi) * tf.cos(theta)
        
        # Directional Feature 2: d2 = sin(phi) * sin(theta)
        # Represents the normalized component pointing towards S22.
        d2 = tf.sin(phi) * tf.sin(theta)
        
        # Directional Feature 3: d3 = cos^2(phi)
        # Represents the component pointing towards Shear (S12).
        # We use squared cosine to enforce f(S12) = f(-S12) symmetry.
        d3 = tf.square(tf.cos(phi))
        
        # Stack coupled features for the network
        features = tf.stack([d1, d2, d3], axis=1)
        
        x = features
        for layer in self.hidden_layers:
            x = layer(x)
            
        return self.radius_out(x)
        
    def call(self, inputs):
        """
        Calculates the equivalent stress for a given 3D stress state.

        Method: Homogeneous Scaling.
        1. Calculate the magnitude (r_total) of the input stress.
        2. Extract the unit direction [d1, d2, d3].
        3. Pass the direction through the network to find the yield radius (R_yield).
        4. Scaled Output: Equivalent Stress = Ref_Stress * (r_total / R_yield).

        Args:
            inputs (tf.Tensor): Stress components [S11, S22, S12].

        Returns:
            tf.Tensor: Equivalent stress (Scalar).
        """
        # Unpack stress components
        s11 = inputs[:, 0:1]
        s22 = inputs[:, 1:2]
        s12 = inputs[:, 2:3]
        
        # 1. CALCULATE INPUT MAGNITUDE (Radius in Stress Space)
        # We add a small epsilon (1e-8) to avoid 0/0 division at the origin.
        r_sq = tf.square(s11) + tf.square(s22) + tf.square(s12)
        r_total = tf.sqrt(r_sq + 1e-8)
        
        # 2. GENERATE SINGULARITY-FREE DIRECTIONAL EMBEDDINGS
        # Using Cartesian ratios (s11/R) is mathematically equivalent to 
        # using spherical angles but avoids the numerical instability at the poles.
        
        # d1: Projection on S11 axis
        d1 = s11 / (r_total + 1e-8)

        # d2: Projection on S22 axis
        d2 = s22 / (r_total + 1e-8)

        # d3: Normalized squared shear component
        # Squaring enforces the physical constraint of shear symmetry.
        d3 = tf.square(s12 / (r_total + 1e-8))
        
        features = tf.concat([d1, d2, d3], axis=1)
        
        # 3. PROPAGATE THROUGH HIDDEN LAYERS
        x = features
        for layer in self.hidden_layers:
            x = layer(x)
        
        # 4. PREDICT YIELD RADIUS
        # R_yield represents the distance to the yield surface in this direction.
        R_yield = self.radius_out(x)
        
        # 5. HOMOGENEOUS SCALING (Final Result)
        # This formula ensures that if the input stress state sits on the 
        # yield surface (r_total == R_yield), the equivalent stress equals ref_stress.
        se = self.ref_stress * (r_total / (R_yield + 1e-8))
        return se