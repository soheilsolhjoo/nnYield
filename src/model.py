import tensorflow as tf
import numpy as np

class HomogeneousYieldModel(tf.keras.Model):
    def __init__(self, config):
        super(HomogeneousYieldModel, self).__init__()
        self.ref_stress = config['model']['ref_stress']
        
        self.hidden_layers = []
        for units in config['model']['hidden_layers']:
            self.hidden_layers.append(
                tf.keras.layers.Dense(units, activation=config['model']['activation'])
            )
        self.radius_out = tf.keras.layers.Dense(1, activation=config['model']['activation'])

    def predict_radius(self, theta, phi):
        """
        Directly predicts Yield Radius from angles.
        Updated to use Coupled Angular Features to eliminate singularity at phi=0.
        """
        # Feature 1: d1 = sin(phi) * cos(theta)
        # Represents the component pointing towards S11.
        # At phi=0 (Pole), sin(phi)=0, so d1 -> 0 smoothly regardless of theta.
        d1 = tf.sin(phi) * tf.cos(theta)
        
        # Feature 2: d2 = sin(phi) * sin(theta)
        # Represents the component pointing towards S22.
        # At phi=0 (Pole), d2 -> 0 smoothly.
        d2 = tf.sin(phi) * tf.sin(theta)
        
        # Feature 3: d3 = cos^2(phi)
        # Represents the component pointing towards Shear (squared for symmetry).
        d3 = tf.square(tf.cos(phi))
        
        # Stack coupled features: [d1, d2, d3]
        features = tf.stack([d1, d2, d3], axis=1)
        
        x = features
        for layer in self.hidden_layers:
            x = layer(x)
            
        return self.radius_out(x)
        
    def call(self, inputs):
        # inputs: [s11, s22, s12]
        s11 = inputs[:, 0:1]
        s22 = inputs[:, 1:2]
        s12 = inputs[:, 2:3]
        
        # 1. Total Magnitude (Radius)
        # We only need R_total now. We do NOT need r_plane anymore.
        r_sq = tf.square(s11) + tf.square(s22) + tf.square(s12)
        r_total = tf.sqrt(r_sq + 1e-8)
        
        # 2. Coupled Directional Embeddings (Singularity-Free)
        # d1 = s11 / R_total  (Equivalent to sin(phi)*cos(theta))
        # At pure shear (s11=0, s22=0), this is 0/R = 0. Smooth.
        d1 = s11 / (r_total + 1e-8)
        
        # d2 = s22 / R_total  (Equivalent to sin(phi)*sin(theta))
        d2 = s22 / (r_total + 1e-8)
        
        # d3 = (s12 / R_total)^2  (Equivalent to cos^2(phi))
        # Squared for physical symmetry (f(s12) = f(-s12))
        d3 = tf.square(s12 / (r_total + 1e-8))
        
        # Concatenate Features [d1, d2, d3]
        features = tf.concat([d1, d2, d3], axis=1)
        
        x = features
        for layer in self.hidden_layers:
            x = layer(x)
        
        R_yield = self.radius_out(x)
        
        # Homogeneous Scaling
        se = self.ref_stress * (r_total / (R_yield + 1e-8))
        return se