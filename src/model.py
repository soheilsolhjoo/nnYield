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
        Directly predicts Yield Radius from angles, skipping Cartesian conversion.
        Used for efficient dynamic sampling.
        """
        # Calculate features directly from angles
        sin_t = tf.sin(theta)
        cos_t = tf.cos(theta)
        sin_p = tf.sin(phi)
        cos_p = tf.cos(phi)
        
        # Match the feature order in 'call': [sin_t, cos_t, sin_p, cos_p]
        # Ensure shapes are (N, 1)
        features = tf.stack([sin_t, cos_t, sin_p, cos_p], axis=1)
        
        # Pass through hidden layers
        x = features
        for layer in self.hidden_layers:
            x = layer(x)
            
        # Output Radius
        return self.radius_out(x)
        
    def call(self, inputs):
        # inputs: [s11, s22, s12]
        s11 = inputs[:, 0:1]
        s22 = inputs[:, 1:2]
        # FIX: Remove tf.abs to allow 2nd derivative (Hessian) to flow
        s12 = inputs[:, 2:3] 
        
        # 1. Magnitudes
        r_plane = tf.sqrt(tf.square(s11) + tf.square(s22) + 1e-8)
        r_total = tf.sqrt(tf.square(r_plane) + tf.square(s12) + 1e-8)
        
        # 2. Algebraic Embeddings (No atan2/acos)
        sin_t = s22 / r_plane
        cos_t = s11 / r_plane
        sin_p = r_plane / r_total
        cos_p = s12 / r_total
        
        # Concatenate Features (4 inputs)
        features = tf.concat([sin_t, cos_t, sin_p, cos_p], axis=1)
        
        x = features
        for layer in self.hidden_layers:
            x = layer(x)
        
        R_yield = self.radius_out(x)
        
        # Homogeneous Scaling
        se = self.ref_stress * (r_total / (R_yield + 1e-8))
        return se