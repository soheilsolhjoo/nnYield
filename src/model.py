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
        Updated to enforce shear symmetry by squaring cos_p.
        """
        sin_t = tf.sin(theta)
        cos_t = tf.cos(theta)
        sin_p = tf.sin(phi)
        
        # CHANGED: Use square of cos(phi) to enforce symmetry and zero gradient at s12=0
        cos_p_sq = tf.square(tf.cos(phi))
        
        # Stack features: [sin_t, cos_t, sin_p, cos_p^2]
        features = tf.stack([sin_t, cos_t, sin_p, cos_p_sq], axis=1)
        
        x = features
        for layer in self.hidden_layers:
            x = layer(x)
            
        return self.radius_out(x)
        
    def call(self, inputs):
        # inputs: [s11, s22, s12]
        s11 = inputs[:, 0:1]
        s22 = inputs[:, 1:2]
        s12 = inputs[:, 2:3]
        
        # 1. Magnitudes
        r_plane = tf.sqrt(tf.square(s11) + tf.square(s22) + 1e-8)
        r_total = tf.sqrt(tf.square(r_plane) + tf.square(s12) + 1e-8)
        
        # 2. Algebraic Embeddings
        sin_t = s22 / (r_plane + 1e-8) # Added epsilon for safety
        cos_t = s11 / (r_plane + 1e-8)
        sin_p = r_plane / (r_total + 1e-8)
        
        # CHANGED: Calculate cos_p but square it immediately
        # cos_p = s12 / r_total  <-- Old
        cos_p_sq = tf.square(s12 / (r_total + 1e-8)) # <-- New (Equivalent to s12^2 physics)
        
        # Concatenate Features [sin_t, cos_t, sin_p, cos_p^2]
        features = tf.concat([sin_t, cos_t, sin_p, cos_p_sq], axis=1)
        
        x = features
        for layer in self.hidden_layers:
            x = layer(x)
        
        R_yield = self.radius_out(x)
        
        # Homogeneous Scaling
        se = self.ref_stress * (r_total / (R_yield + 1e-8))
        return se