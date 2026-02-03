import tensorflow as tf
import numpy as np

class HomogeneousYieldModel(tf.keras.Model):
    """
    Physics-Informed Neural Network for Yield Surface Modeling.
    Strictly defined in Theta-Phi space.
    """
    
    def __init__(self, config):
        super(HomogeneousYieldModel, self).__init__()
        
        # Handle configuration access
        if isinstance(config, dict):
            m_cfg = config['model']
            t_cfg = config.get('training', {})
            
            self.ref_stress = m_cfg['ref_stress']
            self.hidden_sizes = m_cfg['hidden_layers']
            self.activation = m_cfg['activation']
            self.use_icnn = m_cfg.get('use_icnn_constraints', False)
            self.l2_reg = t_cfg.get('l2_regularization', 0.0)
        else:
            m_cfg = config.model
            t_cfg = getattr(config, 'training', None)
            
            self.ref_stress = m_cfg.ref_stress
            self.hidden_sizes = m_cfg.hidden_layers
            self.activation = m_cfg.activation
            self.use_icnn = getattr(m_cfg, 'use_icnn_constraints', False)
            self.l2_reg = getattr(t_cfg, 'l2_regularization', 0.0) if t_cfg else 0.0
        
        # Constraints & Regularizers
        if self.use_icnn:
            self.k_constraint = tf.keras.constraints.NonNeg()
        else:
            self.k_constraint = None
            
        if self.l2_reg > 0:
            self.kernel_regularizer = tf.keras.regularizers.l2(self.l2_reg)
        else:
            self.kernel_regularizer = None
            
        # Build Hidden Layers
        self.hidden_layers = []
        for n_units in self.hidden_sizes:
            layer = tf.keras.layers.Dense(
                units=n_units,
                activation=self.activation,
                kernel_constraint=self.k_constraint,
                kernel_regularizer=self.kernel_regularizer
            )
            self.hidden_layers.append(layer)
            
        # Output Layer (Yield Radius)
        self.radius_out = tf.keras.layers.Dense(
            units=1,
            activation=self.activation,
            kernel_constraint=self.k_constraint,
            kernel_regularizer=self.kernel_regularizer
        )

    def call(self, inputs):
        """
        Forward pass strictly using Theta-Phi transformation.
        Inputs: Stress [s11, s22, s12]
        """
        s11 = inputs[:, 0:1]
        s22 = inputs[:, 1:2]
        s12 = inputs[:, 2:3]
        
        # 1. Calculate Magnitude (Needed for Homogeneity)
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
        
        # 3. Predict Yield Radius
        x = features
        for layer in self.hidden_layers:
            x = layer(x)
        
        R_yield = self.radius_out(x)
        
        # 4. Homogeneous Scaling
        se = self.ref_stress * (r_total / (R_yield + 1e-8))
        return se
    
    def predict_radius(self, theta, phi):
        """
        Predicts radius directly from input angles.
        Used for visualization (Yield Loci Slices).
        """
        theta = tf.convert_to_tensor(theta, dtype=tf.float32)
        phi = tf.convert_to_tensor(phi, dtype=tf.float32)

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