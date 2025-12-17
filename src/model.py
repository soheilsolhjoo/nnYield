import tensorflow as tf
import numpy as np

class HomogeneousYieldModel(tf.keras.Model):
    """
    Physics-Informed Neural Network for Yield Surface Modeling.
    
    This model implements a 'Homogeneous' approach to yield surface modeling.
    Instead of mapping Stress -> Yield Stress directly, it maps:
    Direction -> Yield Radius (Distance from center to yield surface).
    
    Mathematical Formulation:
        1. Calculate input stress magnitude: |sigma|
        2. Calculate input direction: d = sigma / |sigma|
        3. Predict yield radius for that direction: R_yield = NN(d)
        4. Calculate Equivalent Stress: Se = Ref_Stress * (|sigma| / R_yield)
        
    Benefits:
        - Guarantees homogeneity of degree 1 (f(k*sigma) = k*f(sigma)).
        - Solves the 'star-shaped' domain problem effectively.
    """
    def __init__(self, config):
        """
        Initializes the neural network layers based on configuration.
        
        Args:
            config (dict): Configuration dictionary containing model architecture
                           and constraint settings.
        """
        super(HomogeneousYieldModel, self).__init__()
        self.ref_stress = config['model']['ref_stress']
        
        # --- CONSTRAINT LOGIC (ICNN) ---
        # Input Convex Neural Networks (ICNN) require non-negative weights
        # to guarantee that the output is a convex function of the inputs.
        # This is optional but powerful for thermodynamic consistency.
        use_icnn = config['model'].get('use_icnn_constraints', False)
        k_constraint = tf.keras.constraints.NonNeg() if use_icnn else None
        
        if use_icnn:
            print(f"🔒 ICNN Mode Enabled: Weights constrained to be Non-Negative.")

        # --- BUILD HIDDEN LAYERS ---
        self.hidden_layers = []
        for units in config['model']['hidden_layers']:
            self.hidden_layers.append(
                tf.keras.layers.Dense(
                    units, 
                    activation=config['model']['activation'],
                    # Apply constraint if ICNN is enabled
                    kernel_constraint=k_constraint  
                )
            )
        
        # --- OUTPUT LAYER ---
        # Predicts a single scalar: The Yield Radius (Distance to surface)
        # Activation matches hidden layers to ensure smooth gradients.
        self.radius_out = tf.keras.layers.Dense(
            1, 
            activation=config['model']['activation'],
            kernel_constraint=k_constraint
        )

    def predict_radius(self, theta, phi):
        """
        Helper method to predict Yield Radius directly from spherical angles.
        Used primarily for visualization and plotting.
        
        Features:
            Uses 'Coupled Angular Features' to eliminate singularities.
            Standard spherical coordinates (theta, phi) have a singularity at the pole.
            This transformation [d1, d2, d3] ensures smoothness everywhere.
            
        Args:
            theta (tf.Tensor): Angle in the S11-S22 plane.
            phi (tf.Tensor): Angle from the Shear axis.
            
        Returns:
            tf.Tensor: Predicted Yield Radius at these angles.
        """
        # Feature 1: d1 = sin(phi) * cos(theta) -> Proportional to S11 component
        # At phi=0 (Pure Shear), sin(phi)=0, so d1 vanishes smoothly.
        d1 = tf.sin(phi) * tf.cos(theta)
        
        # Feature 2: d2 = sin(phi) * sin(theta) -> Proportional to S22 component
        d2 = tf.sin(phi) * tf.sin(theta)
        
        # Feature 3: d3 = cos^2(phi) -> Proportional to S12 component (Squared)
        # We square the cosine to enforce physical symmetry: f(Shear) = f(-Shear).
        d3 = tf.square(tf.cos(phi))
        
        # Stack coupled features: [d1, d2, d3]
        features = tf.stack([d1, d2, d3], axis=1)
        
        x = features
        for layer in self.hidden_layers:
            x = layer(x)
            
        return self.radius_out(x)
        
    def call(self, inputs):
        """
        Forward Pass (Training Step).
        
        Args:
            inputs (tf.Tensor): Normalized Stress Tensor [Batch, 3] (s11, s22, s12)
            
        Returns:
            se (tf.Tensor): Predicted Equivalent Stress (Yield Metric)
        """
        # Unpack inputs
        s11 = inputs[:, 0:1]
        s22 = inputs[:, 1:2]
        s12 = inputs[:, 2:3]
        
        # 1. CALCULATE MAGNITUDE (R_total)
        # This is the "Length" of the input stress vector.
        # We add epsilon (1e-8) to prevent NaN gradients at zero stress.
        r_sq = tf.square(s11) + tf.square(s22) + tf.square(s12)
        r_total = tf.sqrt(r_sq + 1e-8)
        
        # 2. CALCULATE DIRECTION (Coupled Embeddings)
        # Instead of feeding raw stress, we feed normalized directional components.
        # This separates the "Shape" learning (NN) from the "Size" scaling (Math).
        
        # d1 = Normalized S11 Component
        d1 = s11 / (r_total + 1e-8)
        
        # d2 = Normalized S22 Component
        d2 = s22 / (r_total + 1e-8)
        
        # d3 = Normalized S12 Component (Squared)
        # Squaring enforces the physics that yield surface is symmetric in shear (+/- s12).
        d3 = tf.square(s12 / (r_total + 1e-8))
        
        # Concatenate Features [d1, d2, d3]
        features = tf.concat([d1, d2, d3], axis=1)
        
        # 3. PREDICT YIELD RADIUS
        # The NN answers: "If I go in this direction, how far until I yield?"
        x = features
        for layer in self.hidden_layers:
            x = layer(x)
        
        R_yield = self.radius_out(x)
        
        # 4. HOMOGENEOUS SCALING (The Final Prediction)
        # Equivalent Stress = Ref_Stress * (Current_Radius / Yield_Radius)
        # If Current > Yield, Se > Ref (Yielded)
        # If Current < Yield, Se < Ref (Elastic)
        se = self.ref_stress * (r_total / (R_yield + 1e-8))
        
        return se