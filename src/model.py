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
            config: The strict Config object from src/config.py.
        """
        super(HomogeneousYieldModel, self).__init__()
        
        # Extract Configuration
        # We handle both object (new style) and dict (legacy) access for robustness
        if isinstance(config, dict):
            m_cfg = config['model']
        else:
            m_cfg = config.model

        self.ref_stress = m_cfg.ref_stress
        self.hidden_sizes = m_cfg.hidden_layers
        self.activation = m_cfg.activation
        
        # --- ICNN CONSTRAINT LOGIC ---
        # If enabled, weights are constrained to be non-negative.
        # This, combined with convex activation (Softplus), guarantees the network
        # represents a convex function by construction.
        self.use_icnn = getattr(m_cfg, 'use_icnn_constraints', False)
        
        if self.use_icnn:
            self.k_constraint = tf.keras.constraints.NonNeg()
            print("[Model] ICNN Constraints ENABLED: Weights forced to be Non-Negative.")
        else:
            self.k_constraint = None
            
        # --- BUILD LAYERS ---
        self.hidden_layers = []
        for n_units in self.hidden_sizes:
            layer = tf.keras.layers.Dense(
                units=n_units,
                activation=self.activation,
                kernel_constraint=self.k_constraint, # Apply Constraint Here
                # Bias constraint is usually not strictly required for convexity 
                # if activation is convex and monotonic, but standard ICNN often keeps it simple.
            )
            self.hidden_layers.append(layer)
            
        # Output Layer:
        # Maps hidden features to a single scalar (Yield Radius).
        # Must be positive, so we typically use Softplus or Exp output.
        # For ICNN, the output layer weights must also be non-negative.
        self.radius_out = tf.keras.layers.Dense(
            units=1,
            activation='softplus', # Ensure Radius is always positive
            kernel_constraint=self.k_constraint
        )

    def call(self, inputs):
        """
        Forward pass.
        
        Args:
            inputs: Tensor of shape (Batch, 3) -> [S11, S22, S12]
        
        Returns:
            se_pred: Equivalent Stress prediction.
        """
        # 1. INPUT PARSING
        # Split columns for explicit physics handling
        s11 = inputs[:, 0:1]
        s22 = inputs[:, 1:2]
        s12 = inputs[:, 2:3]
        
        # Calculate Magnitude (Radius) in stress space
        # We add epsilon (1e-8) to prevent NaN gradients at zero stress.
        r_sq = tf.square(s11) + tf.square(s22) + tf.square(s12)
        r_total = tf.sqrt(r_sq + 1e-8)
        
        # 2. CALCULATE DIRECTION (Coupled Embeddings)
        # Instead of feeding raw stress, we feed normalized directional components.
        # This separates the "Shape" learning (NN) from the "Size" scaling (Math).
        
        # d1 = Normalized S11
        d1 = s11 / (r_total + 1e-8)
        
        # d2 = Normalized S22
        d2 = s22 / (r_total + 1e-8)
        
        # d3 = Normalized S12 Component (Squared)
        # IMPOSING SYMMETRY: By feeding s12^2, we force Model(s12) == Model(-s12).
        # This is physically required for orthotropic materials.
        d3 = tf.square(s12 / (r_total + 1e-8))
        
        # Concatenate Features [d1, d2, d3_squared]
        features = tf.concat([d1, d2, d3], axis=1)
        
        # 3. PREDICT YIELD RADIUS
        # The NN answers: "If I go in this direction, how far until I yield?"
        x = features
        for layer in self.hidden_layers:
            x = layer(x)
        
        R_yield = self.radius_out(x)
        
        # 4. CONVERT TO EQUIVALENT STRESS
        # Se = Ref_Stress * (Actual_Radius / Yield_Radius)
        # If Actual < Yield, Se < Ref (Elastic)
        # If Actual = Yield, Se = Ref (Yielding)
        
        # Avoid division by zero
        se_pred = self.ref_stress * (r_total / (R_yield + 1e-8))
        
        return se_pred
    
    def predict_radius(self, inputs):
        """
        Exposes the raw Yield Radius (R_yield) for a given input state.
        Used to identify exactly where the neural network's yield locus is.
        """
        # 1. Standard Normalization
        s11, s22, s12 = inputs[:, 0:1], inputs[:, 1:2], inputs[:, 2:3]
        r_total = tf.sqrt(tf.square(s11) + tf.square(s22) + tf.square(s12) + 1e-8)
        
        # 2. Extract direction components (s12 is squared for symmetry)
        d1, d2, d3 = s11/r_total, s22/r_total, tf.square(s12/r_total)
        features = tf.concat([d1, d2, d3], axis=1)
        
        # 3. Network Forward Pass
        x = features
        for layer in self.hidden_layers:
            x = layer(x)
        return self.radius_out(x)