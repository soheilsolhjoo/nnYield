"""
Barlat 1991 (Yld91) Anisotropic Yield Criterion Module.

This module implements the non-quadratic Yld91 model.
It is characterized by a linear transformation of the stress tensor and 
a power-exponent m, making it superior to Hill48 for many alloys.
"""

import numpy as np
import tensorflow as tf
from .base import BasePhysicsModel

class Yld91Model(BasePhysicsModel):
    """
    Barlat 1991 Yield Criterion (Plane Stress).
    
    Formula: |S1 - S2|^m + |S2 - S3|^m + |S3 - S1|^m = 2 * sigma_eq^m
    where Si are eigenvalues of a transformed stress tensor.
    """
    
    def __init__(self, config_params: dict, ref_stress: float):
        super().__init__(config_params, ref_stress)
        # c1, c2, c3, c6: Directional weight coefficients
        self.c1 = config_params.get('c1', 1.0)
        self.c2 = config_params.get('c2', 1.0)
        self.c3 = config_params.get('c3', 1.0)
        self.c6 = config_params.get('c6', 1.0)
        # m: Exponent (6 for BCC, 8 for FCC)
        self.m = config_params.get('m', 8.0) 

    def equivalent_stress(self, s11, s22, s12):
        """
        Pure NumPy implementation for high-speed data generation and plotting.
        """
        # 1. Transform the stress tensor components
        S_xx = (self.c2 * s11 + self.c3 * (s11 - s22)) / 3.0
        S_yy = (self.c1 * s22 - self.c3 * (s11 - s22)) / 3.0
        S_xy = self.c6 * s12
        
        # 2. Principal values of the transformed deviator
        center = (S_xx + S_yy) / 2.0
        radius = np.sqrt(((S_xx - S_yy) / 2.0)**2 + S_xy**2)
        
        P1 = center + radius
        P2 = center - radius
        P3 = -(S_xx + S_yy) # Deviatoric trace must be 0
        
        # 3. Sum of absolute differences raised to power m
        phi = (np.abs(P1 - P2)**self.m + 
               np.abs(P2 - P3)**self.m + 
               np.abs(P3 - P1)**self.m)
              
        return (phi / 2.0)**(1.0 / self.m)

    def gradients(self, s11, s22, s12):
        """
        TRICK: HYBRID AUTODIFF GRADIENTS.
        
        Manually deriving the derivatives of eigenvalues w.r.t original stress 
        is extremely complex and prone to bugs. 
        
        We solve this by briefly entering TensorFlow's Autodiff engine, 
        calculating the exact mathematical gradient, and returning it to NumPy.
        This ensures 100% precision for the R-value benchmarks.
        """
        s11_tf = tf.convert_to_tensor(s11, dtype=tf.float64)
        s22_tf = tf.convert_to_tensor(s22, dtype=tf.float64)
        s12_tf = tf.convert_to_tensor(s12, dtype=tf.float64)
        
        with tf.GradientTape(persistent=True) as tape:
            tape.watch([s11_tf, s22_tf, s12_tf])
            
            # Re-implement the equivalent stress logic in TF for tracking
            S_xx = (self.c2 * s11_tf + self.c3 * (s11_tf - s22_tf)) / 3.0
            S_yy = (self.c1 * s22_tf - self.c3 * (s11_tf - s22_tf)) / 3.0
            S_xy = self.c6 * s12_tf
            
            center = (S_xx + S_yy) / 2.0
            radius = tf.sqrt(((S_xx - S_yy) / 2.0)**2 + S_xy**2 + 1e-16)
            
            P1 = center + radius
            P2 = center - radius
            P3 = -(S_xx + S_yy)
            
            phi = (tf.abs(P1 - P2)**self.m + 
                   tf.abs(P2 - P3)**self.m + 
                   tf.abs(P3 - P1)**self.m)
            
            sigma_eq = (phi / 2.0)**(1.0 / self.m)
            
        g11 = tape.gradient(sigma_eq, s11_tf).numpy()
        g22 = tape.gradient(sigma_eq, s22_tf).numpy()
        g12 = tape.gradient(sigma_eq, s12_tf).numpy()
        
        del tape
        return g11, g22, g12
