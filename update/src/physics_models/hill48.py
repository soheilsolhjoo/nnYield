"""
Hill 1948 Anisotropic Yield Criterion Module.

This module implements the quadratic Hill48 criterion, which is the 
anisotropic generalization of Von Mises theory.
"""

import numpy as np
from .base import BasePhysicsModel

class Hill48Model(BasePhysicsModel):
    """
    Hill 1948 Yield Criterion (Plane Stress).
    
    Formula: f^2 = F*s22^2 + G*s11^2 + H*(s11-s22)^2 + 2*N*s12^2
    """
    
    def __init__(self, config_params: dict, ref_stress: float):
        super().__init__(config_params, ref_stress)
        # Load Hill parameters (F, G, H, N)
        self.F = config_params.get('F', 0.5)
        self.G = config_params.get('G', 0.5)
        self.H = config_params.get('H', 0.5)
        self.N = config_params.get('N', 1.5)

    def equivalent_stress(self, s11, s22, s12):
        """Standard Hill48 equivalent stress calculation."""
        term = (self.F * s22**2 + 
                self.G * s11**2 + 
                self.H * (s11 - s22)**2 + 
                2.0 * self.N * s12**2)
        return np.sqrt(np.maximum(term, 1e-16))

    def gradients(self, s11, s22, s12):
        """
        TRICK: DERIVATIVE CANCELLATION.
        
        The derivative of sqrt(U) is U' / (2 * sqrt(U)). 
        For the quadratic form U, the derivative U' always produces a factor of 2 
        (e.g., d/dx x^2 = 2x). 
        
        These factors of 2 cancel out perfectly, allowing us to calculate 
        precision gradients without unnecessary multipliers.
        """
        denom = self.equivalent_stress(s11, s22, s12)
        denom = np.where(denom < 1e-12, 1e-12, denom)
        
        # dF/ds11 = [G*s11 + H*(s11-s22)] / sigma_eq
        g11 = (self.G * s11 + self.H * (s11 - s22)) / denom
        g22 = (self.F * s22 - self.H * (s11 - s22)) / denom
        g12 = (2.0 * self.N * s12) / denom
        
        return g11, g22, g12
