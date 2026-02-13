"""
Von Mises Isotropic Yield Criterion Module.

This module implements the standard J2 flow theory for isotropic materials.
"""

import numpy as np
from .base import BasePhysicsModel

class VonMisesModel(BasePhysicsModel):
    """
    Von Mises Yield Criterion (Isotropic).
    
    Formula: f^2 = s11^2 + s22^2 - s11*s22 + 3*s12^2
    """
    
    def __init__(self, config_params: dict, ref_stress: float):
        super().__init__(config_params, ref_stress)

    def equivalent_stress(self, s11, s22, s12):
        """Pure isotropic J2 equivalent stress."""
        term = s11**2 + s22**2 - s11*s22 + 3.0*s12**2
        return np.sqrt(np.maximum(term, 1e-16))

    def gradients(self, s11, s22, s12):
        """Analytical gradients for Von Mises."""
        denom = self.equivalent_stress(s11, s22, s12)
        denom = np.where(denom < 1e-12, 1e-12, denom)
        
        # dF/ds11 = (s11 - 0.5*s22) / sigma_eq
        g11 = (s11 - 0.5 * s22) / denom
        g22 = (s22 - 0.5 * s11) / denom
        g12 = (3.0 * s12) / denom
        
        return g11, g22, g12
