"""
Abstract Base Module for Analytical Physics Models.

This module defines the blueprint for any analytical yield criterion implemented in nnYield.
It ensures that the rest of the codebase can interact with any material model 
(Hill48, Barlat, etc.) using a unified interface.
"""

from abc import ABC, abstractmethod
import numpy as np

class BasePhysicsModel(ABC):
    """
    Unified interface for analytical yield benchmarks.
    """
    
    def __init__(self, config_params: dict, ref_stress: float):
        """
        Args:
            config_params (dict): Dictionary containing model parameters (e.g., F, G, H, N).
            ref_stress (float): The normalization stress level (usually 1.0).
        """
        self.params = config_params
        self.ref_stress = ref_stress

    @abstractmethod
    def equivalent_stress(self, s11, s22, s12):
        """
        Calculates the equivalent stress for a given 3D stress state.
        Must support both scalar and NumPy array inputs.
        """
        pass

    @abstractmethod
    def gradients(self, s11, s22, s12):
        """
        Calculates analytical gradients (partial derivatives) w.r.t [S11, S22, S12].
        These gradients are work-conjugate to engineering shear strain.
        """
        pass

    def solve_radius(self, theta, phi):
        """
        TRICK: THE BISECTION SOLVER.
        
        For non-quadratic models (like Barlat), it is difficult or impossible to 
        algebraically solve for the yield radius R in 3D space. 
        
        This generic solver uses a numerical bisection method to find the point 
        along a unit direction where EqStress(R*u) == RefStress. This allows 
        the pipeline to handle any future material model automatically.
        """
        # 1. Convert spherical angles to unit Cartesian vector
        s12_u = np.cos(phi)            # Vertical axis
        r_p_u = np.sin(phi)            # In-plane projection
        s11_u = r_p_u * np.cos(theta)
        s22_u = r_p_u * np.sin(theta)
        
        # 2. Iterative search
        r_lo = np.zeros_like(theta)
        r_hi = np.full_like(theta, self.ref_stress * 10.0) # High ceiling for search
        
        for _ in range(15): 
            r_mid = (r_lo + r_hi) / 2.0
            # Query the specific model's formula
            val = self.equivalent_stress(r_mid * s11_u, r_mid * s22_u, r_mid * s12_u)
            
            mask = val > self.ref_stress
            r_hi = np.where(mask, r_mid, r_hi)
            r_lo = np.where(mask, r_lo, r_mid)
            
        return (r_lo + r_hi) / 2.0

    def predict_r_value(self, alpha_deg):
        """
        Generic R-value prediction via the Associated Flow Rule.
        
        This method maps a tensile angle to a stress state, calculates the 
        normal to the yield surface (gradients), and applies the rotation 
        transformation to find the strain ratio.
        """
        rads = np.radians(alpha_deg)
        sin_a, cos_a = np.sin(rads), np.cos(rads)
        
        # tensile stress components for angle alpha
        u11, u22, u12 = cos_a**2, sin_a**2, sin_a*cos_a
        
        # Scale to the yield surface
        sigma_y = self.ref_stress / (self.equivalent_stress(u11, u22, u12) + 1e-12)
        s11, s22, s12 = u11 * sigma_y, u22 * sigma_y, u12 * sigma_y
        
        # Get normality (Gradient)
        g11, g22, g12 = self.gradients(s11, s22, s12)
        
        # Strain increments transformation
        # Note: g12 is conjugate to engineering shear, so the factor 2 is NOT needed.
        d_eps_t = -(g11 + g22)
        d_eps_w = g11*sin_a**2 + g22*cos_a**2 - g12*sin_a*cos_a
        
        return d_eps_w / (d_eps_t + 1e-12)
