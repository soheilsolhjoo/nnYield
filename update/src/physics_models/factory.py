"""
Factory Module for Analytical Physics Models.

This module provides a centralized point for instantiating material models.
It decouples the specific mathematical implementations from the rest 
of the training and validation logic.
"""

from .hill48 import Hill48Model
from .yld91 import Yld91Model
from .von_mises import VonMisesModel

def get_physics_model(config):
    """
    TRICK: THE PHYSICS FACTORY.
    
    This function dynamically selects and initializes a material model 
    based on the 'type' string provided in the config.yaml. This is what 
    makes the entire project 'Generic' and easy to extend.
    
    Supported Models:
    - 'von_mises': Isotropic benchmark.
    - 'hill48':    Anisotropic quadratic benchmark.
    - 'yld91':     Anisotropic non-quadratic (exponent-based) benchmark.
    """
    phys_cfg = config.physics
    # Normalize model type string
    model_type = getattr(phys_cfg, 'type', 'hill48').lower()
    
    # Safely extract parameters into a dictionary
    if hasattr(phys_cfg, '__dict__'):
        params = phys_cfg.__dict__
    else:
        params = phys_cfg
        
    ref_stress = config.model.ref_stress
    
    if model_type == 'von_mises':
        print("🏭 Physics Factory: Initializing Isotropic Von Mises Model")
        return VonMisesModel(params, ref_stress)
    
    elif model_type == 'yld91':
        print(f"🏭 Physics Factory: Initializing Barlat Yld91 Model (m={params.get('m', 8)})")
        return Yld91Model(params, ref_stress)
    
    elif model_type == 'hill48':
        print("🏭 Physics Factory: Initializing Hill48 Model")
        return Hill48Model(params, ref_stress)
        
    else:
        # Fallback error for unsupported models
        valid_options = ['von_mises', 'hill48', 'yld91']
        raise ValueError(f"Unknown model type: '{model_type}'. Supported: {valid_options}")
