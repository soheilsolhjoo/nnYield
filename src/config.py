import yaml
from dataclasses import dataclass, field, asdict
from typing import List, Optional
import os

# =============================================================================
# DATA CONFIGURATION
# =============================================================================
@dataclass
class DataConfig:
    """
    Configuration for data generation and domain definition.
    
    Attributes:
        type (str): Source of data. 'synthetic' uses Hill48 physics; 'csv' (future) for experiments.
        symmetry (bool): If True, enforces 0 <= theta <= pi/2 (Quadrants 1 & 3).
                         Crucial for orthotropic materials to reduce learning space.
        samples (dict): Dictionary defining data counts, e.g., {'loci': 1000, 'uniaxial': 15}.
                        - 'loci': Random stress points for shape learning.
                        - 'uniaxial': Experimental tensile test points for R-value learning.
        input_range (List[float]): Normalization range for inputs, typically [-1.0, 1.0].
    """
    type: str
    symmetry: bool
    samples: dict 
    input_range: List[float]

# =============================================================================
# SANITY CHECK CONFIGURATIONS
# =============================================================================
@dataclass
class CheckConfig:
    """
    Settings for periodic sanity checks (Convexity, Symmetry) during training.
    """
    enabled: bool  # Toggle to turn this check on/off
    samples: int   # Number of random points to probe during the check
    interval: int  # Run check every N epochs (0 = disable)

@dataclass
class AnisotropyConfig:
    """
    Settings for the Dual-Stream (R-value) training logic.
    
    This controls how the model learns from the small set of experimental R-values.
    """
    enabled: bool            # If True, mixes uniaxial points into the batch
    batch_r_fraction: float  # Percentage of batch dedicated to physics data (0.0 to 1.0).
                             # Higher values (e.g., 0.5) force the model to focus on derivatives.
    interval: int            # Frequency of specific anisotropy logging

# =============================================================================
# MODEL ARCHITECTURE
# =============================================================================
@dataclass
class ModelConfig:
    """
    Neural Network Architecture settings.
    
    Attributes:
        hidden_layers (List[int]): Neurons per layer (e.g., [64, 64, 64] for cylinder architecture).
        activation (str): Activation function. 'softplus' is recommended for smooth derivatives.
        ref_stress (float): Reference stress (Sigma_0) for normalization.
                            CRITICAL for FEM: This value scales the unitless NN output back to MPa.
        use_icnn_constraints (bool): If True, enforces non-negative weights (Input Convex NN).
                                     Useful for guaranteeing convexity by construction.
    """
    hidden_layers: List[int]
    activation: str
    ref_stress: float
    use_icnn_constraints: bool

# =============================================================================
# PHYSICS PARAMETERS
# =============================================================================
@dataclass
class PhysicsConfig:
    """
    Hill48 Parameters for Synthetic Ground Truth generation.
    Used to create the 'perfect' data the model tries to learn.
    """
    F: float # Controls strength in Transverse direction (90 deg)
    G: float # Controls strength in Rolling direction (0 deg)
    H: float # Controls interaction/biaxial strength
    N: float # Controls Shear strength (45 deg)

# =============================================================================
# LOSS WEIGHTS
# =============================================================================
@dataclass
class WeightsConfig:
    """
    Loss Weights: Controls the importance of each physical constraint.
    """
    stress: float            # Penalty for Yield Surface radius error (Zeroth-order)
    r_value: float           # Penalty for Lankford coefficient error (First-order/Derivative)
    convexity: float         # Penalty for Static Convexity violation (Second-order)
    dynamic_convexity: float # Penalty for sampled convexity violations
    symmetry: float          # Penalty for non-zero shear slope at symmetry planes
    gradient_norm: float     # Penalty for high gradients (Stability/Regularization)

# =============================================================================
# TRAINING SETTINGS
# =============================================================================
@dataclass
class TrainingConfig:
    """
    Main Training Loop settings.
    """
    k_folds: int                # Number of folds for Cross-Validation (None or 1 for standard)
    epochs: int                 # Maximum training epochs
    loss_threshold: float       # Early stopping: Stop if Total Loss < X
    convexity_threshold: float  # Safety Net: Stop if Min Eigenvalue > X (typically negative)
    gnorm_threshold: Optional[float] # Stability: Stop if Gradient Norm < X (or None to disable)
    r_threshold: float          # Physics Accuracy: Stop if R-value MAE < X
    batch_size: int             # Samples per gradient update. Higher = stable derivatives.
    learning_rate: float        # Optimizer step size
    weights: WeightsConfig      # Nested weights configuration
    save_dir: str               # Directory to save logs, plots, and models
    checkpoint_interval: int    # Save model weights every N epochs

# =============================================================================
# MASTER CONFIGURATION
# =============================================================================
@dataclass
class Config:
    """
    Master Configuration Object.
    
    Acts as a structured interface for the 'config.yaml' file.
    Validates that all required fields exist before training starts.
    """
    experiment_name: str
    data: DataConfig
    dynamic_convexity: CheckConfig
    symmetry: CheckConfig
    anisotropy_ratio: AnisotropyConfig
    model: ModelConfig
    physics: PhysicsConfig
    training: TrainingConfig

    @classmethod
    def from_yaml(cls, path: str):
        """
        Factory method to load configuration from a YAML file.
        
        Args:
            path (str): Path to the .yaml configuration file.
            
        Returns:
            Config: Populated configuration object.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Config file not found at: {path}")
            
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict):
        """
        Parses a dictionary into the nested Dataclass structure.
        """
        return cls(
            experiment_name=data.get('experiment_name', 'default_experiment'),
            data=DataConfig(**data['data']),
            dynamic_convexity=CheckConfig(**data['dynamic_convexity']),
            symmetry=CheckConfig(**data['symmetry']),
            anisotropy_ratio=AnisotropyConfig(**data['anisotropy_ratio']),
            model=ModelConfig(
                hidden_layers=data['model']['hidden_layers'],
                activation=data['model']['activation'],
                ref_stress=data['model']['ref_stress'],
                use_icnn_constraints=data['model']['use_icnn_constraints']
            ),
            physics=PhysicsConfig(**data['physics']),
            training=TrainingConfig(
                k_folds=data['training']['k_folds'],
                epochs=data['training']['epochs'],
                loss_threshold=data['training']['loss_threshold'],
                convexity_threshold=data['training']['convexity_threshold'],
                gnorm_threshold=data['training']['gnorm_threshold'],
                r_threshold=data['training']['r_threshold'],
                batch_size=data['training']['batch_size'],
                learning_rate=data['training']['learning_rate'],
                save_dir=data['training']['save_dir'],
                weights=WeightsConfig(**data['training']['weights']),
                checkpoint_interval=data['training']['checkpoint_interval']
            )
        )

    def to_dict(self):
        """Converts the config back to a standard Python dictionary."""
        return asdict(self)

    def get_model_architecture(self):
        """
        Helper to extract just the model-building parameters.
        Useful when initializing the TensorFlow model.
        """
        return {
            'hidden_layers': self.model.hidden_layers,
            'activation': self.model.activation,
            'input_dim': 2, # Fixed for 2D Plane Stress (Normalized)
            'output_dim': 1 # Single scalar yield stress
        }

def load_config(path: str) -> Config:
    """Convenience wrapper for Config.from_yaml."""
    return Config.from_yaml(path)