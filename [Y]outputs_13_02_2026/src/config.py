"""
Configuration Management Module for nnYield.

This module defines the schema for the entire experiment using Python dataclasses.
It provides type-safe access to all parameters, including data pipelines, 
physics constraints, and model hyperparameters.
"""

import yaml
from dataclasses import dataclass, field, asdict
from typing import List, Optional

# =============================================================================
# NESTED CONFIGURATIONS (Sub-schemas)
# =============================================================================

@dataclass
class DataConfig:
    """Settings for the data generation and ingestion pipeline."""
    type: str               # 'synthetic' or 'csv'
    positive_shear: bool    # Whether to sample S12 in [0, max]
    samples: dict           # Mapping of sample counts (loci, uniaxial)
    input_range: List[float] # Normalization range, e.g., [-1.0, 1.0]

@dataclass
class CheckConfig:
    """Standard schema for periodic physics checks (Orthotropy, Convexity)."""
    enabled: bool           # Global switch
    samples: int            # Number of points to probe
    interval: int           # Epoch frequency

@dataclass
class AnisotropyConfig:
    """Settings for R-value (slope) consistency constraints."""
    enabled: bool
    batch_r_fraction: float # portion of training batch for R-values
    interval: int           # Epoch frequency for full path validation

@dataclass
class PhysicsConstraintsConfig:
    """Container for all physics-informed penalty settings."""
    anisotropy: AnisotropyConfig
    orthotropy: CheckConfig
    dynamic_convexity: CheckConfig

@dataclass
class LRSchedulerConfig:
    """Settings for the ReduceLROnPlateau scheduler."""
    enabled: bool
    patience: int           # Epochs to wait before reduction
    factor: float           # Multiplier for reduction
    min_lr: float           # Lower bound for learning rate

@dataclass
class CurriculumConfig:
    """Settings for linear weight ramping during early training."""
    r_warmup: int           # Epochs to ramp anisotropy weight
    convexity_warmup: int   # Epochs to ramp convexity weight

@dataclass
class StoppingCriteriaConfig:
    """Thresholds for early stopping."""
    loss_threshold: float
    r_threshold: float      # Target MAE for R-values
    convexity_threshold: float # Target for minimum eigenvalue
    gnorm_threshold: float  # Target for gradient stability

@dataclass
class ModelConfig:
    """Hyperparameters for the Neural Network architecture."""
    hidden_layers: List[int] # e.g., [8, 8, 8, 8]
    activation: str          # e.g., 'softplus'
    ref_stress: float        # Normalization factor for outputs
    use_icnn_constraints: bool # Advanced: Input Convex constraint

@dataclass
class PhysicsConfig:
    """Material parameters for the target physics model (e.g., Hill48)."""
    F: float
    G: float
    H: float
    N: float

@dataclass
class WeightsConfig:
    """Multipliers for each component of the multi-objective loss function."""
    stress: float
    r_value: float
    batch_convexity: float
    dynamic_convexity: float
    orthotropy: float
    gnorm_penalty: float

@dataclass
class TrainingConfig:
    """Global parameters for the training loop and engine."""
    epochs: int
    batch_size: int
    learning_rate: float
    k_folds: int
    save_dir: str
    checkpoint_interval: int
    print_interval: int
    overwrite: bool
    curriculum: CurriculumConfig
    lr_scheduler: LRSchedulerConfig
    stopping_criteria: StoppingCriteriaConfig
    weights: WeightsConfig

# =============================================================================
# MAIN CONFIG OBJECT
# =============================================================================

@dataclass
class Config:
    """
    Root configuration object representing the entire experiment state.
    
    This class serves as the 'Source of Truth' for all components of nnYield.
    """
    experiment_name: str
    seed: int
    data: DataConfig
    physics: PhysicsConfig
    physics_constraints: PhysicsConstraintsConfig
    model: ModelConfig
    training: TrainingConfig

    @classmethod
    def from_yaml(cls, path: str):
        """Loads configuration from a YAML file path."""
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict):
        """
        Constructs a type-safe Config object from a nested dictionary.
        Handles deep initialization of all sub-dataclasses.
        """
        return cls(
            experiment_name=data['experiment_name'],
            seed=data.get('seed', 42),
            data=DataConfig(**data['data']),
            physics=PhysicsConfig(**data['physics']),
            physics_constraints=PhysicsConstraintsConfig(
                anisotropy=AnisotropyConfig(**data['physics_constraints']['anisotropy']),
                orthotropy=CheckConfig(**data['physics_constraints']['orthotropy']),
                dynamic_convexity=CheckConfig(**data['physics_constraints']['dynamic_convexity'])
            ),
            model=ModelConfig(**data['model']),
            training=TrainingConfig(
                epochs=data['training']['epochs'],
                batch_size=data['training']['batch_size'],
                learning_rate=data['training']['learning_rate'],
                k_folds=data['training'].get('k_folds', 1),
                save_dir=data['training']['save_dir'],
                checkpoint_interval=data['training']['checkpoint_interval'],
                print_interval=data['training']['print_interval'],
                overwrite=data['training'].get('overwrite', False),
                curriculum=CurriculumConfig(**data['training']['curriculum']),
                lr_scheduler=LRSchedulerConfig(**data['training']['lr_scheduler']),
                stopping_criteria=StoppingCriteriaConfig(**data['training']['stopping_criteria']),
                weights=WeightsConfig(**data['training']['weights'])
            )
        )

    def to_dict(self):
        """Converts the entire configuration tree back into a nested dictionary."""
        return asdict(self)

    def get_model_architecture(self):
        """Returns a summary of the architecture suitable for model instantiation."""
        return {
            'hidden_layers': self.model.hidden_layers,
            'activation': self.model.activation,
            'input_dim': 3,  # Stress space: S11, S22, S12
            'output_dim': 1  # Resulting equivalent stress
        }

def load_config(path: str) -> Config:
    """Global helper function to load and validate configuration."""
    return Config.from_yaml(path)