import yaml
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict
import os

# =============================================================================
# DATA CONFIGURATION
# =============================================================================
@dataclass
class DataConfig:
    """
    Configuration for data generation.
    """
    type: str
    symmetry: bool
    samples: dict 
    input_range: List[float]

# =============================================================================
# CHECK CONFIGURATION
# =============================================================================
@dataclass
class CheckConfig:
    """
    Settings for periodic sanity checks.
    
    Attributes:
        enabled (bool): Toggle check on/off.
        samples (int): Number of NEW random points to generate specifically for this check.
        interval (int): Run this check every N epochs.
    """
    enabled: bool
    samples: int
    interval: int

@dataclass
class AnisotropyConfig:
    """
    Configuration for Dual Stream training.
    """
    enabled: bool
    batch_r_fraction: float
    interval: int

# =============================================================================
# PHYSICS CONFIGURATION
# =============================================================================
@dataclass
class PhysicsConfig:
    F: float
    G: float
    H: float
    N: float

# =============================================================================
# MODEL CONFIGURATION
# =============================================================================
@dataclass
class ModelConfig:
    hidden_layers: List[int]
    activation: str
    ref_stress: float
    use_icnn_constraints: bool

# =============================================================================
# WEIGHTS CONFIGURATION
# =============================================================================
@dataclass
class WeightsConfig:
    stress: float
    r_value: float
    symmetry: float
    # convexity: float
    dynamic_convexity: float

# =============================================================================
# TRAINING CONFIGURATION
# =============================================================================
@dataclass
class TrainingConfig:
    k_folds: int
    epochs: int
    batch_size: int
    learning_rate: float
    save_dir: str
    
    # Checkpointing & Logging
    checkpoint_interval: int
    print_interval: int
    overwrite: bool
    
    # Curriculum Learning
    convexity_warmup: int
    r_warmup: int
    
    # Stopping Thresholds (gnorm_threshold removed)
    loss_threshold: Optional[float]
    r_threshold: Optional[float]
    convexity_threshold: Optional[float] # Min Eigenvalue target
    
    weights: WeightsConfig

# =============================================================================
# MAIN CONFIG
# =============================================================================
@dataclass
class Config:
    experiment_name: str
    data: DataConfig
    dynamic_convexity: CheckConfig
    symmetry: CheckConfig
    anisotropy_ratio: AnisotropyConfig
    model: ModelConfig
    physics: PhysicsConfig
    training: TrainingConfig

    @staticmethod
    def from_yaml(path: str) -> 'Config':
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        return Config.from_dict(data)

    @staticmethod
    def from_dict(data: dict) -> 'Config':
        return Config(
            experiment_name=data['experiment_name'],
            data=DataConfig(**data['data']),
            dynamic_convexity=CheckConfig(**data['dynamic_convexity']),
            symmetry=CheckConfig(**data['symmetry']),
            anisotropy_ratio=AnisotropyConfig(**data['anisotropy_ratio']),
            model=ModelConfig(**data['model']),
            physics=PhysicsConfig(**data['physics']),
            
            training=TrainingConfig(
                k_folds=data['training'].get('k_folds', 1),
                epochs=data['training']['epochs'],
                batch_size=data['training']['batch_size'],
                learning_rate=data['training']['learning_rate'],
                save_dir=data['training']['save_dir'],
                
                # Checkpointing
                checkpoint_interval=data['training']['checkpoint_interval'],
                print_interval=data['training'].get('print_interval', 10),
                overwrite=data['training'].get('overwrite', False),
                
                # Curriculum
                convexity_warmup=data['training'].get('convexity_warmup', 0),
                r_warmup=data['training'].get('r_warmup', 0),
                
                # Stopping Thresholds
                loss_threshold=data['training'].get('loss_threshold'),
                r_threshold=data['training'].get('r_threshold'),
                convexity_threshold=data['training'].get('convexity_threshold'),
                
                weights=WeightsConfig(**data['weights'])
            )
        )

    def to_dict(self):
        return asdict(self)