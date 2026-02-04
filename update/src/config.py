import yaml
from dataclasses import dataclass, field, asdict
from typing import List, Optional

@dataclass
class DataConfig:
    type: str
    symmetry: bool
    samples: dict 
    input_range: List[float]

@dataclass
class CheckConfig:
    enabled: bool
    samples: int
    interval: int

@dataclass
class AnisotropyConfig:
    enabled: bool
    batch_r_fraction: float
    interval: int

@dataclass
class ModelConfig:
    hidden_layers: List[int]
    activation: str
    ref_stress: float
    use_icnn_constraints: bool

@dataclass
class PhysicsConfig:
    F: float
    G: float
    H: float
    N: float

@dataclass
class WeightsConfig:
    stress: float
    r_value: float
    batch_convexity: float
    dynamic_convexity: float
    symmetry: float
    gnorm_penalty: float

@dataclass
class TrainingConfig:
    k_folds: int
    epochs: int
    loss_threshold: float
    convexity_threshold: float  # Safety Net
    gnorm_threshold: float      # Stability
    r_threshold: float          # Physics Accuracy
    batch_size: int
    learning_rate: float
    weights: WeightsConfig
    save_dir: str
    checkpoint_interval: int
    overwrite: bool # Added field for directory handling

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

    @classmethod
    def from_yaml(cls, path: str):
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict):
        # Backward Compatibility for Weights
        w_data = data['training']['weights'].copy()
        if 'convexity' in w_data:
            w_data['batch_convexity'] = w_data.pop('convexity')
        if 'gradient_norm' in w_data:
            w_data['gnorm_penalty'] = w_data.pop('gradient_norm')

        return cls(
            experiment_name=data['experiment_name'],
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
                weights=WeightsConfig(**w_data),
                checkpoint_interval=data['training']['checkpoint_interval'],
                overwrite=data['training'].get('overwrite', False)
            )
        )

    def to_dict(self):
        return asdict(self)

    def get_model_architecture(self):
        return {
            'hidden_layers': self.model.hidden_layers,
            'activation': self.model.activation,
            'input_dim': 2, 
            'output_dim': 1 
        }

def load_config(path: str) -> Config:
    return Config.from_yaml(path)