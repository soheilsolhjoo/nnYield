import yaml
from dataclasses import dataclass, field, asdict
from typing import List, Optional

@dataclass
class DataConfig:
    type: str
    symmetry: bool
    samples: dict  # {'loci': int, 'uniaxial': int}
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
    use_icnn_constraints: bool = False  # <--- NEW: ICNN Switch

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
    convexity: float
    dynamic_convexity: float
    symmetry: float

@dataclass
class TrainingConfig:
    k_folds: int
    epochs: int
    loss_threshold: float
    convexity_threshold: float
    batch_size: int
    learning_rate: float
    weights: WeightsConfig
    save_dir: str
    checkpoint_interval: int

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
        """
        Parses a dictionary into the strictly typed Config object.
        Useful for loading config from a saved checkpoint.
        """
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
                batch_size=data['training']['batch_size'],
                learning_rate=data['training']['learning_rate'],
                save_dir=data['training']['save_dir'],
                weights=WeightsConfig(**data['training']['weights']),
                checkpoint_interval=data['training']['checkpoint_interval']
            )
        )

    def to_dict(self):
        """Converts the config back to a dictionary (for saving)."""
        return asdict(self)

    def get_model_architecture(self):
        """Helper to extract only architecture-relevant params."""
        return {
            'hidden_layers': self.model.hidden_layers,
            'activation': self.model.activation,
            'input_dim': 2, 
            'output_dim': 1 
        }

# --- Usage Helper ---
def load_config(path: str) -> Config:
    return Config.from_yaml(path)