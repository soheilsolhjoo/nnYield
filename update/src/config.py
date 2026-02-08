import yaml
from dataclasses import dataclass, field, asdict
from typing import List, Optional

# =============================================================================
# NESTED CONFIGURATIONS
# =============================================================================

@dataclass
class DataConfig:
    type: str
    positive_shear: bool
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
class PhysicsConstraintsConfig:
    anisotropy: AnisotropyConfig
    orthotropy: CheckConfig
    dynamic_convexity: CheckConfig

@dataclass
class LRSchedulerConfig:
    enabled: bool
    patience: int
    factor: float
    min_lr: float

@dataclass
class CurriculumConfig:
    r_warmup: int
    convexity_warmup: int

@dataclass
class StoppingCriteriaConfig:
    loss_threshold: float
    r_threshold: float
    convexity_threshold: float
    gnorm_threshold: float

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
    orthotropy: float
    gnorm_penalty: float

@dataclass
class TrainingConfig:
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
# MAIN CONFIG
# =============================================================================

@dataclass
class Config:
    experiment_name: str
    seed: int
    data: DataConfig
    physics: PhysicsConfig
    physics_constraints: PhysicsConstraintsConfig
    model: ModelConfig
    training: TrainingConfig

    @classmethod
    def from_yaml(cls, path: str):
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict):
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
