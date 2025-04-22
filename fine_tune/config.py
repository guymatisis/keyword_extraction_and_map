from dataclasses import dataclass
from typing import Optional
from omegaconf import DictConfig
import torch
from data import CombinedDataset

@dataclass
class ModelConfig:
    name: str
    max_input_length: int
    max_output_length: int

@dataclass
class DataConfig:
    data_dir: str
    dataset_name: str
    data_separator: str

@dataclass
class TrainingConfig:
    device: str
    num_epochs: int
    batch_size: int
    gradient_accumulation_steps: int
    learning_rate: float
    weight_decay: float
    early_stopping_patience: int
    fp16: bool = False  # Adding fp16 with False as default

@dataclass
class OutputConfig:
    dir: str
    save_total_limit: int

@dataclass
class DebugConfig:
    single_batch: bool

@dataclass
class Config:
    # Top-level configurations
    run_name: str
    model: ModelConfig
    training: TrainingConfig
    output: OutputConfig
    debug: DebugConfig
    datasets: CombinedDataset

    def __post_init__(self):
        if self.debug.single_batch:
            self.training.num_epochs = 1
            self.training.batch_size = 1
            self.training.gradient_accumulation_steps = 1

    @classmethod
    def from_hydra_config(cls, cfg: DictConfig):
        return cls(
            run_name=cfg.run_name,
            model=ModelConfig(**cfg.model),
            training=TrainingConfig(**cfg.training),
            output=OutputConfig(**cfg.output),
            debug=DebugConfig(**cfg.debug),
            datasets=cfg.datasets
        )
