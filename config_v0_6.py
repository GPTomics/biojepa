from dataclasses import dataclass
from pathlib import Path


@dataclass
class PretrainConfig:
    epochs: int = 10
    lr: float = 1e-3
    batch_size: int = 32
    warmup_pct: float = 0.05
    weight_decay: float = 0.05


@dataclass
class AlignmentConfig:
    epochs: int = 1000
    lr: float = 4e-3
    batch_size: int = 32
    weight_decay: float = 0.05


@dataclass
class FullTrainingConfig:
    epochs: int = 10
    predictor_lr: float = 1e-3
    batch_size: int = 32
    weight_decay: float = 0.05


@dataclass
class DecoderConfig:
    epochs: int = 10
    lr: float = 1e-3


@dataclass
class DataConfig:
    data_root: Path
    checkpoint_dir: Path
    eval_results_dir: Path = None


MODALITY_TO_ID = {
    'dna': 0,
    'protein': 1,
    'chemical': 2
}

MODE_TO_ID = {
    'crispri': 0,
    'crispra': 1,
    'overexpression': 2,
    'knockout': 3,
    'inhibitor': 4,
    'agonist': 5,
    'degrader': 6,
    'binder': 7,
    'unknown': 8
}
