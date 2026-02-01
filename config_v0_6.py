from dataclasses import dataclass
from pathlib import Path


@dataclass
class PretrainConfig:
    epochs: int = None #10
    lr: float = 1e-3
    batch_size: int = 32
    warmup_pct: float = 0.05
    weight_decay: float = 0.05
    n_steps: int = None


@dataclass
class AlignmentConfig:
    epochs: int = None #1000
    lr: float = 4e-3
    batch_size: int = 32
    weight_decay: float = 0.05
    n_steps: int = None


@dataclass
class FullTrainingConfig:
    epochs: int = None #10
    predictor_lr: float = 1e-3
    batch_size: int = 32
    weight_decay: float = 0.05
    n_steps: int = None


@dataclass
class DecoderConfig:
    epochs: int = None #10
    lr: float = 1e-3
    batch_size: int = 32
    n_steps: int = None


@dataclass
class DataConfig:
    data_root: Path
    checkpoint_dir: Path
    ref_dir: Path
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

EMBEDDING_DIMS = {
    'dna': 1536,
    'protein': 320,
    'chemical': 1024,
}
MAX_SEQ_DIM = max(EMBEDDING_DIMS.values())
