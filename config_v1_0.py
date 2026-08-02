from dataclasses import dataclass
from pathlib import Path

VERSION = 'v1_0'


@dataclass
class EncoderTrainingConfig:
    epochs: int = None #10
    stop_after_epochs: int = None
    lr: float = 1e-3
    batch_size: int = 32
    warmup_pct: float = 0.05
    weight_decay: float = 0.05
    n_steps: int = None
    phase2_start_pct: float = 0.8
    context_coeff: float = 0.0
    context_ramp_pct: float = 0.2
    context_ramp_start_pct: float = None
    ema_final_momentum: float = None


@dataclass
class ComposerTrainingConfig:
    epochs: int = None #1000
    lr: float = 4e-3
    batch_size: int = 32
    weight_decay: float = 0.05
    n_steps: int = None
    temperature: float = 0.012
    chemical_fraction: float = 0.0


@dataclass
class ACTrainingConfig:
    epochs: int = None #10
    predictor_lr: float = 1e-3
    batch_size: int = 32
    weight_decay: float = 0.05
    n_steps: int = None
    mask_anneal_pct: float = 0.0
    mask_anneal_floor: float = 0.0
    beta_nll_target: float = 0.2
    beta_nll_anneal_pct: float = 0.4
    composer_lr_mult: float = 0.1


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
