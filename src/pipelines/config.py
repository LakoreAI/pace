"""Training-loop hyperparameters — separate from `src.config.PACEConfig`
(model architecture only). Controllable via a YAML file (see
`configs/onepass_tut2017.yaml`) with individual fields overridable from the CLI.
"""

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional, Union

from src.utils.io_utils import read_yaml


@dataclass
class TrainingConfig:
    data_root: str = "data/features"
    train_file: str = "train.npz"
    val_file: Optional[str] = "val.npz"
    val_files: Optional[List[str]] = None
    test_file: Optional[str] = "test.npz"
    seed: int = 42

    experiment: str = "pace"
    dataset: Optional[str] = None
    fold: Optional[int] = None

    ckpt_dir: str = "checkpoints"
    result_dir: str = "results"
    run_name: Optional[str] = None

    epochs: int = 100
    batch_size: int = 128
    accum_steps: int = 1
    num_workers: int = 0
    pin_memory: bool = False
    amp: bool = False
    amp_dtype: str = "bf16"
    lr: float = 1e-3
    optimizer: str = "adam"
    weight_decay: float = 0.0
    log_every: int = 10
    eval_every: int = 1
    ckpt_every: int = 10

    arch: Optional[dict] = None
    n_cells: Optional[int] = None

    target_kind: str = "onepass"
    loss: Optional[dict] = None
    decoder: Optional[dict] = None
    spec_augment: Optional[dict] = None
    recipe: Optional[dict] = None
    postprocess: Optional[dict] = None
    time_resolution: float = 1.0
    eval_batch_size: int = 64
    psds: Optional[dict] = None
    streaming: Optional[dict] = None
    benchmark: Optional[dict] = None
    profile: bool = False
    profile_repeats: int = 20

    lr_scheduler: Optional[dict] = None
    early_stopping: Optional[dict] = None
    save_best: bool = True
    best_metric: str = "error_rate"
    best_mode: str = "min"

    ema: Optional[dict] = None

    wandb: Optional[dict] = None

    resume_from: Optional[str] = None

    def __post_init__(self) -> None:
        positive = (
            "epochs",
            "batch_size",
            "accum_steps",
            "eval_every",
            "ckpt_every",
            "log_every",
            "eval_batch_size",
            "profile_repeats",
        )
        for name in positive:
            value = getattr(self, name)
            if not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive int, got {value!r}")
        if self.num_workers < 0:
            raise ValueError(f"num_workers must be >= 0, got {self.num_workers}")
        if self.lr <= 0:
            raise ValueError(f"lr must be > 0, got {self.lr}")
        if self.optimizer not in ("adam", "adamw"):
            raise ValueError(
                f"optimizer must be 'adam' or 'adamw', got {self.optimizer!r}"
            )
        if self.n_cells is not None and self.n_cells <= 0:
            raise ValueError(f"n_cells must be > 0, got {self.n_cells}")
        if self.best_mode not in ("min", "max"):
            raise ValueError(
                f"best_mode must be 'min' or 'max', got {self.best_mode!r}"
            )
        if self.fold is not None and self.fold < 0:
            raise ValueError(f"fold must be >= 0, got {self.fold}")
        if not self.val_files and not self.val_file:
            raise ValueError("set val_file or val_files (validation is required)")
        if not self.target_kind:
            raise ValueError("target_kind must not be empty")

    @property
    def dataset_family(self) -> str:
        """Dataset tag used in result filenames and summaries."""
        return self.dataset or Path(self.data_root).name

    def result_filename(self) -> str:
        """Unique summary filename: fold is part of the identity, not the tag."""
        stem = f"{self.experiment}_{self.dataset_family}_s{self.seed}"
        if self.fold is not None:
            stem += f"_fold{self.fold}"
        return f"{stem}.json"


def load_training_config(
    config_path: Optional[Union[str, Path]] = None, **cli_overrides
) -> TrainingConfig:
    """Build a TrainingConfig from defaults, a YAML file, then CLI overrides
    (highest precedence, but only applied for keys the caller actually passed —
    argparse defaults of None are treated as "not set").
    """
    cfg = TrainingConfig()

    if config_path is not None:
        yaml_data = read_yaml(config_path)
        known_fields = set(asdict(cfg).keys())
        for key, value in yaml_data.items():
            if key not in known_fields:
                raise ValueError(
                    f"unknown training config key {key!r} in {config_path}"
                )
            setattr(cfg, key, value)

    for key, value in cli_overrides.items():
        if value is not None:
            setattr(cfg, key, value)

    return cfg
