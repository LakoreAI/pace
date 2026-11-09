"""End-to-end smoke test on synthetic audio.

Writes a few noisy clips with sine-tone events, builds train / val / test
feature caches (TUT 2017 settings), runs a couple of training epochs, evaluates
the best checkpoint, and checks that the pipeline produces finite sed_eval
metrics. No external dataset required.

Usage:
    uv run python scripts/training/smoke_test.py
"""

import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.config import PACEConfig  # noqa: E402
from src.data import build_cache, load_cache, make_synthetic_clips  # noqa: E402
from src.features import PRESETS  # noqa: E402
from src.pipelines.config import TrainingConfig  # noqa: E402
from src.pipelines.eval import evaluate, format_report  # noqa: E402
from src.pipelines.infer import load_model  # noqa: E402
from src.pipelines.train import train  # noqa: E402
from src.utils.model_utils import detect_device  # noqa: E402

CLASSES = ("tone_a", "tone_b", "tone_c")
N_FILES = {"train": 8, "val": 3, "test": 3}


def build_data(root: Path) -> None:
    params = PRESETS["tut2017"]
    n_cells = PACEConfig(n_frames=params.n_frames, n_mels=params.n_mels).n_cells
    for seed, (split, n) in enumerate(N_FILES.items()):
        items = make_synthetic_clips(
            root / "audio_root", split, n, classes=CLASSES, duration=5.0, seed=seed
        )
        build_cache(items, CLASSES, params, n_cells, root / f"{split}.npz")


def main() -> None:
    torch.manual_seed(0)
    tmp = Path(tempfile.mkdtemp(prefix="pace_smoke_"))
    try:
        build_data(tmp)
        cfg = TrainingConfig(
            data_root=str(tmp),
            train_file="train.npz",
            val_file="val.npz",
            test_file="test.npz",
            ckpt_dir=str(tmp / "checkpoints"),
            result_dir=str(tmp / "results"),
            epochs=2,
            batch_size=8,
            eval_every=1,
            ckpt_every=1,
            log_every=5,
            lr=1e-3,
            lr_scheduler={"type": "none"},
            spec_augment={"freq_param": 8, "time_param": 25, "time_mask_num": 2},
            save_best=True,
            best_metric="error_rate",
            best_mode="min",
        )
        train(cfg)

        best = list((tmp / "checkpoints").glob("*/best.pt"))
        if not best:
            raise RuntimeError("no best.pt produced")
        device = detect_device()
        model, _ = load_model(best[0], device)
        result = evaluate(model, load_cache(tmp / "test.npz"), device)
        print(format_report(result))
        assert np.isfinite(result["error_rate"]), "error rate is not finite"
        print("\nsmoke test: PASS")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
