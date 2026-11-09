from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from src.config import PACEConfig
from src.modules.augment import spec_augment_batch
from src.modules.model import PACEModel
from src.pipelines._utils import announce_training, label_counts
from src.pipelines.config import TrainingConfig


def test_label_counts_counts_windows_per_class(tiny_cache):
    counts = label_counts(tiny_cache)
    assert counts.shape == (tiny_cache.num_classes,)
    assert int(counts.sum()) >= 1


def test_label_counts_unlabeled_returns_none():
    class Plain(Dataset):
        def __len__(self):
            return 2

        def __getitem__(self, i):
            return torch.zeros(3), 0

    assert label_counts(Plain()) is None


def test_spec_augment_masks_shared_across_batch():
    g = torch.Generator().manual_seed(0)
    x = torch.ones(3, 50, 20)
    out = spec_augment_batch(x, freq_param=8, time_param=10, generator=g)
    assert torch.equal(out[0], out[1]) and torch.equal(out[1], out[2])
    assert (out == 0).any() and torch.equal(x, torch.ones(3, 50, 20))


def test_announce_training_prints_config_data_and_model(capsys, tiny_cache):
    cfg = PACEConfig(num_classes=tiny_cache.num_classes)
    announce_training(
        model=PACEModel(cfg),
        model_cfg=cfg,
        train_cfg=TrainingConfig(),
        train_loader=DataLoader(tiny_cache, batch_size=2),
        train_dataset=tiny_cache,
        device=torch.device("cpu"),
        run_name="test",
        ckpt_dir=Path("ckpt"),
        result_dir=Path("res"),
    )
    out = capsys.readouterr().out
    assert "PACEModel" in out and "TOTAL" in out and "pred" in out
