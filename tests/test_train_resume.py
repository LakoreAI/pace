import json

import pytest
import torch

from src.data import build_cache, make_synthetic_clips
from src.features import PRESETS
from src.pipelines.config import TrainingConfig
from src.pipelines import train as train_module
from src.pipelines.train import train

CLASSES = ("tone_a", "tone_b")


@pytest.fixture(scope="module")
def data_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("resume")
    params = PRESETS["tut2017"]
    for seed, (split, n) in enumerate({"train": 4, "val": 2}.items()):
        items = make_synthetic_clips(
            root / "audio", split, n, CLASSES, duration=3.0, seed=seed
        )
        build_cache(items, CLASSES, params, 9, root / f"{split}.npz")
    return root


def _cfg(root, out, **kw):
    base = dict(
        data_root=str(root),
        train_file="train.npz",
        val_file="val.npz",
        test_file=None,
        ckpt_dir=str(out / "ckpt"),
        result_dir=str(out / "res"),
        run_name="run",
        experiment="onepass",
        dataset="synth",
        epochs=2,
        batch_size=4,
        log_every=100,
        ckpt_every=10,
        lr_scheduler={"type": "none"},
        arch={"dropout": 0.0},
    )
    base.update(kw)
    return TrainingConfig(**base)


def test_resume_matches_uninterrupted_run(data_root, tmp_path, monkeypatch):
    monkeypatch.setattr(train_module, "detect_device", lambda: torch.device("cpu"))
    full = train(_cfg(data_root, tmp_path / "full"))

    train(_cfg(data_root, tmp_path / "split", epochs=1))
    last = tmp_path / "split" / "ckpt" / "run" / "last.pt"
    resumed = train(_cfg(data_root, tmp_path / "split", resume_from=str(last)))

    assert [r["epoch"] for r in resumed] == [1, 2]
    for a, b in zip(full, resumed):
        assert a["loss"] == pytest.approx(b["loss"], rel=1e-6)
        assert a["val_loss"] == pytest.approx(b["val_loss"], rel=1e-6)


def test_summary_file_has_config_and_commit(data_root, tmp_path):
    train(_cfg(data_root, tmp_path, epochs=1))
    summary = json.loads((tmp_path / "res" / "onepass_synth_s42.json").read_text())
    assert summary["experiment"] == "onepass" and summary["seed"] == 42
    assert "git_commit" in summary and summary["train_cfg"]["epochs"] == 1
    assert summary["params"]["n_all"] > 0
    assert (tmp_path / "ckpt" / "run" / "last.pt").exists()
