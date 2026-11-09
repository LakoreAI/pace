import numpy as np
import pytest
import torch

from src.config import PACEConfig
from src.modules.model import PACEModel
from src.pipelines.eval import eval_per_epoch, format_report
from src.sed.metrics import segment_metrics

CLASSES = ["a", "b"]


def test_perfect_estimate_scores_zero_error():
    ref = {"f1.wav": [(0.0, 2.0, "a"), (3.0, 4.0, "b")], "f2.wav": []}
    result = segment_metrics(ref, ref, CLASSES)
    assert result["error_rate"] == pytest.approx(0.0)
    assert result["f_measure"] == pytest.approx(1.0)


def test_missing_estimates_count_as_no_detections():
    ref = {"f1.wav": [(0.0, 2.0, "a")]}
    result = segment_metrics(ref, {}, CLASSES)
    assert result["error_rate"] == pytest.approx(1.0)
    assert np.isnan(result["f_measure"])


def test_false_positive_on_empty_file_is_counted():
    ref = {"f1.wav": [(0.0, 2.0, "a")], "f2.wav": []}
    est = {"f1.wav": [(0.0, 2.0, "a")], "f2.wav": [(0.0, 1.0, "b")]}
    assert segment_metrics(ref, est, CLASSES)["error_rate"] > 0.0


def test_eval_per_epoch_on_cache(tiny_cache):
    torch.manual_seed(0)
    cfg = PACEConfig(num_classes=tiny_cache.num_classes)
    result = eval_per_epoch(
        PACEModel(cfg), tiny_cache, torch.device("cpu"), batch_size=4
    )
    assert np.isfinite(result["error_rate"]) and np.isfinite(result["loss"])
    assert np.isnan(result["f_measure"]) or 0.0 <= result["f_measure"] <= 1.0
    assert result["postprocess_s"] >= 0.0
    assert "ER=" in format_report(result)
