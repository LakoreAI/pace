import numpy as np
import pytest
import torch

from src.modules.loss import iou_1d_loss, sigmoid_focal_loss
from src.sed.postprocess import framewise_events, nms_1d


def test_focal_with_gamma_zero_equals_bce():
    logits = torch.randn(10)
    target = (torch.rand(10) > 0.5).float()
    bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, target)
    assert sigmoid_focal_loss(logits, target, gamma=0.0) == pytest.approx(bce.item())


def test_focal_downweights_easy_examples():
    target = torch.ones(1)
    easy, hard = torch.tensor([4.0]), torch.tensor([-1.0])
    ratio = sigmoid_focal_loss(easy, target) / sigmoid_focal_loss(hard, target)
    bce_ratio = torch.nn.functional.binary_cross_entropy_with_logits(easy, target) / (
        torch.nn.functional.binary_cross_entropy_with_logits(hard, target)
    )
    assert ratio < bce_ratio


@pytest.mark.parametrize("kind", ["iou", "giou", "diou"])
def test_iou_losses_zero_for_identical_segments(kind):
    seg = torch.tensor([[0.0, 1.0], [2.0, 5.0]])
    assert iou_1d_loss(seg, seg, kind=kind).item() == pytest.approx(0.0, abs=1e-6)


def test_giou_and_diou_penalise_distance_when_disjoint():
    target = torch.tensor([[0.0, 1.0]])
    near, far = torch.tensor([[1.5, 2.5]]), torch.tensor([[5.0, 6.0]])
    assert iou_1d_loss(near, target, "iou") == iou_1d_loss(far, target, "iou") == 1.0
    for kind in ("giou", "diou"):
        assert iou_1d_loss(far, target, kind) > iou_1d_loss(near, target, kind)


def test_iou_loss_weighting():
    target = torch.tensor([[0.0, 1.0], [0.0, 10.0]])
    pred = torch.tensor([[0.0, 1.0], [0.0, 5.0]])
    w = 1.0 / (target[:, 1] - target[:, 0])
    assert iou_1d_loss(pred, target, "iou", weight=w) < iou_1d_loss(pred, target, "iou")


def test_hard_nms_keeps_best_per_class():
    events = [
        (0.0, 1.0, "a", 0.9),
        (0.1, 1.1, "a", 0.8),
        (0.1, 1.1, "b", 0.7),
        (3, 4, "a", 0.5),
    ]
    kept = nms_1d(events, iou_threshold=0.5)
    assert kept == [(0.0, 1.0, "a", 0.9), (0.1, 1.1, "b", 0.7), (3, 4, "a", 0.5)]


def test_soft_nms_decays_instead_of_dropping():
    events = [(0.0, 1.0, "a", 0.9), (0.1, 1.1, "a", 0.8)]
    kept = nms_1d(events, soft=True, sigma=0.5)
    assert len(kept) == 2
    assert kept[1][3] < 0.8


def test_framewise_events_and_median_filter():
    scores = np.zeros((10, 2))
    scores[2:6, 0] = 0.9
    scores[7, 0] = 0.9
    scores[0:10, 1] = 0.2
    raw = framewise_events(scores, ["a", "b"], frame_hop=0.1)
    assert raw == [
        (pytest.approx(0.2), pytest.approx(0.6), "a"),
        (pytest.approx(0.7), pytest.approx(0.8), "a"),
    ]
    smoothed = framewise_events(scores, ["a", "b"], 0.1, median_lengths=[3, 1])
    assert len(smoothed) == 1
    assert framewise_events(scores, ["a", "b"], 0.1, threshold=[0.95, 0.1])[0][2] == "b"
