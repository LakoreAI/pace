"""Training losses.

`PACELoss` is the original one-pass detector loss (`my_loss_fn` in the released notebooks):
sum-squared error over every cell, where the onset / offset errors of a class
only count in cells where that class is present. Keras averages the per-example
sum over the batch; so does this.

`sigmoid_focal_loss` and `iou_1d_loss` are building blocks for event-level heads
(presence / objectness classification and boundary regression).
"""

import inspect
from typing import Callable, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class PACELoss(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.num_classes = num_classes

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """pred, target: (B, n_cells, 3 * num_classes)."""
        b, t, _ = target.shape
        target3 = target.reshape(b, t, self.num_classes, 3)
        present = target3[..., 0:1]
        mask = torch.cat([torch.ones_like(present), present, present], dim=-1)
        sq = (pred.reshape(b, t, self.num_classes, 3) - target3).pow(2) * mask
        return sq.sum(dim=(1, 2, 3)).mean()


class BinarySEDLoss(nn.Module):
    """BCE or focal loss for a dense per-cell/per-frame presence output.

    `pred` holds probabilities in [0, 1] (the heads sigmoid their logits), so
    this is the counterpart of `PACELoss` for the frame-wise arm. `target` and
    `pred` both have shape (..., C).
    """

    def __init__(
        self,
        kind: str = "bce",
        gamma: float = 2.0,
        alpha: float = -1.0,
        eps: float = 1e-7,
    ):
        super().__init__()
        if kind not in ("bce", "focal"):
            raise ValueError(f"kind must be 'bce' or 'focal', got {kind!r}")
        self.kind = kind
        self.gamma = gamma
        self.alpha = alpha
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        p = pred.clamp(self.eps, 1.0 - self.eps)
        ce = -(target * p.log() + (1.0 - target) * (1.0 - p).log())
        if self.kind == "focal":
            p_t = p * target + (1.0 - p) * (1.0 - target)
            ce = ce * (1.0 - p_t).pow(self.gamma)
            if self.alpha >= 0:
                ce = ce * (self.alpha * target + (1.0 - self.alpha) * (1.0 - target))
        return ce.mean()


def focal_from_probs(
    probs: torch.Tensor, target: torch.Tensor, alpha: float = 2.0, eps: float = 1e-7
) -> torch.Tensor:
    """Focal loss on probabilities (the RED onset/offset form)."""
    p = probs.clamp(eps, 1.0 - eps)
    pt = torch.where(target > 0, p, 1.0 - p)
    return (-(1.0 - pt).pow(alpha) * pt.log()).mean()


class REDEPNLoss(nn.Module):
    """RED+EPN objective: presence BCE + focal onset/offset + 1D IoU duration.

    `pred` is the head's dict; `target` is (B, T, 5 * C) from
    `src.sed.targets.encode_red_epn`. Default weights follow arXiv:2601.04178
    (lambda_ool=100; lambda_iou is tuned there, 1.0 here as a neutral default).
    """

    def __init__(
        self,
        num_classes: int,
        lambda_ool: float = 100.0,
        lambda_iou: float = 1.0,
        focal_alpha: float = 2.0,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.lambda_ool = lambda_ool
        self.lambda_iou = lambda_iou
        self.focal_alpha = focal_alpha

    def forward(self, pred: dict, target: torch.Tensor) -> torch.Tensor:
        b, t, _ = target.shape
        c = self.num_classes
        target = target.reshape(b, t, 5, c)
        presence_t = target[..., 0, :]
        onset_t = target[..., 1, :]
        offset_t = target[..., 2, :]
        don_t = target[..., 3, :]
        doff_t = target[..., 4, :]

        presence_p = pred["pres"].clamp(1e-7, 1.0 - 1e-7)
        presence = F.binary_cross_entropy(presence_p, presence_t)
        onset = focal_from_probs(pred["on"], onset_t, self.focal_alpha)
        offset = focal_from_probs(pred["off"], offset_t, self.focal_alpha)

        frames = torch.arange(t, dtype=target.dtype, device=target.device).view(1, t, 1)
        true_seg = torch.stack([frames - don_t, frames + doff_t], dim=-1)
        pred_seg = torch.stack([frames - pred["don"], frames + pred["doff"]], dim=-1)
        active = presence_t > 0
        if active.any():
            weight = 1.0 / (don_t + doff_t).clamp(min=1e-3)
            iou = iou_1d_loss(
                pred_seg[active], true_seg[active], kind="iou", weight=weight[active]
            )
        else:
            iou = torch.zeros((), dtype=target.dtype, device=target.device)
        return presence + self.lambda_ool * (onset + offset) + self.lambda_iou * iou


LOSSES: Dict[str, Callable[..., nn.Module]] = {
    "onepass": PACELoss,
    "bce": BinarySEDLoss,
    "focal": BinarySEDLoss,
    "red_epn": REDEPNLoss,
}


def register_loss(name: str):
    def deco(cls):
        LOSSES[name] = cls
        return cls

    return deco


def build_loss(
    cfg: dict | str | None, num_classes: int, target_kind: str = "onepass"
) -> nn.Module:
    """Build a loss from a config.

    `cfg=None` picks `onepass` for one-pass targets and `bce` otherwise. `{"type":
    "focal", "gamma": 2}` maps to `BinarySEDLoss(kind="focal")`.
    """
    if cfg is None:
        cfg = {"type": "onepass" if target_kind == "onepass" else "bce"}
    if isinstance(cfg, str):
        cfg = {"type": cfg}
    name = cfg.get("type")
    if name not in LOSSES:
        raise ValueError(f"unknown loss {name!r}, expected one of {sorted(LOSSES)}")
    kwargs = {k: v for k, v in cfg.items() if k != "type"}
    if name == "onepass":
        return PACELoss(num_classes)
    if name in ("bce", "focal"):
        kwargs.setdefault("kind", name)
        return BinarySEDLoss(**kwargs)
    if name == "red_epn":
        return REDEPNLoss(num_classes=num_classes, **kwargs)
    cls = LOSSES[name]
    if "num_classes" in inspect.signature(cls).parameters:
        kwargs.setdefault("num_classes", num_classes)
    return cls(**kwargs)


def sigmoid_focal_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    gamma: float = 2.0,
    alpha: float = -1.0,
    reduction: str = "mean",
) -> torch.Tensor:
    """Focal loss on logits (Lin et al., 2017), for presence / objectness / class outputs.

    `alpha < 0` disables class weighting. `gamma = 0` reduces to BCE.
    """
    p = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    p_t = p * target + (1 - p) * (1 - target)
    loss = ce * (1 - p_t).pow(gamma)
    if alpha >= 0:
        loss = loss * (alpha * target + (1 - alpha) * (1 - target))
    if reduction == "sum":
        return loss.sum()
    if reduction == "none":
        return loss
    return loss.mean()


def iou_1d_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    kind: str = "diou",
    weight: torch.Tensor | None = None,
    eps: float = 1e-7,
) -> torch.Tensor:
    """1 - IoU-family overlap between 1D segments, averaged over segments.

    pred, target: (..., 2) as (onset, offset). `kind`: "iou", "giou" (Rezatofighi
    et al., 2019) or "diou" (Zheng et al., 2020; ActionFormer's choice). `weight`
    (...,) e.g. 1 / duration to stop long events dominating, as in RED+EPN.
    """
    if kind not in ("iou", "giou", "diou"):
        raise ValueError(f"kind must be iou, giou or diou, got {kind!r}")
    p_on, p_off = pred[..., 0], pred[..., 1]
    t_on, t_off = target[..., 0], target[..., 1]
    inter = (torch.min(p_off, t_off) - torch.max(p_on, t_on)).clamp(min=0)
    union = (p_off - p_on).clamp(min=0) + (t_off - t_on) - inter
    iou = inter / union.clamp(min=eps)
    hull = (torch.max(p_off, t_off) - torch.min(p_on, t_on)).clamp(min=eps)
    if kind == "giou":
        iou = iou - (hull - union) / hull
    elif kind == "diou":
        centre_dist = 0.5 * ((p_on + p_off) - (t_on + t_off))
        iou = iou - centre_dist.pow(2) / hull.pow(2)
    loss = 1.0 - iou
    if weight is not None:
        return (loss * weight).sum() / weight.sum().clamp(min=eps)
    return loss.mean()
