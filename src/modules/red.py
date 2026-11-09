"""Boundary-aware event head: RED + Onset-Offset Loss + Event Proposal Network.

Credit: Schmid et al., "Sound Event Detection with Boundary-Aware Optimization
and Inference", arXiv:2601.04178. RED (Recurrent Event Detection) is a
parameter-free probabilistic recurrence; the proposal network and IoU loss are
adapted here.

    backbone seq (B, T, D)
      -> 1x1 conv -> start / end logits (B, T, 2C)
      -> RED scan -> presence / onset / offset probabilities
      -> EPN (conv or BiGRU) -> time-since-onset / time-to-offset (B, T, 2C)
      -> NMS-like proposal inference -> events

The default EPN is a dilated Conv1d stack to keep the detector convolutional
and cheap. `epn_type: "gru"` selects the bidirectional-GRU variant as a
reference; it is not the default so the event head stays purely convolutional.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def red_scan(
    start_logits: torch.Tensor, end_logits: torch.Tensor
) -> dict[str, torch.Tensor]:
    """Parameter-free RED recurrence.

    `start_logits` model P(e_t | not e_{t-1}); `end_logits` model
    P(not e_t | e_{t-1}). Returns onset, offset, presence probabilities.
    """
    p_start = torch.sigmoid(start_logits)
    p_end = torch.sigmoid(end_logits)
    b, t, c = p_start.shape
    presence, onset, offset = [], [], []
    prev = torch.zeros((b, c), dtype=p_start.dtype, device=p_start.device)
    for step in range(t):
        start = p_start[:, step]
        end = p_end[:, step]
        on = start * (1.0 - prev)
        off = end * prev
        pres = on + (1.0 - end) * prev
        onset.append(on)
        offset.append(off)
        presence.append(pres)
        prev = pres
    return {
        "pres": torch.stack(presence, dim=1),
        "on": torch.stack(onset, dim=1),
        "off": torch.stack(offset, dim=1),
        "start": p_start,
        "end": p_end,
    }


class ConvEPN(nn.Module):
    """Dilated Conv1d proposal network over [presence, onset, offset]."""

    def __init__(
        self,
        num_classes: int,
        hidden: int = 64,
        layers: int = 2,
        kernel: int = 3,
        causal: bool = False,
        future_pads: list[int] | None = None,
    ):
        super().__init__()
        self.causal = causal
        self.kernel = kernel
        self.future_pads = list(future_pads or [0] * layers)
        if len(self.future_pads) != layers:
            raise ValueError("future_pads must have one entry per EPN layer")
        in_ch = 3 * num_classes
        blocks = []
        for i in range(layers):
            dilation = 2**i
            blocks += [
                nn.Conv1d(
                    in_ch if i == 0 else hidden,
                    hidden,
                    kernel,
                    padding=0 if causal else dilation * (kernel // 2),
                    dilation=dilation,
                ),
                nn.ReLU(inplace=True),
            ]
        self.net = nn.Sequential(*blocks, nn.Conv1d(hidden, 2 * num_classes, 1))

    def forward(self, presence, onset, offset):
        x = torch.cat([presence, onset, offset], dim=-1).transpose(1, 2)
        if self.causal:
            for future, layer in zip(self.future_pads, self.net[:-1:2]):
                dilation = layer.dilation[0]
                kernel = layer.kernel_size[0]
                x = F.pad(x, (dilation * (kernel - 1 - future), future * dilation))
                x = F.relu(layer(x))
            x = self.net[-1](x)
            return F.softplus(x).transpose(1, 2)
        return F.softplus(self.net(x)).transpose(1, 2)


class GRUEPN(nn.Module):
    """Bidirectional-GRU proposal network; unidirectional if causal."""

    def __init__(
        self,
        num_classes: int,
        hidden: int = 64,
        layers: int = 2,
        kernel: int = 3,
        causal: bool = False,
    ):
        super().__init__()
        self.bidirectional = not causal
        self.gru = nn.GRU(
            3 * num_classes,
            hidden,
            num_layers=layers,
            batch_first=True,
            bidirectional=self.bidirectional,
        )
        factor = 2 if self.bidirectional else 1
        self.out = nn.Linear(factor * hidden, 2 * num_classes)

    def forward(self, presence, onset, offset):
        x = torch.cat([presence, onset, offset], dim=-1)
        y, _ = self.gru(x)
        return F.softplus(self.out(y))


def _make_epn(cfg, future_pads=None):
    kwargs = dict(
        num_classes=cfg.num_classes,
        hidden=cfg.epn_hidden,
        layers=cfg.epn_layers,
        kernel=cfg.epn_kernel,
        causal=cfg.causal,
    )
    if cfg.epn_type == "conv":
        return ConvEPN(**kwargs, future_pads=future_pads)
    if cfg.epn_type == "gru":
        return GRUEPN(**kwargs)
    raise ValueError(f"unknown epn_type {cfg.epn_type!r}, expected 'conv' or 'gru'")


class REDEPNHead(nn.Module):
    """Start/end logits -> RED -> proposal network, returning a dict of maps."""

    def __init__(self, in_features: int, cfg):
        super().__init__()
        self.num_classes = cfg.num_classes
        self.proj = nn.Conv1d(in_features, 2 * cfg.num_classes, 1)
        _, epn_pads, _ = cfg.future_padding()
        self.epn = _make_epn(cfg, epn_pads)

    def forward(self, seq: torch.Tensor) -> dict[str, torch.Tensor]:
        """seq: (B, T, D) -> dict of (B, T, C) maps plus durations."""
        logits = self.proj(seq.transpose(1, 2)).transpose(1, 2)
        red = red_scan(logits[..., : self.num_classes], logits[..., self.num_classes :])
        duration = self.epn(red["pres"], red["on"], red["off"])
        return {
            **red,
            "don": duration[..., : self.num_classes],
            "doff": duration[..., self.num_classes :],
        }
