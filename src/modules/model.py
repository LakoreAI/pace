"""PACEModel model, ported from the original Keras notebooks.

Credit: Venkatesh, Moffat, Miranda, "You Only Hear Once", Applied Sciences 2022;
github.com/satvik-venkatesh/you-only-hear-once (MIT).

    mel (B, n_frames, n_mels)
      -> PACEBackbone            -> feature (B, C, n_cells, F')
      -> flatten freq x channels -> (B, n_cells, F' * C)
      -> PACEHead (1x1 Conv1d + sigmoid) -> pred (B, n_cells, 3 * num_classes)

`forward` returns a dict so other heads (event-level, frame-wise) can
add outputs without changing the training loop: {"pred", "feature"}.

`SEDModel` is the generic backbone -> optional context -> head stack; `PACEModel` is its
original instance. Components are looked up by name in `BACKBONES`, `CONTEXTS` and
`HEADS` (`register_backbone` etc.), and `build_model(cfg)` assembles them:

    backbone : (B, n_frames, n_mels) -> (B, C, T, F), exposes `out_channels`
    context  : (B, T, D) -> (B, T, D), built as `cls(dim, cfg)`
    head     : (B, T, D) -> pred, built as `cls(dim, cfg)`, `D = C * F`
"""

from typing import Callable, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import PACEConfig
from src.modules.red import REDEPNHead


def _bn(channels: int, cfg: PACEConfig) -> nn.BatchNorm2d:
    bn = nn.BatchNorm2d(channels, eps=cfg.bn_eps, momentum=cfg.bn_momentum)
    bn.weight.requires_grad_(False)
    return bn


class SeparableBlock(nn.Module):
    """Depthwise 3x3 -> BN -> ReLU -> pointwise 1x1 -> BN -> ReLU -> Dropout2d."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel: int,
        stride: int,
        cfg: PACEConfig,
        future_pad: int = 0,
    ):
        super().__init__()
        self.causal = cfg.causal
        self.kernel = kernel
        self.future_pad = future_pad
        self.depthwise = nn.Conv2d(
            in_ch,
            in_ch,
            kernel,
            stride=stride,
            padding=(0, kernel // 2) if self.causal else kernel // 2,
            groups=in_ch,
            bias=False,
        )
        self.bn1 = _bn(in_ch, cfg)
        self.pointwise = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn2 = _bn(out_ch, cfg)
        self.act = nn.ReLU(inplace=True)
        self.drop = nn.Dropout2d(cfg.dropout) if cfg.dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.causal:
            x = F.pad(x, (0, 0, self.kernel - 1 - self.future_pad, self.future_pad))
        x = self.act(self.bn1(self.depthwise(x)))
        x = self.act(self.bn2(self.pointwise(x)))
        return self.drop(x)


class PACEBackbone(nn.Module):
    def __init__(self, cfg: PACEConfig):
        super().__init__()
        self.causal = cfg.causal
        self.future_pads, _, self.future_reach_frames = cfg.future_padding()
        self.stem_future_pad = self.future_pads[0]
        self.stem = nn.Conv2d(
            1,
            cfg.stem_channels,
            3,
            stride=2,
            padding=(0, 1) if self.causal else 1,
            bias=False,
        )
        self.stem_bn = _bn(cfg.stem_channels, cfg)
        self.act = nn.ReLU(inplace=True)
        blocks = []
        in_ch = cfg.stem_channels
        for i, (kernel, stride, out_ch) in enumerate(cfg.layer_defs):
            blocks.append(
                SeparableBlock(
                    in_ch,
                    out_ch,
                    kernel,
                    stride,
                    cfg,
                    future_pad=self.future_pads[i + 1],
                )
            )
            in_ch = out_ch
        self.blocks = nn.Sequential(*blocks)
        self.out_channels = in_ch
        self._cfg = cfg

    def regularization_loss(self) -> torch.Tensor:
        """Weight-decay terms of the original model (stem and pointwise kernels only)."""
        reg = self._cfg.l2_stem * self.stem.weight.pow(2).sum()
        for block in self.blocks:
            reg = reg + self._cfg.l2_pointwise * block.pointwise.weight.pow(2).sum()
        return reg

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, n_frames, n_mels) -> (B, C, n_cells, F')."""
        x = x.unsqueeze(1)
        if self.causal:
            x = F.pad(x, (0, 0, 2 - self.stem_future_pad, self.stem_future_pad))
        x = self.act(self.stem_bn(self.stem(x)))
        return self.blocks(x)


class PACEHead(nn.Module):
    """Per-cell presence / onset / offset for every class, all sigmoid."""

    def __init__(self, in_features: int, cfg: PACEConfig):
        super().__init__()
        self.proj = nn.Conv1d(in_features, 3 * cfg.num_classes, 1)

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        """seq: (B, T, D) -> (B, T, 3 * num_classes)."""
        return torch.sigmoid(self.proj(seq.transpose(1, 2))).transpose(1, 2)


class FrameWiseHead(nn.Module):
    """Per-cell class presence: the matched dense (frame-wise) baseline head.

    It shares the PACEModel backbone's output grid so the only difference from the
    event head is the head and decoder, which is what the A-ladder compares.
    """

    def __init__(self, in_features: int, cfg: PACEConfig):
        super().__init__()
        self.proj = nn.Conv1d(in_features, cfg.num_classes, 1)

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        """seq: (B, T, D) -> (B, T, num_classes) probabilities."""
        return torch.sigmoid(self.proj(seq.transpose(1, 2))).transpose(1, 2)


class CausalMaxPool2d(nn.Module):
    """Max pooling with a causal time axis.

    Left-pads `size - 1` time steps with -inf so output t covers the current
    block of frames ending at t, never a future one. The output length stays
    ceil(T / size), matching the non-causal pooled grid.
    """

    def __init__(self, size: int = 2, future: int = 0, freq_ceil: bool = False):
        super().__init__()
        self.size = size
        self.future = future
        self.freq_ceil = freq_ceil

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        freq_pad = (-x.shape[-1]) % self.size if self.freq_ceil else 0
        x = F.pad(
            x,
            (0, freq_pad, self.size - 1 - self.future, self.future),
            value=float("-inf"),
        )
        return F.max_pool2d(x, (self.size, self.size), stride=(self.size, self.size))


class DynamicConv2d(nn.Module):
    """Frequency-dynamic convolution (Nam et al., Interspeech 2022).

    A bank of ``n_basis`` kernels is mixed per frequency bin with attention
    pooled over time, so the operator is no longer translation-equivariant
    along frequency. Time padding follows the causal/non-causal convention of
    ``CRNNBlock``; the caller has already applied the look-ahead padding.
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int = 3,
        n_basis: int = 3,
        causal: bool = True,
    ):
        super().__init__()
        self.in_ch, self.out_ch = in_ch, out_ch
        self.kernel_size, self.n_basis = kernel_size, n_basis
        self.padding = (0, kernel_size // 2) if causal else kernel_size // 2
        self.weight = nn.Parameter(
            torch.empty(n_basis, out_ch, in_ch, kernel_size, kernel_size)
        )
        nn.init.kaiming_normal_(
            self.weight.view(n_basis * out_ch, in_ch, kernel_size, kernel_size)
        )
        hidden = max(in_ch // 4, 4)
        self.attention = nn.Sequential(
            nn.Conv1d(in_ch, hidden, kernel_size, padding=kernel_size // 2, bias=False),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden, n_basis, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = torch.softmax(self.attention(x.mean(dim=2)), dim=1)
        weights = weights.unsqueeze(2).unsqueeze(3)
        batch = x.shape[0]
        kernels = self.weight.reshape(
            self.n_basis * self.out_ch, self.in_ch, self.kernel_size, self.kernel_size
        )
        out = F.conv2d(x, kernels, padding=self.padding)
        out = out.view(batch, self.n_basis, self.out_ch, out.shape[-2], out.shape[-1])
        return (out * weights).sum(dim=1)


class CRNNBlock(nn.Module):
    """Conv -> BN -> ReLU -> pool, optionally causal on the time axis."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        cfg: PACEConfig,
        future_conv: int = 0,
        future_pool: int = 0,
        pool: int | None = None,
    ):
        super().__init__()
        self.causal = cfg.causal
        self.future_conv = future_conv
        self.pool_size = pool if pool is not None else cfg.crnn_pool
        if getattr(cfg, "crnn_conv", "conv") == "fdy":
            self.conv = DynamicConv2d(
                in_ch,
                out_ch,
                kernel_size=3,
                n_basis=cfg.fdy_basis,
                causal=self.causal,
            )
        else:
            self.conv = nn.Conv2d(
                in_ch, out_ch, 3, padding=(0, 1) if self.causal else 1, bias=False
            )
        self.bn = nn.BatchNorm2d(out_ch, eps=cfg.bn_eps, momentum=cfg.bn_momentum)
        self.act = nn.ReLU(inplace=True)
        self.pool = (
            CausalMaxPool2d(
                self.pool_size,
                future=future_pool,
                freq_ceil=getattr(cfg, "causal_pool_freq_ceil", False),
            )
            if cfg.causal
            else nn.MaxPool2d(self.pool_size, ceil_mode=True)
        )
        self.drop = (
            nn.Dropout2d(cfg.crnn_dropout) if cfg.crnn_dropout > 0 else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.causal:
            x = F.pad(x, (0, 0, 2 - self.future_conv, self.future_conv))
        return self.drop(self.pool(self.act(self.bn(self.conv(x)))))


class CRNNBackbone(nn.Module):
    """Small conv stack with pooling; the frame-level CRNN regime.

    The RED+EPN CRNN reaches PSDS1 48.0 on AS-Strong-10 with 1.4M parameters,
    so capacity is not the constraint for this task. Four 2x-pooled blocks give
    a temporal stride of 16 (63 cells on 10 s scenes) instead of the MobileNet's
    32 (32 cells), and BatchNorm keeps its learnable scale (unlike the
    Keras-ported MobileNet). `causal: true` uses left-only time padding and
    causal max pooling, so output cell t never sees a future frame.
    """

    def __init__(self, cfg: PACEConfig):
        super().__init__()
        self.causal = cfg.causal
        self.future_pads, self.future_reach_frames = cfg.crnn_future_padding()
        blocks = []
        in_ch = 1
        for ch, pool, (future_conv, future_pool) in zip(
            cfg.crnn_channels, cfg.crnn_pools, self.future_pads
        ):
            blocks.append(
                CRNNBlock(
                    in_ch,
                    ch,
                    cfg,
                    future_conv=future_conv,
                    future_pool=future_pool,
                    pool=pool,
                )
            )
            in_ch = ch
        self.blocks = nn.ModuleList(blocks)
        self.out_channels = in_ch

    def regularization_loss(self) -> torch.Tensor:
        return torch.zeros((), device=next(self.parameters()).device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)
        for block in self.blocks:
            x = block(x)
        return x


class GRUContext(nn.Module):
    """Residual BiGRU (unidirectional when causal) over the flattened features.

    A linear projection keeps the recurrent cost independent of the
    frequency x channel width; the residual keeps the head's input stable.
    """

    def __init__(self, dim: int, cfg: PACEConfig):
        super().__init__()
        self.bidirectional = not cfg.causal
        factor = 2 if self.bidirectional else 1
        self.proj = nn.Linear(dim, cfg.context_hidden)
        self.gru = nn.GRU(
            cfg.context_hidden,
            cfg.context_hidden,
            num_layers=cfg.context_layers,
            batch_first=True,
            bidirectional=self.bidirectional,
            dropout=cfg.dropout if cfg.context_layers > 1 else 0.0,
        )
        self.out = nn.Linear(factor * cfg.context_hidden, dim)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        y, _ = self.gru(self.proj(seq))
        return seq + self.drop(self.out(y))


BACKBONES: Dict[str, Callable] = {
    "onepass": PACEBackbone,
    "crnn": CRNNBackbone,
}


class LimitedFutureGRU(nn.Module):
    """Bidirectional GRU whose backward pass sees at most K future cells.

    The forward direction is an ordinary unidirectional GRU over the whole
    history. For every cell t the backward direction runs over cells
    t+K, ..., t only (zero initial state, zero padding past the end of the
    clip), so the recurrent layer adds exactly K cells of look-ahead per layer
    stack instead of the whole remaining clip. With K = 0 the backward half
    sees only the current cell. Parameter count equals the stock
    bidirectional ``GRUContext`` at the same ``context_hidden``.
    """

    def __init__(self, dim: int, cfg: PACEConfig):
        super().__init__()
        self.k = int(cfg.context_future_cells)
        h = cfg.context_hidden
        self.proj = nn.Linear(dim, h)
        self.fwd = nn.ModuleList()
        self.bwd = nn.ModuleList()
        for layer in range(cfg.context_layers):
            d_in = h if layer == 0 else 2 * h
            self.fwd.append(nn.GRU(d_in, h, batch_first=True))
            self.bwd.append(nn.GRU(d_in, h, batch_first=True))
        self.inter_drop = nn.Dropout(cfg.dropout if cfg.context_layers > 1 else 0.0)
        self.out = nn.Linear(2 * h, dim)
        self.drop = nn.Dropout(cfg.dropout)

    def _backward(self, gru: nn.GRU, x: torch.Tensor) -> torch.Tensor:
        b, t, d = x.shape
        xp = F.pad(x, (0, 0, 0, self.k))
        win = xp.unfold(1, self.k + 1, 1)
        win = win.permute(0, 1, 3, 2).reshape(b * t, self.k + 1, d)
        y, _ = gru(win.flip(1))
        return y[:, -1].reshape(b, t, -1)

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        x = self.proj(seq)
        for i, (f, bk) in enumerate(zip(self.fwd, self.bwd)):
            y = torch.cat([f(x)[0], self._backward(bk, x)], dim=-1)
            x = self.inter_drop(y) if i < len(self.fwd) - 1 else y
        return seq + self.drop(self.out(x))


class AttentionContext(nn.Module):
    """Causal self-attention with a bounded look-ahead (Emformer-style).

    By default every output cell attends to past cells only: the convolutional
    stack has already spent the look-ahead budget, so attending to a future
    cell would extend the commit delay by that cell's own future reach.
    ``attn_lookahead_cells`` can add future cells when the CNN budget is zero.
    A learned positional embedding is added; the residual keeps the head stable.
    """

    def __init__(self, dim: int, cfg: PACEConfig):
        super().__init__()
        if cfg.context_hidden % cfg.attn_heads != 0:
            raise ValueError("context_hidden must be divisible by attn_heads")
        self.proj = nn.Linear(dim, cfg.context_hidden)
        self.pos = nn.Parameter(torch.zeros(1, 1024, cfg.context_hidden))
        nn.init.normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=cfg.context_hidden,
            nhead=cfg.attn_heads,
            dim_feedforward=cfg.context_hidden * 2,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=cfg.context_layers)
        self.out = nn.Linear(cfg.context_hidden, dim)
        self.drop = nn.Dropout(cfg.dropout)
        self.lookahead_cells = cfg.attn_lookahead_cells if cfg.causal else 10**6

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        length = seq.shape[1]
        x = self.proj(seq) + self.pos[:, :length]
        idx = torch.arange(length, device=seq.device)
        mask = idx[None, :] > (idx[:, None] + self.lookahead_cells)
        x = self.encoder(x, mask=mask)
        return seq + self.drop(self.out(x))


CONTEXTS: Dict[str, Callable] = {
    "gru": GRUContext,
    "attn": AttentionContext,
}
HEADS: Dict[str, Callable] = {
    "onepass": PACEHead,
    "framewise": FrameWiseHead,
    "red_epn": REDEPNHead,
}


CONTEXTS["gru_lc"] = LimitedFutureGRU


def register_backbone(name: str):
    def deco(cls):
        BACKBONES[name] = cls
        return cls

    return deco


def register_context(name: str):
    def deco(cls):
        CONTEXTS[name] = cls
        return cls

    return deco


def register_head(name: str):
    def deco(cls):
        HEADS[name] = cls
        return cls

    return deco


def _lookup(registry: Dict[str, Callable], kind: str, name: str) -> Callable:
    if name not in registry:
        raise ValueError(f"unknown {kind} {name!r}, expected one of {sorted(registry)}")
    return registry[name]


def to_sequence(feature: torch.Tensor) -> torch.Tensor:
    """(B, C, T, F) -> (B, T, F * C), the Keras reshape order of the original."""
    b, c, t, f = feature.shape
    return feature.permute(0, 2, 3, 1).reshape(b, t, f * c)


class SEDModel(nn.Module):
    """Backbone -> flatten -> optional context -> head."""

    def __init__(self, cfg: PACEConfig):
        super().__init__()
        self.cfg = cfg
        self.backbone = _lookup(BACKBONES, "backbone", cfg.backbone)(cfg)
        was_training = self.backbone.training
        self.backbone.eval()
        try:
            with torch.no_grad():
                probe = self.backbone(torch.zeros(1, cfg.n_frames, cfg.n_mels))
        finally:
            self.backbone.train(was_training)
        _, c, t, f = probe.shape
        if t != cfg.n_cells:
            raise ValueError(f"backbone gives {t} cells, config expects {cfg.n_cells}")
        dim = c * f
        self.context = (
            _lookup(CONTEXTS, "context", cfg.context)(dim, cfg) if cfg.context else None
        )
        self.head = _lookup(HEADS, "head", cfg.head)(dim, cfg)

    def regularization_loss(self) -> torch.Tensor:
        """Sum of the weight penalties its components define (zero if none)."""
        reg = torch.zeros((), device=next(self.parameters()).device)
        for part in (self.backbone, self.context, self.head):
            if part is not None and hasattr(part, "regularization_loss"):
                reg = reg + part.regularization_loss()
        return reg

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """x: (B, n_frames, n_mels) log-mel.

        Returns {"pred": head output, "feature": (B, C, n_cells, F')}.
        """
        feature = self.backbone(x)
        seq = to_sequence(feature)
        if self.context is not None:
            seq = self.context(seq)
        return {"pred": self.head(seq), "feature": feature}


class PACEModel(SEDModel):
    """The original one-pass baseline: `SEDModel` with the legacy backbone and head."""


def build_model(cfg: PACEConfig) -> SEDModel:
    """Model for `cfg`; the default config is the original one-pass detector."""
    return SEDModel(cfg)
