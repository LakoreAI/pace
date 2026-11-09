"""PACE model architecture, with the original one-pass baseline.

This is the ARCHITECTURE-ONLY config: input shape, layer stack, regularization.
Training-loop hyperparameters (epochs, lr, data paths, callback wiring) live in
`src.pipelines.config.TrainingConfig`. Keep the split so checkpoints can rebuild
the architecture without the training YAML.

Defaults follow the original one-pass detector (Venkatesh, Moffat, Miranda, "You Only Hear
Once", Applied Sciences 2022; github.com/satvik-venkatesh/you-only-hear-once,
MIT) for TUT Sound Events 2017: 2.56 s windows, 257 x 40 log-mel input,
9 output cells.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

PACE_LAYER_DEFS: Tuple[Tuple[int, int, int], ...] = (
    (3, 1, 64),
    (3, 2, 128),
    (3, 1, 128),
    (3, 2, 256),
    (3, 1, 256),
    (3, 2, 512),
    (3, 1, 512),
    (3, 1, 512),
    (3, 1, 512),
    (3, 1, 512),
    (3, 1, 512),
    (3, 2, 1024),
    (3, 1, 1024),
    (3, 1, 512),
    (3, 1, 256),
    (3, 1, 128),
)


@dataclass
class PACEConfig:
    """MobileNet-style backbone plus the original one-pass detector head.

    Layout:

        mel (n_frames, n_mels)
          -> Conv3x3/2 (stem_channels) -> [DWConv3x3 -> PWConv1x1 -> Dropout2d] * len(layer_defs)
          -> flatten (freq x channels) per time step
          -> Conv1d 1x1 -> sigmoid -> (n_cells, 3 * num_classes)

    Each class owns three outputs per cell: presence, onset and offset, the
    latter two relative to the cell start and divided by the cell width.
    """

    backbone: str = "onepass"
    context: Optional[str] = None
    head: str = "onepass"

    num_classes: int = 6
    epn_type: str = "conv"
    epn_hidden: int = 64
    epn_layers: int = 2
    epn_kernel: int = 3

    n_frames: int = 257
    n_mels: int = 40

    crnn_channels: List[int] = field(default_factory=lambda: [64, 128, 128, 256])
    crnn_pool: int = 2
    crnn_pools: Optional[List[int]] = None
    crnn_dropout: float = 0.1
    crnn_conv: str = "conv"
    fdy_basis: int = 3

    context_hidden: int = 128
    context_layers: int = 2
    context_future_cells: int = 6
    attn_heads: int = 4
    attn_lookahead_cells: int = 0

    output_stride: Optional[int] = None

    causal: bool = False
    causal_pool_freq_ceil: bool = False
    lookahead_frames: int = 0
    lookahead_order: str = "deep"
    stem_channels: int = 32
    layer_defs: List[Tuple[int, int, int]] = field(
        default_factory=lambda: [tuple(d) for d in PACE_LAYER_DEFS]
    )
    dropout: float = 0.1
    bn_eps: float = 1e-4
    bn_momentum: float = 0.01

    l2_stem: float = 1e-3
    l2_pointwise: float = 1e-2

    def __post_init__(self) -> None:
        if self.lookahead_frames < 0:
            raise ValueError(
                f"lookahead_frames must be >= 0, got {self.lookahead_frames}"
            )
        if self.lookahead_order not in ("deep", "shallow"):
            raise ValueError(
                f"lookahead_order must be 'deep' or 'shallow', got {self.lookahead_order!r}"
            )
        if self.lookahead_frames > 0 and not self.causal:
            raise ValueError(
                "lookahead_frames > 0 requires causal=True; a non-causal model "
                "already uses unlimited future context"
            )
        if self.num_classes <= 0:
            raise ValueError(f"num_classes must be > 0, got {self.num_classes}")
        if self.n_frames <= 0 or self.n_mels <= 0:
            raise ValueError(
                f"n_frames and n_mels must be > 0, got {self.n_frames}, {self.n_mels}"
            )
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout}")
        if self.epn_type not in ("conv", "gru"):
            raise ValueError(f"epn_type must be 'conv' or 'gru', got {self.epn_type!r}")
        if self.epn_hidden <= 0 or self.epn_layers <= 0 or self.epn_kernel <= 0:
            raise ValueError("epn_hidden, epn_layers and epn_kernel must be positive")
        if self.output_stride is not None and self.output_stride <= 0:
            raise ValueError(f"output_stride must be > 0, got {self.output_stride}")
        if not self.crnn_channels or any(c <= 0 for c in self.crnn_channels):
            raise ValueError("crnn_channels must be a non-empty list of positives")
        if self.crnn_pool < 1:
            raise ValueError(f"crnn_pool must be >= 1, got {self.crnn_pool}")
        if self.crnn_pools is None:
            self.crnn_pools = [self.crnn_pool] * len(self.crnn_channels)
        else:
            self.crnn_pools = [int(p) for p in self.crnn_pools]
            if len(self.crnn_pools) != len(self.crnn_channels):
                raise ValueError(
                    "crnn_pools must have one entry per crnn_channels block"
                )
            if any(p < 1 for p in self.crnn_pools):
                raise ValueError(f"crnn_pools must all be >= 1, got {self.crnn_pools}")
        if not 0.0 <= self.crnn_dropout < 1.0:
            raise ValueError(f"crnn_dropout must be in [0, 1), got {self.crnn_dropout}")
        if self.context_hidden <= 0 or self.context_layers <= 0:
            raise ValueError("context_hidden and context_layers must be positive")
        self.layer_defs = [tuple(int(v) for v in d) for d in self.layer_defs]
        for d in self.layer_defs:
            if len(d) != 3:
                raise ValueError(
                    f"layer_defs entries are (kernel, stride, channels), got {d}"
                )

    @property
    def total_stride(self) -> int:
        if self.output_stride is not None:
            return self.output_stride
        stride = 2
        for _, s, _ in self.layer_defs:
            stride *= s
        return stride

    @property
    def n_cells(self) -> int:
        """Output time steps. 'same' padding with stride 2 gives ceil(n / 2)."""
        if self.output_stride is not None:
            return -(-self.n_frames // self.output_stride)
        n = self.n_frames
        for s in [2] + [s for _, s, _ in self.layer_defs]:
            n = -(-n // s)
        return n

    def future_padding(self) -> Tuple[List[int], List[int], int]:
        """Split `lookahead_frames` across layers: (backbone pads, EPN pads, reach).

        Each convolution pads `kernel - 1` steps in total; a layer with future
        pad `a` splits that as `(kernel - 1 - a)` past and `a` future steps, so
        the output grid and alignment are unchanged. A layer's future reach in
        input frames is `a * dilation * S_in` (`S_in` = cumulative stride before
        the layer). The budget is assigned by exact subset-sum over those
        weights, deepest layers first, so the realised reach is the largest
        value <= `lookahead_frames` the architecture can represent.

        Returns three values: per-backbone-layer pads (stem first, then each
        `layer_defs` entry), per-EPN-layer pads (empty for the GRU EPN), and the
        realised reach in input frames. All zeros when not causal or L == 0.
        """
        n_backbone = 1 + len(self.layer_defs)
        if not self.causal or self.lookahead_frames <= 0:
            return [0] * n_backbone, [0] * self.epn_layers, 0

        items = [("stem", 0, 2, 1)]
        s_in = 2
        for i, (kernel, stride, _) in enumerate(self.layer_defs):
            items.append(("bb", i, kernel - 1, s_in))
            s_in *= stride
        if self.epn_type == "conv":
            for i in range(self.epn_layers):
                items.append(("epn", i, self.epn_kernel - 1, 2**i * self.total_stride))

        best: dict[int, list] = {0: []}
        ordered = items if self.lookahead_order == "shallow" else list(reversed(items))
        for kind, index, max_a, weight in ordered:
            nxt = dict(best)
            for reach, chosen in best.items():
                for a in range(1, max_a + 1):
                    new = reach + a * weight
                    if new <= self.lookahead_frames and new not in nxt:
                        nxt[new] = chosen + [(kind, index, a)]
            best = nxt
        reach = max(best)
        backbone_pads, epn_pads = [0] * n_backbone, [0] * self.epn_layers
        for kind, index, a in best[reach]:
            if kind == "stem":
                backbone_pads[0] = a
            elif kind == "bb":
                backbone_pads[index + 1] = a
            else:
                epn_pads[index] = a
        return backbone_pads, epn_pads, reach

    @property
    def future_reach_frames(self) -> int:
        """Future input frames the architecture can actually see (<= budget)."""
        return self.future_padding()[2]

    def crnn_future_padding(self) -> Tuple[List[Tuple[int, int]], int]:
        """Split `lookahead_frames` across CRNN conv and pool layers.

        Returns `([(future_conv, future_pool), ...], reach)`. A conv can spend
        0..kernel-1 future steps and a pool 0..size-1; each step at cumulative
        stride `S_in` adds `S_in` input frames. The four-block CRNN at stride 16
        can represent at most 45 frames (~0.45 s at 10 ms/frame), so the
        realised reach is reported and may be smaller than the request.
        """
        n = len(self.crnn_channels)
        if not self.causal or self.lookahead_frames <= 0:
            return [(0, 0)] * n, 0
        items = []
        s_in = 1
        for i in range(n):
            pool = self.crnn_pools[i]
            items.append(("conv", i, 2, s_in))
            items.append(("pool", i, pool - 1, s_in))
            s_in *= pool
        best: dict[int, list] = {0: []}
        ordered = items if self.lookahead_order == "shallow" else list(reversed(items))
        for kind, index, max_a, weight in ordered:
            nxt = dict(best)
            for reach, chosen in best.items():
                for a in range(1, max_a + 1):
                    new = reach + a * weight
                    if new <= self.lookahead_frames and new not in nxt:
                        nxt[new] = chosen + [(kind, index, a)]
            best = nxt
        reach = max(best)
        pads = [(0, 0)] * n
        for kind, index, a in best[reach]:
            conv, pool = pads[index]
            pads[index] = (a, pool) if kind == "conv" else (conv, a)
        return pads, reach
