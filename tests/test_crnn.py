"""CRNN + BiGRU regime: frame-level grid and recurrent context."""

import pytest
import torch

from src.callbacks.lr_scheduler import build_lr_scheduler
from src.config import PACEConfig
from src.modules.model import build_model
from src.pipelines.config import TrainingConfig


def test_crnn_gru_grid_and_forward():
    cfg = PACEConfig(
        backbone="crnn",
        context="gru",
        head="framewise",
        num_classes=10,
        n_frames=1001,
        n_mels=128,
        output_stride=16,
        crnn_channels=[8, 16, 16, 32],
        context_hidden=16,
        context_layers=1,
        causal=False,
    )
    assert cfg.n_cells == 63
    model = build_model(cfg).eval()
    with torch.no_grad():
        out = model(torch.randn(2, 1001, 128))
    assert out["pred"].shape == (2, 63, 10)
    assert torch.isfinite(out["pred"]).all()


def test_causal_crnn_keeps_grid_and_never_sees_the_future():
    torch.manual_seed(0)
    cfg = PACEConfig(
        backbone="crnn",
        context="gru",
        head="framewise",
        num_classes=2,
        n_frames=1001,
        n_mels=128,
        output_stride=16,
        crnn_channels=[8, 16, 16, 32],
        context_hidden=16,
        context_layers=1,
        causal=True,
    )
    model = build_model(cfg).eval()
    x = torch.randn(1, 1001, 128)
    with torch.no_grad():
        base = model(x)["pred"]
    assert base.shape == (1, 63, 2)
    for t in (0, 5, 20):
        j = 16 * (t + 1)
        perturbed = x.clone()
        perturbed[:, j, :] += 10.0
        with torch.no_grad():
            out = model(perturbed)["pred"]
        assert torch.equal(out[:, t], base[:, t]), (
            f"future frame {j} leaked into cell {t}"
        )


def test_cosine_epochs_scheduler_steps_on_validation():
    opt = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=1.0)
    cb = build_lr_scheduler(opt, {"type": "cosine_epochs", "t_max": 10})
    assert cb is not None and cb.step_kind == "epoch"


def test_optimizer_config_validation():
    assert TrainingConfig(optimizer="adamw").optimizer == "adamw"
    with pytest.raises(ValueError):
        TrainingConfig(optimizer="lamb")


def test_high_resolution_crnn_grid_and_pools():
    cfg = PACEConfig(
        backbone="crnn",
        context="gru",
        head="framewise",
        num_classes=2,
        n_frames=1001,
        n_mels=128,
        output_stride=8,
        crnn_channels=[8, 16, 16, 32],
        crnn_pools=[2, 2, 2, 1],
        context_hidden=16,
        context_layers=1,
        causal=False,
    )
    assert cfg.n_cells == 126
    model = build_model(cfg).eval()
    with torch.no_grad():
        out = model(torch.randn(1, 1001, 128))["pred"]
    assert out.shape == (1, 126, 2)


def test_bounded_crnn_lookahead_reach_and_leak():
    torch.manual_seed(0)
    cfg = PACEConfig(
        backbone="crnn",
        context="gru",
        head="framewise",
        num_classes=2,
        n_frames=1001,
        n_mels=128,
        output_stride=16,
        crnn_channels=[8, 16, 16, 32],
        context_hidden=16,
        context_layers=1,
        causal=True,
        lookahead_frames=25,
    )
    pads, reach = cfg.crnn_future_padding()
    assert reach == 25
    assert any(conv or pool for conv, pool in pads)
    model = build_model(cfg).eval()
    assert model.backbone.future_reach_frames == 25
    x = torch.randn(1, 1001, 128)
    with torch.no_grad():
        base = model(x)["pred"]
    for t in (0, 5, 20):
        j = 16 * t + reach + 1
        if j >= 1001:
            continue
        perturbed = x.clone()
        perturbed[:, j, :] += 10.0
        with torch.no_grad():
            out = model(perturbed)["pred"]
        assert torch.equal(out[:, t], base[:, t]), f"frame {j} leaked into cell {t}"


def test_causal_pool_freq_ceil_matches_noncausal_width_and_stays_causal():
    import torch

    from src.config import PACEConfig
    from src.modules.model import CRNNBackbone

    kw = dict(
        crnn_channels=[8, 8, 8, 8],
        crnn_pools=[2, 2, 2, 2],
        output_stride=16,
        n_mels=40,
        n_frames=320,
    )
    nc = CRNNBackbone(PACEConfig(causal=False, **kw)).eval()
    floor = CRNNBackbone(PACEConfig(causal=True, lookahead_frames=25, **kw)).eval()
    ceil = CRNNBackbone(
        PACEConfig(causal=True, lookahead_frames=25, causal_pool_freq_ceil=True, **kw)
    ).eval()
    x = torch.randn(1, 320, 40)
    assert floor(x).shape[-1] == 2
    assert ceil(x).shape == nc(x).shape
    assert ceil(x).shape[-1] == 3


def test_causal_pool_freq_ceil_does_not_leak_future():
    import torch

    from src.config import PACEConfig
    from src.modules.model import CRNNBackbone

    cfg = PACEConfig(
        causal=True,
        lookahead_frames=25,
        causal_pool_freq_ceil=True,
        crnn_channels=[8, 8, 8, 8],
        crnn_pools=[2, 2, 2, 2],
        output_stride=16,
        n_mels=40,
        n_frames=320,
    )
    net = CRNNBackbone(cfg).eval()
    x = torch.randn(1, 320, 40)
    t = 8
    cut = t * 16 + cfg.future_reach_frames + 1
    y = x.clone()
    y[:, cut:] = torch.randn_like(y[:, cut:]) * 5
    with torch.no_grad():
        a, b = net(x), net(y)
    assert (a[:, :, : t + 1] - b[:, :, : t + 1]).abs().max() < 1e-5


def test_limited_future_gru_param_count_and_future_limit():
    import torch

    from src.config import PACEConfig
    from src.modules.model import GRUContext, LimitedFutureGRU

    dim, cells, k = 24, 30, 4
    base = dict(context_hidden=16, context_layers=2, dropout=0.0)
    lc = LimitedFutureGRU(dim, PACEConfig(context_future_cells=k, **base)).eval()
    stock = GRUContext(dim, PACEConfig(causal=False, **base))
    assert sum(p.numel() for p in lc.parameters()) == sum(
        p.numel() for p in stock.parameters()
    )

    x = torch.randn(2, cells, dim)
    t = 10
    y = x.clone()
    y[:, t + 2 * k + 1 :] = torch.randn_like(y[:, t + 2 * k + 1 :]) * 5
    with torch.no_grad():
        a, b = lc(x), lc(y)
    assert (a[:, : t + 1] - b[:, : t + 1]).abs().max() < 1e-5
    z = x.clone()
    z[:, t + 1] += 3.0
    with torch.no_grad():
        assert (lc(z)[:, t] - a[:, t]).abs().max() > 1e-4


def test_limited_future_gru_in_full_model():
    import torch

    from src.config import PACEConfig
    from src.modules.model import build_model

    cfg = PACEConfig(
        num_classes=10,
        n_frames=1001,
        n_mels=40,
        backbone="crnn",
        context="gru_lc",
        head="framewise",
        causal=True,
        lookahead_frames=25,
        causal_pool_freq_ceil=True,
        crnn_channels=[8, 8, 8, 8],
        crnn_pools=[2, 2, 2, 2],
        output_stride=16,
        context_hidden=16,
        context_future_cells=3,
    )
    out = build_model(cfg).eval()(torch.randn(2, 1001, 40))["pred"]
    assert out.shape[0] == 2 and out.shape[1] == 63
