import pytest
import torch

from src.config import PACEConfig
from src.modules.loss import PACELoss
from src.modules.model import PACEModel

B = 2


@pytest.mark.parametrize(
    "n_frames, n_mels, num_classes, n_cells",
    [
        (257, 40, 6, 9),
        (1001, 40, 10, 32),
        (801, 64, 2, 26),
    ],
)
def test_output_grid_matches_original(n_frames, n_mels, num_classes, n_cells):
    cfg = PACEConfig(num_classes=num_classes, n_frames=n_frames, n_mels=n_mels)
    assert cfg.n_cells == n_cells
    model = PACEModel(cfg).eval()
    with torch.no_grad():
        out = model(torch.randn(B, n_frames, n_mels))
    assert out["pred"].shape == (B, n_cells, 3 * num_classes)
    assert ((out["pred"] >= 0) & (out["pred"] <= 1)).all()


def test_bn_has_no_learnable_scale():
    model = PACEModel(PACEConfig())
    assert not model.backbone.stem_bn.weight.requires_grad
    assert model.backbone.stem_bn.bias.requires_grad


def test_one_batch_one_step_reduces_loss():
    torch.manual_seed(0)
    cfg = PACEConfig(dropout=0.0)
    model = PACEModel(cfg)
    criterion = PACELoss(cfg.num_classes)
    x = torch.randn(4, cfg.n_frames, cfg.n_mels)
    y = torch.zeros(4, cfg.n_cells, 3 * cfg.num_classes)
    y[:, 2:5, 0:3] = torch.tensor([1.0, 0.0, 1.0])
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    def total():
        return criterion(model(x)["pred"], y) + model.regularization_loss()

    before = total()
    opt.zero_grad()
    before.backward()
    opt.step()
    assert torch.isfinite(before)
    with torch.no_grad():
        assert total() < before


def test_loss_masks_regression_where_absent():
    criterion = PACELoss(num_classes=1)
    target = torch.zeros(1, 2, 3)
    target[0, 0] = torch.tensor([1.0, 0.2, 0.8])
    pred = target.clone()
    pred[0, 1, 1:] = 0.9
    assert criterion(pred, target).item() == pytest.approx(0.0)
    pred[0, 0, 1] = 0.7
    assert criterion(pred, target).item() == pytest.approx(0.25)


def test_config_rejects_invalid_values():
    with pytest.raises(ValueError):
        PACEConfig(num_classes=0)
    with pytest.raises(ValueError):
        PACEConfig(dropout=1.0)


def test_onepass_matches_original_definition_and_parameters():
    from src.modules.model import to_sequence

    torch.manual_seed(0)
    model = PACEModel(PACEConfig()).eval()
    assert sum(p.numel() for p in model.parameters()) == 3_920_658
    keys = list(model.state_dict())
    assert keys[0] == "backbone.stem.weight" and keys[-2:] == [
        "head.proj.weight",
        "head.proj.bias",
    ]
    x = torch.randn(2, 257, 40)
    with torch.no_grad():
        out = model(x)
        feature = out["feature"]
        b, c, t, f = feature.shape
        ref = feature.permute(0, 2, 3, 1).reshape(b, t, f * c)
        ref = torch.sigmoid(model.head.proj(ref.transpose(1, 2))).transpose(1, 2)
        assert torch.equal(out["pred"], ref)
        assert torch.equal(
            to_sequence(feature), feature.permute(0, 2, 3, 1).reshape(b, t, f * c)
        )


def test_build_model_defaults_to_onepass_and_checkpoints_without_new_fields_load():
    from dataclasses import asdict

    from src.modules.model import build_model

    default = build_model(PACEConfig())
    assert default.cfg.backbone == "onepass" and default.cfg.head == "onepass"
    torch.manual_seed(0)
    a = PACEModel(PACEConfig())
    torch.manual_seed(0)
    b = default.__class__(PACEConfig())
    assert all(torch.equal(p, q) for p, q in zip(a.parameters(), b.parameters()))
    saved = asdict(PACEConfig())
    for key in ("backbone", "context", "head"):
        saved.pop(key)
    assert PACEConfig(**saved).backbone == "onepass"


def test_registered_context_and_head_are_used():
    import torch.nn as nn

    from src.modules.model import (
        CONTEXTS,
        HEADS,
        build_model,
        register_context,
        register_head,
    )

    @register_context("test_identity")
    class Identity(nn.Module):
        def __init__(self, dim, cfg):
            super().__init__()
            self.fc = nn.Linear(dim, dim)

        def forward(self, seq):
            return seq + self.fc(seq)

    @register_head("test_linear")
    class Linear(nn.Module):
        def __init__(self, dim, cfg):
            super().__init__()
            self.fc = nn.Linear(dim, cfg.num_classes)

        def forward(self, seq):
            return self.fc(seq)

    try:
        cfg = PACEConfig(context="test_identity", head="test_linear", num_classes=3)
        model = build_model(cfg)
        out = model(torch.randn(2, cfg.n_frames, cfg.n_mels))
        assert out["pred"].shape == (2, cfg.n_cells, 3)
        assert torch.equal(
            model.regularization_loss(), model.backbone.regularization_loss()
        )
    finally:
        CONTEXTS.pop("test_identity")
        HEADS.pop("test_linear")


def test_unknown_component_name_raises():
    from src.modules.model import build_model

    with pytest.raises(ValueError):
        build_model(PACEConfig(context="nope"))
