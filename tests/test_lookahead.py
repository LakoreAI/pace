"""Bounded look-ahead: the model may use at most L future input frames.

These tests pin the property the streaming protocol depends on: output cell
`t` is a function of input frames `<= t * total_stride + reach`, and of nothing
later. Without it, "look-ahead L" would only be a commit delay, not a context
budget.
"""

import pytest
import torch

from src.config import PACEConfig
from src.modules.model import build_model

S = PACEConfig().total_stride


def test_zero_lookahead_is_strictly_causal():
    cfg = PACEConfig(causal=True, lookahead_frames=0)
    pads, epn_pads, reach = cfg.future_padding()
    assert reach == 0
    assert not any(pads) and not any(epn_pads)


@pytest.mark.parametrize("budget", [25, 50, 100])
def test_requested_lookahead_is_realised_exactly(budget):
    cfg = PACEConfig(causal=True, lookahead_frames=budget, head="red_epn")
    assert cfg.future_reach_frames == budget
    pads, epn_pads, reach = cfg.future_padding()
    for kernel, pad in zip([3] + [d[0] for d in cfg.layer_defs], pads):
        assert 0 <= pad <= kernel - 1
    assert sum(pads) + sum(epn_pads) > 0


def test_lookahead_reach_never_exceeds_budget():
    for budget in range(0, 120, 7):
        cfg = PACEConfig(causal=True, lookahead_frames=budget)
        assert 0 <= cfg.future_reach_frames <= budget


def test_output_grid_is_unchanged_by_lookahead():
    for head in ("onepass", "framewise", "red_epn"):
        cfg = PACEConfig(
            num_classes=2, causal=True, lookahead_frames=50, head=head, epn_hidden=8
        )
        model = build_model(cfg)
        out = model(torch.randn(1, cfg.n_frames, cfg.n_mels))["pred"]
        t = out["pres"].shape[1] if isinstance(out, dict) else out.shape[1]
        assert t == cfg.n_cells


def test_frames_beyond_the_budget_do_not_change_the_output():
    torch.manual_seed(0)
    budget = 25
    cfg = PACEConfig(head="framewise", causal=True, lookahead_frames=budget)
    model = build_model(cfg).eval()
    x = torch.randn(1, cfg.n_frames, cfg.n_mels)
    with torch.no_grad():
        base = model(x)["pred"]

    reach = cfg.future_reach_frames
    checked = 0
    with torch.no_grad():
        for t in range(cfg.n_cells):
            j = t * S + reach + 1
            if j >= cfg.n_frames:
                continue
            perturbed = x.clone()
            perturbed[:, j, :] += 10.0
            out = model(perturbed)["pred"]
            assert torch.equal(out[:, t], base[:, t]), (
                f"future frame {j} leaked into cell {t}"
            )
            checked += 1
    assert checked > 0


def test_gru_epn_builds_with_backbone_only_budget():
    cfg = PACEConfig(
        num_classes=2,
        causal=True,
        head="red_epn",
        epn_type="gru",
        epn_hidden=8,
        lookahead_frames=50,
    )
    model = build_model(cfg)
    out = model(torch.randn(1, cfg.n_frames, cfg.n_mels))["pred"]
    assert out["pres"].shape == (1, cfg.n_cells, 2)
    assert 0 < cfg.future_reach_frames <= 50


def test_non_causal_model_rejects_a_lookahead_budget():
    with pytest.raises(ValueError):
        PACEConfig(causal=False, lookahead_frames=25)
    with pytest.raises(ValueError):
        PACEConfig(causal=True, lookahead_frames=-1)
