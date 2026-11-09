import torch

from src.modules.augment import filter_augment, freq_mixstyle, mixup


def test_mixup_is_convex_combination():
    g = torch.Generator().manual_seed(0)
    x = torch.randn(4, 10, 8)
    mixed, perm, lam = mixup(x, alpha=0.4, generator=g)
    assert torch.allclose(mixed, lam * x + (1 - lam) * x[perm])


def test_filter_augment_adds_bandwise_constant_gain():
    g = torch.Generator().manual_seed(0)
    x = torch.zeros(2, 5, 40)
    out = filter_augment(x, generator=g)
    assert out.shape == x.shape
    assert torch.allclose(out, out[:, :1, :].expand_as(out))
    assert out.abs().max() <= 6.0 + 1e-6


def test_copy_paste_moves_an_event_and_its_label():
    from src.modules.augment import copy_paste_events

    g = torch.Generator().manual_seed(0)
    x = torch.zeros(2, 20, 8)
    x[0, 2:6, :] = 1.0
    target = torch.zeros(2, 20, 2)
    target[0, 2:6, 1] = 1.0
    out_x, out_t = copy_paste_events(x, target, classes=2, p=1.0, generator=g)
    assert (out_t[1, :, 1] > 0).any()
    assert out_x[1].abs().sum() > 0
    assert out_x.shape == x.shape and out_t.shape == target.shape


def test_copy_paste_rejects_non_framewise_targets():
    import pytest

    from src.modules.augment import copy_paste_events

    with pytest.raises(ValueError):
        copy_paste_events(torch.zeros(1, 4, 2), torch.zeros(1, 4, 10), classes=2)


def test_freq_mixstyle_keeps_shape_and_can_be_skipped():
    g = torch.Generator().manual_seed(0)
    x = torch.randn(4, 20, 8)
    assert freq_mixstyle(x, p=0.0, generator=g) is x
    out = freq_mixstyle(x, p=1.0, generator=g)
    assert out.shape == x.shape and torch.isfinite(out).all()
