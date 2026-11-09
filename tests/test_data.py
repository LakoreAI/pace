import numpy as np
import pytest

from src.data import hold_out_files, load_cache
from src.features import PRESETS, window_audio


def test_window_audio_pads_tail_like_original():
    sr = 100
    windows, starts = window_audio(np.ones(500), sr, win_len=2.56, hop_len=1.96)
    assert len(windows) == 3 and all(len(w) == 256 for w in windows)
    assert starts == pytest.approx([0.0, 1.96, 3.92])
    assert windows[-1][-1] == 0.0


def test_short_audio_gives_one_padded_window():
    windows, starts = window_audio(np.ones(10), 100, 2.56, 1.96)
    assert len(windows) == 1 and starts == [0.0]


def test_presets_match_original_frame_counts():
    assert PRESETS["tut2017"].n_frames == 257
    assert PRESETS["urbansed"].n_frames == 1001


def test_cache_contents(tiny_cache):
    ds = tiny_cache
    x, y = ds[0]
    assert x.shape == (257, 40) and x.dtype.is_floating_point
    assert y.shape == (9, 9)
    assert len(ds.meta["files"]) == 4
    assert set(np.unique(ds.file_idx)) == {0, 1, 2, 3}
    assert ds.num_classes == 3
    assert y[:, 0::3].max() <= 1.0


def test_missing_cache_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_cache(tmp_path / "nope.npz")


def test_hold_out_is_file_level_and_deterministic():
    items = [(f"f{i}.wav", []) for i in range(10)]
    train, val = hold_out_files(items, 0.2, seed=0)
    assert len(val) == 2 and len(train) == 8
    assert not {a for a, _ in train} & {a for a, _ in val}
    assert hold_out_files(items, 0.2, seed=0) == (train, val)


def test_targets_are_encoded_from_window_events(tiny_cache):
    from src.sed.targets import encode_onepass

    ds = tiny_cache
    classes = {c: i for i, c in enumerate(ds.classes)}
    for i in (0, len(ds) - 1):
        events = [tuple(e) for e in ds.meta["win_events"][i]]
        expected = encode_onepass(events, classes, ds.features.win_len, ds.n_cells)
        np.testing.assert_array_equal(ds[i][1].numpy(), expected)


def test_cache_has_no_baked_targets(tiny_cache, tmp_path):
    assert len(tiny_cache.meta["win_events"]) == len(tiny_cache)


def test_legacy_cache_with_y_still_loads(tiny_cache, tmp_path):
    import json

    path = tmp_path / "legacy.npz"
    np.savez(
        path,
        X=tiny_cache.x.numpy(),
        Y=tiny_cache.y.numpy(),
        file_idx=tiny_cache.file_idx,
        win_start=tiny_cache.win_start,
        meta=np.array(json.dumps(tiny_cache.meta)),
    )
    legacy = load_cache(path)
    np.testing.assert_array_equal(legacy.y.numpy(), tiny_cache.y.numpy())
    assert load_cache(path, target_kind="onepass").y.shape == tiny_cache.y.shape
    with pytest.raises(ValueError):
        load_cache(path, n_cells=5)


def test_unknown_target_kind_raises(tiny_cache):
    from src.sed.targets import get_encoder

    with pytest.raises(ValueError):
        get_encoder("nope")


def test_urbansed_listing_skips_hidden_files(tmp_path):
    from src.data import make_synthetic_clips, urbansed_items

    make_synthetic_clips(tmp_path, "train", n_files=2, duration=2.0)
    junk = tmp_path / "audio" / "train" / "._train_0000.wav"
    junk.write_bytes(b"\x00\xa2binary")
    (tmp_path / "annotations" / "train" / "._train_0000.txt").write_bytes(b"\xa2")
    assert len(urbansed_items(tmp_path, "train")) == 2
