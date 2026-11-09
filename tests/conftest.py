import pytest

from src.config import PACEConfig
from src.data import build_cache, load_cache, make_synthetic_clips
from src.features import PRESETS

CLASSES = ("tone_a", "tone_b", "tone_c")


@pytest.fixture(scope="session")
def tiny_cache(tmp_path_factory):
    """A 4-file synthetic cache with TUT 2017 feature settings (257 x 40, 9 cells)."""
    root = tmp_path_factory.mktemp("synthetic")
    items = make_synthetic_clips(
        root, "train", n_files=4, classes=CLASSES, duration=4.0
    )
    params = PRESETS["tut2017"]
    n_cells = PACEConfig(n_frames=params.n_frames, n_mels=params.n_mels).n_cells
    path = build_cache(items, CLASSES, params, n_cells, root / "train.npz")
    return load_cache(path)
