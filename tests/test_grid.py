"""Target-grid changes: a finer output grid must be consistent end to end."""

from src.config import PACEConfig, PACE_LAYER_DEFS
from src.pipelines.config import TrainingConfig


def test_finer_grid_from_layer_defs():
    defs = [list(d) for d in PACE_LAYER_DEFS]
    defs[11] = [3, 1, 1024]
    cfg = PACEConfig(
        num_classes=10, n_frames=1001, n_mels=128, layer_defs=defs, causal=True
    )
    assert cfg.total_stride == 16
    assert cfg.n_cells == 63


def test_training_config_accepts_n_cells():
    cfg = TrainingConfig(n_cells=63)
    assert cfg.n_cells == 63
