"""Target encoders, selected by name through `ENCODERS`. Currently: the original one-pass grid.

`encode_onepass` reproduces `get_universal_labels` from the original notebooks:
every cell an event touches gets `[1, onset, offset]` for that class, with onset
and offset relative to the cell start and divided by the cell width. A long
event therefore appears in many cells; decoding merges them back.
"""

from typing import Callable, Dict, Iterable

import numpy as np

from src.sed.events import Event


def encode_onepass(
    events: Iterable[Event],
    class_index: Dict[str, int],
    win_len: float,
    n_cells: int,
) -> np.ndarray:
    """Window-relative events -> (n_cells, 3 * num_classes) float32 target."""
    width = win_len / n_cells
    labels = np.zeros((n_cells, 3 * len(class_index)), dtype=np.float32)

    for onset, offset, label in events:
        c = class_index[label] * 3
        start_bin = min(int(onset // width), n_cells - 1)
        stop_bin = int(offset // width)
        start_rel = onset - start_bin * width
        stop_rel = offset - stop_bin * width
        if stop_bin >= n_cells:
            stop_bin, stop_rel = n_cells - 1, width

        n_bins = stop_bin - start_bin
        if n_bins == 0:
            labels[start_bin, c : c + 3] = [1.0, start_rel, stop_rel]
        else:
            labels[start_bin, c : c + 3] = [1.0, start_rel, width]
            for i in range(1, n_bins):
                labels[start_bin + i, c : c + 3] = [1.0, 0.0, width]
            if stop_rel > 0.0:
                labels[stop_bin, c : c + 3] = [1.0, 0.0, stop_rel]

    labels[:, 1::3] /= width
    labels[:, 2::3] /= width
    return labels


def encode_framewise(
    events: Iterable[Event],
    class_index: Dict[str, int],
    win_len: float,
    n_cells: int,
) -> np.ndarray:
    """Window-relative events -> (n_cells, num_classes) binary presence target.

    A cell is positive for a class if the event overlaps it at all. This is the
    matched frame-wise baseline's target on the same output grid.
    """
    width = win_len / n_cells
    labels = np.zeros((n_cells, len(class_index)), dtype=np.float32)
    for onset, offset, label in events:
        c = class_index[label]
        start_bin = int(np.floor(max(onset, 0.0) / width))
        stop_bin = int(np.ceil(min(offset, win_len) / width)) - 1
        start_bin = min(max(start_bin, 0), n_cells - 1)
        stop_bin = min(max(stop_bin, start_bin), n_cells - 1)
        labels[start_bin : stop_bin + 1, c] = 1.0
    return labels


def encode_red_epn(
    events: Iterable[Event],
    class_index: Dict[str, int],
    win_len: float,
    n_cells: int,
) -> np.ndarray:
    """RED+EPN targets: (n_cells, 5 * C) = presence, onset, offset, don, doff.

    `don`/`doff` are the distances from the cell start to the event onset and
    to the event offset, in cell widths (the intervals used by the 1D IoU loss).
    """
    width = win_len / n_cells
    c_count = len(class_index)
    labels = np.zeros((n_cells, 5 * c_count), dtype=np.float32)
    for onset, offset, label in events:
        if offset <= onset:
            continue
        c = class_index[label]
        start_bin = min(max(int(onset // width), 0), n_cells - 1)
        stop_bin = min(max(int(np.ceil(offset / width)) - 1, start_bin), n_cells - 1)
        labels[start_bin : stop_bin + 1, c] = 1.0
        labels[start_bin, c_count + c] = 1.0
        labels[stop_bin, 2 * c_count + c] = 1.0
        for t in range(start_bin, stop_bin + 1):
            labels[t, 3 * c_count + c] = max(0.0, t * width - onset) / width
            labels[t, 4 * c_count + c] = max(0.0, offset - t * width) / width
    return labels


ENCODERS: Dict[str, Callable[..., np.ndarray]] = {
    "onepass": encode_onepass,
    "framewise": encode_framewise,
    "red_epn": encode_red_epn,
}


def get_encoder(kind: str) -> Callable[..., np.ndarray]:
    if kind not in ENCODERS:
        raise ValueError(
            f"unknown target kind {kind!r}, expected one of {sorted(ENCODERS)}"
        )
    return ENCODERS[kind]
