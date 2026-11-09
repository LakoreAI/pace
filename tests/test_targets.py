import numpy as np
import pytest

from src.sed.events import clip_to_window, merge_events
from src.sed.postprocess import decode_onepass, onepass_file_events
from src.sed.targets import encode_onepass

CLASSES = ["a", "b"]
INDEX = {c: i for i, c in enumerate(CLASSES)}
WIN, CELLS = 2.56, 9
W = WIN / CELLS


def test_encode_single_cell_event():
    y = encode_onepass([(0.30, 0.50, "b")], INDEX, WIN, CELLS)
    assert y.shape == (CELLS, 6)
    j = int(0.30 // W)
    np.testing.assert_allclose(
        y[j, 3:6], [1, (0.30 - j * W) / W, (0.50 - j * W) / W], atol=1e-6
    )
    assert y[:, 3].sum() == 1 and y[:, 0].sum() == 0


def test_encode_long_event_fills_every_cell():
    y = encode_onepass([(0.0, WIN, "a")], INDEX, WIN, CELLS)
    assert y[:, 0].sum() == CELLS
    np.testing.assert_allclose(y[:, 1], 0.0)
    np.testing.assert_allclose(y[:, 2], 1.0, atol=1e-6)


@pytest.mark.parametrize(
    "event", [(0.30, 0.50, "a"), (0.10, 1.90, "b"), (0.0, WIN, "a")]
)
def test_cell_decode_round_trips(event):
    y = encode_onepass([event], INDEX, WIN, CELLS)
    events = merge_events(
        decode_onepass(y, CLASSES, WIN, end_mode="cell"), max_gap=1e-6
    )
    assert len(events) == 1
    assert events[0][2] == event[2]
    assert events[0][0] == pytest.approx(event[0], abs=1e-3)
    assert events[0][1] == pytest.approx(event[1], abs=1e-3)


def test_notebook_decode_overshoots_first_cell_offset():
    y = encode_onepass([(0.30, 0.50, "a")], INDEX, WIN, CELLS)
    ((on, off, _),) = decode_onepass(y, CLASSES, WIN, end_mode="notebook")
    assert on == pytest.approx(0.30, abs=1e-6)
    assert off == pytest.approx(0.50 + (0.30 - W), abs=1e-6)


def test_window_offset_and_merge_across_windows():
    y = encode_onepass([(0.0, WIN, "a")], INDEX, WIN, CELLS)
    events = onepass_file_events(
        [y, y], [0.0, 1.96], CLASSES, WIN, merge_gap=1.0, end_mode="cell"
    )
    assert events == [(0.0, round(1.96 + WIN, 3), "a")]


def test_merge_gap():
    evs = [(0.0, 1.0, "a"), (1.5, 2.0, "a"), (5.0, 6.0, "a"), (1.2, 1.4, "b")]
    assert merge_events(evs, max_gap=1.0) == [
        (0.0, 2.0, "a"),
        (1.2, 1.4, "b"),
        (5.0, 6.0, "a"),
    ]
    assert len(merge_events(evs, max_gap=0.0)) == 4


def test_clip_to_window():
    evs = [(1.0, 3.0, "a"), (2.5, 2.7, "a"), (10.0, 11.0, "b"), (4.0, 5.0, "b")]
    assert clip_to_window(evs, 2.0, WIN) == [(0.0, 1.0, "a"), (2.0, 2.56, "b")]


def test_clip_drops_event_starting_at_window_end():
    assert clip_to_window([(WIN, 3.0, "a")], 0.0, WIN) == []
