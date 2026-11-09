"""Decoding model outputs into events, and post-processing.

Kept separate from the forward pass so its cost can be timed on its own: the
latency analysis depends on what post-processing costs.

`decode_onepass` follows the released one-pass decoder from the original notebooks. In
`end_mode="notebook"` the offset output is added to the decoded onset
(`end = start + p_off * width`), which is what the released code does even though
the encoder stores the offset relative to the cell start. `end_mode="cell"` is the
label-consistent decode (`end = cell_start + p_off * width`).

Also here: per-class 1D NMS / Soft-NMS for event-level heads, and the
frame-wise baseline decoder (median filter + threshold).
"""

import inspect
import math
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from src.sed.events import Event, merge_events

END_MODES = ("notebook", "cell")


def decode_onepass(
    pred: np.ndarray,
    classes: Sequence[str],
    win_len: float,
    win_start: float = 0.0,
    threshold: float = 0.5,
    end_mode: str = "notebook",
) -> List[Event]:
    """One window's (n_cells, 3 * num_classes) output -> events in file time, unmerged."""
    if end_mode not in END_MODES:
        raise ValueError(f"end_mode must be one of {END_MODES}, got {end_mode!r}")
    n_cells = pred.shape[0]
    width = win_len / n_cells
    idx = np.argwhere(pred[:, 0::3] >= threshold)
    if idx.size == 0:
        return []
    idx = idx[np.lexsort((idx[:, 1], idx[:, 0]))]
    cells, class_ids = idx[:, 0], idx[:, 1]
    onsets = width * cells + width * pred[cells, 3 * class_ids + 1] + win_start
    ends = width * cells + width * pred[cells, 3 * class_ids + 2] + win_start
    if end_mode == "notebook":
        ends = onsets + width * pred[cells, 3 * class_ids + 2]
    return [
        (float(on), float(off), classes[int(k)])
        for on, off, k in zip(onsets, ends, class_ids)
    ]


def onepass_file_events(
    preds: np.ndarray,
    win_starts: Sequence[float],
    classes: Sequence[str],
    win_len: float,
    threshold: float = 0.5,
    merge_gap: float = 1.0,
    end_mode: str = "notebook",
) -> List[Event]:
    """All windows of one file (in window order) -> merged events.

    `merge_gap` defaults to 1.0 s, the value the original one-pass detector uses for
    environmental sounds; set it to 0 to remove merging.
    """
    events: List[Event] = []
    for pred, start in zip(np.asarray(preds), win_starts):
        events += decode_onepass(pred, classes, win_len, start, threshold, end_mode)
    return merge_events(events, max_gap=merge_gap)


def framewise_file_events(
    preds: np.ndarray,
    win_starts: Sequence[float],
    classes: Sequence[str],
    win_len: float,
    threshold: Union[float, Sequence[float]] = 0.5,
    median_lengths: Optional[Sequence[int]] = None,
    merge_gap: float = 0.0,
) -> List[Event]:
    """Per-cell presence windows of one file -> merged events.

    The frame hop is the model's own output resolution (`win_len / n_cells`),
    so this decoder shares the temporal grid of the event head it is compared
    against. `merge_gap=0` keeps adjacent-window duplicates from being merged
    into one long event; raise it only when the protocol requires it.
    """
    preds = np.asarray(preds)
    if len(preds) == 0:
        return []
    frame_hop = win_len / preds.shape[1]
    events: List[Event] = []
    for pred, start in zip(preds, win_starts):
        events += framewise_events(
            pred,
            classes,
            frame_hop,
            threshold=threshold,
            median_lengths=median_lengths,
            offset=float(start),
        )
    return merge_events(events, max_gap=merge_gap)


def red_epn_file_events(
    preds: dict,
    win_starts: Sequence[float],
    classes: Sequence[str],
    win_len: float,
    threshold: float = 0.5,
    iou_threshold: float = 0.5,
    k: int = 15,
    m: Optional[int] = None,
    merge_gap: float = 0.0,
) -> List[Event]:
    """RED+EPN inference: NMS over duration-aware proposals (arXiv:2601.04178).

    `preds` holds `pres`, `don`, `doff` with shape (windows, cells, C). Each
    active cell proposes the interval `[cell - don, cell + doff]`; proposals are
    sorted by presence and kept greedily up to `k` per class and `m` classes.
    """
    if not isinstance(preds, dict):
        raise TypeError("red_epn decoder needs the head's dict of maps")
    presence = np.asarray(preds["pres"])
    don = np.asarray(preds["don"])
    doff = np.asarray(preds["doff"])
    n_windows, n_cells, c_count = presence.shape
    width = win_len / n_cells
    selected = list(range(c_count))
    if m is not None and m < c_count:
        mean_presence = presence.mean(axis=(0, 1))
        selected = list(np.argsort(mean_presence)[-m:])

    by_class: Dict[int, List[Tuple[float, float, float]]] = {}
    for wi, start in enumerate(win_starts):
        for t in range(n_cells):
            for c in selected:
                score = float(presence[wi, t, c])
                if score < threshold:
                    continue
                on = max(0.0, start + t * width - float(don[wi, t, c]) * width)
                off = start + t * width + float(doff[wi, t, c]) * width
                if off > on:
                    by_class.setdefault(c, []).append((on, off, score))

    kept: List[Event] = []
    for c, proposals in by_class.items():
        proposals.sort(key=lambda p: p[2], reverse=True)
        chosen: List[Tuple[float, float, float]] = []
        for proposal in proposals:
            if len(chosen) >= k:
                break
            if all(_iou_1d(proposal[:2], q[:2]) < iou_threshold for q in chosen):
                chosen.append(proposal)
        kept.extend((round(on, 3), round(off, 3), classes[c]) for on, off, _ in chosen)
    if merge_gap > 0:
        kept = merge_events(kept, max_gap=merge_gap)
    return sorted(kept, key=lambda e: (e[0], e[2]))


def red_presence_file_events(
    preds: dict,
    win_starts: Sequence[float],
    classes: Sequence[str],
    win_len: float,
    threshold: Union[float, Sequence[float]] = 0.5,
    median_lengths: Optional[Sequence[int]] = 3,
    merge_gap: float = 0.0,
) -> List[Event]:
    """Decode RED's presence map with threshold + median filtering.

    This is the no-EPN ablation: it keeps RED/onset-offset training but drops
    the duration proposals. It answers whether the event-proposal decoder is
    the part that hurts, separately from the RED presence representation.
    """
    if not isinstance(preds, dict):
        raise TypeError("red_presence decoder needs the head's dict of maps")
    presence = np.asarray(preds["pres"])
    frame_hop = win_len / presence.shape[1]
    events: List[Event] = []
    for frame_scores, start in zip(presence, win_starts):
        events += framewise_events(
            frame_scores,
            classes,
            frame_hop,
            threshold=threshold,
            median_lengths=median_lengths,
            offset=float(start),
        )
    return merge_events(events, max_gap=merge_gap)


DECODERS: Dict[str, Callable[..., List[Event]]] = {
    "onepass": onepass_file_events,
    "framewise": framewise_file_events,
    "red_epn": red_epn_file_events,
    "red_presence": red_presence_file_events,
}


def register_decoder(name: str):
    def deco(fn):
        DECODERS[name] = fn
        return fn

    return deco


def decode_file_events(
    decoder: Union[str, dict, None],
    preds: np.ndarray,
    win_starts: Sequence[float],
    classes: Sequence[str],
    win_len: float,
    **postprocess,
) -> List[Event]:
    """Dispatch to a registered decoder, dropping keys it does not accept.

    `decoder` is a name or `{"type": name, ...}`; extra decoder parameters are
    merged over the shared `postprocess` dict.
    """
    if decoder is None:
        name, extra = "onepass", {}
    elif isinstance(decoder, str):
        name, extra = decoder, {}
    else:
        name = decoder.get("type", "onepass")
        extra = {k: v for k, v in decoder.items() if k != "type"}
    if name not in DECODERS:
        raise ValueError(
            f"unknown decoder {name!r}, expected one of {sorted(DECODERS)}"
        )
    fn = DECODERS[name]
    accepted = inspect.signature(fn).parameters
    kwargs = {k: v for k, v in {**postprocess, **extra}.items() if k in accepted}
    return fn(preds, win_starts, classes, win_len, **kwargs)


ScoredEvent = Tuple[float, float, str, float]


def _iou_1d(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = (a[1] - a[0]) + (b[1] - b[0]) - inter
    return inter / union if union > 0 else 0.0


def nms_1d(
    events: Sequence[ScoredEvent],
    iou_threshold: float = 0.5,
    soft: bool = False,
    sigma: float = 0.5,
    score_threshold: float = 1e-3,
) -> List[ScoredEvent]:
    """Per-class 1D NMS; with `soft=True`, Gaussian Soft-NMS (Bodla et al., 2017).

    Hard NMS drops lower-scored events overlapping a kept one by more than
    `iou_threshold`. Soft-NMS decays their score by exp(-IoU^2 / sigma) instead and
    drops them only below `score_threshold`. Returns kept events sorted by onset.
    """
    by_class: Dict[str, List[ScoredEvent]] = {}
    for e in events:
        by_class.setdefault(e[2], []).append(e)

    kept: List[ScoredEvent] = []
    for evs in by_class.values():
        pool = sorted(evs, key=lambda e: e[3], reverse=True)
        while pool:
            best = pool.pop(0)
            kept.append(best)
            rest = []
            for e in pool:
                iou = _iou_1d(best[:2], e[:2])
                if soft:
                    score = e[3] * math.exp(-(iou**2) / sigma)
                    if score >= score_threshold:
                        rest.append((e[0], e[1], e[2], score))
                elif iou <= iou_threshold:
                    rest.append(e)
            pool = sorted(rest, key=lambda e: e[3], reverse=True)
    return sorted(kept, key=lambda e: (e[0], e[2]))


def median_filter_scores(scores: np.ndarray, lengths) -> np.ndarray:
    """Class-wise median filter over time. scores: (n_frames, C); lengths in frames.

    `lengths` is either one length shared by all classes or one per class.
    """
    from scipy.ndimage import median_filter

    if isinstance(lengths, (int, np.integer)):
        lengths = [int(lengths)] * scores.shape[1]
    out = np.empty_like(scores)
    for c, n in enumerate(lengths):
        out[:, c] = median_filter(scores[:, c], size=max(int(n), 1), mode="nearest")
    return out


def framewise_events(
    scores: np.ndarray,
    classes: Sequence[str],
    frame_hop: float,
    threshold: Union[float, Sequence[float]] = 0.5,
    median_lengths: Optional[Sequence[int]] = None,
    offset: float = 0.0,
) -> List[Event]:
    """Frame-wise baseline decoding: (optional) median filter, threshold, group runs.

    scores: (n_frames, C) in [0, 1]; `frame_hop` in seconds. Thresholds may be
    per class. Tune `threshold` / `median_lengths` on validation only.
    """
    if median_lengths is not None:
        scores = median_filter_scores(scores, median_lengths)
    thr = np.broadcast_to(np.asarray(threshold, dtype=float), (scores.shape[1],))
    active = scores >= thr
    events: List[Event] = []
    for c, label in enumerate(classes):
        padded = np.concatenate([[False], active[:, c], [False]])
        edges = np.flatnonzero(np.diff(padded.astype(np.int8)))
        for start, stop in zip(edges[::2], edges[1::2]):
            events.append(
                (offset + start * frame_hop, offset + stop * frame_hop, label)
            )
    return sorted(events, key=lambda e: (e[0], e[2]))
