"""Threshold-independent metrics (PSDS) and event-based F-score.

PSDS is computed with `sed_scores_eval` from per-file score timelines, so it
does not depend on the event decoder or its threshold. The default parameter
sets are the DCASE conventions:

- PSDS1: dtc=0.7, gtc=0.7, no cross-trigger penalty.
- PSDS2: dtc=0.1, gtc=0.1, cttc=0.1, alpha_ct=0.5.

They are configuration values, not tuned here. `sed_scores_eval` must be
installed (`uv sync` installs it); importing this module is lazy.
"""

from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from src.sed.events import Event, merge_events

PSDS1 = {"dtc_threshold": 0.7, "gtc_threshold": 0.7, "alpha_ct": 0.0, "alpha_st": 0.0}
PSDS2 = {
    "dtc_threshold": 0.1,
    "gtc_threshold": 0.1,
    "cttc_threshold": 0.1,
    "alpha_ct": 0.5,
    "alpha_st": 0.0,
}

Scores = Tuple[np.ndarray, np.ndarray]


def file_scores_from_windows(
    pred: np.ndarray,
    win_start: float,
    win_len: float,
    target_kind: str = "onepass",
    n_bins: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """One window's output -> global score bins for a file-level timeline.

    Windows overlap, so a bin can be covered more than once; the maximum
    presence score wins. `timestamps` has `n_bins + 1` edges.
    """
    if target_kind == "onepass":
        pred = pred[:, 0::3]
    n_cells = pred.shape[0]
    width = win_len / n_cells
    first = int(round(win_start / width))
    n_bins = n_bins or first + n_cells
    scores = np.zeros((n_bins, pred.shape[1]), dtype=np.float64)
    scores[first : first + n_cells] = pred
    return np.arange(n_bins + 1) * width, scores


def merge_window_scores(
    preds: np.ndarray,
    win_starts: Sequence[float],
    win_len: float,
    target_kind: str = "onepass",
) -> Tuple[np.ndarray, np.ndarray]:
    """All windows of one file -> a single non-overlapping score timeline."""
    if isinstance(preds, dict):
        presence = np.asarray(preds["pres"])
    elif target_kind == "onepass":
        presence = np.asarray(preds)[..., 0::3]
    else:
        presence = np.asarray(preds)
    n_cells = presence.shape[1]
    width = win_len / n_cells
    last = max(int(round(s / width)) + n_cells for s in win_starts)
    scores = np.zeros((last, presence.shape[2]), dtype=np.float64)
    for pred, start in zip(presence, win_starts):
        first = int(round(start / width))
        block = scores[first : first + n_cells]
        np.maximum(block, pred, out=block)
    return np.arange(last + 1) * width, scores


def psds_metrics(
    references: Mapping[str, Sequence[Event]],
    scores: Mapping[str, Scores],
    durations: Mapping[str, float],
    classes: Sequence[str],
    params: Optional[Mapping[str, dict]] = None,
) -> Dict[str, float]:
    """PSDS1/PSDS2 from per-file score timelines.

    `scores[file] = (timestamps, (n_bins, C) array)`. Every class must occur in
    the references (a `sed_scores_eval` requirement); otherwise that class is
    dropped from evaluation and the omission is reported via
    `classes_evaluated`.
    """
    from sed_scores_eval import io
    from sed_scores_eval.intersection_based import psds

    params = params or {"psds1": PSDS1, "psds2": PSDS2}
    present = {label for events in references.values() for _, _, label in events}
    kept = [c for c in classes if c in present]
    dropped = [c for c in classes if c not in present]
    if not kept:
        return {"psds1": float("nan"), "psds2": float("nan"), "classes_evaluated": []}

    score_dfs = {}
    for name in references:
        if name not in scores:
            continue
        timestamps, values = scores[name]
        duration = durations.get(name)
        if duration:
            keep = timestamps[1:] <= float(duration) + 1e-6
            if keep.any():
                n = int(keep.sum())
                timestamps, values = timestamps[: n + 1], values[:n]
        score_dfs[name] = io.create_score_dataframe(
            values[:, [list(classes).index(c) for c in kept]], timestamps, kept
        )
    ground_truth = {
        name: merge_events(events, max_gap=0.0) for name, events in references.items()
    }
    result: Dict[str, float] = {"classes_evaluated": kept, "classes_dropped": dropped}
    for key, kw in params.items():
        value = psds(score_dfs, ground_truth, dict(durations), **kw)[0]
        result[key] = float(value)
    return result


def event_based_metrics(
    references: Mapping[str, Sequence[Event]],
    estimates: Mapping[str, Sequence[Event]],
    classes: Sequence[str],
    collar: float = 0.2,
) -> Dict[str, float]:
    """Collar-based event F-score (sed_eval), independent of segment length."""
    import dcase_util
    import sed_eval

    env = dcase_util.containers.MetaDataContainer

    def container(mapping):
        return env(
            [
                {"filename": name, "onset": on, "offset": off, "event_label": label}
                for name, events in mapping.items()
                for on, off, label in events
            ]
        )

    metrics = sed_eval.sound_event.EventBasedMetrics(
        event_label_list=list(classes),
        t_collar=collar,
        evaluate_onset=True,
        evaluate_offset=True,
    )
    for name in references:
        ref = container({name: references[name]})
        est = container({name: estimates.get(name, [])})
        ref = ref if len(ref) else env([{"filename": name}])
        est = est if len(est) else env([{"filename": name}])
        metrics.evaluate(reference_event_list=ref, estimated_event_list=est)
    overall = metrics.results_overall_metrics()
    return {
        "event_f_measure": float(overall["f_measure"]["f_measure"]),
        "event_error_rate": float(overall["error_rate"]["error_rate"]),
    }
