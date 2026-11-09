"""Bounded-look-ahead streaming inference and latency metrics.

The offline decoders consume a whole file. This module simulates deployment:
windows arrive in time order, and an event may be committed only once the
system has seen `lookahead` seconds of audio *after* the predicted offset. The
latency therefore includes the look-ahead plus window/hop quantisation.

PSDS remains a retrospective score-quality metric; the latency numbers are
reported alongside it, and the streamed event list is what an application can
actually emit. This is the protocol used by on-line SELD work, adapted to
event-level SED.
"""

from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence

import numpy as np

from src.sed.events import Event
from src.sed.metrics import segment_metrics
from src.sed.postprocess import decode_file_events


@dataclass
class StreamResult:
    events: List[Event]
    commit_times: List[float]
    forced_at_end: int


def _iou(a, b) -> float:
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = (a[1] - a[0]) + (b[1] - b[0]) - inter
    return inter / union if union > 0 else 0.0


def stream_file_events(
    preds,
    win_starts: Sequence[float],
    classes: Sequence[str],
    win_len: float,
    decoder: str | dict,
    postprocess: Optional[Mapping] = None,
    lookahead: float = 0.0,
    duplicate_iou: float = 0.5,
) -> StreamResult:
    """Decode one file window-by-window and commit events under `lookahead`.

    `lookahead` is the future audio required after an event offset before the
    event is emitted. A larger value improves boundary context at the cost of
    latency. Events still pending at the end of the file are force-committed and
    counted in `forced_at_end`.
    """
    starts = np.asarray(win_starts, dtype=float)
    order = np.argsort(starts, kind="stable")
    pending: List[Event] = []
    committed: List[Event] = []
    commit_times: List[float] = []
    forced = 0
    final_end = float(starts[order[-1]] + win_len) if len(order) else 0.0

    def commit(event: Event, when: float) -> None:
        if any(
            event[2] == prev[2] and _iou(event, prev) > duplicate_iou
            for prev in committed
        ):
            return
        committed.append(event)
        commit_times.append(when)

    if len(order) == 0:
        return StreamResult([], [], 0)
    for i in order:
        window_end = float(starts[i] + win_len)
        window_preds = (
            {k: v[i : i + 1] for k, v in preds.items()}
            if isinstance(preds, dict)
            else preds[i : i + 1]
        )
        new = decode_file_events(
            decoder,
            window_preds,
            [float(starts[i])],
            classes,
            win_len,
            **(postprocess or {}),
        )
        pending.extend(new)
        ready = [e for e in pending if e[1] <= window_end - lookahead + 1e-9]
        pending = [e for e in pending if e[1] > window_end - lookahead + 1e-9]
        for event in ready:
            commit(event, window_end)
    for event in pending:
        commit(event, final_end)
        forced += 1
    order_out = sorted(
        range(len(committed)), key=lambda j: (committed[j][0], committed[j][2])
    )
    return StreamResult(
        events=[committed[j] for j in order_out],
        commit_times=[commit_times[j] for j in order_out],
        forced_at_end=forced,
    )


def latency_stats(
    events: Sequence[Event], commit_times: Sequence[float]
) -> Dict[str, float]:
    if not events:
        return {
            "n_events": 0,
            "mean_delay_after_offset": float("nan"),
            "median_delay_after_offset": float("nan"),
            "mean_delay_after_onset": float("nan"),
            "max_delay_after_offset": float("nan"),
        }
    after_offset = np.asarray([ct - e[1] for e, ct in zip(events, commit_times)])
    after_onset = np.asarray([ct - e[0] for e, ct in zip(events, commit_times)])
    return {
        "n_events": len(events),
        "mean_delay_after_offset": float(after_offset.mean()),
        "median_delay_after_offset": float(np.median(after_offset)),
        "mean_delay_after_onset": float(after_onset.mean()),
        "max_delay_after_offset": float(after_offset.max()),
    }


def evaluate_streaming(
    model,
    dataset,
    device,
    lookaheads: Sequence[float],
    decoder: str | dict,
    postprocess: Optional[Mapping] = None,
    batch_size: int = 64,
    time_resolution: float = 1.0,
) -> Dict[float, dict]:
    """Run one model pass and evaluate every look-ahead in `lookaheads`."""
    from src.pipelines.eval import group_windows, predict_dataset, slice_predictions

    preds, _, _ = predict_dataset(model, dataset, device, batch_size=batch_size)
    windows = group_windows(dataset)
    references = dataset.references()
    win_len = dataset.features.win_len
    results: Dict[float, dict] = {}
    for lookahead in lookaheads:
        estimates = {}
        all_events: List[Event] = []
        all_times: List[float] = []
        forced = 0
        for fi, idx in windows.items():
            stream = stream_file_events(
                slice_predictions(preds, idx),
                dataset.win_start[idx],
                dataset.classes,
                win_len,
                decoder,
                postprocess,
                lookahead=lookahead,
            )
            estimates[dataset.meta["files"][fi]] = stream.events
            all_events.extend(stream.events)
            all_times.extend(stream.commit_times)
            forced += stream.forced_at_end
        metrics = segment_metrics(
            references, estimates, dataset.classes, time_resolution
        )
        metrics["latency"] = latency_stats(all_events, all_times)
        metrics["forced_at_end"] = forced
        metrics["lookahead"] = float(lookahead)
        results[float(lookahead)] = metrics
    return results


def format_streaming(results: Mapping[float, dict]) -> str:
    lines = [
        f"{'lookahead':>9s}  {'ER':>7s}  {'F':>6s}  {'lat_off_ms':>10s}  {'forced':>6s}"
    ]
    for lookahead, r in sorted(results.items()):
        lat = r["latency"]
        f = r["f_measure"]
        lines.append(
            f"{lookahead:9.2f}  {r['error_rate']:7.4f}  "
            f"{(f * 100 if f == f else float('nan')):6.2f}  "
            f"{lat['mean_delay_after_offset'] * 1000:10.1f}  {r['forced_at_end']:6d}"
        )
    return "\n".join(lines)
