"""Segment-based metrics with sed_eval: error rate and F-score.

Same library and settings as the original one-pass detector evaluation (1 s segments for
TUT 2017 and URBAN-SED), so numbers are comparable to the original one-pass detector.
"""

from typing import Dict, List, Mapping, Sequence

from src.sed.events import Event


def _container(events_by_file: Mapping[str, List[Event]]):
    import dcase_util

    rows = [
        {"filename": name, "onset": on, "offset": off, "event_label": label}
        for name, events in events_by_file.items()
        for on, off, label in events
    ]
    return dcase_util.containers.MetaDataContainer(rows)


def segment_metrics(
    references: Mapping[str, List[Event]],
    estimates: Mapping[str, List[Event]],
    classes: Sequence[str],
    time_resolution: float = 1.0,
) -> Dict[str, object]:
    """Overall segment-based ER / F over all files in `references`, plus per-class F.

    Files present in `references` but missing from `estimates` count as having
    no detections. Returns {"error_rate", "f_measure", "per_class_f"}.
    """
    import sed_eval

    metrics = sed_eval.sound_event.SegmentBasedMetrics(
        event_label_list=list(classes), time_resolution=time_resolution
    )
    for name in references:
        ref = _container({name: references[name]})
        est = _container({name: estimates.get(name, [])})
        ref = ref if len(ref) else _empty(name)
        est = est if len(est) else _empty(name)
        metrics.evaluate(reference_event_list=ref, estimated_event_list=est)

    overall = metrics.results_overall_metrics()
    class_wise = metrics.results_class_wise_metrics()
    return {
        "error_rate": float(overall["error_rate"]["error_rate"]),
        "f_measure": float(overall["f_measure"]["f_measure"]),
        "per_class_f": {
            label: float(class_wise[label]["f_measure"]["f_measure"])
            for label in classes
        },
    }


def _empty(name: str):
    import dcase_util

    return dcase_util.containers.MetaDataContainer([{"filename": name}])
