"""Event lists: reading annotations, clipping to windows, merging.

An event is `(onset_s, offset_s, label)`. Behaviour follows the original one-pass detector
notebooks (github.com/satvik-venkatesh/you-only-hear-once, MIT) unless noted.
"""

import csv
from pathlib import Path
from typing import Dict, Iterable, List, Tuple, Union

Event = Tuple[float, float, str]


def read_tut2017_ann(path: Union[str, Path]) -> List[Event]:
    """TUT Sound Events 2017 `.ann`: file, scene, onset, offset, label (tab-separated)."""
    events = []
    with open(path, newline="") as fh:
        for row in csv.reader(fh, delimiter="\t"):
            if len(row) >= 5 and row[4].strip():
                events.append((float(row[2]), float(row[3]), row[4].strip()))
    return events


def read_onset_offset_label(path: Union[str, Path]) -> List[Event]:
    """URBAN-SED / DESED style: onset, offset, label (tab-separated)."""
    events = []
    with open(path, newline="") as fh:
        for row in csv.reader(fh, delimiter="\t"):
            if len(row) >= 3 and row[2].strip():
                events.append((float(row[0]), float(row[1]), row[2].strip()))
    return events


def merge_events(
    events: Iterable[Event], max_gap: float, sort: bool = False
) -> List[Event]:
    """Per class, merge events that overlap or are separated by <= `max_gap` s.

    The original merges in the order the events arrive, without sorting first
    (`sort=False` keeps that). Output is rounded to 3 decimals and sorted by onset.
    """
    by_class: Dict[str, List[List]] = {}
    for onset, offset, label in events:
        by_class.setdefault(label, []).append([onset, offset, label])

    merged: List[Event] = []
    for evs in by_class.values():
        if sort:
            evs.sort(key=lambda e: e[0])
        if not evs:
            continue
        current = evs[0]
        for nxt in evs[1:]:
            if current[1] >= nxt[0] or nxt[0] - current[1] <= max_gap:
                current[1] = max(nxt[1], current[1])
            else:
                merged.append(current)
                current = nxt
        merged.append(current)
    merged = [(round(e[0], 3), round(e[1], 3), e[2]) for e in merged]

    merged.sort(key=lambda e: e[0])
    return merged


def clip_to_window(
    events: Iterable[Event], win_start: float, win_len: float
) -> List[Event]:
    """Events overlapping `[win_start, win_start + win_len]`, in window time.

    Same-class overlaps are merged (gap 0) as in the original `construct_labels`.
    Deviation: zero-length results (an event starting exactly at the window end)
    are dropped instead of indexing past the last cell.
    """
    win_end = win_start + win_len
    clipped = []
    for onset, offset, label in events:
        if offset > win_start and onset <= win_end:
            start = max(onset - win_start, 0.0)
            end = min(offset - win_start, win_len)
            if end > start:
                clipped.append((start, end, label))
    return merge_events(clipped, max_gap=0.0)
