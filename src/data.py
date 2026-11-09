"""Dataset listings, the feature cache, and the window dataset.

One cache file per split (`.npz`), written by `scripts/data/extract_features.py`:

    X          : (N, n_frames, n_mels) float16  log-mel per window
    Y          : (N, n_cells, 3 * C)   float32  one-pass targets (legacy caches only)
    file_idx   : (N,)                  int64    index into meta["files"]
    win_start  : (N,)                  float64  window start in its file, s
    meta       : JSON string           classes, files, reference events per file,
                                       events of every window (window time, "win_events"),
                                       feature params, n_cells

Targets are encoded at load time from the window events (`src.sed.targets.ENCODERS`),
so one cache serves every head. Caches written before this change hold `Y` and still load.

Evaluation needs whole files (events are merged across windows), so the dataset
keeps the per-window file index and start time alongside the tensors.
"""

import csv
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset

from src.features import FeatureParams, load_mono, log_mel, window_audio
from src.sed.events import (
    Event,
    clip_to_window,
    read_onset_offset_label,
    read_tut2017_ann,
)
from src.sed.targets import get_encoder

Item = Tuple[Path, List[Event]]

TUT2017_CLASSES = (
    "brakes squeaking",
    "car",
    "children",
    "large vehicle",
    "people speaking",
    "people walking",
)
URBANSED_CLASSES = (
    "air_conditioner",
    "car_horn",
    "children_playing",
    "dog_bark",
    "drilling",
    "engine_idling",
    "gun_shot",
    "jackhammer",
    "siren",
    "street_music",
)


def tut2017_items(root: Union[str, Path], fold: int, split: str) -> List[Item]:
    """TUT Sound Events 2017 development set, official 4-fold setup.

    `root` is the extracted `TUT-sound-events-2017-development/` directory;
    `split` is "train" or "evaluate".
    """
    root = Path(root)
    setup = root / "evaluation_setup" / f"street_fold{fold}_{split}.txt"
    rel_paths = []
    with open(setup) as fh:
        for line in fh:
            rel = line.split("\t")[0].strip()
            if rel and rel not in rel_paths:
                rel_paths.append(rel)
    items = []
    for rel in sorted(rel_paths):
        audio = root / rel
        ann = root / "meta" / Path(rel).relative_to("audio").with_suffix(".ann")
        items.append((audio, read_tut2017_ann(ann)))
    return items


def urbansed_items(root: Union[str, Path], split: str) -> List[Item]:
    """URBAN-SED predefined splits: "train", "validate", "test"."""
    root = Path(root)
    items = []
    for audio in sorted((root / "audio" / split).glob("*.wav")):
        if audio.name.startswith("."):
            continue
        ann = root / "annotations" / split / f"{audio.stem}.txt"
        items.append((audio, read_onset_offset_label(ann)))
    return items


_PATH_ALIASES = ("path", "audio", "filename", "file", "ytid", "clip")
_LABEL_ALIASES = ("label", "event_label", "class", "positive_labels", "labels")


def read_manifest(
    manifest: Union[str, Path],
    root: Union[str, Path],
    split: Optional[str] = None,
    split_column: str = "split",
    delimiter: Optional[str] = None,
    labels_separator: Optional[str] = None,
) -> List[Item]:
    """Read a CSV/TSV manifest into (audio path, events) items.

    Required columns (aliases accepted): a path column (`path`, `audio`,
    `filename`, `file`, `YTID`, `clip`), `onset`, `offset`, and a label column
    (`label`, `event_label`, `class`, `positive_labels`, `labels`). An optional
    `split` column filters the rows. `labels_separator` splits multi-label
    cells (e.g. AudioSet's `positive_labels`).

    This is the generic adapter for datasets whose exact on-disk layout varies
    (VOICe, AudioSet-Strong, custom splits); export the official split instead
    of guessing paths.
    """
    manifest = Path(manifest)
    root = Path(root)
    text = manifest.read_text()
    delimiter = delimiter or ("\t" if "\t" in text.splitlines()[0] else ",")
    rows = list(csv.DictReader(text.splitlines(), delimiter=delimiter))
    if not rows:
        return []
    lower = {k.lower(): k for k in rows[0] if k}

    def column(aliases):
        for alias in aliases:
            if alias in lower:
                return lower[alias]
        raise ValueError(
            f"{manifest}: none of {aliases} found in columns {sorted(lower)}"
        )

    path_col = column(_PATH_ALIASES)
    label_col = column(_LABEL_ALIASES)
    split_col = lower.get(split_column.lower())
    grouped: Dict[Path, List[Event]] = defaultdict(list)
    for row in rows:
        if split is not None:
            if split_col is None:
                raise ValueError(
                    f"{manifest}: no {split_column!r} column for split={split!r}"
                )
            if (row.get(split_col) or "").strip() != split:
                continue
        name = (row.get(path_col) or "").strip()
        if not name:
            continue
        audio = Path(name)
        if not audio.is_absolute():
            audio = root / audio
        onset = float(row["onset"])
        offset = float(row["offset"])
        labels = (row.get(label_col) or "").strip()
        parts = labels.split(labels_separator) if labels_separator else [labels]
        for label in (p.strip() for p in parts):
            if label:
                grouped[audio].append((onset, offset, label))
    return [(path, events) for path, events in sorted(grouped.items())]


def audioset_strong_items(
    csv_path: Union[str, Path],
    audio_root: Union[str, Path],
    labels_map: Optional[Dict[str, str]] = None,
) -> List[Item]:
    """AudioSet-Strong CSV -> items, skipping clips missing from `audio_root`.

    The CSV has no header or a `#`-prefixed header and columns
    `YTID,start_time_seconds,end_time_seconds,positive_labels`. Labels are
    space-separated ontology ids; `labels_map` can translate them to readable
    names. Audio files are looked up as `<audio_root>/<YTID>.(wav|flac|mp3)`.
    """
    audio_root = Path(audio_root)
    grouped: Dict[Path, List[Event]] = defaultdict(list)
    for line in Path(csv_path).read_text().splitlines():
        line = line.strip()
        if not line or line.lower().startswith(("ytid", "#ytid")):
            continue
        parts = line.split(",")
        if len(parts) < 4:
            continue
        ytid, onset, offset, labels = parts[0], parts[1], parts[2], ",".join(parts[3:])
        audio = next(
            (
                audio_root / f"{ytid}{ext}"
                for ext in (".wav", ".flac", ".mp3")
                if (audio_root / f"{ytid}{ext}").exists()
            ),
            None,
        )
        if audio is None:
            continue
        for label in labels.split():
            grouped[audio].append(
                (float(onset), float(offset), (labels_map or {}).get(label, label))
            )
    return [(path, events) for path, events in sorted(grouped.items())]


def audioset_missing_clips(
    csv_path: Union[str, Path], audio_root: Union[str, Path]
) -> List[str]:
    """YTIDs referenced by the CSV but absent from `audio_root` (record them)."""
    audio_root = Path(audio_root)
    missing = set()
    for line in Path(csv_path).read_text().splitlines():
        line = line.strip()
        if not line or line.lower().startswith(("ytid", "#ytid")):
            continue
        ytid = line.split(",", 1)[0]
        if not any(
            (audio_root / f"{ytid}{ext}").exists() for ext in (".wav", ".flac", ".mp3")
        ):
            missing.add(ytid)
    return sorted(missing)


def desed_jams_items(
    jams_dir: Union[str, Path], audio_dir: Union[str, Path]
) -> List[Item]:
    """DESED synthetic scenes from SCAPER JAMS annotations.

    Parses the JAMS JSON directly (no `jams` dependency): every annotation
    namespace's `data` entries with a `time`, `duration` and label are events.
    Audio is `<audio_dir>/<jams stem>.wav`. The exact DESED release must be
    recorded in the benchmark block before numbers enter the ledger.
    """
    jams_dir, audio_dir = Path(jams_dir), Path(audio_dir)
    items: List[Item] = []
    for jams_path in sorted(jams_dir.glob("*.jams")):
        data = json.loads(jams_path.read_text())
        events: List[Event] = []
        for annotation in data.get("annotations", []):
            for item in annotation.get("data", []):
                value = item.get("value", {})
                labels = value.get("labels") or (
                    [value["label"]] if value.get("label") else []
                )
                if not labels or "time" not in item:
                    continue
                duration = item.get("duration", 0.0)
                for label in labels:
                    events.append(
                        (
                            float(item["time"]),
                            float(item["time"]) + float(duration),
                            label,
                        )
                    )
        audio = audio_dir / f"{jams_path.stem}.wav"
        if events and audio.exists():
            items.append((audio, events))
    return items


def desed_tsv_items(
    tsv_path: Union[str, Path], audio_root: Union[str, Path]
) -> List[Item]:
    """DESED strong-label TSV (`filename onset offset event_label`) -> items."""
    audio_root = Path(audio_root)
    rows = list(csv.DictReader(Path(tsv_path).read_text().splitlines(), delimiter="\t"))
    if not rows:
        return []
    keys = {k.lower(): k for k in rows[0] if k}
    grouped: Dict[Path, List[Event]] = defaultdict(list)
    for row in rows:
        name = (row.get(keys.get("filename", "filename")) or "").strip()
        if not name:
            continue
        audio = Path(name)
        if not audio.is_absolute():
            audio = audio_root / audio
        label = (row.get(keys.get("event_label", "event_label")) or "").strip()
        if audio.exists() and label:
            grouped[audio].append(
                (
                    float(row[keys.get("onset", "onset")]),
                    float(row[keys.get("offset", "offset")]),
                    label,
                )
            )
    return [(path, events) for path, events in sorted(grouped.items())]


def hold_out_files(
    items: Sequence[Item], fraction: float, seed: int
) -> Tuple[List[Item], List[Item]]:
    """File-level split into (train, val). Windows of one file never straddle both."""
    order = list(range(len(items)))
    random.Random(seed).shuffle(order)
    n_val = max(1, round(len(items) * fraction)) if fraction > 0 else 0
    val_idx = set(order[:n_val])
    return (
        [it for i, it in enumerate(items) if i not in val_idx],
        [it for i, it in enumerate(items) if i in val_idx],
    )


def stratified_holdout(
    items: Sequence[Item], fraction: float, seed: int
) -> Tuple[List[Item], List[Item]]:
    """File-level hold-out stratified by each clip's set of labels.

    Approximates AudioSet-Strong's multilabel-stratified split: clips with the
    same label set are shuffled together and sampled at `fraction`, so rare
    label combinations stay represented on both sides.
    """
    if fraction <= 0:
        return list(items), []
    rng = random.Random(seed)
    groups: Dict[frozenset, List[Item]] = defaultdict(list)
    for item in items:
        groups[frozenset(label for _, _, label in item[1])].append(item)
    train, val = [], []
    for group in groups.values():
        group = list(group)
        rng.shuffle(group)
        n_val = max(1, round(len(group) * fraction))
        val.extend(group[:n_val])
        train.extend(group[n_val:])
    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


def build_cache(
    items: Sequence[Item],
    classes: Sequence[str],
    params: FeatureParams,
    n_cells: int,
    out_path: Union[str, Path],
    resample: bool = False,
) -> Path:
    xs, win_events, file_idx, win_start = [], [], [], []
    files, references, durations = [], {}, {}
    for fi, (audio_path, events) in enumerate(items):
        name = Path(audio_path).name
        files.append(name)
        references[name] = [list(e) for e in events]
        audio = load_mono(audio_path, params.sr, resample=resample)
        durations[name] = audio.shape[0] / params.sr
        windows, starts = window_audio(audio, params.sr, params.win_len, params.hop_len)
        for w, s in zip(windows, starts):
            xs.append(log_mel(w, params).astype(np.float16))
            win_events.append(
                [list(e) for e in clip_to_window(events, s, params.win_len)]
            )
            file_idx.append(fi)
            win_start.append(s)

    meta = {
        "classes": list(classes),
        "files": files,
        "references": references,
        "durations": durations,
        "win_events": win_events,
        "features": params.to_dict(),
        "n_cells": n_cells,
    }
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path,
        X=np.stack(xs),
        file_idx=np.asarray(file_idx, dtype=np.int64),
        win_start=np.asarray(win_start, dtype=np.float64),
        meta=np.array(json.dumps(meta)),
    )
    return out_path


class WindowDataset(Dataset):
    """Yields `(mel (n_frames, n_mels) float32, target float32)`.

    The target is encoded from the window events with `target_kind` (see
    `src.sed.targets.ENCODERS`) on `n_cells` output steps (default: the cache's).
    Legacy caches carry `Y` already in the one-pass grid and only serve `target_kind="onepass"`.
    """

    def __init__(
        self,
        x: np.ndarray,
        y: Optional[np.ndarray],
        file_idx: np.ndarray,
        win_start: np.ndarray,
        meta: Dict,
        target_kind: str = "onepass",
        n_cells: Optional[int] = None,
    ):
        if not (len(x) == len(file_idx) == len(win_start)):
            raise ValueError("cache arrays must have the same length")
        self.x = torch.as_tensor(x)
        self.meta = meta
        self._target_kind = target_kind
        self.n_cells = int(n_cells or meta["n_cells"])
        if y is not None:
            if len(y) != len(x):
                raise ValueError("cache arrays must have the same length")
            if target_kind != "onepass" or self.n_cells != int(meta["n_cells"]):
                raise ValueError(
                    "legacy cache holds one-pass targets only; re-extract features to use "
                    f"target_kind={target_kind!r} with n_cells={self.n_cells}"
                )
            self.y = torch.as_tensor(y, dtype=torch.float32)
        else:
            self.y = self._encode_all()
        self.file_idx = np.asarray(file_idx)
        self.win_start = np.asarray(win_start)

    def _encode_all(self) -> torch.Tensor:
        encode = get_encoder(self.target_kind)
        classes = {c: i for i, c in enumerate(self.meta["classes"])}
        win_len = self.meta["features"]["win_len"]
        ys = [
            encode([tuple(e) for e in events], classes, win_len, self.n_cells)
            for events in self.meta["win_events"]
        ]
        return torch.as_tensor(np.stack(ys), dtype=torch.float32)

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, index: int):
        return self.x[index].float(), self.y[index]

    @property
    def target_kind(self) -> str:
        return self._target_kind

    @property
    def classes(self) -> List[str]:
        return list(self.meta["classes"])

    @property
    def num_classes(self) -> int:
        return len(self.meta["classes"])

    @property
    def features(self) -> FeatureParams:
        return FeatureParams(**self.meta["features"])

    @property
    def n_frames(self) -> int:
        return int(self.x.shape[1])

    @property
    def n_mels(self) -> int:
        return int(self.x.shape[2])

    def references(self) -> Dict[str, List[Event]]:
        return {k: [tuple(e) for e in v] for k, v in self.meta["references"].items()}

    def durations(self) -> Dict[str, float]:
        """Audio duration per file; falls back to the window extent for old caches."""
        saved = self.meta.get("durations")
        if saved:
            return {k: float(v) for k, v in saved.items()}
        gaps = defaultdict(list)
        for fi, start in zip(self.file_idx, self.win_start):
            gaps[int(fi)].append(float(start))
        win_len = float(self.meta["features"]["win_len"])
        return {
            self.meta["files"][fi]: max(starts) + win_len for fi, starts in gaps.items()
        }


def load_cache(
    path: Union[str, Path],
    target_kind: str = "onepass",
    n_cells: Optional[int] = None,
) -> WindowDataset:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"feature cache not found: {path}")
    with np.load(path) as data:
        return WindowDataset(
            data["X"],
            data["Y"] if "Y" in data.files else None,
            data["file_idx"],
            data["win_start"],
            json.loads(str(data["meta"])),
            target_kind=target_kind,
            n_cells=n_cells,
        )


def resolve(data_root: str, name: Optional[str]) -> Optional[Path]:
    """`<data_root>/<name>` if it exists, else None (optional splits)."""
    if not name:
        return None
    path = Path(data_root) / name
    return path if path.exists() else None


def make_synthetic_clips(
    root: Union[str, Path],
    split: str,
    n_files: int,
    classes: Sequence[str] = ("tone_a", "tone_b", "tone_c"),
    duration: float = 5.0,
    sr: int = 44100,
    seed: int = 0,
) -> List[Item]:
    """Write noisy clips with one sine tone per class at random times.

    Layout matches URBAN-SED (`audio/<split>/*.wav`, `annotations/<split>/*.txt`),
    so `urbansed_items(root, split)` reads it back. For smoke tests only.
    """
    import soundfile as sf

    root = Path(root)
    (root / "audio" / split).mkdir(parents=True, exist_ok=True)
    (root / "annotations" / split).mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    freqs = np.linspace(500.0, 4000.0, len(classes))
    t = np.arange(int(duration * sr)) / sr
    for n in range(n_files):
        audio = 0.01 * rng.standard_normal(t.shape[0])
        events = []
        for _ in range(int(rng.integers(1, 4))):
            k = int(rng.integers(len(classes)))
            onset = float(rng.uniform(0.0, duration - 0.5))
            offset = float(min(duration, onset + rng.uniform(0.2, 2.0)))
            mask = (t >= onset) & (t < offset)
            audio[mask] += 0.3 * np.sin(2 * np.pi * freqs[k] * t[mask])
            events.append((round(onset, 3), round(offset, 3), classes[k]))
        name = f"{split}_{n:04d}"
        sf.write(root / "audio" / split / f"{name}.wav", audio, sr)
        with open(root / "annotations" / split / f"{name}.txt", "w") as fh:
            fh.writelines(f"{on}\t{off}\t{label}\n" for on, off, label in events)
    return urbansed_items(root, split)
