"""Build log-mel + target caches for one dataset/benchmark split.

Writes `<out>/train.npz`, validation cache(s) and `<out>/test.npz` (see
src/data.py). Supported inputs:

- tut2017: official fold `--fold`. test = the fold's evaluate files; val = a
  file-level hold-out of the fold's train files (selection never sees them).
- urbansed: predefined train / validate / test splits.
- manifest: generic CSV/TSV (`path,onset,offset,label[,split]`) for VOICe,
  custom splits, or any dataset whose official split has been exported.
- audioset_strong: AudioSet-Strong CSV + downloaded 10 s clips. Train/val are
  a multilabel-stratified file-level split unless explicit files are given;
  missing clips are recorded in `<out>/missing_clips.json`.
- desed: synthetic scenes from SCAPER JAMS plus strong-label TSV val/test.

`--n_val_splits` partitions the held-out files into disjoint validation caches;
`--val_files` in the training config then averages metrics over them.

Usage:
    uv run python scripts/data/extract_features.py --dataset tut2017 --fold 1 \
        --root data/raw/TUT-sound-events-2017-development --out data/features/tut2017_fold1
    uv run python scripts/data/extract_features.py --dataset manifest \
        --root data/raw/VOICe --train_manifest train.tsv \
        --val_manifest val.tsv --test_manifest test.tsv --out data/features/voice
"""

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.config import PACEConfig  # noqa: E402
from src.data import (  # noqa: E402
    TUT2017_CLASSES,
    URBANSED_CLASSES,
    audioset_missing_clips,
    audioset_strong_items,
    build_cache,
    desed_jams_items,
    desed_tsv_items,
    hold_out_files,
    read_manifest,
    stratified_holdout,
    tut2017_items,
    urbansed_items,
)
from src.features import PRESETS  # noqa: E402


def _classes(args, splits) -> list[str]:
    if args.classes_file:
        return [
            line.strip()
            for line in Path(args.classes_file).read_text().splitlines()
            if line.strip()
        ]
    if args.classes:
        return [c.strip() for c in args.classes.split(",") if c.strip()]
    inferred = sorted(
        {
            label
            for items in splits.values()
            for _, events in items
            for _, _, label in events
        }
    )
    print(
        f"classes inferred from data ({len(inferred)}); freeze this list in the config"
    )
    return inferred


def _manifest(args, name: str):
    path = getattr(args, f"{name}_manifest")
    if not path:
        return None
    return read_manifest(
        path,
        args.root,
        split=args.split,
        split_column=args.split_column,
        labels_separator=args.labels_separator,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        choices=("tut2017", "urbansed", "manifest", "audioset_strong", "desed"),
        required=True,
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--fold", type=int, default=1, help="tut2017 only (1-4)")
    parser.add_argument("--val_fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0, help="hold-out seed")
    parser.add_argument(
        "--n_val_splits",
        type=int,
        default=1,
        help="partition the held-out files into this many disjoint val caches",
    )
    parser.add_argument("--classes", default=None, help="comma-separated class list")
    parser.add_argument("--classes_file", type=Path, default=None)
    parser.add_argument("--features", choices=sorted(PRESETS), default=None)
    parser.add_argument("--resample", action="store_true")
    parser.add_argument("--train_manifest", type=Path, default=None)
    parser.add_argument("--val_manifest", type=Path, default=None)
    parser.add_argument("--test_manifest", type=Path, default=None)
    parser.add_argument(
        "--split", default=None, help="value of the manifest split column"
    )
    parser.add_argument("--split_column", default="split")
    parser.add_argument("--labels_separator", default=None)
    parser.add_argument("--labels_map", type=Path, default=None, help="JSON code->name")
    parser.add_argument("--jams_dir", type=Path, default=None)
    parser.add_argument("--audio_dir", type=Path, default=None)
    parser.add_argument("--val_tsv", type=Path, default=None)
    parser.add_argument("--test_tsv", type=Path, default=None)
    args = parser.parse_args()
    if args.n_val_splits < 1:
        parser.error("--n_val_splits must be >= 1")

    preset = args.features or (args.dataset if args.dataset in PRESETS else "tut2017")
    params = PRESETS[preset]
    n_cells = PACEConfig(n_frames=params.n_frames, n_mels=params.n_mels).n_cells

    splits: dict[str, list] = {}
    if args.dataset == "tut2017":
        train, val = hold_out_files(
            tut2017_items(args.root, args.fold, "train"), args.val_fraction, args.seed
        )
        splits = {
            "train": train,
            "val": val,
            "test": tut2017_items(args.root, args.fold, "evaluate"),
        }
        classes = TUT2017_CLASSES
    elif args.dataset == "urbansed":
        splits = {
            "train": urbansed_items(args.root, "train"),
            "val": urbansed_items(args.root, "validate"),
            "test": urbansed_items(args.root, "test"),
        }
        classes = URBANSED_CLASSES
    elif args.dataset == "manifest":
        splits = {
            name: items
            for name in ("train", "val", "test")
            if (items := _manifest(args, name)) is not None
        }
        if "train" not in splits or "val" not in splits:
            parser.error("manifest mode needs --train_manifest and --val_manifest")
        classes = _classes(args, splits)
    elif args.dataset == "audioset_strong":
        if not args.train_manifest:
            parser.error("audioset_strong needs --train_manifest (CSV)")
        labels_map = (
            json.loads(args.labels_map.read_text()) if args.labels_map else None
        )
        audio_root = args.audio_dir or args.root
        items = audioset_strong_items(args.train_manifest, audio_root, labels_map)
        train, val = stratified_holdout(items, args.val_fraction, args.seed)
        splits = {"train": train, "val": val}
        if args.test_manifest:
            splits["test"] = audioset_strong_items(
                args.test_manifest, audio_root, labels_map
            )
        missing = audioset_missing_clips(args.train_manifest, audio_root)
        if args.test_manifest:
            missing += audioset_missing_clips(args.test_manifest, audio_root)
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "missing_clips.json").write_text(json.dumps(sorted(set(missing))))
        classes = _classes(args, splits)
    else:
        if not (args.jams_dir and args.audio_dir):
            parser.error("desed needs --jams_dir and --audio_dir")
        splits = {"train": desed_jams_items(args.jams_dir, args.audio_dir)}
        if args.val_tsv:
            splits["val"] = desed_tsv_items(args.val_tsv, args.audio_dir)
        if args.test_tsv:
            splits["test"] = desed_tsv_items(args.test_tsv, args.audio_dir)
        if "val" not in splits:
            parser.error("desed needs --val_tsv")
        classes = _classes(args, splits)

    val = splits.pop("val")
    n = min(args.n_val_splits, len(val))
    if n < args.n_val_splits:
        print(f"only {len(val)} validation files; using n_val_splits={n}")
    val_parts = [val[i::n] for i in range(n)] if val else [[]]
    for i, part in enumerate(val_parts):
        splits["val" if len(val_parts) == 1 else f"val{i}"] = part

    if "train" not in splits:
        parser.error("no training items found")
    if not classes:
        parser.error("no classes available; pass --classes or --classes_file")

    for split, items in splits.items():
        path = build_cache(
            items,
            classes,
            params,
            n_cells,
            args.out / f"{split}.npz",
            resample=args.resample,
        )
        print(f"{split}: {len(items)} files -> {path}")


if __name__ == "__main__":
    main()
