"""Summarise result files: mean +- std over seeds, per experiment and dataset.

Reads only `results/<experiment>_<dataset>_s<seed>.json`. Datasets that differ only by a
`_fold<N>` suffix are one family: metrics are averaged over the folds of each seed first, then
mean and std are taken over seeds. With `--compare A B`, flags the gap between two experiments
as indistinguishable when it is smaller than either one's std.

Usage:
    uv run python scripts/analysis/summarize.py --results results
    uv run python scripts/analysis/summarize.py --results results --compare onepass other --metric f_measure
"""

import argparse
import json
import re
import statistics as st
import sys
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

FOLD = re.compile(r"_fold\d+$")


def fold_key(d: dict) -> str:
    """Fold identity: explicit `fold` field, else the dataset tag."""
    if d.get("fold") is not None:
        return f"fold{d['fold']}"
    return str(d.get("dataset"))


def load(results_dir: Path, metric: str) -> Dict[Tuple[str, str], Dict[int, dict]]:
    """(experiment, dataset family) -> seed -> {fold: metric value}."""
    out: Dict[Tuple[str, str], Dict[int, dict]] = defaultdict(lambda: defaultdict(dict))
    for path in sorted(results_dir.glob("*.json")):
        d = json.loads(path.read_text())
        if "experiment" not in d or "test" not in d or metric not in d["test"]:
            continue
        family = FOLD.sub("", str(d["dataset"]))
        key = fold_key(d)
        bucket = out[(d["experiment"], family)][int(d["seed"])]
        if key in bucket:
            warnings.warn(
                f"duplicate {d['experiment']}/{family} seed {d['seed']} "
                f"{key}: overwriting {path.name}",
                stacklevel=2,
            )
        bucket[key] = float(d["test"][metric])
    return out


def per_seed(values: Dict[int, dict]) -> List[float]:
    return [st.mean(v.values()) for _, v in sorted(values.items())]


def mean_std(xs: List[float]) -> Tuple[float, float]:
    return st.mean(xs), (st.stdev(xs) if len(xs) > 1 else float("nan"))


def compare(a: List[float], b: List[float]) -> str:
    (ma, sa), (mb, sb) = mean_std(a), mean_std(b)
    gap = abs(ma - mb)
    spread = (
        max(s for s in (sa, sb) if s == s) if (sa == sa or sb == sb) else float("nan")
    )
    verdict = "indistinguishable" if spread == spread and gap < spread else "separated"
    return f"gap {gap:.4f} vs std {spread:.4f}: {verdict}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--metric", default="error_rate")
    parser.add_argument("--compare", nargs=2, metavar=("A", "B"))
    args = parser.parse_args()

    table = load(args.results, args.metric)
    if not table:
        sys.exit(f"no results with metric {args.metric!r} in {args.results}")
    print(
        f"metric: {args.metric}  (mean +- std over seeds; folds averaged within a seed)"
    )
    summary = {}
    for (exp, ds), seeds in sorted(table.items()):
        xs = per_seed(seeds)
        m, s = mean_std(xs)
        summary[(exp, ds)] = xs
        n_runs = sum(len(v) for v in seeds.values())
        print(
            f"  {exp:<24s} {ds:<14s} {m:8.4f} +- {s:6.4f}   seeds={len(xs)} runs={n_runs}"
        )
    if args.compare:
        a, b = args.compare
        for ds in sorted({d for _, d in summary}):
            if (a, ds) in summary and (b, ds) in summary:
                print(
                    f"  {a} vs {b} on {ds}: {compare(summary[(a, ds)], summary[(b, ds)])}"
                )


if __name__ == "__main__":
    main()
