"""Compare our result JSONs against the benchmark ledger, tier-aware.

Only results with the same benchmark id and metric are compared. Literature
rows are printed by tier: M (matched), E (more capacity/pretraining/data) and
X (different protocol, never in the main table). A result without a
`benchmark` block is matched by dataset tag but flagged as unverified protocol.

Usage:
    uv run python scripts/analysis/compare_benchmarks.py --results results
    uv run python scripts/analysis/compare_benchmarks.py --results results --benchmark desed
"""

import argparse
import json
import re
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

import yaml

FOLD = re.compile(r"_fold\d+$")


def family(dataset) -> str:
    return FOLD.sub("", str(dataset))


def load_results(results_dir: Path):
    """benchmark id / dataset family -> list of (seed, fold, value, efficiency)."""
    rows = []
    for path in sorted(results_dir.glob("*.json")):
        try:
            d = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if "test" not in d:
            continue
        benchmark = d.get("benchmark") or {}
        rows.append(
            {
                "path": path,
                "id": benchmark.get("id"),
                "reported_metric": benchmark.get("metric"),
                "experiment": d.get("experiment", "unknown"),
                "dataset": family(d.get("dataset")),
                "dataset_raw": d.get("dataset"),
                "seed": int(d.get("seed", 0)),
                "fold": d.get("fold"),
                "test": d["test"],
                "efficiency": d.get("efficiency"),
                "benchmark": benchmark,
            }
        )
    return rows


def mean_std(values):
    return st.mean(values), (st.stdev(values) if len(values) > 1 else float("nan"))


def ours_for(rows, key, spec, metric, scale, experiment=None):
    """Return one group per experiment; never pool different experiments."""
    matched = [
        r
        for r in rows
        if (r["id"] == key)
        or (
            r["id"] is None
            and r["dataset"] in [str(a) for a in spec.get("aliases", [key])]
        )
    ]
    if experiment:
        matched = [r for r in matched if r.get("experiment") == experiment]
    if not matched:
        return {}
    groups = defaultdict(list)
    for row in matched:
        groups[row.get("experiment", "unknown")].append(row)

    out = {}
    for name, group in sorted(groups.items()):
        by_seed = defaultdict(dict)
        for row in group:
            fold = row["fold"] if row["fold"] is not None else row["dataset_raw"]
            by_seed[row["seed"]][fold] = float(row["test"][metric]) * scale
        per_seed = {seed: st.mean(folds.values()) for seed, folds in by_seed.items()}
        out[name] = {
            "per_seed": per_seed,
            "mean_std": mean_std(list(per_seed.values())),
            "n_runs": len(group),
            "unverified_protocol": any(r["id"] is None for r in group),
        }
    return out


def compare(ours, entries, metric, direction="max"):
    if ours is None:
        return []
    mean, std = ours["mean_std"]
    lines = []
    for entry in entries:
        if entry.get("tier") == "X":
            continue
        gap = mean - float(entry["value"])
        spread = std if std == std else 0.0
        if abs(gap) < spread:
            verdict = "indistinguishable"
        elif (gap > 0) == (direction == "max"):
            verdict = "ours better"
        else:
            verdict = "ours worse"
        lines.append(
            f"    vs [{entry['tier']}] {entry['system']:<42s} "
            f"{float(entry['value']):8.3f}  gap {gap:+.3f}  {verdict}"
        )
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--ledger", type=Path, default=Path("benchmarks/ledger.yaml"))
    parser.add_argument("--benchmark", default=None)
    parser.add_argument(
        "--experiment", default=None, help="only this experiment name (e.g. onepass)"
    )
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()

    ledger = yaml.safe_load(args.ledger.read_text())
    rows = load_results(args.results)
    output = {}
    print(f"ledger: {args.ledger}  (cut-off {ledger.get('as_of')})")
    for key, spec in ledger["benchmarks"].items():
        if args.benchmark and key != args.benchmark:
            continue
        metric = spec["metric"]
        scale = float(spec.get("scale", 1.0))
        groups = ours_for(rows, key, spec, metric, scale, args.experiment)
        print(
            f"\n{key}: {spec['dataset']} | {spec['release']} | {spec['split']}"
            f"\n  metric={metric} ({spec.get('metric_label', metric)})"
            f" class_count={spec.get('class_count')} supervision={spec.get('supervision')}"
        )
        bench_out = {"spec": spec, "ours": {}, "literature": spec.get("entries", [])}
        if not groups:
            print("  ours: no result file for this benchmark/metric")
        for name, ours in groups.items():
            mean, std = ours["mean_std"]
            print(
                f"  ours [{name}]: {mean:8.3f} +/- {std:6.3f}   "
                f"seeds={len(ours['per_seed'])} runs={ours['n_runs']}"
            )
            if ours["unverified_protocol"]:
                print(
                    "  WARNING: matched by dataset tag only; result has no "
                    "benchmark block, so protocol match is unverified"
                )
            for line in compare(
                ours, spec.get("entries", []), metric, spec.get("direction", "max")
            ):
                print(line)
            bench_out["ours"][name] = ours
        if spec.get("notes"):
            print(f"  note: {' '.join(spec['notes'].split())}")
        output[key] = bench_out

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(output, indent=2, default=str))
        print(f"\nsaved: {args.json}")


if __name__ == "__main__":
    sys.exit(main())
