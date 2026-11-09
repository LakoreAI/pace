"""Run the benchmark matrix described by a YAML file.

Each unit is one (benchmark, seed, fold) training run. Units whose summary
JSON already exists are skipped, so a matrix can be resumed after preemption.
The runner only launches processes; comparison is done afterwards by
`scripts/analysis/compare_benchmarks.py` against `benchmarks/ledger.yaml`.

Example matrix (see configs/benchmarks/matrix_example.yaml):

    benchmarks:
      - config: configs/benchmark_desed.yaml
        data_root: data/features/desed
        dataset: desed
        benchmark: {id: desed, release: DESED 2020, split: public eval,
                    metric: psds1, supervision: synthetic+weak+unlabelled}
        seeds: [1, 2, 3, 4, 5]

Usage:
    uv run python scripts/benchmarks/run_matrix.py --matrix configs/benchmarks/matrix_example.yaml
    uv run python scripts/benchmarks/run_matrix.py --matrix ... --dry-run
"""

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))


def read_config(path: Path) -> dict:
    return yaml.safe_load(path.read_text()) or {}


def expected_summary(config: dict, data_root: str, seed: int, fold) -> Path:
    from src.pipelines.config import TrainingConfig

    cfg = TrainingConfig(
        **{
            **config,
            "data_root": data_root,
            "seed": seed,
            "fold": fold,
        }
    )
    return Path(cfg.result_dir) / cfg.result_filename()


def run_unit(unit: dict, log_dir: Path, dry_run: bool) -> dict:
    config_path = Path(unit["config"])
    config = read_config(config_path)
    fold = unit.get("fold")
    if "data_root_template" in unit:
        data_root = str(unit["data_root_template"]).format(fold=fold)
    else:
        data_root = str(unit["data_root"])
    seed = int(unit["seed"])
    run_name = unit["run_name"]
    summary = expected_summary(config, data_root, seed, fold)
    if summary.exists():
        return {"run": run_name, "status": "skipped", "summary": str(summary)}
    log_dir.mkdir(parents=True, exist_ok=True)
    log = log_dir / f"{run_name}.log"
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts/training/train.py"),
        "--config",
        str(config_path),
        "--data_root",
        data_root,
        "--seed",
        str(seed),
        "--run_name",
        run_name,
    ]
    if unit.get("dataset") or config.get("dataset"):
        cmd += ["--dataset", str(unit.get("dataset") or config["dataset"])]
    if fold is not None:
        cmd += ["--fold", str(fold)]
    if unit.get("experiment") or config.get("experiment"):
        cmd += ["--experiment", str(unit.get("experiment") or config["experiment"])]
    env = {
        **os.environ,
        "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS", "6"),
        "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS", "6"),
        "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS", "6"),
        "PYTHONUNBUFFERED": "1",
    }
    if dry_run:
        print(" ".join(cmd))
        return {"run": run_name, "status": "dry-run"}
    with open(log, "w") as fh:
        proc = subprocess.run(
            cmd, cwd=REPO_ROOT, stdout=fh, stderr=subprocess.STDOUT, env=env
        )
    return {
        "run": run_name,
        "status": "ok" if proc.returncode == 0 else f"failed ({proc.returncode})",
        "log": str(log),
        "summary": str(summary),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--log_dir", type=Path, default=Path("logs/benchmarks"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()

    matrix = read_config(args.matrix)
    units = []
    for bench in matrix.get("benchmarks", []):
        config_path = Path(bench["config"])
        for seed in bench.get("seeds", [1]):
            for fold in bench.get("folds", [None]):
                suffix = f"f{fold}" if fold is not None else ""
                run_name = f"{bench['id']}_s{seed}" + (f"_{suffix}" if suffix else "")
                units.append(
                    {
                        **bench,
                        "config": config_path,
                        "seed": seed,
                        "fold": fold,
                        "run_name": run_name,
                    }
                )
    if not units:
        sys.exit("matrix has no benchmark units")

    results = []
    if args.workers <= 1:
        results = [run_unit(u, args.log_dir, args.dry_run) for u in units]
        for result in results:
            print(json.dumps(result))
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [
                pool.submit(run_unit, u, args.log_dir, args.dry_run) for u in units
            ]
            for future in as_completed(futures):
                results.append(future.result())
                print(json.dumps(results[-1]), flush=True)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
