# scripts/

Entry points. Run them from the repo root, e.g. `uv run python scripts/<category>/<name>.py ...`.

| Script | Purpose |
|---|---|
| `data/extract_features.py` | Build train/val/test log-mel caches and targets (TUT 2017, URBAN-SED, manifest, DESED, AudioSet-Strong) |
| `training/train.py` | Thin wrapper over `src.pipelines.train` (the real CLI) |
| `training/evaluate.py` | Segment ER / F of a checkpoint on a cache split |
| `training/smoke_test.py` | End-to-end run on synthetic audio: train, evaluate, assert finite metrics |
| `benchmarks/run_matrix.py` | Run a config over several seeds in parallel |
| `analysis/summarize.py` | Mean ± std over seeds from result JSONs |
| `analysis/compare_benchmarks.py` | Tier-aware comparison against `benchmarks/ledger.yaml` |
| `analysis/evaluate_streaming.py` | Evaluate a checkpoint at several look-ahead values |
| `analysis/profile_model.py` | Parameters, MACs and latency of a checkpoint |
