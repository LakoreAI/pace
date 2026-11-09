# Getting started

Requires Python 3.12 or newer and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/LakoreAI/pace.git
cd pace
uv sync                 # core dependencies
uv sync --extra rich    # optional: formatted training summary
uv sync --extra wandb   # optional: Weights & Biases logging
uv run pytest           # test suite
```

Check the whole pipeline on synthetic audio, with no dataset needed:

```bash
uv run python scripts/training/smoke_test.py
```

## Train and evaluate

```bash
uv run python scripts/training/train.py --config configs/onepass_urbansed.yaml
uv run python scripts/training/train.py --config configs/urbansed_causal.yaml --seed 1

uv run python scripts/training/evaluate.py \
    --ckpt checkpoints/<run>/best.pt --data data/features/urbansed/test.npz
```

Any key in a config can be overridden on the command line. A run writes
`checkpoints/<run_name>/best.pt` and `results/<experiment>_<dataset>_s<seed>.json`.

## Several seeds

```bash
uv run python scripts/benchmarks/run_matrix.py \
    --matrix configs/benchmarks/matrix_example.yaml --workers 2
uv run python scripts/analysis/summarize.py --results results
```

`summarize.py` reports mean ± sample standard deviation over seeds. With few
seeds, treat differences smaller than the spread as indistinguishable.
