# PACE

**PACE** is a bounded-latency streaming event-level sound event detector. It
extends the one-pass detector of Venkatesh, Moffat and Miranda (Applied
Sciences, 2022) with a causal backbone, RED + onset/offset + event proposals,
and an explicit look-ahead commit protocol. Events are predicted directly and
compared with a matched causal frame-wise detector on PSDS, commit latency and
end-to-end cost.

The comparison is run on URBAN-SED (segment F), with TUT 2017, DESED and
AudioSet-Strong supported through the same feature-cache interface. The causal
backbone takes a bounded look-ahead, and the streaming evaluator replays a
checkpoint cell by cell to measure the commit delay.

## Credits

- **Original one-pass detector:** Venkatesh, Moffat, Miranda. *Applied Sciences*
  12(7), 3293, 2022 (MIT-licensed code). Its model, targets, loss, SpecAugment
  and post-processing are ported here as the baseline.
- **EfficientSED**: Morocutti, Schmid, Greif, Foscarin, Widmer. "Exploring
  Performance-Complexity Trade-Offs in Sound Event Detection." arXiv:2503.11373.
  Code: github.com/theMoro/EfficientSED.

## Quickstart

Requires Python ≥ 3.12 and [`uv`](https://docs.astral.sh/uv/).

```bash
uv sync                # core: torch, librosa, sed_eval, ...
uv sync --extra rich   # optional: pretty training summary (panels / tables)
uv sync --extra wandb  # optional: Weights & Biases logging
uv run pytest          # test suite

# end-to-end on synthetic audio, no dataset required
uv run python scripts/training/smoke_test.py
```

## Data

Audio and features are never committed. Extract log-mel windows and one-pass targets
into one cache file per split:

```bash
# TUT Sound Events 2017, fold 1 (repeat for folds 2-4)
uv run python scripts/data/extract_features.py --dataset tut2017 --fold 1 \
    --root <path>/TUT-sound-events-2017-development --out data/features/tut2017_fold1

# URBAN-SED predefined splits
uv run python scripts/data/extract_features.py --dataset urbansed \
    --root <path>/URBAN-SED_v2.0.0 --out data/features/urbansed
```

Each `<split>.npz` holds `X` (windows × frames × mels, fp16), window events,
the file index, start time and duration of every window, and a JSON `meta` with
classes, reference events and feature settings (see `src/data.py`). Targets are
encoded at load time from `--target_kind` (one-pass or frame-wise). For TUT 2017 the
validation split is a file-level hold-out of the fold's train files; the fold's
evaluate files are the test split and are never used for checkpoint selection.
`--n_val_splits K` partitions the held-out files into disjoint `val0..K-1`
caches; the training config's `val_files` averages selection metrics over them.

Beyond TUT/URBAN, `--dataset` also accepts:

- `manifest` — a generic CSV/TSV (`path,onset,offset,label[,split]`) for VOICe,
  exported AudioSet splits, or any dataset whose official split is frozen.
- `desed` — SCAPER JAMS synthetic scenes plus strong-label TSV val/test.
- `audioset_strong` — the official CSV + downloaded clips, with a
  multilabel-stratified file-level train/val split and a missing-clip report.

Pick the front end with `--features {tut2017,urbansed,music_speech,voice,desed}`.

## Benchmarks

`benchmarks/ledger.yaml` is the machine-readable mirror of the benchmark ledger
(dataset, release, split, metric, class count, supervision, tier M/E/X). Run a
matrix and compare without crossing protocols:

```bash
# 1. extract features for each dataset (see Data above)
# 2. launch the benchmark matrix (resumable; skips existing result JSONs)
uv run python scripts/benchmarks/run_matrix.py \
    --matrix configs/benchmarks/matrix_example.yaml --workers 2

# 3. tier-aware table: ours mean +/- std vs ledger M/E rows, same metric only
uv run python scripts/analysis/compare_benchmarks.py --results results
uv run python scripts/analysis/compare_benchmarks.py --results results --benchmark desed
```

A result only joins a benchmark table when its config carries a `benchmark:`
block (`id`, `release`, `split`, `metric`, `supervision`). The comparison script
flags results matched by dataset tag alone as having an unverified protocol.
The number is never compared across datasets or metric families. PSDS1/PSDS2,
event-F1 and per-stage efficiency are written into the result JSON when
`psds.enabled` and `profile: true`.

For bounded-latency evaluation, enable `streaming.lookaheads` in the config or
run the checkpoint directly:

```bash
uv run python scripts/analysis/evaluate_streaming.py \
    --ckpt checkpoints/<run>/best.pt --data data/features/desed/test.npz \
    --lookaheads 0,0.1,0.25,0.5,1.0 --decoder red_epn
```

`PSDS` is retrospective; report it with the measured commit latency and the
`forced_at_end` count.

## Training, evaluation, inference

```bash
uv run python scripts/training/train.py --config configs/onepass_tut2017.yaml
uv run python scripts/training/train.py --config configs/onepass_urbansed.yaml --epochs 50
uv run python scripts/training/train.py --config configs/urbansed_causal.yaml --seed 1

uv run python scripts/training/evaluate.py \
    --ckpt checkpoints/<run>/best.pt --data data/features/tut2017_fold1/test.npz

uv run python -m src.pipelines.infer --ckpt checkpoints/<run>/best.pt \
    --audio a001.wav --features tut2017 --classes "brakes squeaking,car,..."
```

A run writes `checkpoints/<run_name>/best.pt` (best validation segment ER),
`epoch_*.pt`, `train_log.json`, and `results/<run_name>/test_metrics.json` when a
test split is present.

## Repository layout

```
src/
├── config.py            # PACEConfig: model architecture only
├── data.py              # dataset listings, feature cache, WindowDataset
├── features.py          # audio loading, windowing, log-mel
├── modules/
│   ├── model.py         # backbone/context/head registries (one-pass, frame-wise)
│   ├── red.py           # RED + onset/offset + event-proposal head
│   ├── loss.py          # one-pass, BCE/focal, RED+EPN losses
│   └── augment.py       # SpecAugment (original per-batch variant)
├── sed/
│   ├── events.py        # annotation readers, window clipping, merging
│   ├── targets.py       # one-pass target encoder
│   ├── postprocess.py   # decoder registry: one-pass merge, frame-wise, NMS
│   ├── metrics.py       # sed_eval segment ER / F
│   └── psds.py          # PSDS1/2 (sed_scores_eval) + event-F1
├── pipelines/           # TrainingConfig, train, eval, infer
├── callbacks/           # checkpoint, early_stopping, lr_scheduler, wandb, EMA
└── utils/               # io, model, device, efficiency (params/MACs/latency)
benchmarks/ledger.yaml   # machine-readable benchmark tiers and references
configs/                 # annotated base configs (every parameter explained in place)
scripts/                 # data/, training/, benchmarks/, analysis/
tests/                   # pytest suite
notebooks/               # exploratory notebooks
```

## Configuration

- **`src/config.py` → `PACEConfig`** holds the architecture. It is saved into
  every checkpoint.
- **`src/pipelines/config.py` → `TrainingConfig`** holds the training loop (data
  paths, optimizer, SpecAugment, post-processing, callbacks), loaded from a YAML
  in `configs/` and overridable from the CLI.

## License

Released under the repository [LICENSE](LICENSE).
