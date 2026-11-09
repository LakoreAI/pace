# Models and configs

Architecture lives in `src/config.py` (`PACEConfig`) and is saved in every
checkpoint. Training settings live in `src/pipelines/config.py`
(`TrainingConfig`), loaded from a YAML file and overridable from the CLI.

## Base configs

Each config explains its parameters in place.

| Config | Use |
|---|---|
| `configs/onepass_urbansed.yaml` | one-pass baseline on URBAN-SED |
| `configs/onepass_tut2017.yaml` | one-pass baseline on one TUT 2017 fold |
| `configs/urbansed_causal.yaml` | causal CRNN with a frame-wise head and streaming evaluation |
| `configs/urbansed_streaming_red_epn.yaml` | causal RED + onset/offset + event-proposal head |

## Switching variants (`arch:` block)

| Parameter | Effect |
|---|---|
| `causal: false` | bidirectional reference |
| `lookahead_frames` | input frames of future audio a causal model may use (25 is about 0.25 s) |
| `output_stride` and `n_cells` | output grid: 16 and 63 cells give 160 ms cells, 8 and 126 give 80 ms |
| `context_hidden` | GRU width; use a smaller value to match the parameter count of a bidirectional model |
| `context: gru_lc`, `context_future_cells` | forward GRU on the full history plus a backward GRU over K future cells per layer |
| `causal_pool_freq_ceil: true` | round frequency pooling up, giving a wider frequency axis |
| `epn_type: conv or gru` | convolutional or bidirectional-GRU proposal network |

With two recurrent layers, `gru_lc` reaches up to 2K cells into the future.
