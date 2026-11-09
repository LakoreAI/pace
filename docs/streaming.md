# Streaming evaluation

A causal model can be run offline over a whole window, but a deployed one sees
audio cell by cell. `src/sed/streaming.py` and
`scripts/analysis/evaluate_streaming.py` replay a checkpoint that way:

```bash
uv run python scripts/analysis/evaluate_streaming.py \
    --ckpt checkpoints/<run>/best.pt --data data/features/urbansed/test.npz \
    --lookaheads 0,0.25 --decoder framewise
```

Reported per look-ahead: the metric, commit delay (mean, median, p95), the
number of events still open when the window ends, and the difference from the
offline score.

## Commit delay

An event can be reported once the model has seen its end plus the look-ahead.
For a frame-wise model on the 63-cell grid the delay is roughly
`2S + L + 0.013 s`, where `S` is the cell length and `L` the look-ahead.

| Look-ahead L | Delay |
|---|---|
| 0 s | 0.333 s |
| 0.25 s | 0.583 s |
| 0.35 s | 0.683 s |
| 0.45 s | 0.783 s |

PSDS is retrospective. Report it together with the measured delay and the
forced-at-end count, never as a latency-aware metric. Do not compare an
offline model with a streaming one without stating the look-ahead.
