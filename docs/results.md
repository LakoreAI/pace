# Results

Segment F (percent) on the URBAN-SED test split, mean ± sample standard
deviation over five seeds, running-gain peak normalisation.

| Model | Cells | Hidden | F |
|---|---|---|---|
| Causal, floor pooling | 63 | 128 | 63.78 ± 0.37 |
| Causal, ceil pooling | 63 | 112 | 64.10 ± 0.49 |
| Bidirectional | 63 | 128 | 64.38 ± 1.12 |
| Bidirectional, parameter-matched | 63 | 72 | 65.08 ± 0.49 |
| Causal | 126 | 128 | 64.01 ± 0.69 |
| Bidirectional | 126 | 128 | 65.23 ± 0.58 |
| Bidirectional, parameter-matched | 126 | 83 | 64.92 ± 0.22 |

## Limited-future GRU

| Future cells K | F | Formula delay |
|---|---|---|
| 3 | 65.11 ± 0.43 | 1.54 s |
| 6 | 65.23 ± 0.49 | 2.50 s |
| 12 | 64.76 ± 0.26 | 4.42 s |

The delay of these models follows the formula `0.583 + 2KS` s and was not
replayed.

## Reading the numbers

- Against a parameter-matched bidirectional model the causal penalty is about
  one F point.
- Where the difference is smaller than the seed spread, read it as
  indistinguishable.
- All claims are for URBAN-SED; they are not shown to hold on real recordings.

The full analysis is in the accompanying paper.
