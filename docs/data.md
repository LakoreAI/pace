# Data

Audio and features are never committed. Extract log-mel windows and targets
into one cache file per split:

```bash
# URBAN-SED predefined splits
uv run python scripts/data/extract_features.py --dataset urbansed \
    --root <path>/URBAN-SED_v2.0.0 --out data/features/urbansed

# TUT Sound Events 2017, one fold (repeat for folds 2-4)
uv run python scripts/data/extract_features.py --dataset tut2017 --fold 1 \
    --root <path>/TUT-sound-events-2017-development --out data/features/tut2017_fold1
```

Each `<split>.npz` holds the windows (fp16), window events, file index, start
times and a JSON `meta` entry with classes, reference events and feature
settings. Targets are encoded at load time from `target_kind`.

Other `--dataset` values:

| Value | Input |
|---|---|
| `manifest` | CSV/TSV with `path,onset,offset,label[,split]` |
| `desed` | SCAPER JAMS synthetic scenes plus strong-label TSV val/test |
| `audioset_strong` | official CSV plus downloaded clips, with a missing-clip report |

Choose the front end with `--features {tut2017,urbansed,music_speech,voice,desed}`.

## URBAN-SED

URBAN-SED v2.0.0 is redistributed on Hugging Face as
[`LakoreAI/URBAN-SED`](https://huggingface.co/datasets/LakoreAI/URBAN-SED).
All credit belongs to the original authors (Salamon et al.), and it keeps its
CC BY 4.0 license.
