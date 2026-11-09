"""Single-checkpoint inference: audio file -> events.

Loads a checkpoint, rebuilds the model from its saved architecture config,
windows the audio exactly as the feature cache does, and decodes + merges the
per-window outputs into `(onset, offset, label)` events.

Usage:
    uv run python -m src.pipelines.infer \\
        --ckpt checkpoints/<run>/best.pt --audio a001.wav --features tut2017 \\
        --classes "brakes squeaking,car,children,large vehicle,people speaking,people walking"
"""

import argparse
from dataclasses import fields
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import torch

from src.config import PACEConfig
from src.features import PRESETS, FeatureParams, load_mono, log_mel, window_audio
from src.modules.model import SEDModel, build_model
from src.pipelines.eval import DEFAULT_POSTPROCESS
from src.sed.events import Event
from src.sed.postprocess import decode_file_events
from src.utils.model_utils import detect_device


def config_from_checkpoint(ckpt: dict) -> PACEConfig:
    """Rebuild PACEConfig from a checkpoint's saved `extra["cfg"]`."""
    saved = ckpt.get("extra", {}).get("cfg", {})
    known = {f.name for f in fields(PACEConfig)}
    return PACEConfig(**{k: v for k, v in saved.items() if k in known})


def load_model(ckpt_path: Path, device: torch.device) -> tuple[SEDModel, PACEConfig]:
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")
    raw = torch.load(ckpt_path, map_location=str(device), weights_only=False)
    cfg = config_from_checkpoint(raw)
    model = build_model(cfg).to(device)
    model.load_state_dict(raw["model"])
    model.eval()
    return model, cfg


@torch.no_grad()
def infer(
    ckpt_path: Path,
    audio_path: Path,
    params: FeatureParams,
    classes: Sequence[str],
    postprocess: Optional[dict] = None,
    decoder: Optional[str] = None,
) -> List[Event]:
    device = detect_device()
    model, cfg = load_model(ckpt_path, device)
    if len(classes) != cfg.num_classes:
        raise ValueError(
            f"{len(classes)} class names given, model has {cfg.num_classes}"
        )
    if params.n_frames != cfg.n_frames or params.n_mels != cfg.n_mels:
        raise ValueError(
            f"features give {params.n_frames} x {params.n_mels}, "
            f"model expects {cfg.n_frames} x {cfg.n_mels}"
        )

    audio = load_mono(audio_path, params.sr)
    windows, starts = window_audio(audio, params.sr, params.win_len, params.hop_len)
    x = torch.as_tensor(np.stack([log_mel(w, params) for w in windows])).to(device)
    raw = model(x)["pred"]
    if isinstance(raw, dict):
        preds = {key: value.float().cpu().numpy() for key, value in raw.items()}
    else:
        preds = raw.float().cpu().numpy()
    decoder = decoder or ("red_epn" if cfg.head == "red_epn" else "onepass")

    pp = {**DEFAULT_POSTPROCESS, **(postprocess or {})}
    return decode_file_events(
        {"type": decoder}, preds, starts, classes, params.win_len, **pp
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--features", choices=sorted(PRESETS), default="tut2017")
    parser.add_argument("--classes", type=str, required=True, help="comma-separated")
    parser.add_argument("--decoder", default=None)
    args = parser.parse_args()
    events = infer(
        args.ckpt,
        args.audio,
        PRESETS[args.features],
        args.classes.split(","),
        decoder=args.decoder,
    )
    for onset, offset, label in events:
        print(f"{onset:.3f}\t{offset:.3f}\t{label}")
