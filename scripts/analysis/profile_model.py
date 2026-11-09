"""Profile a model: parameters, MACs, forward latency and decode latency.

Works from a checkpoint (architecture rebuilt from it) or from a config/feature
preset for an untrained model. Latency is wall-clock on the current device;
report it with the device name and never compare across hardware.

Usage:
    uv run python scripts/analysis/profile_model.py --ckpt checkpoints/<run>/best.pt
    uv run python scripts/analysis/profile_model.py --features tut2017 --batch_size 1
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.config import PACEConfig  # noqa: E402
from src.data import TUT2017_CLASSES  # noqa: E402
from src.features import PRESETS  # noqa: E402
from src.modules.model import build_model  # noqa: E402
from src.pipelines.eval import DEFAULT_POSTPROCESS  # noqa: E402
from src.pipelines.infer import config_from_checkpoint  # noqa: E402
from src.sed.postprocess import decode_file_events  # noqa: E402
from src.utils.efficiency import count_macs, measure_latency  # noqa: E402
from src.utils.io_utils import save_json  # noqa: E402
from src.utils.model_utils import count_parameters, detect_device  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", type=Path, default=None)
    parser.add_argument("--features", choices=sorted(PRESETS), default="tut2017")
    parser.add_argument(
        "--head", choices=("onepass", "framewise", "red_epn"), default="onepass"
    )
    parser.add_argument("--epn_type", choices=("conv", "gru"), default="conv")
    parser.add_argument("--epn_hidden", type=int, default=64)
    parser.add_argument("--causal", action="store_true")
    parser.add_argument("--num_classes", type=int, default=len(TUT2017_CLASSES))
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--device", default=None)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()

    device = torch.device(args.device) if args.device else detect_device()
    if args.ckpt:
        raw = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        cfg = config_from_checkpoint(raw)
    else:
        params = PRESETS[args.features]
        cfg = PACEConfig(
            num_classes=args.num_classes,
            n_frames=params.n_frames,
            n_mels=params.n_mels,
            head=args.head,
            epn_type=args.epn_type,
            epn_hidden=args.epn_hidden,
            causal=args.causal,
        )
    model = build_model(cfg).to(device)
    params = PRESETS[args.features]
    sample = torch.randn(args.batch_size, cfg.n_frames, cfg.n_mels, device=device)

    forward = measure_latency(model, sample, repeats=args.repeats)
    macs = count_macs(model, sample)
    sec_per_batch = forward["median_ms"] / 1000.0
    audio_s = args.batch_size * params.win_len
    result = {
        "device": str(device),
        "params": count_parameters(model, verbose=False),
        "macs_per_window": macs // max(args.batch_size, 1),
        "forward": forward,
        "realtime_factor": sec_per_batch / audio_s if audio_s else None,
        "windows_per_s": (args.batch_size / sec_per_batch) if sec_per_batch else None,
        "features": params.to_dict(),
        "model_cfg": cfg.__dict__,
    }

    preds = np.random.default_rng(0).random((32, cfg.n_cells, 3 * cfg.num_classes))
    preds = preds.astype(np.float32)
    t0 = time.perf_counter()
    decode_file_events(
        {"type": "onepass"},
        preds,
        np.arange(32) * params.hop_len,
        [f"class{i}" for i in range(cfg.num_classes)],
        params.win_len,
        **DEFAULT_POSTPROCESS,
    )
    result["decode_ms_per_32_windows"] = (time.perf_counter() - t0) * 1000.0

    print(json.dumps(result, indent=2, default=str))
    if args.json:
        save_json(result, args.json)
        print(f"saved: {args.json}")


if __name__ == "__main__":
    main()
