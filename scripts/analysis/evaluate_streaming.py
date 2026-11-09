"""Evaluate a checkpoint under bounded look-ahead (streaming inference).

Reports event metrics and commit latency for each look-ahead. PSDS, when the
cache/config supports it, remains a retrospective score-quality measure; this
script reports what a streaming application can actually emit.

Usage:
    uv run python scripts/analysis/evaluate_streaming.py \
        --ckpt checkpoints/<run>/best.pt --data data/features/desed/test.npz \
        --lookaheads 0,0.1,0.25,0.5,1.0 --decoder red_epn
"""

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.data import load_cache  # noqa: E402
from src.pipelines.eval import DEFAULT_POSTPROCESS  # noqa: E402
from src.pipelines.infer import load_model  # noqa: E402
from src.sed.streaming import evaluate_streaming, format_streaming  # noqa: E402
from src.utils.io_utils import save_json  # noqa: E402
from src.utils.model_utils import detect_device  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--lookaheads", default="0,0.1,0.25,0.5,1.0")
    parser.add_argument("--decoder", default=None)
    parser.add_argument("--target_kind", default=None)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--time_resolution", type=float, default=1.0)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()

    lookaheads = [float(v) for v in args.lookaheads.split(",")]
    device = detect_device()
    model, cfg = load_model(args.ckpt, device)
    target_kind = args.target_kind or (
        cfg.head if cfg.head in ("onepass", "framewise", "red_epn") else "onepass"
    )
    decoder = args.decoder or ("red_epn" if target_kind == "red_epn" else target_kind)
    dataset = load_cache(args.data, target_kind=target_kind)
    postprocess = dict(DEFAULT_POSTPROCESS)
    if decoder == "red_epn":
        postprocess.pop("end_mode", None)
    results = evaluate_streaming(
        model,
        dataset,
        device,
        lookaheads,
        decoder,
        postprocess,
        batch_size=args.batch_size,
        time_resolution=args.time_resolution,
    )
    print(format_streaming(results))
    if args.json:
        save_json({str(k): v for k, v in results.items()}, args.json)
        print(f"saved: {args.json}")


if __name__ == "__main__":
    main()
