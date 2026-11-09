"""Evaluate a trained checkpoint on a feature-cache split.

Reports segment-based ER / F (sed_eval) and per-class F, optionally writing a
JSON summary.

Usage:
    uv run python scripts/training/evaluate.py \
        --ckpt checkpoints/<run>/best.pt --data data/features/tut2017_fold1/test.npz
"""

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.data import load_cache  # noqa: E402
from src.pipelines.eval import DEFAULT_POSTPROCESS, evaluate, format_report  # noqa: E402
from src.pipelines.infer import load_model  # noqa: E402
from src.utils.model_utils import detect_device  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument(
        "--threshold", type=float, default=DEFAULT_POSTPROCESS["threshold"]
    )
    parser.add_argument(
        "--merge_gap", type=float, default=DEFAULT_POSTPROCESS["merge_gap"]
    )
    parser.add_argument(
        "--end_mode",
        choices=("notebook", "cell"),
        default=DEFAULT_POSTPROCESS["end_mode"],
    )
    parser.add_argument("--time_resolution", type=float, default=1.0)
    parser.add_argument(
        "--target_kind",
        default=None,
        help="default: infer from the checkpoint head (onepass/framewise/red_epn)",
    )
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()

    device = detect_device()
    model, model_cfg = load_model(args.ckpt, device)
    target_kind = args.target_kind or (
        model_cfg.head
        if model_cfg.head in ("onepass", "framewise", "red_epn")
        else "onepass"
    )
    dataset = load_cache(args.data, target_kind=target_kind)
    result = evaluate(
        model,
        dataset,
        device,
        batch_size=args.batch_size,
        postprocess={
            "threshold": args.threshold,
            "merge_gap": args.merge_gap,
            "end_mode": args.end_mode,
        },
        time_resolution=args.time_resolution,
        save_json_path=args.json,
    )
    print(format_report(result))

    if args.json:
        print(f"saved: {args.json}")


if __name__ == "__main__":
    main()
