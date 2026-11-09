"""SED evaluation: loss over windows, then events per file and sed_eval metrics.

The per-pass work lives in `eval_per_epoch`, which runs the model over one
window dataset, decodes and merges events per file, and returns a plain metrics
dict. `evaluate` is the public entry point: it calls `eval_per_epoch` and
optionally persists the result.
"""

import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data import WindowDataset
from src.modules.loss import build_loss
from src.sed.metrics import segment_metrics
from src.sed.postprocess import decode_file_events
from src.utils.io_utils import save_json

DEFAULT_POSTPROCESS = {"threshold": 0.5, "merge_gap": 1.0, "end_mode": "notebook"}


@torch.no_grad()
def predict_dataset(
    model: torch.nn.Module,
    dataset: WindowDataset,
    device: torch.device,
    batch_size: int = 64,
    criterion: Optional[torch.nn.Module] = None,
) -> tuple[object, bool, Optional[float]]:
    """Run the model once over a dataset.

    Returns `(predictions, is_dict, loss)`; dict predictions are the event
    head's maps, otherwise a `(N, T, ...)` array. `loss` is None without a
    criterion.
    """
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    preds_list, losses = [], []
    is_dict = False
    for x, y in loader:
        out = model(x.to(device))["pred"]
        if isinstance(out, dict):
            is_dict = True
            out = {key: value.float() for key, value in out.items()}
            preds_list.append({key: value.cpu().numpy() for key, value in out.items()})
        else:
            out = out.float()
            preds_list.append(out.cpu().numpy())
        if criterion is not None:
            losses.append(criterion(out, y.to(device)).item() * len(x))
    if not preds_list:
        return ({} if is_dict else np.empty((0,)), is_dict, None)
    if is_dict:
        preds = {
            key: np.concatenate([batch[key] for batch in preds_list])
            for key in preds_list[0]
        }
    else:
        preds = np.concatenate(preds_list)
    loss = float(np.sum(losses) / len(dataset)) if losses else None
    return preds, is_dict, loss


def group_windows(dataset: WindowDataset) -> dict[int, list[int]]:
    """File index -> window indices, in cache order."""
    windows = defaultdict(list)
    for i, fi in enumerate(dataset.file_idx):
        windows[int(fi)].append(i)
    return dict(windows)


def slice_predictions(preds, index):
    if isinstance(preds, dict):
        return {key: value[index] for key, value in preds.items()}
    return preds[index]


@torch.no_grad()
def eval_per_epoch(
    model: torch.nn.Module,
    dataset: WindowDataset,
    device: torch.device,
    batch_size: int = 64,
    postprocess: Optional[dict] = None,
    decoder: Optional[dict] = None,
    time_resolution: float = 1.0,
    criterion: Optional[torch.nn.Module] = None,
    psds: Optional[dict] = None,
) -> Dict[str, object]:
    """Returns {"error_rate", "f_measure", "per_class_f", "loss", "postprocess_s"}.

    `postprocess_s` is the wall time of decoding + merging alone (CPU), kept
    separate from the forward pass for the latency analysis. `decoder` is a
    name or {"type": name, ...} from `src.sed.postprocess.DECODERS`; it
    defaults to the dataset's target kind.
    """
    model.eval()
    pp = {**DEFAULT_POSTPROCESS, **(postprocess or {})}
    if decoder is None:
        default = {
            "onepass": "onepass",
            "framewise": "framewise",
            "red_epn": "red_epn",
        }
        decoder = {"type": default.get(dataset.target_kind, "framewise")}
    criterion = criterion or build_loss(None, dataset.num_classes, dataset.target_kind)
    preds, _, loss = predict_dataset(
        model, dataset, device, batch_size=batch_size, criterion=criterion
    )
    if isinstance(preds, dict) and not preds:
        return {
            "error_rate": float("nan"),
            "f_measure": float("nan"),
            "per_class_f": {},
            "loss": float("nan"),
            "postprocess_s": 0.0,
        }
    if not isinstance(preds, dict) and len(preds) == 0:
        return {
            "error_rate": float("nan"),
            "f_measure": float("nan"),
            "per_class_f": {},
            "loss": float("nan"),
            "postprocess_s": 0.0,
        }

    files = dataset.meta["files"]
    windows = group_windows(dataset)
    slice_preds = lambda index: slice_predictions(preds, index)  # noqa: E731

    t0 = time.perf_counter()
    estimates = {
        files[fi]: decode_file_events(
            decoder,
            slice_preds(idx),
            dataset.win_start[idx],
            dataset.classes,
            dataset.features.win_len,
            **pp,
        )
        for fi, idx in windows.items()
    }
    postprocess_s = time.perf_counter() - t0

    references = dataset.references()
    result = segment_metrics(references, estimates, dataset.classes, time_resolution)
    result["loss"] = float(loss) if loss is not None else float("nan")
    result["postprocess_s"] = postprocess_s

    if psds and psds.get("enabled", True):
        from src.sed.psds import event_based_metrics, merge_window_scores, psds_metrics

        params = psds.get("params")
        file_scores = {
            files[fi]: merge_window_scores(
                slice_preds(idx),
                dataset.win_start[idx],
                dataset.features.win_len,
                dataset.target_kind,
            )
            for fi, idx in windows.items()
        }
        result.update(
            psds_metrics(
                references,
                file_scores,
                dataset.durations(),
                dataset.classes,
                params=params,
            )
        )
        result.update(
            event_based_metrics(
                references,
                estimates,
                dataset.classes,
                collar=float(psds.get("event_collar", 0.2)),
            )
        )
    return result


def evaluate(
    model: torch.nn.Module,
    dataset: WindowDataset,
    device: torch.device,
    batch_size: int = 64,
    postprocess: Optional[dict] = None,
    decoder: Optional[dict] = None,
    time_resolution: float = 1.0,
    save_json_path: Optional[Union[str, Path]] = None,
) -> Dict[str, object]:
    """Evaluate `model` and optionally write the metrics dict to JSON."""
    result = eval_per_epoch(
        model,
        dataset,
        device,
        batch_size,
        postprocess,
        decoder,
        time_resolution,
    )
    if save_json_path is not None:
        save_json(result, save_json_path)
    return result


def format_report(result: Dict[str, object]) -> str:
    lines = [
        f"ER={result['error_rate']:.4f}  "
        f"F={result['f_measure'] * 100:6.2f}  "
        f"loss={result['loss']:.4f}"
    ]
    for label, f in result["per_class_f"].items():
        lines.append(f"  {label:<20s} F={f * 100:6.2f}")
    return "\n".join(lines)
