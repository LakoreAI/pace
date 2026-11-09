"""Presentation helpers for the training entrypoint.

Kept separate from `src.pipelines.train` so the training loop stays readable and
this introspection can be reused (a debug script) or tested on its own. Nothing
here affects training — it only prints.

Output is rendered with `rich` when it is installed (panels, tables, JSON
highlighting) and falls back to aligned plain text otherwise. `rich` is an
optional extra: `uv sync --extra rich`.
"""

import json
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader, Dataset

from src.config import PACEConfig
from src.pipelines.config import TrainingConfig

try:
    from rich.console import Console
    from rich.json import JSON
    from rich.panel import Panel
    from rich.rule import Rule
    from rich.table import Table

    _HAS_RICH = True
except ImportError:  # pragma: no cover - exercised only without the extra
    _HAS_RICH = False

WIDTH = 72


def banner(title: str) -> None:
    print(f"\n{title}\n{'-' * WIDTH}")


def indent(text: str, spaces: int = 2) -> str:
    pad = " " * spaces
    return "\n".join(pad + line for line in text.splitlines())


def presence_targets(dataset: Dataset) -> Optional[torch.Tensor]:
    """The per-class presence slice of a dataset's targets, else None.

    one-pass targets are `(cells, 3 * C)` with presence every third column;
    frame-wise targets are already `(cells, C)`.
    """
    y = getattr(dataset, "y", None)
    if y is None:
        return None
    kind = getattr(dataset, "target_kind", "onepass")
    if kind == "onepass":
        return y[..., 0::3]
    if kind == "red_epn":
        return y[..., : y.shape[-1] // 5]
    return y


def presence_targets_batch(y: torch.Tensor, kind: str) -> torch.Tensor:
    if kind == "onepass":
        return y[:, 0::3]
    if kind == "red_epn":
        return y[:, : y.shape[-1] // 5]
    return y


def label_counts(dataset: Dataset) -> Optional[torch.Tensor]:
    """Per-class count of windows containing that class, else None."""
    presence = presence_targets(dataset)
    if presence is None:
        return None
    return (presence > 0).any(dim=1).sum(dim=0)


def announce_training(
    *,
    model: torch.nn.Module,
    model_cfg: PACEConfig,
    train_cfg: TrainingConfig,
    train_loader: DataLoader,
    train_dataset: Dataset,
    device: torch.device,
    run_name: str,
    ckpt_dir: Path,
    result_dir: Path,
    sample_rows: int = 3,
    sample_cols: int = 8,
) -> None:
    """Print configs, a data sample, and the model layout before training."""
    common = dict(
        model=model,
        model_cfg=model_cfg,
        train_cfg=train_cfg,
        train_loader=train_loader,
        train_dataset=train_dataset,
        device=device,
        run_name=run_name,
        ckpt_dir=ckpt_dir,
        result_dir=result_dir,
        sample_rows=sample_rows,
        sample_cols=sample_cols,
    )
    if _HAS_RICH:
        _announce_rich(**common)
    else:
        _announce_plain(**common)


def _announce_plain(
    *,
    model,
    model_cfg,
    train_cfg,
    train_loader,
    train_dataset,
    device,
    run_name,
    ckpt_dir,
    result_dir,
    sample_rows,
    sample_cols,
) -> None:
    print("\n" + "=" * WIDTH)
    print(f"RUN  {run_name}")
    print("=" * WIDTH)
    print(f"device:      {device}   seed: {train_cfg.seed}")
    print(f"checkpoints: {ckpt_dir}")
    print(f"results:     {result_dir}")

    banner("model config (PACEConfig)")
    print(indent(json.dumps(asdict(model_cfg), indent=2, default=str)))
    banner("training config (TrainingConfig)")
    print(indent(json.dumps(asdict(train_cfg), indent=2, default=str)))

    counts = label_counts(train_dataset)
    banner(
        f"data: train sample  ({len(train_dataset)} examples"
        + (f", {counts.numel()} classes)" if counts is not None else ")")
    )
    if counts is not None:
        summary = "  ".join(
            f"{cls}:{int(n)}" for cls, n in enumerate(counts) if int(n) > 0
        )
        print(f"class counts: {summary}")

    x, y = next(iter(train_loader))
    kind = getattr(train_dataset, "target_kind", "onepass")
    print(f"batch: x {tuple(x.shape)} {x.dtype}   y {tuple(y.shape)} {y.dtype}")
    rows = min(sample_rows, x.shape[0])
    cols = min(sample_cols, x.shape[-1])
    for r in range(rows):
        values = " ".join(f"{v:+.3f}" for v in x[r, 0, :cols].tolist())
        tail = " ..." if x.shape[-1] > cols else ""
        p = presence_targets_batch(y[r], kind)
        print(
            f"  x[{r}][0][:{cols}] = [{values}{tail}]   "
            f"active cells={int((p > 0).sum())}"
        )

    banner(f"model: {type(model).__name__}")
    total = 0
    for name, param in model.named_parameters():
        total += param.numel()
        print(f"  {name:<34s} {str(tuple(param.shape)):>16s}  {param.numel():>10,d}")
    print(f"  {'TOTAL':<34s} {'':>16s}  {total:>10,d}")
    largest = max(model.parameters(), key=lambda p: p.numel())
    print(
        f"initial weights of {largest.shape}: "
        f"mean={largest.mean():+.4f}  std={largest.std():.4f}"
    )

    banner("forward check (untrained)")
    print(_forward_check(model, x, device))

    print("\n" + "=" * WIDTH + "\n")


def _forward_check(model: torch.nn.Module, x: torch.Tensor, device) -> str:
    """A no-grad forward on a real batch; never raises."""
    try:
        was_training = model.training
        model.eval()
        with torch.no_grad():
            out = model(x.to(device))
        model.train(was_training)
        pred, feature = out["pred"], out["feature"]
        if isinstance(pred, dict):
            shapes = ", ".join(f"{k}{tuple(v.shape)}" for k, v in pred.items())
            return f"x {tuple(x.shape)} -> pred dict[{shapes}]  feature {tuple(feature.shape)}"
        preview = " ".join(
            f"{v:+.3f}" for v in pred[0, 0, : min(6, pred.shape[-1])].tolist()
        )
        return (
            f"x {tuple(x.shape)} -> pred {tuple(pred.shape)}  "
            f"feature {tuple(feature.shape)}\npred[0][0][:6] = [{preview}]"
        )
    except Exception as e:  # noqa: BLE001 - a shape peek must never block training
        return f"forward check skipped: {e}"


def _announce_rich(
    *,
    model,
    model_cfg,
    train_cfg,
    train_loader,
    train_dataset,
    device,
    run_name,
    ckpt_dir,
    result_dir,
    sample_rows,
    sample_cols,
) -> None:
    console = Console(highlight=False)
    console.print()
    console.print(Rule(f"[bold]RUN  {run_name}[/bold]"))

    info = Table.grid(padding=(0, 2))
    info.add_column(style="bold cyan", justify="right")
    info.add_column()
    info.add_row("device", str(device))
    info.add_row("seed", str(train_cfg.seed))
    info.add_row("checkpoints", str(ckpt_dir))
    info.add_row("results", str(result_dir))
    console.print(info)

    console.print(
        Panel(
            JSON(json.dumps(asdict(model_cfg), default=str)),
            title="model config (PACEConfig)",
            border_style="blue",
        )
    )
    console.print(
        Panel(
            JSON(json.dumps(asdict(train_cfg), default=str)),
            title="training config (TrainingConfig)",
            border_style="blue",
        )
    )

    counts = label_counts(train_dataset)
    kind = getattr(train_dataset, "target_kind", "onepass")
    x, y = next(iter(train_loader))
    caption = f"batch: x {tuple(x.shape)} {x.dtype}   y {tuple(y.shape)} {y.dtype}"
    title = f"data: train sample  ({len(train_dataset)} examples" + (
        f", {counts.numel()} classes)" if counts is not None else ")"
    )
    table = Table(title=title, border_style="green", caption=caption)
    rows = min(sample_rows, x.shape[0])
    cols = min(sample_cols, x.shape[-1])
    table.add_column("#", justify="right", style="dim")
    for c in range(cols):
        table.add_column(f"x{c}", justify="right")
    table.add_column("active cells", style="bold")
    for r in range(rows):
        p = presence_targets_batch(y[r], kind)
        table.add_row(
            str(r),
            *[f"{v:+.3f}" for v in x[r, 0, :cols].tolist()],
            str(int((p > 0).sum())),
        )
    if counts is not None:
        table.caption = (
            caption
            + "   "
            + "  ".join(f"{cls}:{int(n)}" for cls, n in enumerate(counts) if int(n) > 0)
        )
    console.print(table)

    model_table = Table(title=f"model: {type(model).__name__}", border_style="magenta")
    model_table.add_column("layer")
    model_table.add_column("shape", justify="right")
    model_table.add_column("params", justify="right")
    total = 0
    for name, param in model.named_parameters():
        total += param.numel()
        model_table.add_row(name, str(tuple(param.shape)), f"{param.numel():,d}")
    model_table.add_section()
    model_table.add_row("TOTAL", "", f"{total:,d}", style="bold")
    largest = max(model.parameters(), key=lambda p: p.numel())
    model_table.caption = (
        f"initial weights of {largest.shape}: "
        f"mean={largest.mean():+.4f}  std={largest.std():.4f}"
    )
    console.print(model_table)

    console.print(
        Panel(
            _forward_check(model, x, device),
            title="forward check (untrained)",
            border_style="yellow",
        )
    )
    console.print()
