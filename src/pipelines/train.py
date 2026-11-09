"""PACEModel training entrypoint.

Trains on cached log-mel windows with the original one-pass detector loss plus its weight-decay terms.
Validation decodes events per file and scores them with sed_eval; checkpoint
selection and early stopping see the validation split only. The test split is
scored once, with the best checkpoint, after training. Controlled via a YAML
config (see configs/onepass_tut2017.yaml) with individual CLI overrides.

Usage:
    uv run python -m src.pipelines.train --config configs/onepass_tut2017.yaml
    uv run python -m src.pipelines.train --config configs/onepass_tut2017.yaml --epochs 50
"""

import argparse
import math
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.callbacks import (
    BestCheckpoint,
    EarlyStopping,
    EMACallback,
    TrainContext,
    TrainerState,
    build_lr_scheduler,
)
from src.callbacks.wandb_callback import WandbCallback
from src.config import PACEConfig
from src.data import WindowDataset, load_cache, resolve
from src.modules.augment import apply_recipe, spec_augment_batch
from src.modules.loss import build_loss
from src.modules.model import SEDModel, build_model as build_sed_model
from src.pipelines._utils import announce_training
from src.pipelines.config import TrainingConfig, load_training_config
from src.pipelines.eval import eval_per_epoch, format_report
from src.utils.efficiency import count_macs, measure_latency
from src.utils.io_utils import env_info, git_info, load_env, save_checkpoint, save_json
from src.utils.model_utils import (
    autocast_dtype,
    count_parameters,
    detect_device,
    get_run_name,
    rng_state,
    seed_everything,
    seed_worker,
    set_rng_state,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def build_model(model_cfg: PACEConfig, device: torch.device) -> SEDModel:
    return build_sed_model(model_cfg).to(device)


def build_callbacks(
    train_cfg: TrainingConfig,
    optimizer: torch.optim.Optimizer,
    run_name: str,
    wandb_config: dict,
    wandb_resume_id: Optional[str] = None,
):
    callbacks = []

    lr_cb = build_lr_scheduler(optimizer, train_cfg.lr_scheduler)
    if lr_cb is not None:
        callbacks.append(lr_cb)

    if train_cfg.early_stopping and train_cfg.early_stopping.get("enabled", True):
        es_kwargs = {
            k: v for k, v in train_cfg.early_stopping.items() if k != "enabled"
        }
        callbacks.append(EarlyStopping(**es_kwargs))

    if train_cfg.save_best:
        callbacks.append(
            BestCheckpoint(monitor=train_cfg.best_metric, mode=train_cfg.best_mode)
        )

    if train_cfg.wandb and train_cfg.wandb.get("enabled", False):
        wb = train_cfg.wandb
        callbacks.append(
            WandbCallback(
                project_name=wb.get("project", "pace"),
                run_name=run_name,
                config=wandb_config,
                entity=wb.get("entity"),
                monitor=wb.get("monitor", train_cfg.best_metric),
                mode=wb.get("mode", train_cfg.best_mode),
                log_artifacts=wb.get("log_artifacts", True),
                group=wb.get("group"),
                resume_id=wandb_resume_id,
            )
        )

    if train_cfg.ema:
        callbacks.append(EMACallback(**train_cfg.ema))

    return callbacks


def _load_callback_states(callbacks, states) -> None:
    """Match saved states to callbacks by position and type."""
    if len(states) != len(callbacks):
        raise ValueError(
            f"checkpoint has {len(states)} callback states, run has {len(callbacks)} "
            "callbacks; resume with the same callback config"
        )
    for cb, saved in zip(callbacks, states):
        if saved["type"] != type(cb).__name__:
            raise ValueError(
                f"callback mismatch on resume: {saved['type']} vs {type(cb).__name__}"
            )
        cb.load_state_dict(saved["state"])


def build_loaders(
    train_cfg: TrainingConfig,
    device: torch.device,
) -> tuple[DataLoader, list[WindowDataset], Optional[WindowDataset], WindowDataset]:
    """Returns (train_loader, val_datasets, test_dataset, train_dataset).

    Evaluation consumes whole datasets (events are merged per file), so only
    the training split gets a loader here. `val_files` (if set) gives several
    disjoint validation caches; metrics are averaged over them for selection.
    """
    train_path = resolve(train_cfg.data_root, train_cfg.train_file)
    if train_path is None:
        raise FileNotFoundError(
            f"train split not found: {Path(train_cfg.data_root) / str(train_cfg.train_file)}"
        )
    val_names = train_cfg.val_files or [train_cfg.val_file]
    val_paths = []
    for name in val_names:
        path = resolve(train_cfg.data_root, name)
        if path is None:
            raise FileNotFoundError(
                f"val split not found: {Path(train_cfg.data_root) / str(name)}"
            )
        val_paths.append(path)

    target_kind = train_cfg.target_kind
    n_cells = train_cfg.n_cells
    train_dataset = load_cache(train_path, target_kind=target_kind, n_cells=n_cells)
    val_datasets = [
        load_cache(p, target_kind=target_kind, n_cells=n_cells) for p in val_paths
    ]
    for val_dataset in val_datasets:
        if val_dataset.classes != train_dataset.classes:
            raise ValueError("train and val caches have different class lists")

    test_path = resolve(train_cfg.data_root, train_cfg.test_file)
    test_dataset = (
        load_cache(test_path, target_kind=target_kind, n_cells=n_cells)
        if test_path is not None
        else None
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=train_cfg.num_workers,
        pin_memory=train_cfg.pin_memory and device.type == "cuda",
        worker_init_fn=seed_worker,
        generator=torch.Generator().manual_seed(train_cfg.seed),
    )
    return train_loader, val_datasets, test_dataset, train_dataset


def _optimizer_step(
    optimizer, scaler, callbacks, ctx, pending_losses, step, epoch, log_every
):
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad()
    step += 1
    avg_loss = sum(pending_losses) / len(pending_losses)
    state = TrainerState(step=step, train_loss=avg_loss, epoch=epoch)
    for cb in callbacks:
        cb.on_step_end(ctx, state)
    if step % log_every == 0 or step == 1:
        print(f"  epoch {epoch:3d}  step {step:5d}  loss={avg_loss:.4f}")
    return step, avg_loss


def train_per_epoch(
    model,
    criterion,
    loader,
    optimizer,
    scaler,
    accum_steps,
    device,
    callbacks,
    ctx,
    step,
    epoch,
    log_every,
    spec_augment=None,
    recipe=None,
    amp_dtype=None,
):
    model.train()
    epoch_losses, pending = [], []
    optimizer.zero_grad()

    for x, target in loader:
        x = x.float().to(device)
        target = target.to(device)
        if spec_augment:
            x = spec_augment_batch(x, **spec_augment)
        if recipe:
            x, target = apply_recipe(x, target, recipe)

        with torch.autocast(
            device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None
        ):
            pred = model(x)["pred"]
        if isinstance(pred, dict):
            pred = {key: value.float() for key, value in pred.items()}
        else:
            pred = pred.float()
        loss = criterion(pred, target) + model.regularization_loss()

        scaler.scale(loss / accum_steps).backward()
        pending.append(loss.item())

        if len(pending) == accum_steps:
            step, avg_loss = _optimizer_step(
                optimizer, scaler, callbacks, ctx, pending, step, epoch, log_every
            )
            epoch_losses.append(avg_loss)
            pending = []

    if pending:
        step, avg_loss = _optimizer_step(
            optimizer, scaler, callbacks, ctx, pending, step, epoch, log_every
        )
        epoch_losses.append(avg_loss)

    return (
        sum(epoch_losses) / len(epoch_losses) if epoch_losses else float("nan"),
        step,
    )


def aggregate_results(results: list[dict]) -> dict:
    """Average validation metrics over disjoint splits for checkpoint selection."""
    if len(results) == 1:
        return results[0]
    numeric = [
        key
        for key, value in results[0].items()
        if isinstance(value, (int, float)) and key != "per_class_f"
    ]
    out = {key: float(np.mean([r[key] for r in results])) for key in numeric}
    per_class = {}
    for label in results[0]["per_class_f"]:
        values = [r["per_class_f"][label] for r in results]
        finite = [v for v in values if math.isfinite(v)]
        per_class[label] = float(np.mean(finite)) if finite else float("nan")
    out["per_class_f"] = per_class
    return out


def train(train_cfg: TrainingConfig):
    load_env(REPO_ROOT / ".env")
    device = detect_device()
    print(f"device: {device}")

    seed_everything(train_cfg.seed)
    resume = (
        torch.load(train_cfg.resume_from, map_location=str(device), weights_only=False)
        if train_cfg.resume_from
        else None
    )

    train_loader, val_datasets, test_dataset, train_dataset = build_loaders(
        train_cfg, device
    )
    num_classes = train_dataset.num_classes
    print(
        f"classes: {num_classes}  input: {train_dataset.n_frames} x {train_dataset.n_mels}"
        f"  target={train_cfg.target_kind}"
    )

    model_cfg = PACEConfig(
        num_classes=num_classes,
        n_frames=train_dataset.n_frames,
        n_mels=train_dataset.n_mels,
        **(train_cfg.arch or {}),
    )
    if train_dataset.n_cells != model_cfg.n_cells:
        raise ValueError(
            f"target grid mismatch: data encodes {train_dataset.n_cells} cells, "
            f"architecture produces {model_cfg.n_cells}; set n_cells in the "
            "training config to the architecture's grid"
        )
    model = build_model(model_cfg, device)
    criterion = build_loss(train_cfg.loss, num_classes, train_cfg.target_kind).to(
        device
    )
    decoder = train_cfg.decoder or {
        "type": (
            "onepass"
            if train_cfg.target_kind == "onepass"
            else "red_epn"
            if train_cfg.target_kind == "red_epn"
            else "framewise"
        )
    }
    eval_kwargs = dict(
        batch_size=train_cfg.eval_batch_size,
        postprocess=train_cfg.postprocess,
        decoder=decoder,
        time_resolution=train_cfg.time_resolution,
        criterion=criterion,
        psds=train_cfg.psds,
    )

    optimizer_cls = (
        torch.optim.AdamW if train_cfg.optimizer == "adamw" else torch.optim.Adam
    )
    optimizer = optimizer_cls(
        model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay
    )
    use_amp = train_cfg.amp and device.type == "cuda"
    amp_dtype = autocast_dtype(train_cfg.amp_dtype, device) if use_amp else None
    scaler = torch.amp.GradScaler(
        "cuda" if device.type == "cuda" else "cpu",
        enabled=amp_dtype == torch.float16,
    )

    dataset_tag = train_cfg.dataset_family
    run_name = (
        train_cfg.run_name
        or (resume["extra"].get("run_name") if resume else None)
        or get_run_name("pace", dataset_tag, train_cfg.lr, train_cfg.batch_size)
    )
    ckpt_dir = Path(train_cfg.ckpt_dir) / run_name
    result_dir = Path(train_cfg.result_dir) / run_name

    callbacks = build_callbacks(
        train_cfg,
        optimizer,
        run_name,
        {**asdict(model_cfg), **asdict(train_cfg)},
        wandb_resume_id=resume["extra"].get("wandb_run_id") if resume else None,
    )
    early_stoppers = [cb for cb in callbacks if isinstance(cb, EarlyStopping)]

    wandb_run_id = None
    for cb in callbacks:
        if isinstance(cb, WandbCallback) and cb.enabled and cb.run is not None:
            wandb_run_id = cb.run.id

    run_meta = {
        "train_cfg": asdict(train_cfg),
        **git_info(REPO_ROOT),
        **env_info(),
    }
    ctx = TrainContext(
        model=model,
        optimizer=optimizer,
        ckpt_dir=ckpt_dir,
        cfg_dict=asdict(model_cfg),
        run_meta=run_meta,
        wandb_run_id=wandb_run_id,
    )
    for cb in callbacks:
        cb.on_train_start(ctx)

    announce_training(
        model=model,
        model_cfg=model_cfg,
        train_cfg=train_cfg,
        train_loader=train_loader,
        train_dataset=train_dataset,
        device=device,
        run_name=run_name,
        ckpt_dir=ckpt_dir,
        result_dir=result_dir,
    )

    history = []
    if resume is not None:
        model.load_state_dict(resume["model"])
        optimizer.load_state_dict(resume["optimizer"])
        extra = resume["extra"]
        if "scaler" in extra:
            scaler.load_state_dict(extra["scaler"])
        _load_callback_states(callbacks, extra.get("callbacks", []))
        if "rng" in extra:
            set_rng_state(extra["rng"])
        if "loader_rng" in extra:
            train_loader.generator.set_state(extra["loader_rng"].cpu())
        history = list(extra.get("history", []))
        step = int(resume.get("step", 0))
        start_epoch = int(extra.get("epoch", 0)) + 1
        print(f"resumed at step {step}, continuing from epoch {start_epoch}")
    else:
        step, start_epoch = 0, 1

    stopped_early = any(es.should_stop for es in early_stoppers)
    last_epoch = train_cfg.epochs if not stopped_early else start_epoch - 1
    epoch = start_epoch - 1
    for epoch in range(start_epoch, last_epoch + 1):
        train_loss, step = train_per_epoch(
            model,
            criterion,
            train_loader,
            optimizer,
            scaler,
            train_cfg.accum_steps,
            device,
            callbacks,
            ctx,
            step,
            epoch,
            train_cfg.log_every,
            train_cfg.spec_augment,
            train_cfg.recipe,
            amp_dtype,
        )
        record = {"epoch": epoch, "step": step, "loss": train_loss}
        print(f"epoch {epoch:3d}/{train_cfg.epochs}  train_loss={train_loss:.4f}")

        should_validate = epoch % train_cfg.eval_every == 0 or epoch == last_epoch
        if should_validate:
            for cb in callbacks:
                cb.on_validation_start(ctx)
            split_results = [
                eval_per_epoch(model, ds, device, **eval_kwargs) for ds in val_datasets
            ]
            val_result = aggregate_results(split_results)
            extra = {
                key: value
                for key, value in val_result.items()
                if isinstance(value, (int, float))
            }
            record.update(extra)
            if len(split_results) > 1:
                record["val_splits"] = split_results
            print(format_report(val_result))
            state = TrainerState(
                step=step,
                train_loss=train_loss,
                epoch=epoch,
                val_loss=val_result["loss"],
                extra=extra,
            )
            record["val_loss"] = state.val_loss
            for cb in callbacks:
                cb.on_validation_end(ctx, state)
            if any(es.should_stop for es in early_stoppers):
                stopped_early = True

        history.append(record)

        extra = {
            "cfg": asdict(model_cfg),
            **run_meta,
            "epoch": epoch,
            "run_name": run_name,
            "wandb_run_id": ctx.wandb_run_id,
        }
        save_checkpoint(
            model,
            optimizer,
            step,
            ckpt_dir / "last.pt",
            extra={
                **extra,
                "scaler": scaler.state_dict(),
                "callbacks": [
                    {"type": type(cb).__name__, "state": cb.state_dict()}
                    for cb in callbacks
                ],
                "rng": rng_state(),
                "loader_rng": train_loader.generator.get_state(),
                "history": history,
            },
        )
        if epoch % train_cfg.ckpt_every == 0 or epoch == train_cfg.epochs:
            ckpt_path = ckpt_dir / f"epoch_{epoch}.pt"
            save_checkpoint(model, optimizer, step, ckpt_path, extra=extra)
            print(f"  saved checkpoint: {ckpt_path}")

        if stopped_early:
            break

    for cb in callbacks:
        cb.on_train_end(ctx)

    best_path = ckpt_dir / "best.pt"
    best = (
        torch.load(best_path, map_location=str(device), weights_only=False)
        if best_path.exists()
        else None
    )
    test_result = None
    streaming_result = None
    if test_dataset is not None:
        if best is not None:
            model.load_state_dict(best["model"])
        test_result = eval_per_epoch(model, test_dataset, device, **eval_kwargs)
        print(f"test ({'best.pt' if best is not None else 'last epoch'}):")
        print(format_report(test_result))
        save_json(test_result, result_dir / "test_metrics.json")
        if train_cfg.streaming and train_cfg.streaming.get("enabled", True):
            from src.sed.streaming import evaluate_streaming, format_streaming

            lookaheads = train_cfg.streaming.get("lookaheads", [0.0, 0.25, 0.5, 1.0])
            streaming_result = evaluate_streaming(
                model,
                test_dataset,
                device,
                lookaheads,
                decoder,
                train_cfg.postprocess,
                batch_size=train_cfg.eval_batch_size,
                time_resolution=train_cfg.time_resolution,
            )
            print("streaming (test):")
            print(format_streaming(streaming_result))
            save_json(
                {str(k): v for k, v in streaming_result.items()},
                result_dir / "test_streaming.json",
            )

    efficiency = None
    if train_cfg.profile:
        sample = torch.randn(1, train_dataset.n_frames, train_dataset.n_mels)
        sample = sample.to(device)
        macs = count_macs(model, sample)
        latency = measure_latency(model, sample, repeats=train_cfg.profile_repeats)
        efficiency = {
            "device": str(device),
            "macs_per_window": macs,
            "latency": latency,
            "features": train_dataset.features.to_dict(),
        }
        print(f"efficiency: {macs:,d} MACs  {latency['median_ms']:.2f} ms/window")

    save_json(history, ckpt_dir / "train_log.json")
    save_json(history, result_dir / "train_log.json")

    summary_path = Path(train_cfg.result_dir) / train_cfg.result_filename()
    save_json(
        {
            "schema_version": 2,
            "experiment": train_cfg.experiment,
            "dataset": dataset_tag,
            "fold": train_cfg.fold,
            "seed": train_cfg.seed,
            "run_name": run_name,
            **git_info(REPO_ROOT),
            **env_info(),
            "model_cfg": asdict(model_cfg),
            "train_cfg": asdict(train_cfg),
            "benchmark": train_cfg.benchmark,
            "params": count_parameters(model, verbose=False),
            "efficiency": efficiency,
            "best_epoch": best["extra"].get("epoch") if best else None,
            "best_val": (
                {best["extra"]["monitor"]: best["extra"]["monitor_value"]}
                if best
                else None
            ),
            "epochs_run": epoch,
            "stopped_early": stopped_early,
            "test": test_result,
            "test_streaming": (
                {str(k): v for k, v in streaming_result.items()}
                if streaming_result
                else None
            ),
        },
        summary_path,
    )
    print(f"summary: {summary_path}")
    print(
        f"\n{'stopped early' if stopped_early else 'finished'} "
        f"at epoch {epoch} (step {step})"
    )
    return history


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="YAML, see configs/onepass_tut2017.yaml",
    )
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--train_file", type=str, default=None)
    parser.add_argument("--val_file", type=str, default=None)
    parser.add_argument(
        "--val_files", nargs="*", default=None, help="multiple validation caches"
    )
    parser.add_argument("--test_file", type=str, default=None)
    parser.add_argument("--ckpt_dir", type=str, default=None)
    parser.add_argument("--result_dir", type=str, default=None)
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--experiment", type=str, default=None)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--fold", type=int, default=None)
    parser.add_argument("--target_kind", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--accum_steps", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--log_every", type=int, default=None)
    parser.add_argument("--eval_every", type=int, default=None)
    parser.add_argument("--ckpt_every", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--resume_from", type=str, default=None)
    parser.add_argument("--amp", action="store_true", default=None)
    parser.add_argument("--amp_dtype", choices=("bf16", "fp16"), default=None)
    parser.add_argument("--pin_memory", action="store_true", default=None)
    args = parser.parse_args()

    overrides = {k: v for k, v in vars(args).items() if k != "config"}
    train_cfg = load_training_config(args.config, **overrides)
    train(train_cfg)
