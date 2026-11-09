import torch

from src.callbacks import (
    BestCheckpoint,
    EarlyStopping,
    EMACallback,
    LRSchedulerCallback,
    TrainContext,
    TrainerState,
    build_lr_scheduler,
)


class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(2, 2)


def make_ctx(tmp_path):
    model = Tiny()
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    return model, opt, TrainContext(model=model, optimizer=opt, ckpt_dir=tmp_path)


def test_best_checkpoint_saves_and_improves(tmp_path):
    model, opt, ctx = make_ctx(tmp_path)
    cb = BestCheckpoint(monitor="auc", mode="max")
    cb.on_validation_end(ctx, TrainerState(step=1, train_loss=1.0, extra={"auc": 0.5}))
    assert (tmp_path / "best.pt").exists()
    first_mtime = (tmp_path / "best.pt").stat().st_mtime_ns
    cb.on_validation_end(ctx, TrainerState(step=2, train_loss=1.0, extra={"auc": 0.4}))
    assert (tmp_path / "best.pt").stat().st_mtime_ns == first_mtime
    cb.on_validation_end(ctx, TrainerState(step=3, train_loss=1.0, extra={"auc": 0.9}))
    assert cb.best == 0.9


def test_early_stopping_triggers(tmp_path):
    _, _, ctx = make_ctx(tmp_path)
    es = EarlyStopping(monitor="auc", mode="max", patience=2)
    es.on_validation_end(ctx, TrainerState(step=1, train_loss=1.0, extra={"auc": 0.5}))
    assert not es.should_stop
    es.on_validation_end(ctx, TrainerState(step=2, train_loss=1.0, extra={"auc": 0.4}))
    assert not es.should_stop
    es.on_validation_end(ctx, TrainerState(step=3, train_loss=1.0, extra={"auc": 0.3}))
    assert es.should_stop


def test_cosine_scheduler_respects_start_epoch(tmp_path):
    model, opt, ctx = make_ctx(tmp_path)
    cb = build_lr_scheduler(
        opt, {"type": "cosine", "t_max": 10, "eta_min": 0.0, "start_epoch": 2}
    )
    assert isinstance(cb, LRSchedulerCallback)
    initial = opt.param_groups[0]["lr"]
    opt.step()
    cb.on_step_end(ctx, TrainerState(step=1, train_loss=1.0, epoch=0))
    assert opt.param_groups[0]["lr"] == initial
    opt.step()
    cb.on_step_end(ctx, TrainerState(step=2, train_loss=1.0, epoch=2))
    assert opt.param_groups[0]["lr"] < initial


def test_build_lr_scheduler_none():
    model = Tiny()
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    assert build_lr_scheduler(opt, None) is None
    assert build_lr_scheduler(opt, {"type": "none"}) is None


def test_best_checkpoint_state_survives_resume(tmp_path):
    model, opt, ctx = make_ctx(tmp_path)
    cb = BestCheckpoint(monitor="error_rate", mode="min")
    cb.on_validation_end(
        ctx, TrainerState(step=1, train_loss=1.0, extra={"error_rate": 0.5})
    )
    first = (tmp_path / "best.pt").stat().st_mtime_ns

    resumed = BestCheckpoint(monitor="error_rate", mode="min")
    resumed.load_state_dict(cb.state_dict())
    resumed.on_validation_end(
        ctx, TrainerState(step=2, train_loss=1.0, extra={"error_rate": 0.9})
    )
    assert (tmp_path / "best.pt").stat().st_mtime_ns == first


def test_nan_is_never_best_and_counts_as_bad_check(tmp_path):
    model, opt, ctx = make_ctx(tmp_path)
    cb = BestCheckpoint(monitor="error_rate", mode="min")
    cb.on_validation_end(
        ctx, TrainerState(step=1, train_loss=1.0, extra={"error_rate": float("nan")})
    )
    assert not (tmp_path / "best.pt").exists() and cb.best is None

    es = EarlyStopping(monitor="error_rate", mode="min", patience=2)
    for v in (0.5, float("nan"), float("nan")):
        es.on_validation_end(
            ctx, TrainerState(step=1, train_loss=1.0, extra={"error_rate": v})
        )
    assert es.should_stop and es.best == 0.5


def test_early_stopping_respects_min_epochs(tmp_path):
    _, _, ctx = make_ctx(tmp_path)
    es = EarlyStopping(monitor="auc", mode="max", patience=2, min_epochs=5)
    for epoch, auc in [(1, 0.9), (2, 0.8), (3, 0.7), (4, 0.6), (5, 0.5)]:
        es.on_validation_end(
            ctx,
            TrainerState(step=epoch, train_loss=1.0, epoch=epoch, extra={"auc": auc}),
        )
    assert not es.should_stop
    es.on_validation_end(
        ctx, TrainerState(step=6, train_loss=1.0, epoch=6, extra={"auc": 0.4})
    )
    assert es.should_stop


def test_warmup_cosine_epochs_scheduler(tmp_path):
    model, opt, ctx = make_ctx(tmp_path)
    cb = build_lr_scheduler(
        opt, {"type": "warmup_cosine_epochs", "t_max": 10, "warmup_epochs": 3}
    )
    lrs = []
    for epoch in range(1, 11):
        opt.step()
        cb.on_validation_end(ctx, TrainerState(step=epoch, train_loss=0.0, epoch=epoch))
        lrs.append(opt.param_groups[0]["lr"])
    assert lrs[1] > lrs[0]
    assert lrs[-1] < lrs[3]


def test_early_stopping_state_roundtrip(tmp_path):
    _, _, ctx = make_ctx(tmp_path)
    es = EarlyStopping(monitor="val_loss", patience=3)
    for v in (1.0, 2.0):
        es.on_validation_end(ctx, TrainerState(step=1, train_loss=1.0, val_loss=v))
    other = EarlyStopping(monitor="val_loss", patience=3)
    other.load_state_dict(es.state_dict())
    assert other.best == 1.0 and other.num_bad_checks == 1


def test_lr_scheduler_state_roundtrip(tmp_path):
    model, opt, ctx = make_ctx(tmp_path)
    cb = build_lr_scheduler(opt, {"type": "cosine", "t_max": 10})
    for s in range(3):
        cb.on_step_end(ctx, TrainerState(step=s, train_loss=0.0))
    model2, opt2, _ = make_ctx(tmp_path)
    cb2 = build_lr_scheduler(opt2, {"type": "cosine", "t_max": 10})
    cb2.load_state_dict(cb.state_dict())
    assert cb2.scheduler.last_epoch == cb.scheduler.last_epoch


def test_ema_swaps_weights_for_validation(tmp_path):
    model, opt, ctx = make_ctx(tmp_path)
    ema = EMACallback(decay=0.5)
    ema.on_step_end(ctx, TrainerState(step=1, train_loss=0.0))
    with torch.no_grad():
        model.lin.weight.add_(1.0)
    raw = model.lin.weight.clone()
    ema.on_step_end(ctx, TrainerState(step=2, train_loss=0.0))
    ema.on_validation_start(ctx)
    assert torch.allclose(model.lin.weight, raw - 0.5)
    ema.on_validation_end(ctx, TrainerState(step=2, train_loss=0.0))
    assert torch.allclose(model.lin.weight, raw)
