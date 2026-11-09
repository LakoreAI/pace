import math
from typing import Optional

from src.callbacks.base import Callback, TrainContext, TrainerState


class EarlyStopping(Callback):
    """Stop training when `monitor` hasn't improved for `patience` validation
    checks in a row. Set `enabled: false` (or omit the block) to disable.
    """

    def __init__(
        self,
        monitor: str = "val_loss",
        patience: int = 10,
        min_delta: float = 0.0,
        mode: str = "min",
        min_epochs: int = 0,
    ):
        assert mode in ("min", "max"), f"mode must be 'min' or 'max', got {mode!r}"
        assert min_epochs >= 0, f"min_epochs must be >= 0, got {min_epochs}"
        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.min_epochs = min_epochs
        self.best: Optional[float] = None
        self.num_bad_checks = 0
        self.should_stop = False

    def _is_improvement(self, value: float) -> bool:
        if self.best is None:
            return True
        if self.mode == "min":
            return value < self.best - self.min_delta
        return value > self.best + self.min_delta

    def state_dict(self) -> dict:
        return {
            "best": self.best,
            "num_bad_checks": self.num_bad_checks,
            "should_stop": self.should_stop,
        }

    def load_state_dict(self, state: dict) -> None:
        self.best = state.get("best")
        self.num_bad_checks = int(state.get("num_bad_checks", 0))
        self.should_stop = bool(state.get("should_stop", False))

    def on_validation_end(self, ctx: TrainContext, state: TrainerState) -> None:
        value = state.get(self.monitor)
        if value is None:
            return
        if not math.isfinite(value):
            value = math.inf if self.mode == "min" else -math.inf

        if self._is_improvement(value):
            self.best = value
            self.num_bad_checks = 0
            return
        if state.epoch < self.min_epochs:
            return
        self.num_bad_checks += 1
        if self.num_bad_checks >= self.patience:
            self.should_stop = True
            print(
                f"  [EarlyStopping] {self.monitor} did not improve for "
                f"{self.patience} checks (best={self.best:.4f}) — stopping"
            )
