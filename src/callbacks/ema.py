"""Exponential moving average of model weights.

Updates a shadow copy after every optimizer step and swaps it into the model
for validation, so `BestCheckpoint` selects and saves the EMA weights. Training
resumes on the raw weights after validation. Buffers (BatchNorm statistics) are
copied from the model rather than averaged.
"""

from typing import Dict, Optional

import torch

from src.callbacks.base import Callback, TrainContext, TrainerState


class EMACallback(Callback):
    def __init__(self, decay: float = 0.999, start_step: int = 0):
        assert 0.0 < decay < 1.0, f"decay must be in (0, 1), got {decay}"
        self.decay = decay
        self.start_step = start_step
        self.shadow: Optional[Dict[str, torch.Tensor]] = None
        self._backup: Optional[Dict[str, torch.Tensor]] = None

    def _params(self, model: torch.nn.Module) -> Dict[str, torch.Tensor]:
        return {
            k: v for k, v in model.state_dict().items() if v.dtype.is_floating_point
        }

    @torch.no_grad()
    def on_step_end(self, ctx: TrainContext, state: TrainerState) -> None:
        current = self._params(ctx.model)
        if self.shadow is None or state.step <= self.start_step:
            self.shadow = {k: v.detach().clone() for k, v in current.items()}
            return
        params = dict(ctx.model.named_parameters())
        for k, v in current.items():
            if k in params:
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1.0 - self.decay)
            else:
                self.shadow[k].copy_(v)

    @torch.no_grad()
    def on_validation_start(self, ctx: TrainContext) -> None:
        if self.shadow is None:
            return
        self._backup = {
            k: v.detach().clone() for k, v in self._params(ctx.model).items()
        }
        ctx.model.load_state_dict(self.shadow, strict=False)

    @torch.no_grad()
    def on_validation_end(self, ctx: TrainContext, state: TrainerState) -> None:
        if self._backup is not None:
            ctx.model.load_state_dict(self._backup, strict=False)
            self._backup = None

    def state_dict(self) -> dict:
        return {"shadow": self.shadow}

    def load_state_dict(self, state: dict) -> None:
        self.shadow = state.get("shadow")
