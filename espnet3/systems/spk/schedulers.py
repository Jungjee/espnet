"""Learning-rate schedulers used by the ReDimNet2 speaker recipes.

ESPnet ships nine schedulers, none of which is the exponential decay the
ReDimNet2 recipes use, so a faithful reproduction was not expressible.

Transcribed from ``wespeaker_lite/utils/schedulers.py`` in PalabraAI/redimnet2.
"""

import math
from typing import List

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler


class ExponentialDecrease(LRScheduler):
    """Exponential interpolation from ``initial_lr`` to ``final_lr``.

    The reference's formula, per optimiser step::

        lr = coeff * initial_lr * exp((iter / max_iter) * log(final_lr / initial_lr))

    which is a geometric sweep: ``initial_lr`` at step 0, ``final_lr`` at
    ``max_iter``. ``max_iter = num_epochs * epoch_iter``, so this must be
    stepped per iteration -- ``scheduler_interval: step``, which is the
    ESPnet3 default.

    **Two warm-up modes, and the released recipe uses the second.**
    ``warm_up_epoch`` governs a coefficient over the warm-up window:

    * ``warm_from_zero=False``: ``coeff = (scale_ratio - 1) * iter/warm_up + 1``,
      which at ``scale_ratio=1.0`` is identically ``1.0`` -- a **no-op**. This
      is the mode that implements the reference's linear LR scaling for large
      effective batch size, active only when ``scale_ratio > 1``.
    * ``warm_from_zero=True``: ``coeff = scale_ratio * iter/warm_up``, a
      conventional linear ramp from 0.

    The released b6 recipe sets ``warm_from_zero: true``, so its 6-epoch
    warm-up is a real ramp from zero, and that is the default here. (An earlier
    reading of only the scheduler source concluded the warm-up was vestigial;
    the recipe shows otherwise, which is why the recipe and not the paper is
    the source of truth.)

    Args:
        optimizer: Wrapped optimizer.
        num_epochs: Total epochs the sweep spans.
        epoch_iter: Optimiser steps per epoch.
        initial_lr: LR at step 0.
        final_lr: LR at ``num_epochs * epoch_iter``.
        warm_up_epoch: Length of the coefficient ramp, in epochs.
        scale_ratio: Coefficient reached at the end of the ramp. 1.0 = no-op.
        warm_from_zero: Ramp linearly from 0. True in the released b6 recipe,
            and the default here.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        num_epochs: int,
        epoch_iter: int,
        initial_lr: float,
        final_lr: float,
        warm_up_epoch: int = 6,
        scale_ratio: float = 1.0,
        warm_from_zero: bool = True,
        last_epoch: int = -1,
    ):
        if initial_lr <= 0 or final_lr <= 0:
            raise ValueError(
                f"initial_lr and final_lr must be positive (the schedule is "
                f"geometric), got {initial_lr} and {final_lr}"
            )
        if epoch_iter <= 0 or num_epochs <= 0:
            raise ValueError(
                f"num_epochs and epoch_iter must be positive, got "
                f"{num_epochs} and {epoch_iter}"
            )
        self.max_iter = num_epochs * epoch_iter
        self.initial_lr = initial_lr
        self.final_lr = final_lr
        self.warm_up_iter = warm_up_epoch * epoch_iter
        self.scale_ratio = scale_ratio
        self.warm_from_zero = warm_from_zero
        super().__init__(optimizer, last_epoch)

    def _coeff(self, step: int) -> float:
        if step < self.warm_up_iter and self.warm_up_iter > 0:
            if self.warm_from_zero:
                return self.scale_ratio * step / self.warm_up_iter
            return (self.scale_ratio - 1) * step / self.warm_up_iter + 1.0
        return self.scale_ratio

    def lr_at(self, step: int) -> float:
        """LR for a given optimiser step. Pure, so it can be tested directly."""
        progress = min(step, self.max_iter) / self.max_iter
        return (
            self._coeff(step)
            * self.initial_lr
            * math.exp(progress * math.log(self.final_lr / self.initial_lr))
        )

    def get_lr(self) -> List[float]:
        lr = self.lr_at(self.last_epoch)
        return [lr for _ in self.optimizer.param_groups]
