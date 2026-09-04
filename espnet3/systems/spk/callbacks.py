"""Validation-time callbacks for speaker verification training."""

import logging
import math

import torch
from lightning.pytorch.callbacks import Callback

from espnet3.systems.spk.scoring import compute_eer, compute_min_dcf

logger = logging.getLogger(__name__)


class SpeakerVerificationScoring(Callback):
    """Turn one epoch of validation trial scores into EER and minDCF.

    :class:`espnet3.systems.spk.espnet_model.ESPnetSpeakerVerificationModel`
    buffers a similarity score and a target/nontarget label for every trial it
    sees during validation. This callback gathers those buffers across ranks
    and logs ``valid/eer`` and ``valid/mindcf``, so that ``best_model_criterion``
    can select checkpoints on open-set verification performance instead of
    closed-set classification loss.

    Args:
        p_target: Prior probability of a target trial used by minDCF.
        c_miss: Cost of a missed detection used by minDCF.
        c_fa: Cost of a false alarm used by minDCF.

    Examples:
        Enable it from a training config:

        ```yaml
        trainer:
          callbacks:
            - _target_: espnet3.systems.spk.callbacks.SpeakerVerificationScoring

        best_model_criterion:
          - - valid/eer
            - 3
            - min
        ```
    """

    def __init__(
        self,
        p_target: float = 0.05,
        c_miss: float = 1.0,
        c_fa: float = 1.0,
    ) -> None:
        """Initialize the callback with the minDCF operating point."""
        self.p_target = float(p_target)
        self.c_miss = float(c_miss)
        self.c_fa = float(c_fa)

    def on_validation_epoch_start(self, trainer, pl_module) -> None:
        """Drop trial scores left over from a previous validation run."""
        pl_module.model.reset_trials()

    def on_validation_batch_end(
        self,
        trainer,
        pl_module,
        outputs,
        batch,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Score the collected trials once the last validation batch is done.

        The reduction runs here rather than in ``on_validation_epoch_end``
        because callbacks configured by a recipe are appended after the default
        ESPnet3 callbacks, and ``MetricsLogger`` prints its validation summary
        from that hook. Logging on the final batch keeps ``valid/eer`` in
        ``trainer.callback_metrics`` in time for that summary line.
        """
        num_batches = trainer.num_val_batches
        if isinstance(num_batches, (list, tuple)):
            num_batches = num_batches[dataloader_idx]
        if not math.isfinite(num_batches) or batch_idx + 1 < num_batches:
            return
        self.score_epoch(pl_module)

    def score_epoch(self, pl_module) -> None:
        """Gather the buffered trials across ranks and log the metrics.

        Args:
            pl_module: LightningModule wrapping the speaker model.
        """
        scores, labels = pl_module.model.pop_trials()
        if scores.numel() == 0:
            return

        scores = pl_module.all_gather(scores).flatten().cpu().numpy()
        labels = pl_module.all_gather(labels).flatten().cpu().numpy()

        # The sanity-check run only sees a couple of batches, which may not
        # contain both target and nontarget trials.
        if len(set(labels.tolist())) < 2:
            logger.info(
                "Skipping verification scoring: %d trial(s) of a single class.",
                len(labels),
            )
            return

        metrics = {
            "valid/eer": compute_eer(scores, labels),
            "valid/mindcf": compute_min_dcf(
                scores,
                labels,
                p_target=self.p_target,
                c_miss=self.c_miss,
                c_fa=self.c_fa,
            ),
        }
        pl_module.log_dict(
            {k: torch.tensor(v) for k, v in metrics.items()},
            prog_bar=True,
            logger=True,
            sync_dist=False,
        )


class MarginScheduler(Callback):
    """Ramp the additive-margin loss's margin during training.

    Large-margin losses converge poorly when the margin starts at its final
    value, so the reference recipes hold it at ``initial_margin``, raise it over
    a window of epochs, then fix it at ``final_margin``. ESPnet ships nine
    learning-rate schedulers and none for the margin, so a recipe that wants
    the reference's schedule has no way to express it.

    The curve follows ReDimNet2's ``wespeaker_lite`` MarginScheduler, including
    two details that are easy to get wrong:

    * it is **saturating**, not linear -- ``ratio = 1 - exp(progress * ln(1e-3))``,
      so the margin rises quickly at the start of the window and flattens
      towards the end (half way through the window it is already at 0.97 of its
      final value, not 0.5);
    * it steps per **iteration**, not per epoch, so with several thousand steps
      to an epoch the margin moves smoothly rather than in visible jumps.

    Epoch arguments are 1-indexed, again matching the reference:
    ``increase_start_epoch: 20`` begins at the start of the twentieth epoch.

    The margin is applied through the loss's own ``update(margin=...)``. A loss
    without that method is left alone and reported once -- silently training at
    a constant margin while a schedule is configured would be the worse
    failure.

    Example::

        trainer:
          callbacks:
            - _target_: espnet3.systems.spk.callbacks.MarginScheduler
              initial_margin: 0.0
              final_margin: 0.2
              increase_start_epoch: 20
              fix_start_epoch: 40
    """

    def __init__(
        self,
        initial_margin: float = 0.0,
        final_margin: float = 0.2,
        increase_start_epoch: int = 20,
        fix_start_epoch: int = 40,
        increase_type: str = "exp",
        log_every_n_steps: int = 1000,
    ):
        if fix_start_epoch <= increase_start_epoch:
            raise ValueError(
                f"fix_start_epoch ({fix_start_epoch}) must be after "
                f"increase_start_epoch ({increase_start_epoch})"
            )
        if increase_type not in ("exp", "linear"):
            raise ValueError(
                f"unknown increase_type {increase_type!r}; expected 'exp' or 'linear'"
            )
        self.initial_margin = float(initial_margin)
        self.final_margin = float(final_margin)
        self.increase_start_epoch = int(increase_start_epoch)
        self.fix_start_epoch = int(fix_start_epoch)
        self.increase_type = increase_type
        self.log_every_n_steps = int(log_every_n_steps)
        self._warned = False
        self._last_logged = -1

    def margin_at(self, step: int, steps_per_epoch: int) -> float:
        """Return the margin for a global step.

        Pure, so the schedule can be tested without a Trainer.
        """
        if steps_per_epoch <= 0:
            # num_training_batches is not always known at on_train_start.
            return self.initial_margin

        start = (self.increase_start_epoch - 1) * steps_per_epoch
        fix = (self.fix_start_epoch - 1) * steps_per_epoch
        if step < start:
            return self.initial_margin
        if step >= fix:
            return self.final_margin

        progress = (step - start) / (fix - start)
        if self.increase_type == "exp":
            # The reference's constants: initial_val 1.0, final_val 1e-3, and a
            # 1e-6 guard in the denominator.
            ratio = 1.0 - math.exp(progress * math.log(1e-3 / (1.0 + 1e-6)))
        else:
            ratio = progress
        return self.initial_margin + (self.final_margin - self.initial_margin) * ratio

    def _check_endpoint(self, loss) -> None:
        """Warn if the recipe's `loss_conf.margin` disagrees with the ramp.

        Once a schedule is installed it overwrites the margin every step, so
        `loss_conf.margin` no longer has any effect on training -- the config
        would claim one endpoint while the run used another. That kind of dead
        key is hard to spot by reading the file, so say it out loud instead.
        Only the value the loss was constructed with is compared, which is what
        `loss.margin` still holds the first time this runs.
        """
        configured = getattr(loss, "margin", None)
        if configured is None or abs(configured - self.final_margin) <= 1e-9:
            return
        logging.warning(
            "MarginScheduler ramps to final_margin=%g, but the loss was built "
            "with margin=%g; the schedule wins and loss_conf.margin has no "
            "effect. Set them to the same value.",
            self.final_margin,
            configured,
        )

    def _apply(self, trainer, pl_module) -> None:
        loss = getattr(getattr(pl_module, "model", None), "loss", None)
        if loss is None or not hasattr(loss, "update"):
            if not self._warned:
                self._warned = True
                logging.warning(
                    "MarginScheduler is configured but %s has no loss exposing "
                    "update(margin=...); the margin will stay constant.",
                    type(getattr(pl_module, "model", pl_module)).__name__,
                )
            return

        steps_per_epoch = getattr(trainer, "num_training_batches", 0)
        if not isinstance(steps_per_epoch, int):
            # Lightning uses float('inf') for an unsized dataloader.
            steps_per_epoch = 0
        margin = self.margin_at(trainer.global_step, steps_per_epoch)
        loss.update(margin=margin)

        if (
            self.log_every_n_steps > 0
            and trainer.global_step - self._last_logged >= self.log_every_n_steps
        ):
            self._last_logged = trainer.global_step
            pl_module.log("train/margin", margin, prog_bar=False)

    def on_train_start(self, trainer, pl_module) -> None:
        loss = getattr(getattr(pl_module, "model", None), "loss", None)
        if loss is not None:
            self._check_endpoint(loss)
        self._apply(trainer, pl_module)

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx) -> None:
        self._apply(trainer, pl_module)

