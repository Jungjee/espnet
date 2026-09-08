"""Training entrypoint for ESPnet3 systems."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict

import lightning as L
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from espnet3.components.modeling.lightning_module import ESPnetLightningModule
from espnet3.components.trainers.trainer import ESPnet3LightningTrainer
from espnet3.parallel.parallel import set_parallel
from espnet3.utils.task_utils import get_espnet_model, save_espnet_config

logger = logging.getLogger(__name__)


def _instantiate_model(config: DictConfig) -> Any:
    task = config.get("task")
    if task:
        model_config = OmegaConf.to_container(config.model, resolve=True)
        return get_espnet_model(task, model_config)
    return instantiate(config.model)


def _init_param_spec(config: DictConfig) -> Dict[str, Any] | None:
    """Normalise `init_param` into {path, strict, exclude}, or None."""
    spec = config.get("init_param")
    if spec is None:
        return None
    if isinstance(spec, str):
        spec = {"path": spec}
    elif isinstance(spec, DictConfig):
        spec = OmegaConf.to_container(spec, resolve=True)
    else:
        spec = dict(spec)
    if not spec.get("path"):
        raise ValueError("init_param needs a `path` (a .ckpt or a state dict)")
    spec.setdefault("strict", True)
    spec.setdefault("exclude", [])
    if isinstance(spec["exclude"], str):
        spec["exclude"] = [spec["exclude"]]
    return spec


def _load_init_param(model: Any, spec: Dict[str, Any]) -> None:
    """Initialise a model's weights from a checkpoint -- and nothing else.

    Fine-tuning is not resuming, and `fit(ckpt_path=...)` is the wrong tool for
    it: that restores the optimiser, the scheduler and the epoch counter, so a
    large-margin stage asked to run 5 epochs at lr 1e-4 from an arm that
    stopped at epoch 40 would restore epoch 40, find max_epochs already
    exceeded, and exit having trained nothing. It also cannot work here at all
    -- the top-k `*valid.eer.ckpt` files this recipe keeps are weights-only and
    carry no optimiser state to restore.

    Loud on purpose. A partially loaded speaker model is the failure that does
    not look like one: the encoder loads, the projector or the loss head
    silently stays random, and the run reports a plausible-looking EER that
    means nothing. So every key is accounted for, and an unexplained mismatch
    is fatal unless `strict: false` says otherwise.
    """
    path = Path(spec["path"])
    if not path.is_file():
        raise FileNotFoundError(f"init_param: no checkpoint at {path}")

    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    state = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    if not isinstance(state, dict) or not state:
        raise ValueError(f"init_param: {path} holds no state dict")

    # This project's checkpoints are unprefixed, because ESPnetLightningModule
    # delegates state_dict() straight to the wrapped model. Other savers keep
    # Lightning's "model." prefix, and a prefixed dict would land as 823
    # unexpected keys and 823 missing ones -- which strict=False would happily
    # ignore.
    prefix = "model."
    if all(k.startswith(prefix) for k in state):
        state = {k[len(prefix):]: v for k, v in state.items()}

    exclude = list(spec["exclude"])
    dropped = [k for k in state if any(k.startswith(p) for p in exclude)]
    for key in dropped:
        state.pop(key)

    incompatible = model.load_state_dict(state, strict=False)
    missing = list(incompatible.missing_keys)
    unexpected = list(incompatible.unexpected_keys)
    expected_missing = [k for k in missing if any(k.startswith(p) for p in exclude)]
    unexplained = [k for k in missing if k not in expected_missing]

    total = len(list(model.state_dict()))
    logger.info(
        "init_param: loaded %d/%d tensors from %s%s",
        len(state) - len(unexpected),
        total,
        path,
        (
            f" (checkpoint epoch {ckpt.get('epoch')}, step {ckpt.get('global_step')})"
            if isinstance(ckpt, dict) and "epoch" in ckpt
            else ""
        ),
    )
    if dropped:
        logger.info(
            "init_param: excluded %d tensors matching %s (e.g. %s)",
            len(dropped),
            exclude,
            dropped[:3],
        )
    if expected_missing:
        logger.info(
            "init_param: %d excluded tensors keep their fresh initialisation",
            len(expected_missing),
        )
    if unexplained or unexpected:
        message = (
            f"init_param: {len(unexplained)} missing and {len(unexpected)} "
            f"unexpected tensors loading {path}. "
            f"missing (up to 5): {unexplained[:5]}. "
            f"unexpected (up to 5): {unexpected[:5]}."
        )
        if spec["strict"]:
            raise RuntimeError(
                message + " Set init_param.strict=false, or list the prefixes "
                "in init_param.exclude, if this is intended."
            )
        logger.warning("%s Continuing because strict=false.", message)


def _build_trainer(config: DictConfig) -> ESPnet3LightningTrainer:
    model = _instantiate_model(config)
    init_param = _init_param_spec(config)
    if init_param is not None:
        _load_init_param(model, init_param)
    lit_model = ESPnetLightningModule(model, config)
    trainer = ESPnet3LightningTrainer(
        model=lit_model,
        exp_dir=config.exp_dir,
        config=config.trainer,
        best_model_criterion=config.best_model_criterion,
    )
    return trainer


def _ensure_directories(config: DictConfig) -> None:
    Path(config.exp_dir).mkdir(parents=True, exist_ok=True)
    if hasattr(config, "stats_dir"):
        Path(config.stats_dir).mkdir(parents=True, exist_ok=True)


def collect_stats(config: DictConfig) -> None:
    """Collect statistics required by the training pipeline."""
    _ensure_directories(config)
    start = time.perf_counter()

    if config.get("parallel"):
        set_parallel(config.parallel)

    if config.get("seed") is not None:
        L.seed_everything(int(config.seed), workers=True)

    torch.set_float32_matmul_precision("high")

    if "normalize" in config.model:
        config.model.pop("normalize")
    if "normalize_conf" in config.model:
        config.model.pop("normalize_conf")

    trainer = _build_trainer(config)
    trainer.collect_stats()
    logger.info(
        "Collect stats finished in %.2fs | exp_dir=%s stats_dir=%s",
        time.perf_counter() - start,
        config.exp_dir,
        getattr(config, "stats_dir", None),
    )


def train(config: DictConfig) -> None:
    """Run the training loop."""
    _ensure_directories(config)
    start = time.perf_counter()

    if config.get("parallel"):
        set_parallel(config.parallel)

    if config.get("seed") is not None:
        L.seed_everything(int(config.seed), workers=True)

    torch.set_float32_matmul_precision("high")

    task = config.get("task")
    if task:
        save_espnet_config(task, config, config.exp_dir)

    trainer = _build_trainer(config)

    fit_kwargs: Dict[str, Any] = {}
    if hasattr(config, "fit") and config.fit:
        fit_kwargs = OmegaConf.to_container(config.fit, resolve=True)

    trainer.fit(**fit_kwargs)
    logger.info(
        "Training finished in %.2fs | exp_dir=%s model=%s",
        time.perf_counter() - start,
        config.exp_dir,
        (
            config.model.get("_target_", None)
            if isinstance(config.model, DictConfig)
            else None
        ),
    )
