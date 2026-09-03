"""Map a ReDimNet2 checkpoint onto ESPnet's encoder/pooling/projector split.

``ReDimNet2Wrap`` is one module; ESPnet composes three. The parameters are
identical, so the translation is a prefix rewrite:

    spec.*     -> encoder.spec.*
    backbone.* -> encoder.backbone.*
    pool.*     -> pooling.*
    bn.*       -> projector.bn.*
    linear.*   -> projector.fc.*

The last two land on ESPnet's existing ``RawNet3Projector``, which is already
``BatchNorm1d`` followed by ``Linear`` -- only the attribute name differs.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch

# Longest source prefix first. "backbone." must be tried before "bn.", or
# "backbone.bn.weight" would be rewritten as a projector parameter.
_PREFIX_MAP: Tuple[Tuple[str, str], ...] = (
    ("backbone.", "encoder.backbone."),
    ("linear.", "projector.fc."),
    ("spec.", "encoder.spec."),
    ("pool.", "pooling."),
    ("bn.", "projector.bn."),
)

#: Config keys consumed by pooling rather than the encoder.
_POOLING_KEYS = frozenset({"global_context_att"})
#: Config keys consumed by the projector.
_PROJECTOR_KEYS = frozenset({"embed_dim"})
#: Config keys that exist only on the monolithic wrapper.
_WRAP_ONLY_KEYS = frozenset(
    {
        "pooling_func",
        "emb_bn",
        "num_classes",
        "feat_agg_dropout",
        "head_activation",
        "return_all_outputs",
    }
)


def remap_redimnet2_state_dict(
    state_dict: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """Rewrite ``ReDimNet2Wrap`` parameter names to the ESPnet composition's.

    Args:
        state_dict: as produced by ``ReDimNet2Wrap.state_dict()`` or read from a
            released ``.pt``.

    Returns:
        A new dict with rewritten keys; tensors are passed through unchanged.

    Raises:
        KeyError: if any key matches no known prefix. ``bn2`` (present only when
            ``emb_bn=True``) has no ESPnet counterpart and is reported here
            rather than dropped -- a silently discarded parameter would let a
            broken port pass the parity test.
    """
    out: Dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        for src, dst in _PREFIX_MAP:
            if key.startswith(src):
                out[dst + key[len(src) :]] = value
                break
        else:
            raise KeyError(
                f"no ESPnet destination for checkpoint key {key!r}; "
                f"known prefixes: {[p for p, _ in _PREFIX_MAP]}"
            )
    return out


def split_model_config(model_config: dict) -> Tuple[dict, dict, dict]:
    """Route a ``ReDimNet2Wrap`` config into encoder, pooling and projector kwargs.

    Args:
        model_config: the ``model_config`` dict stored in a released checkpoint,
            or an equivalent hand-written config.

    Returns:
        ``(encoder_kwargs, pooling_kwargs, projector_kwargs)``.

    Raises:
        ValueError: for a pooling variant or head configuration that has no
            ESPnet counterpart, rather than silently building a different model.
    """
    pooling_func = model_config.get("pooling_func", "ASTP")
    if pooling_func != "ASTP":
        raise ValueError(
            f"only ASTP pooling is ported; this config requests {pooling_func!r}"
        )
    if model_config.get("emb_bn", False):
        raise ValueError(
            "emb_bn=True adds a bn2 layer after the projector with no ESPnet "
            "counterpart; not supported"
        )

    dropped = _POOLING_KEYS | _PROJECTOR_KEYS | _WRAP_ONLY_KEYS
    encoder_kwargs = {k: v for k, v in model_config.items() if k not in dropped}
    pooling_kwargs = {
        "global_context_att": model_config.get("global_context_att", True)
    }
    projector_kwargs = {"output_size": model_config["embed_dim"]}
    return encoder_kwargs, pooling_kwargs, projector_kwargs
