"""Attentive statistics pooling, ported from ReDimNet2 for weight compatibility.

Destined for ``espnet2/spk/pooling/astp_pooling.py``.

This is **not** interchangeable with ESPnet's existing ``ChnAttnStatPooling``:

    ChnAttnStatPooling:  Conv1d -> ReLU -> BatchNorm1d(128) -> Conv1d -> softmax
    ASTP (this module):  Conv1d -> tanh                     -> Conv1d -> softmax

Different activation, and no normalisation in the attention branch. The
reference implementation carries an explicit warning against the ReLU variant
(``redimnet2/layers/poolings.py``: "DON'T use ReLU here! ReLU may be hard to
converge"). The two are also state-dict incompatible.

Parameter names ``linear1``/``linear2`` are kept identical to upstream so that a
released ReDimNet2 checkpoint maps onto the ESPnet composition by a pure prefix
rename.
"""

from __future__ import annotations

from typing import Optional

import torch

from espnet2.spk.pooling.abs_pooling import AbsPooling


class AstpPooling(AbsPooling):
    """Channel- and context-dependent attentive statistics pooling.

    Reference:
        Desplanques et al., "ECAPA-TDNN: Emphasized Channel Attention,
        Propagation and Aggregation in TDNN Based Speaker Verification",
        https://arxiv.org/pdf/2005.07143

    Args:
        input_size: channel dimension of the frame-level input.
        bottleneck_dim: width of the attention bottleneck.
        global_context_att: concatenate the context mean and standard deviation
            to the attention input, tripling its width.
    """

    def __init__(
        self,
        input_size: int = 1536,
        bottleneck_dim: int = 128,
        global_context_att: bool = True,
    ):
        super().__init__()
        self.in_dim = input_size
        self.global_context_att = global_context_att

        attn_in = input_size * 3 if global_context_att else input_size
        self.linear1 = torch.nn.Conv1d(attn_in, bottleneck_dim, kernel_size=1)
        self.linear2 = torch.nn.Conv1d(bottleneck_dim, input_size, kernel_size=1)

        self._output_size = 2 * input_size

    def output_size(self) -> int:
        return self._output_size

    def forward(
        self,
        x: torch.Tensor,
        feat_lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Pool ``(B, C, T)`` frame-level features to ``(B, 2C)``.

        A 4-D ``(B, C, F, T)`` input is folded to ``(B, C*F, T)`` first, matching
        the reference.

        Args:
            x: frame-level features, channel-major.
            feat_lengths: accepted for signature compatibility with the other
                poolings -- `ESPnetSpeakerModel` passes it -- and deliberately
                unused. `SpkPreprocessor` emits fixed-length crops, so every
                utterance in a batch has the same length and there is nothing to
                mask; the reference implementation does not mask either. If a
                variable-length batch is ever fed here, the statistics would
                include padding and this would need a real mask.
        """
        if x.dim() == 4:
            x = x.reshape(x.shape[0], x.shape[1] * x.shape[2], x.shape[3])
        assert x.dim() == 3, f"expected (B, C, T), got {tuple(x.shape)}"

        if self.global_context_att:
            context_mean = torch.mean(x, dim=-1, keepdim=True).expand_as(x)
            context_std = torch.sqrt(
                torch.var(x, dim=-1, keepdim=True) + 1e-7
            ).expand_as(x)
            x_in = torch.cat((x, context_mean, context_std), dim=1)
        else:
            x_in = x

        # tanh, not ReLU: see the module docstring.
        alpha = torch.tanh(self.linear1(x_in))
        alpha = torch.softmax(self.linear2(alpha), dim=2)

        mean = torch.sum(alpha * x, dim=2)
        var = torch.sum(alpha * (x**2), dim=2) - mean**2
        std = torch.sqrt(var.clamp(min=1e-7))
        return torch.cat([mean, std], dim=1)
