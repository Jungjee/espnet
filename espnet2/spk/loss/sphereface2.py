"""SphereFace2 loss, ported from the ReDimNet2 reference implementation.

ESPnet ships AAM-softmax, its sub-center/inter-topk variant, and plain softmax.
ReDimNet2's published recipes use SphereFace2 with ``margin_type: C``, so
reproducing their numbers is impossible without it.

The maths is transcribed from ``wespeaker_lite/models/projections.py`` in
PalabraAI/redimnet2 rather than rewritten from the paper, which is what makes a
numerical parity test against that file meaningful. Two adaptations were
required, both interface rather than behaviour:

* ESPnet calls ``loss(embedding, label)`` and expects
  ``(loss, accuracy, predictions)``; the reference returns ``(logits, loss)``.
* ESPnet constructs losses as ``cls(nout=..., nclasses=...)``.

The reference's ``dynamic_margin`` path exists for mixup/cutmix soft labels. It
is kept, and defaults to True as it does there, because for the one-hot targets
this recipe uses it reduces exactly to vanilla SphereFace2 -- the reference says
so in a comment, and the tests check it rather than taking its word.

References:
    Exploring Binary Classification Loss for Speaker Verification,
    https://ieeexplore.ieee.org/abstract/document/10094954
    SphereFace2: Binary Classification is All You Need,
    https://arxiv.org/pdf/2108.01513
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from espnet2.spk.loss.abs_loss import AbsLoss


class SphereFace2(AbsLoss):
    """SphereFace2 with the reference's margin-scheduling hook.

    Args:
        nout: Embedding dimension.
        nclasses: Number of speaker classes.
        scale: Feature-norm scale. Reference b6 uses 32.
        margin: Initial margin. The reference ramps it 0.0 -> 0.2 during
            pretraining via MarginScheduler, and fixes it at 0.3 for
            large-margin finetuning.
        lanbuda: Weight between the positive and negative binary terms. The
            reference's spelling is kept so its configs transfer verbatim.
        t: Exponent of the score-shaping function ``fun_g``.
        margin_type: ``"C"`` for cos(theta) - m, ``"A"`` for cos(theta + m).
            Reference b6 uses C.
        dynamic_margin: Scale the positive margin by the soft-label weight.
            A no-op for one-hot targets.

    Examples:
        >>> loss = SphereFace2(nout=192, nclasses=17982, margin_type="C")
        >>> loss.update(margin=0.2)  # what MarginScheduler calls each step
    """

    def __init__(
        self,
        nout: int,
        nclasses: int,
        scale: float = 32.0,
        margin: float = 0.2,
        lanbuda: float = 0.7,
        t: int = 3,
        margin_type: str = "C",
        dynamic_margin: bool = True,
        **kwargs,
    ):
        super().__init__(nout)
        if margin_type not in ("A", "C"):
            raise ValueError(f"margin_type must be 'A' or 'C', got {margin_type!r}")
        self.in_features = nout
        self.out_features = nclasses
        self.scale = scale
        self.weight = nn.Parameter(torch.FloatTensor(nclasses, nout))
        nn.init.xavier_uniform_(self.weight)
        self.bias = nn.Parameter(torch.zeros(1, 1))
        self.t = t
        self.lanbuda = lanbuda
        self.margin_type = margin_type
        self.dynamic_margin = dynamic_margin
        self.update(margin)

    def update(self, margin: float = 0.2) -> None:
        """Set the margin. Called every step by MarginScheduler."""
        self.margin = margin
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin)
        self.mmm = 1.0 + math.cos(math.pi - margin)

    def fun_g(self, z: torch.Tensor, t: int) -> torch.Tensor:
        """The reference's score-shaping function."""
        return 2 * torch.pow((z + 1) / 2, t) - 1

    def forward(
        self, input: torch.Tensor, label: Optional[torch.Tensor] = None
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], torch.Tensor]:
        """Compute the SphereFace2 loss.

        Args:
            input: Speaker embeddings, shape (batch, nout).
            label: Speaker indices, shape (batch,). ``None`` puts the loss in
                inference mode and returns predictions only.

        Returns:
            ``(loss, accuracy, predictions)``, matching the other losses here.
        """
        cos = F.linear(F.normalize(input), F.normalize(self.weight))

        if label is None:
            return None, None, torch.argmax(cos, dim=1)

        target_mask = input.new_zeros(cos.size())
        target_mask.scatter_(1, label.view(-1, 1).long(), 1.0)

        if self.dynamic_margin:
            # For a one-hot target_mask this is identical to the branch below;
            # it differs only for soft labels, which this recipe does not use.
            khot = (target_mask > 0).to(target_mask.dtype)
            row_active = (khot.sum(1, keepdim=True) > 0).to(target_mask.dtype)
            nontarget_mask = (1.0 - khot) * row_active
            margin_p = self.margin * target_mask
        else:
            nontarget_mask = 1.0 - target_mask
            margin_p = self.margin

        if self.margin_type == "A":  # arcface type
            sin = torch.sqrt(1.0 - torch.pow(cos, 2))
            if self.dynamic_margin:
                cos_m_p = torch.cos(margin_p)
                sin_m_p = torch.sin(margin_p)
                th_p = torch.cos(math.pi - margin_p)
                mmm_p = 1.0 + torch.cos(math.pi - margin_p)
                cos_m_theta_p = (
                    self.scale
                    * self.fun_g(
                        torch.where(
                            cos > th_p, cos * cos_m_p - sin * sin_m_p, cos - mmm_p
                        ),
                        self.t,
                    )
                    + self.bias[0][0]
                )
            else:
                cos_m_theta_p = (
                    self.scale
                    * self.fun_g(
                        torch.where(
                            cos > self.th,
                            cos * self.cos_m - sin * self.sin_m,
                            cos - self.mmm,
                        ),
                        self.t,
                    )
                    + self.bias[0][0]
                )
            cos_m_theta_n = (
                self.scale * self.fun_g(cos * self.cos_m + sin * self.sin_m, self.t)
                + self.bias[0][0]
            )
        else:  # cosface type, margin_type "C"
            cos_m_theta_p = (
                self.scale * (self.fun_g(cos, self.t) - margin_p) + self.bias[0][0]
            )
            cos_m_theta_n = (
                self.scale * (self.fun_g(cos, self.t) + self.margin) + self.bias[0][0]
            )

        cos_p_theta = self.lanbuda * torch.log(1 + torch.exp(-1.0 * cos_m_theta_p))
        cos_n_theta = (1 - self.lanbuda) * torch.log(1 + torch.exp(cos_m_theta_n))

        # Logits for the accuracy meter, exactly as the reference forms them.
        cos1 = (cos - margin_p) * (1.0 - nontarget_mask) + cos * nontarget_mask
        output = self.scale * cos1
        preds = torch.argmax(output, dim=1)
        accuracy = (preds == label).float().mean()

        # N independent one-vs-all binary terms, summed over classes then
        # averaged over the batch -- the reference's reduction.
        loss = (target_mask * cos_p_theta + nontarget_mask * cos_n_theta).sum(1).mean()
        return loss, accuracy, preds
