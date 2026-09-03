"""Adaptive score normalisation (AS-Norm) for the speaker task.

ESPnet3 ships EER and minDCF but no score normalisation; ESPnet2 has it in
``egs2/TEMPLATE/asr1/pyscripts/utils/spk_apply_score_norm.py``, driven by
``spk.sh`` stage 7-b. This is a faithful port with the loop hoisted out of the
script and the numerical edge cases guarded.

Given a trial ``(e, t)`` with raw score ``s``, and a cohort ``C``::

    s_norm = 1/2 * [ (s - mu_e) / sigma_e  +  (s - mu_t) / sigma_t ]

where ``mu_e``/``sigma_e`` are the mean and standard deviation of the top-k
scores of ``e`` against ``C``, and likewise for ``t``.

One detail worth stating because the literature is not consistent about it:
ESPnet2 scores the cohort with **negative Euclidean distance**, not cosine. On
L2-normalised embeddings ``||a-b||^2 = 2 - 2*cos``, so the two are monotonically
related, but AS-norm takes a mean and a standard deviation over cohort scores
and those are not preserved by a nonlinear monotone map. ``similarity="cosine"``
is available, but ``"neg_euclidean"`` is the default because it is what ESPnet2
does and what its published numbers reflect.
"""

from __future__ import annotations

from typing import Dict, Mapping, Optional, Sequence, Tuple

import torch

#: Smallest standard deviation treated as meaningful. Below it -- a degenerate
#: cohort of identical embeddings, or a cohort of one, where the unbiased std is
#: NaN -- we fall back to 1.0 and centre without rescaling. Clamping to a tiny
#: epsilon instead would be finite but useless: it multiplies each utterance's
#: offset by ~1e6, and since the scale is per-utterance it destroys the ordering
#: between trials, which is the only thing EER depends on.
_STD_EPS = 1e-6
_STD_FALLBACK = 1.0

_SIMILARITIES = ("neg_euclidean", "cosine")


def l2_normalize(x: torch.Tensor) -> torch.Tensor:
    """L2-normalise along the last dimension."""
    return torch.nn.functional.normalize(x.float(), p=2, dim=-1)


def build_cohort(
    embeddings: Mapping[str, torch.Tensor],
    utt2spk: Optional[Mapping[str, str]] = None,
    average_spk: bool = True,
) -> torch.Tensor:
    """Assemble the cohort matrix from utterance embeddings.

    Args:
        embeddings: utterance id to embedding.
        utt2spk: utterance id to speaker id. Required when ``average_spk``.
        average_spk: average each speaker's embeddings into one cohort row, as
            ESPnet2's ``average_spk`` option does. This makes the cohort a set
            of speakers rather than a set of utterances, which is the usual
            choice and keeps the cohort size independent of how many utterances
            each speaker happens to have.

    Returns:
        ``(n_cohort, dim)``, L2-normalised, with deterministic row order.

    Raises:
        ValueError: if averaging is requested without ``utt2spk``.
        KeyError: if an utterance is missing from ``utt2spk``. Dropping it would
            silently shrink the cohort.
    """
    if not average_spk:
        keys = sorted(embeddings)
        return l2_normalize(torch.stack([l2_normalize(embeddings[k]) for k in keys]))

    if utt2spk is None:
        raise ValueError("utt2spk is required when average_spk=True")

    per_spk: Dict[str, list[torch.Tensor]] = {}
    for utt in sorted(embeddings):
        if utt not in utt2spk:
            raise KeyError(f"cohort utterance {utt!r} is missing from utt2spk")
        per_spk.setdefault(utt2spk[utt], []).append(l2_normalize(embeddings[utt]))

    rows = [torch.stack(per_spk[spk]).mean(0) for spk in sorted(per_spk)]
    return l2_normalize(torch.stack(rows))


class AdaptiveScoreNorm:
    """Adaptive score normalisation against a fixed cohort.

    Args:
        cohort: ``(n_cohort, dim)``. L2-normalised on construction.
        cohort_size: the ``k`` of the top-k. Clamped to the cohort size, as
            ESPnet2 does.
        similarity: ``"neg_euclidean"`` (ESPnet2's choice, the default) or
            ``"cosine"``.
        device: where to hold the cohort and run the distance computation.
    """

    def __init__(
        self,
        cohort: torch.Tensor,
        cohort_size: int,
        similarity: str = "neg_euclidean",
        device: str | torch.device = "cpu",
    ):
        if cohort.ndim != 2 or cohort.shape[0] == 0:
            raise ValueError(
                f"cohort must be a non-empty (n, dim) matrix, got {tuple(cohort.shape)}"
            )
        if similarity not in _SIMILARITIES:
            raise ValueError(
                f"unknown similarity {similarity!r}; known: {list(_SIMILARITIES)}"
            )

        self.cohort = l2_normalize(cohort).to(device)
        self.similarity = similarity
        self.device = device
        self.cohort_size = min(int(cohort_size), self.cohort.shape[0])

    def _cohort_stats(self, embedding: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Mean and standard deviation of the top-k cohort scores."""
        emb = l2_normalize(embedding).to(self.device)
        if emb.dim() == 1:
            emb = emb[None, :]

        if self.similarity == "neg_euclidean":
            scores = -1.0 * torch.cdist(emb, self.cohort).mean(0)
        else:
            scores = (emb @ self.cohort.T).mean(0)

        top = torch.topk(scores, k=self.cohort_size)[0]
        mean = torch.mean(top)
        std = torch.std(top) if top.numel() > 1 else torch.zeros((), device=top.device)
        if not torch.isfinite(std) or std < _STD_EPS:
            std = torch.tensor(_STD_FALLBACK, device=top.device)
        return mean, std

    def score(self, enroll: torch.Tensor, test: torch.Tensor) -> float:
        """Raw trial score, using this instance's similarity.

        Use this rather than computing a cosine by hand. ESPnet2 scores trials
        with the same negative Euclidean distance it uses for the cohort
        (``spk_calculate_scores_from_embeddings.py``), and feeding a cosine into
        a neg-Euclidean normaliser mixes two different scales: the subtraction
        ``score - mu`` is then meaningless and EER collapses.
        """
        e = l2_normalize(enroll).to(self.device)
        t = l2_normalize(test).to(self.device)
        if e.dim() == 1:
            e = e[None, :]
        if t.dim() == 1:
            t = t[None, :]
        if self.similarity == "neg_euclidean":
            return float(-1.0 * torch.mean(torch.cdist(e, t)))
        return float(torch.mean(e @ t.T))

    def normalize(
        self, score: float, enroll: torch.Tensor, test: torch.Tensor
    ) -> float:
        """Normalise one trial score.

        Args:
            score: the raw similarity for this trial.
            enroll: enrollment embedding, ``(dim,)`` or ``(n_crops, dim)``.
            test: test embedding, same shapes.
        """
        e_m, e_s = self._cohort_stats(enroll)
        t_m, t_s = self._cohort_stats(test)
        normed = ((score - e_m) / e_s + (score - t_m) / t_s) / 2
        return float(normed)

    def normalize_trials(
        self,
        trials: Sequence[Tuple[str, str]],
        scores: Sequence[float],
        embeddings: Mapping[str, torch.Tensor],
    ) -> list[float]:
        """Normalise a list of trials, computing each utterance's stats once.

        Trial lists reuse utterances heavily -- the cleaned Vox1-O protocol has
        37,611 trials over 4,708 utterances -- so caching the per-utterance
        cohort statistics is roughly an eightfold saving.
        """
        if len(trials) != len(scores):
            raise ValueError(
                f"got {len(trials)} trials but {len(scores)} scores"
            )

        cache: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}

        def stats(utt: str) -> Tuple[torch.Tensor, torch.Tensor]:
            if utt not in cache:
                cache[utt] = self._cohort_stats(embeddings[utt])
            return cache[utt]

        out = []
        for (enroll, test), score in zip(trials, scores, strict=True):
            e_m, e_s = stats(enroll)
            t_m, t_s = stats(test)
            out.append(float(((score - e_m) / e_s + (score - t_m) / t_s) / 2))
        return out
