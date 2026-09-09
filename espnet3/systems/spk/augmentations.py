"""Reference-parity augmentation tree for speaker-embedding pretraining.

Contract:
    ``HardAugmentation(config)(speech)`` takes a 1-D float array holding one
    training crop and returns an array of the same length and dtype.

Runtime assumptions:
    Every ``scp`` named in the config exists on the training host and lists
    one audio file per line, either as a bare path or in ESPnet's
    ``<id> <path>`` form. Files are read with ``soundfile``; there is no
    network access on this path.

Failure modes:
    ``FileNotFoundError`` if an scp is missing and ``ValueError`` if a
    probability falls outside [0, 1], an SNR range is inverted, or a branch
    list is empty. All are raised while the preprocessor is being
    constructed, so a misconfigured run dies at startup rather than in the
    middle of an epoch on one unlucky rank.

Why this exists:
    ESPnet's ``SpkPreprocessor`` draws noise and reverberation independently
    at p=0.5 each, so a quarter of every batch receives both -- a combination
    neither released ReDimNet2 recipe ever produces -- while half receives no
    additive noise at all. The authors' ``vb2_vox2_cnc2/b6/ptn.yaml``, the
    config behind their best published number, instead composes

        Sequential(0.9)
          |- OneOf(0.8)  music SNR 2-12 | noise SNR 0-12 | babble SNR 7-18
          |- Reverb(0.2)
          |- OneOf(0.25) LowPass(cutoff_ratio 0.6)

    This module reproduces that tree. The order is theirs and is load-bearing:
    reverberation is applied *after* noise, so it acts on the noisy mixture
    rather than noise being added to an already-reverberant signal.

    Ported from ``wespeaker_lite/dataset/augs.py`` in the authors' release.
    Their classes operate on a dict carrying a torch tensor; these operate on
    the bare numpy array ESPnet's ``_apply_data_augmentation`` passes around,
    but the arithmetic is line-for-line the same.

Colocated docs: ``docs/spkid-reference-parity.md``.
"""

from dataclasses import dataclass, field

import numpy as np
import scipy.signal
import soundfile

# The authors' CustomNoise floors the power terms with this before taking a
# log, and the value changes the SNR of quiet crops enough to be worth keeping.
_POWER_EPS = 1e-5


def _read_scp(path: str) -> list[str]:
    """Audio paths from an scp, accepting both bare and ``<id> <path>`` lines."""
    paths: list[str] = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            fields = line.strip().split(None, 1)
            if not fields:
                continue
            paths.append(fields[-1])
    if not paths:
        raise ValueError(f"{path} lists no audio files")
    return paths


def _check_prob(name: str, value: float) -> float:
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be in [0, 1], got {value}")
    return value


def _crop_or_pad(audio: np.ndarray, length: int) -> np.ndarray:
    """Match an interferer to the crop length, wrapping when it is too short."""
    if len(audio) > length:
        start = np.random.randint(0, len(audio) - length + 1)
        return audio[start : start + length]
    if len(audio) < length:
        return np.pad(audio, (0, length - len(audio) + 1), "wrap")[:length]
    return audio


@dataclass
class NoiseBranch:
    """One arm of the noise ``OneOf``: a corpus, an SNR range, a mix count.

    ``num_mix`` is (3, 7) for babble, which is what makes babble babble --
    several speakers averaged together rather than one interfering voice.
    """

    scp: str
    snr: tuple[float, float]
    name: str = ""
    num_mix: tuple[int, int] = (1, 1)
    paths: list[str] = field(default_factory=list, repr=False)

    def __post_init__(self) -> None:
        # Hydra hands these over as ListConfig, and a list compares unequal to
        # the (1, 1) default that decides whether a branch is babble.
        self.snr = tuple(self.snr)
        self.num_mix = tuple(self.num_mix)
        low, high = self.snr
        if low > high:
            raise ValueError(f"branch {self.name!r}: snr {self.snr} is inverted")
        if self.num_mix[0] > self.num_mix[1] or self.num_mix[0] < 1:
            raise ValueError(f"branch {self.name!r}: bad num_mix {self.num_mix}")
        self.paths = _read_scp(self.scp)

    def draw(self, speech: np.ndarray) -> np.ndarray:
        """The interferer to add, already scaled to a randomly drawn SNR."""
        count = (
            self.num_mix[0]
            if self.num_mix[0] == self.num_mix[1]
            else np.random.randint(self.num_mix[0], self.num_mix[1] + 1)
        )
        drawn = []
        for _ in range(count):
            audio, _ = soundfile.read(np.random.choice(self.paths), dtype="float64")
            if audio.ndim > 1:
                audio = audio[:, 0]
            drawn.append(_crop_or_pad(audio, len(speech)))
        # Averaged, not summed. Summing would make the babble SNR depend on how
        # many speakers happened to be drawn; the authors average so that the
        # requested SNR is the SNR regardless of the count.
        noise = np.mean(drawn, axis=0)

        snr = np.random.uniform(self.snr[0], self.snr[1])
        speech_db = 10 * np.log10(np.mean(speech**2) + _POWER_EPS)
        noise_db = 10 * np.log10(np.mean(noise**2) + _POWER_EPS)
        return noise * np.sqrt(10 ** ((speech_db - noise_db - snr) / 10))


@dataclass
class HardAugConfig:
    """Probabilities and sources for the authors' hard-augmentation tree.

    Defaults are the released ``vb2_vox2_cnc2/b6/ptn.yaml`` values. The one
    departure is ``lowpass_cutoff_ratio``: their file offers 0.5 (which
    band-limits to roughly telephone bandwidth) and 0.6 (a softer limit used
    in place of SpecAugment), with 0.5 commented out. We default to their
    active value.
    """

    branches: list[NoiseBranch]
    rir_scp: str | None = None
    sequential_prob: float = 0.9
    noise_prob: float = 0.8
    reverb_prob: float = 0.2
    reverb_length_sec: float = 1.0
    lowpass_prob: float = 0.25
    lowpass_cutoff_ratio: float = 0.6
    lowpass_transition_width: float = 0.08

    def __post_init__(self) -> None:
        if not self.branches:
            raise ValueError("hard augmentation needs at least one noise branch")
        # Hydra hands nested dataclasses back as omegaconf nodes rather than
        # instances -- `branch.draw(...)` then raises "Key 'draw' not in
        # NoiseBranch" on the first batch, on the pod, an hour into a launch.
        # Coerce here so the type is guaranteed however the config arrived.
        self.branches = [
            branch
            if isinstance(branch, NoiseBranch)
            else NoiseBranch(
                **{k: v for k, v in dict(branch).items() if not k.startswith("_")}
            )
            for branch in self.branches
        ]
        for name in ("sequential_prob", "noise_prob", "reverb_prob", "lowpass_prob"):
            _check_prob(name, getattr(self, name))
        _check_prob("lowpass_cutoff_ratio", self.lowpass_cutoff_ratio)
        if self.reverb_prob > 0 and not self.rir_scp:
            raise ValueError("reverb_prob > 0 but no rir_scp was given")

    def summary_str(self) -> str:
        """One line per stage, for the startup log."""
        arms = ", ".join(
            f"{b.name or 'noise'} SNR {b.snr[0]:g}-{b.snr[1]:g}"
            f"{'' if b.num_mix == (1, 1) else f' x{b.num_mix[0]}-{b.num_mix[1]}'}"
            for b in self.branches
        )
        return (
            f"hard augmentation: Sequential(p={self.sequential_prob}) of "
            f"OneOf(p={self.noise_prob})[{arms}] -> "
            f"Reverb(p={self.reverb_prob}, {self.reverb_length_sec}s) -> "
            f"LowPass(p={self.lowpass_prob}, cutoff={self.lowpass_cutoff_ratio}, "
            f"width={self.lowpass_transition_width})"
        )


class HardAugmentation:
    """The augmentation tree from the authors' best released recipe."""

    def __init__(self, config: HardAugConfig) -> None:
        self.config = config
        self.rirs = _read_scp(config.rir_scp) if config.rir_scp else []

    def __repr__(self) -> str:
        return f"HardAugmentation({self.config.summary_str()})"

    def _reverb(self, speech: np.ndarray) -> np.ndarray:
        rir, rate = soundfile.read(np.random.choice(self.rirs), dtype="float64")
        if rir.ndim > 1:
            rir = rir[:, 0]
        rir = rir[: int(self.config.reverb_length_sec * rate)]
        rir = rir / np.sqrt(np.sum(rir**2) + 1e-10)
        return scipy.signal.convolve(speech, rir, mode="full")[: len(speech)]

    def _lowpass(self, speech: np.ndarray) -> np.ndarray:
        """Smooth spectral roll-off, the authors' stand-in for SpecAugment.

        At cutoff_ratio 0.5 this is also how they produce band-limited
        (telephone-like) speech, which is worth knowing: it is an alternative
        to baking separate 8 kHz copies of every corpus.
        """
        length = len(speech)
        spectrum = np.fft.rfft(speech)
        freqs = np.fft.rfftfreq(length)
        cutoff = self.config.lowpass_cutoff_ratio * freqs[-1]
        width = self.config.lowpass_transition_width * freqs[-1]

        mask = np.ones_like(freqs)
        mask[freqs > cutoff + width / 2] = 0.0
        band = np.logical_and(freqs >= cutoff - width / 2, freqs <= cutoff + width / 2)
        mask[band] = 0.5 * (1 + np.cos(np.pi * (freqs[band] - cutoff) / width))
        return np.fft.irfft(spectrum * mask, n=length)

    def __call__(self, speech: np.ndarray) -> np.ndarray:
        cfg = self.config
        if np.random.rand() > cfg.sequential_prob:
            return speech

        original_dtype = speech.dtype
        speech = speech.astype(np.float64)

        if np.random.rand() <= cfg.noise_prob:
            branch = cfg.branches[np.random.randint(0, len(cfg.branches))]
            speech = speech + branch.draw(speech)

        if self.rirs and np.random.rand() <= cfg.reverb_prob:
            speech = self._reverb(speech)

        if np.random.rand() <= cfg.lowpass_prob:
            speech = self._lowpass(speech)

        return speech.astype(original_dtype)
