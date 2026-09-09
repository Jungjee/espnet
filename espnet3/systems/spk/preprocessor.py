"""Speaker preprocessor adjustments for the ESPnet3 data pipeline."""

import numpy as np
from espnet2.train.preprocessor import SpkPreprocessor as _SpkPreprocessor

from espnet3.systems.spk.augmentations import HardAugConfig, HardAugmentation

_WAVEFORM_KEYS = ("speech", "speech2")


class SpkPreprocessor(_SpkPreprocessor):
    """Speaker preprocessor that always emits float32 waveforms.

    ESPnet2 routes preprocessor output through ``ESPnetDataset``, which casts
    float64 arrays down to float32 before batching. ESPnet3 hands the output
    straight to the collate function instead, and the noise and reverberation
    augmentation of the base class promotes waveforms to float64. Since the
    augmentations are applied probabilistically, that would otherwise leave the
    dtype of a minibatch dependent on which samples happened to be augmented.

    Passing ``hard_aug`` swaps the base class's independent noise and
    reverberation draws for the augmentation tree in
    ``espnet3.systems.spk.augmentations`` -- see that module for why. The base
    class's ``noise_apply_prob`` and ``rir_apply_prob`` are then unused, and
    are asserted to be zero rather than silently ignored, because a config
    carrying both would read as though both were in effect.

    Examples:
        >>> preprocessor = SpkPreprocessor(train=True, target_duration=3.0)
        >>> preprocessor("utt1", {"speech": wav, "spk_labels": "id10001"})
    """

    def __init__(self, *args, hard_aug: HardAugConfig | None = None, **kwargs):
        """Build the preprocessor, optionally replacing its augmentation.

        Args:
            hard_aug: The augmentation tree to use instead of the base
                class's independent draws. ``None`` keeps base behaviour.

        Raises:
            ValueError: If ``hard_aug`` is given alongside a non-zero
                ``noise_apply_prob`` or ``rir_apply_prob``.
        """
        # Checked before super().__init__, which has validation of its own that
        # would otherwise trip first and report a missing noise_info rather
        # than the actual mistake. Both base probabilities default to 1.0, so
        # a config that simply forgets to zero them is caught here too.
        if hard_aug is not None:
            leftovers = {
                name: kwargs.get(name, 1.0)
                for name in ("noise_apply_prob", "rir_apply_prob")
                if kwargs.get(name, 1.0)
            }
            if leftovers:
                raise ValueError(
                    "hard_aug replaces the base augmentation, so "
                    f"{leftovers} must be 0; leaving them set would describe a "
                    "pipeline that is not the one being run"
                )
        super().__init__(*args, **kwargs)
        self.hard_aug = HardAugmentation(hard_aug) if hard_aug is not None else None

    def _speech_process(self, data):
        """Crop as the base class does, then augment through the hard tree.

        Hooking here rather than at ``_apply_data_augmentation`` because the
        base class only reaches that method when ``noise_apply_prob`` or
        ``rir_apply_prob`` is non-zero -- and this path requires both to be
        zero. Overriding the augmentation hook alone would leave a run that
        looks configured for hard augmentation and trains on clean audio.
        """
        data = super()._speech_process(data)
        if self.train and self.hard_aug is not None:
            data["speech"] = self.hard_aug(data["speech"])
        return data

    def __call__(self, uid, data):
        """Preprocess one sample and normalize its waveform dtype.

        Args:
            uid: Sample identifier, unused by the base class.
            data: Sample holding the waveform(s) and the speaker label.

        Returns:
            The preprocessed sample, with every waveform as float32.
        """
        data = super().__call__(uid, data)
        for key in _WAVEFORM_KEYS:
            array = data.get(key)
            if array is not None and array.dtype != np.float32:
                data[key] = array.astype(np.float32)
        return data
