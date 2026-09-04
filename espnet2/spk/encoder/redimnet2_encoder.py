"""ReDimNet2 encoder for ESPnet's speaker task.

Destined for ``espnet2/spk/encoder/redimnet2_encoder.py``.

Paper: I. Yakovlev and A. Okhotnikov, "ReDimNet2: Scaling Speaker Verification
via Time-Pooled Dimension Reshaping", Interspeech 2026, arXiv:2603.11841.

Wraps the featuriser and backbone of ``redimnet2.ReDimNet2Wrap``. Pooling and
projection are supplied separately by ``AstpPooling`` and ESPnet's existing
``RawNet3Projector``, so a released ReDimNet2 checkpoint maps onto the ESPnet
composition by renaming keys rather than by reshaping anything.

Consumes raw waveform, like ``RawNet3Encoder``: ReDimNet2 does its own
featurisation, so ESPnet's frontend must be disabled (``frontend: null``, which
``espnet2/tasks/spk.py`` allows -- its ``frontend_choices`` is
``optional=True``).
"""

from __future__ import annotations


import torch

from espnet2.asr.encoder.abs_encoder import AbsEncoder


class RedimNet2Encoder(AbsEncoder):
    """Extract frame-level ReDimNet2 features from raw waveform.

    Args:
        input_size: unused; present for ESPnet encoder-interface compatibility.
        F: number of mel bins, and the frequency extent of the 2-D pathway.
        C: base channel count.
        out_channels: channel count entering pooling. ``None`` means ``C * F``.
        stages_setup: per-stage ``[stride, num_blocks, expansion, kernels,
            channels]``.
        feat_type: featuriser variant. ``"tf"``/``"tf_mel"`` use
            ``TFMelBanks``; ``"pt"``/``"pt_mel"`` use ``MelBanks``.
        spec_params: forwarded to the featuriser.
        before_pool_offset: leading frames to drop before pooling.
        pad_right_samples: right zero-padding applied to the waveform.
    """

    def __init__(
        self,
        input_size: Optional[int] = None,
        F: int = 72,
        C: int = 64,
        spec_in_channels: int = 1,
        causal: str = "none",
        out_channels: Optional[int] = 224,
        block_1d_type: str = "conv+att",
        block_2d_type: str = "basic_resnet",
        return_2d_output: bool = True,
        fm_weigthing_type: str = "NC",
        use_freq_pos_enc: bool = False,
        compress_tconvs: bool = True,
        stages_setup: Optional[list] = None,
        group_divisor: int = 1,
        dual_agg: bool = False,
        agg_gnorm: bool = False,
        att_dos=None,
        hop_length: int = 160,
        feat_type: str = "tf",
        spec_params: Optional[dict] = None,
        pad_right_samples: Optional[int] = None,
        before_pool_offset: Optional[int] = None,
        **kwargs,
    ):
        super().__init__()

        try:
            from redimnet2.layers import features, features_tf
            from redimnet2.redimnet2 import ReDimNet2
        except ImportError as exc:  # pragma: no cover - environment guard
            raise ImportError(
                "RedimNet2Encoder requires the redimnet2 package. Install it, or "
                "put the vendored checkout on PYTHONPATH."
            ) from exc

        if spec_params is None:
            spec_params = dict(do_spec_aug=False, do_preemph=True, norm_signal=True)

        self.backbone = ReDimNet2(
            F=F,
            C=C,
            spec_in_channels=spec_in_channels,
            causal=causal,
            out_channels=out_channels,
            block_1d_type=block_1d_type,
            block_2d_type=block_2d_type,
            return_2d_output=return_2d_output,
            fm_weigthing_type=fm_weigthing_type,
            use_freq_pos_enc=use_freq_pos_enc,
            compress_tconvs=compress_tconvs,
            stages_setup=stages_setup,
            group_divisor=group_divisor,
            dual_agg=dual_agg,
            agg_gnorm=agg_gnorm,
            att_dos=att_dos,
        )

        if feat_type in ("pt", "pt_mel"):
            self.spec = features.MelBanks(n_mels=F, hop_length=hop_length, **spec_params)
        elif feat_type in ("tf", "tf_mel"):
            self.spec = features_tf.TFMelBanks(
                n_mels=F, hop_length=hop_length, **spec_params
            )
        else:
            raise ValueError(f"unsupported feat_type, got: {feat_type}")

        # Mirrors ReDimNet2Wrap's own output-width derivation. Getting this
        # wrong is invisible until pooling receives the wrong channel count.
        if out_channels is None:
            resolved = C * F
        elif return_2d_output:
            resolved = (F // self.backbone.freq_stride) * out_channels
        else:
            resolved = out_channels

        self._output_size = resolved
        self.pad_right_samples = pad_right_samples
        self.before_pool_offset = before_pool_offset

    def output_size(self) -> int:
        return self._output_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode raw waveform ``(B, T)`` into frame-level features ``(B, C, T')``.

        The speaker task's encoder contract is one tensor in, one tensor out --
        `ESPnetSpeakerModel.encode_frame` calls `self.encoder(feats)` and hands
        the result straight to the aggregator. That differs from the ASR
        contract of the shared `AbsEncoder` base, which takes `(xs_pad, ilens,
        prev_states)` and returns a tuple; following the base class here rather
        than the task cost a training run with

            TypeError: RedimNet2Encoder.forward() missing 1 required
            positional argument: 'ilens'

        With `frontend: null` -- which this encoder requires, since it owns its
        own feature extractor -- `feats` is the raw waveform, so `x` is `(B, T)`.
        Output stays channel-major because that is what the poolings consume.
        """
        if self.pad_right_samples is not None:
            x = torch.nn.functional.pad(x, (0, self.pad_right_samples), value=0.0)

        x = self.spec(x)
        if x.ndim == 3:
            x = x.unsqueeze(1)

        out = self.backbone(x)
        if out.ndim == 4:
            bs, c, f, t = out.size()
            out = out.reshape(bs, c * f, t)
        if self.before_pool_offset is not None:
            out = out[:, :, self.before_pool_offset :]

        return out
