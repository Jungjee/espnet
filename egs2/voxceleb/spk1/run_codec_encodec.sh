#!/usr/bin/env bash
set -e
set -u
set -o pipefail


spk_config=conf/train_ECAPA_codec_encodec16k.yaml

train_set="voxceleb1_dev_sp_codec_encodec16k"
valid_set="voxceleb1_test_codec_encodec16k"
cohort_set="voxceleb1_test_codec_encodec16k"
test_sets="voxceleb1_test_codec_encodec16k"
feats_type="extracted"
spk_stats_dir="exp/spk_stats_codec"

./spk_codec.sh \
    --feats_type ${feats_type} \
    --spk_config ${spk_config} \
    --train_set ${train_set} \
    --valid_set ${valid_set} \
    --cohort_set ${cohort_set} \
    --test_sets ${test_sets} \
    --spk_stats_dir ${spk_stats_dir} \
    "$@"