#!/usr/bin/env bash

python ./scripts/metadata_to_text.py "ylacombe/mls-eng-10k-tags+ylacombe/libritts_r_tags+ylacombe/libritts_r_tags" \
    --configuration "default+clean+other" \
    --output_dir "./tmp_mls+./tmp_tts_clean+./tmp_tts_other" \
    --cpu_num_workers "8" \
    --leading_split_for_bins "train" \
    --plot_directory "./plots/" \
    --save_bin_edges "./examples/tags_to_annotations/v01_bin_edges.json" \
    --only_save_plot
