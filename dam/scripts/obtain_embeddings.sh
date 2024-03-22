#!/usr/bin/env bash

datasets=$1
outdir=$2

layerid=-4
for dataset in $datasets; do
    ngpus=1
    jobdir=$outdir/zs-$dataset
    mkdir -p $jobdir
    echo "extracting embeddings for dataset:" $dataset, $jobdir
    python -m torch.distributed.launch --master_port 0 --nproc_per_node $ngpus --use_env \
        dam/extract_embedding.py --test --eval \
        --combine_datasets $dataset --combine_datasets_val $dataset --save_dir=$jobdir \
        --ds_factor_ff=8 --ds_factor_attn=8 --suffix='.' \
        --batch_size_val=32 --max_tokens=256 --load=$CVL_DATA_DIR/frozen_bilm/checkpoints/frozenbilm.pth  \
        --embedding_layer $layerid \
        --concat_video_feat

done
