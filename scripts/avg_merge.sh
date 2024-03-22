#!/usr/bin/env sh

jobdir=$1

dsets="ivqa msvd msrvtt lsmdc activitynet tgif"
port="`shuf -i 20000-30000 -n 1`"
w="[1/6, 1/6, 1/6, 1/6, 1/6, 1/6]"
expname="one_adapter_cl/freeze_ln"
python -m torch.distributed.launch --master_port $port --nproc_per_node 1 --use_env\
    scripts/model_merge.py \
    --eval --test \
    --ds_factor_ff=8 --ds_factor_attn=8 --suffix='.' \
    --batch_size_val=32 --max_tokens=256 \
    --unified_vocab_path $CVL_DATA_DIR/frozen_bilm/data/vocab_6db.json \
    --combine_datasets $dsets \
    --combine_datasets_val $dsets \
    --model_weight "$w" \
    --load=$jobdir  \
    --save_dir=$jobdir/avg_merge
