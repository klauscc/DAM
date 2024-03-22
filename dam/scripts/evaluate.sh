#!/usr/bin/env sh

ckptdir=$1
savedir=$2
logits_dir=$3
args=$4

order="ivqa msvd msrvtt lsmdc activitynet tgif"
echo savedir: $savedir
mkdir -p $savedir

python -m torch.distributed.launch --nproc_per_node 1 --master_port 0 --use_env \
    dam/videoqa_inference.py \
    --test --eval \
    --combine_datasets $order --combine_datasets_val $order \
    --lr=5e-5 --schedule=linear_with_warmup  \
    --ds_factor_ff=8 --ds_factor_attn=8 --suffix=\".\" \
    --batch_size=8 --batch_size_val=32 --max_tokens 256 --epochs=20  \
    --unified_vocab_path data/vocab_6db.json \
    --num_models 6  --merge_model \
    --all_datasets $order \
    --router_results $logits_dir \
    --load $ckptdir \
    --save_dir $savedir \
    $args
