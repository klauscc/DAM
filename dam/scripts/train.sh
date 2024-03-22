#!/usr/bin/env bash

outdir=$1
ngpus=$2
order=$3
model_args=$4

pre_ckpt=$CVL_DATA_DIR/frozen_bilm/checkpoints/frozenbilm.pth

codedir=`pwd`

taskid=0
for dataset in $order; do

    port="`shuf -i 20000-30000 -n 1`"
    jobdir=$outdir/ft-$dataset
    mkdir -p $jobdir
    echo $dataset, $port, $jobdir

    # finetune
    python -m torch.distributed.launch --nproc_per_node $ngpus --master_port $port --use_env \
        videoqa.py \
        --combine_datasets $dataset --combine_datasets_val $dataset --save_dir=$jobdir \
        --lr=5e-5 --schedule=linear_with_warmup  \
        --ds_factor_ff=8 --ds_factor_attn=8 --suffix=\".\" \
        --batch_size=8 --batch_size_val=32 --max_tokens 256 --epochs=20  \
        --unified_vocab_path data/vocab_6db.json \
        --freeze_ln \
        --load=${pre_ckpt} \
        $model_args

    eval_ckpt=$jobdir/best_model.pth

    # evaluate
    eval_db=$order
    eval_jobdir=$jobdir/eval
    mkdir -p $eval_jobdir
    port="`shuf -i 30000-40000 -n 1`"
    python -m torch.distributed.launch --nproc_per_node 1 --master_port $port --use_env \
        videoqa.py \
        --combine_datasets $eval_db --combine_datasets_val $eval_db --save_dir=$jobdir \
        --lr=5e-5 --schedule=linear_with_warmup  \
        --ds_factor_ff=8 --ds_factor_attn=8 --suffix="." \
        --batch_size=8 --batch_size_val=32 --max_tokens 256 --epochs=20  \
        --unified_vocab_path data/vocab_6db.json \
        --load=${eval_ckpt} \
        --freeze_ln \
        --eval --test \
        $model_args

    taskid=$((taskid+1))

    pre_ckpt=$eval_ckpt
done

