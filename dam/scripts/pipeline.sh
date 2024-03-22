#!/usr/bin/env bash

ngpus=4
expname="dam/vidqa-1.0"
datasets="ivqa msvd msrvtt lsmdc activitynet tgif"
args=" " # additional arguments passed to model training script.
ckptdir=$CVL_EXP_DIR/frozen_bilm/${expname}/train
embedding_dir=$CVL_EXP_DIR/frozen_bilm/${expname}/embeddings
logits_dir=$CVL_EXP_DIR/frozen_bilm/${expname}/routers
eval_res_dir=$CVL_EXP_DIR/frozen_bilm/${expname}/evaluate

# step 1: continual finetuning on each dataset
bash dam/scripts/train.sh $ckptdir $ngpus "$datasets" "$args"

# step 2: extract embeddings for each dataset
bash dam/scripts/obtain_embeddings.sh "$datasets" $embedding_dir

# step 3: predict dataset identity for each sample in each dataset
python dam/router_cls.py --embedding_dir  $embedding_dir --save_dir $logits_dir

# step 4: inference and evaluation
eval_args="--router_temp 0.01 --topk 2"
bash dam/scripts/evaluate.sh $ckptdir $eval_res_dir $logits_dir "${eval_args}"

# step 5: print the evaluation accuracies
tail $eval_res_dir/*ry.json
