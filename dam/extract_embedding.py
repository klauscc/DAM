import argparse
import datetime
import json
import math
import os
import pickle
import random
import sys
import time
from collections import namedtuple
from functools import reduce
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, DistributedSampler

from args import get_args_parser
from datasets import build_videoqa_dataset, videoqa_collate_fn
from model import build_model, get_tokenizer
from util import dist
from util.metrics import MetricLogger
from util.misc import adjust_learning_rate, get_mask


def train_one_epoch(
    model: torch.nn.Module,
    tokenizer,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    dataset_name,
    args,
    max_norm: float = 0,
):
    model.train()
    metric_logger = MetricLogger(delimiter="  ")
    header = "Epoch: [{}]".format(epoch)
    num_training_steps = int(len(data_loader) * args.epochs)

    for i_batch, batch_dict in enumerate(
        metric_logger.log_every(data_loader, args.print_freq, header)
    ):
        video = batch_dict["video"].to(device)
        video_len = batch_dict["video_len"]
        video_mask = get_mask(video_len, video.size(1)).to(device)
        text = batch_dict["text"]
        encoded = tokenizer(
            text,
            add_special_tokens=True,
            max_length=args.max_tokens,
            padding="longest",
            truncation=True,
            return_tensors="pt",
        )

        inputs = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)

        # forward
        output = model(
            video=video,
            video_mask=video_mask,
            input_ids=inputs,
            attention_mask=attention_mask,
        )
        delay = args.max_feats if args.use_video else 0
        logits = output["logits"][:, delay : encoded["input_ids"].size(1) + delay][
            encoded["input_ids"] == tokenizer.mask_token_id
        ]
        answer_id = batch_dict["answer_id"].to(device)
        if dataset_name == "ivqa":
            a = (answer_id / 2).clamp(max=1)
            nll = -F.log_softmax(logits, 1, _stacklevel=5)
            loss = (nll * a / a.sum(1, keepdim=True).clamp(min=1)).sum(dim=1).mean()
        elif dataset_name == "vqa":
            a = (answer_id / 3).clamp(max=1)
            nll = -F.log_softmax(logits, 1, _stacklevel=5)
            loss = (nll * a / a.sum(1, keepdim=True).clamp(min=1)).sum(dim=1).mean()
        else:
            loss = F.cross_entropy(logits, answer_id)

        loss_dict = {"cls_loss": loss}

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = dist.reduce_dict(loss_dict)
        loss_reduced = sum(loss_dict_reduced.values())
        loss_value = loss_reduced.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        loss.backward()
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        adjust_learning_rate(
            optimizer,
            curr_step=epoch * len(data_loader) + i_batch,
            num_training_steps=num_training_steps,
            args=args,
        )

        metric_logger.update(loss=loss_value, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    tokenizer,
    data_loader,
    device: torch.device,
    dataset_name,
    args,
    thresholds=[1, 10],
    split="test",
    type_map={0: "all"},
):
    model.eval()
    metric_logger = MetricLogger(delimiter="  ")
    header = f"{split}:"

    res = {}
    embeddings = []

    for i_batch, batch_dict in enumerate(
        metric_logger.log_every(data_loader, args.print_freq, header)
    ):
        video = batch_dict["video"].to(device)
        video_len = batch_dict["video_len"]
        video_mask = get_mask(video_len, video.size(1)).to(device)
        text = batch_dict["text"]
        encoded = tokenizer(
            text,
            add_special_tokens=True,
            max_length=args.max_tokens,
            padding="longest",
            truncation=True,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)
        if (
            not args.suffix and not args.use_context
        ):  # remove sep token if not using the suffix
            attention_mask[input_ids == tokenizer.sep_token_id] = 0
            input_ids[input_ids == tokenizer.sep_token_id] = tokenizer.pad_token_id

        output = model(
            video=video,
            video_mask=video_mask,
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        # obtain the representation for the input.
        layer_id = args.embedding_layer
        last_hidden_states = output["hidden_states"][layer_id]  # (B, L, C)
        # input_feature = last_hidden_states.mean(dim=1)  # [B,C]

        mask = torch.cat([video_mask, attention_mask], dim=1)[:, :, None].to(
            last_hidden_states
        )  # [B,L,1]

        input_feature = torch.sum(last_hidden_states * mask, dim=1) / torch.sum(mask, dim=1)

        if args.concat_video_feat:
            if args.use_projected_vid_feat:
                video = output.video
            vid_mask = video_mask[:,:,None]
            vid_feature = (video * vid_mask).sum(dim=1) / vid_mask.sum(dim=1)
            embedding = torch.cat([vid_feature, input_feature], dim=1)
        else:
            embedding = input_feature

        embedding = embedding.detach().cpu()

        embeddings.append(embedding)

        logits = output["logits"]
        delay = args.max_feats if args.use_video else 0
        logits = logits[:, delay : encoded["input_ids"].size(1) + delay][
            encoded["input_ids"] == tokenizer.mask_token_id
        ]  # get the prediction on the mask token
        logits = logits.softmax(-1)
        topk_aids = torch.topk(logits, max(thresholds), -1).indices

        answer_id, qids = batch_dict["answer_id"].to(device), batch_dict["qid"]
        types = batch_dict["type"]
        if "sub" in batch_dict:
            subs = batch_dict["sub"]
        else:
            subs = [0] * len(types)
        if dataset_name not in ["ivqa", "vqa"]:
            answer_id_expanded = answer_id.view(-1, 1).expand_as(topk_aids).to(device)
        elif dataset_name == "ivqa":
            answer_id = (answer_id / 2).clamp(max=1)
            answer_id_expanded = answer_id.to(device)
        elif dataset_name == "vqa":
            answer_id = (answer_id / 3).clamp(max=1)
            answer_id_expanded = answer_id.to(device)

        agreeings = {}
        for x in thresholds:
            if dataset_name not in ["ivqa", "vqa"]:
                agreeings[x] = topk_aids[:, :x] == answer_id_expanded[:, :x]
            else:
                predicted = F.one_hot(
                    topk_aids[:, :x], num_classes=answer_id_expanded.shape[-1]
                ).sum(1)
                agreeings[x] = (predicted * answer_id_expanded).max(1)[0]

        for i, (qid, gt, pred, type, sub) in enumerate(
            zip(qids, answer_id, topk_aids, types, subs)
        ):
            res[qid] = {
                "pred": pred.tolist(),
                "gt": gt.tolist() if dataset_name in ["ivqa", "vqa"] else gt.item(),
                "type": int(type),
                "sub": sub,
            }
            for x in thresholds:
                res[qid][f"acc{x}"] = agreeings[x][i].sum().detach().cpu().item()

        dico = {"acc": agreeings[1].sum() / len(qids)}
        dico_reduced = dist.reduce_dict(dico)
        acc_value = dico_reduced["acc"].item()
        metric_logger.update(acc=acc_value)

    all_res = dist.all_gather(res)
    results = reduce(lambda a, b: a.update(b) or a, all_res, {})
    assert len(results) == len(data_loader.dataset)
    out = {}
    for x in thresholds:
        out[f"acc{x}"] = sum(results[qid][f"acc{x}"] for qid in results) / len(results)
    if type_map is not None and len(type_map) > 1:
        acc_type = {}
        for i in type_map:
            if len([x for x in results.values() if x["type"] == i]) > 0:
                acc_type[type_map[i]] = sum(
                    results[qid][f"acc1"] for qid in results if results[qid]["type"] == i
                ) / len([x for x in results.values() if x["type"] == i])
        # acc_type = {
        #     type_map[i]: sum(
        #         results[qid][f"acc1"] for qid in results if results[qid]["type"] == i
        #     )
        #     / len([x for x in results.values() if x["type"] == i])
        #     for i in type_map
        # }
    n_sub = len([x for x in results.values() if x["sub"]])
    if n_sub:
        acc_sub = sum(results[qid][f"acc1"] for qid in results if results[qid]["sub"]) / n_sub
    if dist.is_main_process():
        print(dataset_name)
        for x in thresholds:
            print(f"{split} acc{x}: {out[f'acc{x}']: .2%}")
        if type_map is not None and len(type_map) > 1:
            for x in acc_type:
                print(f"acc {x}: {acc_type[x]: .2%}")
            out.update(acc_type)
        if n_sub:
            print(f"acc sub: {acc_sub: .2%}; proportion {n_sub / len(results): .2%}")
            out["acc_sub"] = acc_sub

    return results, out, embeddings


def main(args):
    # Init distributed mode
    dist.init_distributed_mode(args)

    if dist.is_main_process():
        if args.save_dir and not (os.path.isdir(args.save_dir)):
            os.makedirs(os.path.join(args.save_dir), exist_ok=True)
        print(args)

    device = torch.device(args.device)

    # Fix seeds
    seed = args.seed + dist.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Build model
    tokenizer = get_tokenizer(args)

    nt = namedtuple(
        typename="data",
        field_names=[
            "dataset_name",
            "dataloader_test",
            "dataloader_train",
        ],
    )

    tuples = []
    for dset_name in args.combine_datasets_val:
        dataloaders = {}
        for split in ["train", "test"]:
            dataset = build_videoqa_dataset(dset_name, split, args, tokenizer)
            sampler_test = (
                DistributedSampler(dataset, shuffle=False)
                if args.distributed
                else torch.utils.data.SequentialSampler(dataset)
            )
            dataloader = DataLoader(
                dataset,
                batch_size=args.batch_size_val,
                sampler=sampler_test,
                collate_fn=videoqa_collate_fn,
                num_workers=args.num_workers,
            )
            dataloaders[split] = dataloader
        tuples.append(
            nt(
                dataset_name=dset_name,
                dataloader_train=dataloaders["train"],
                dataloader_test=dataloaders["test"],
            )
        )

    args.n_ans = len(tuples[0].dataloader_test.dataset.a2id)
    model = build_model(args)
    model.to(device)
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if dist.is_main_process():
        print("number of params:", n_parameters)

    # Load pretrained checkpoint
    if args.load:
        if dist.is_main_process():
            print("loading from", args.load)
        checkpoint = torch.load(args.load, map_location="cpu")
        model.load_state_dict(checkpoint["model"], strict=False)
        if args.resume and not args.eval:
            optimizer.load_state_dict(checkpoint["optimizer"])
            args.start_epoch = checkpoint["epoch"] + 1

    for i, item in enumerate(tuples):
        aid2tokid = torch.zeros(
            len(item.dataloader_test.dataset.a2id), args.max_atokens
        ).long()
        for a, aid in item.dataloader_test.dataset.a2id.items():
            tok = torch.tensor(
                tokenizer(
                    a,
                    add_special_tokens=False,
                    max_length=args.max_atokens,
                    truncation=True,
                    padding="max_length",
                )["input_ids"],
                dtype=torch.long,
            )
            aid2tokid[aid] = tok
        model.set_answer_embeddings(
            aid2tokid.to(model.device), freeze_last=args.freeze_last
        )  # init answer embedding module

        dataloaders = {"train": item.dataloader_train, "test": item.dataloader_test}

        for split, dataloader in dataloaders.items():
            results, out, embeddings = evaluate(
                model=model,
                tokenizer=tokenizer,
                data_loader=dataloader,
                device=device,
                dataset_name=item.dataset_name,
                args=args,
                split=split,
                type_map=item.dataloader_test.dataset.type_map,
            )

            embeddings = torch.cat(embeddings)  # [N,C]
            embeddings = embeddings.numpy()
            if dist.is_main_process():
                print(f"total embeddings: {embeddings.shape}")

            if args.save_dir and dist.is_main_process():
                pickle.dump(
                    embeddings,
                    open(
                        os.path.join(
                            args.save_dir, f"{split}_embeddings_{item.dataset_name}.pkl"
                        ),
                        "wb",
                    ),
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(parents=[get_args_parser()])

    parser.add_argument(
        "--embedding_layer",
        default=-1,
        type=int,
        help="The hidden states of this layer to use as embedding.",
    )

    parser.add_argument(
        "--concat_video_feat",
        action="store_true",
        help="whether to concat video features to the embedding",
    )
    parser.add_argument(
        "--use_projected_vid_feat",
        action="store_true",
        help="whether to concat video features to the embedding",
    )

    args = parser.parse_args()
    if args.save_dir:
        args.save_dir = os.path.join(args.presave_dir, args.save_dir)
    # will download huggingface model.
    # args.model_name = os.path.join(os.environ["TRANSFORMERS_CACHE"], args.model_name)
    main(args)
