import argparse
import os
from typing import List

import numpy as np
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--anno_dir", type=str, help="The annotation directory")
parser.add_argument("--save_dir", type=str, help="The directory to save the new splits")
parser.add_argument("--num_tasks", type=int, default=5)
args = parser.parse_args()

anno_dir = args.anno_dir
save_dir = args.save_dir
num_tasks = args.num_tasks

"""
anno_dir = os.path.join(os.environ["CVL_DATA_DIR"], "frozen_bilm/data/iVQA")
save_dir = os.path.join(os.environ["CVL_DATA_DIR"], "frozen_bilm/data/iVQA/time_based_splits")
num_splits = 5
"""

os.makedirs(save_dir, exist_ok=True)


def filter_anno(anno_filepath: str, video_ids: List[str], save_path: str):
    """filter annotations and save to `save_path`

    Args:
        anno_filepath (str): The original annotation files
        video_ids (List[str]): The items whose video_ids are in `video_ids` will be kept.
        save_path (str): The path to save the filtered annotations.

    """
    annos_df = pd.read_csv(anno_filepath)
    filtered_df = annos_df[annos_df["video_id"].isin(video_ids)]
    filtered_df.to_csv(save_path, index=False)


# first split train set
train_upload_date = pd.read_csv(os.path.join(anno_dir, "upload_date_train.csv"))
train_subsets = np.array_split(train_upload_date, num_tasks)
train_subsets_vid_ids = [list(s["video_id"].values) for s in train_subsets]

# save train subsets
anno_filepath = os.path.join(anno_dir, "train.csv")
for i, video_ids in enumerate(train_subsets_vid_ids):
    save_path = os.path.join(save_dir, f"train-{i}.csv")
    filter_anno(anno_filepath, video_ids, save_path)


# split val and test set according to train set's splits
subset_end_date = [s.tail(1)["upload_date"].values[0] for s in train_subsets]
print(subset_end_date)


splits = ["val", "test"]
for split in splits:
    vid_info = pd.read_csv(os.path.join(anno_dir, f"upload_date_{split}.csv"))
    subsets_df = []
    for i, end_date in enumerate(subset_end_date):
        if i == 0:
            subset = vid_info[vid_info["upload_date"] <= end_date]
        elif i == len(subset_end_date) - 1:
            subset = vid_info[vid_info["upload_date"] > subset_end_date[i - 1]]
        else:
            subset = vid_info[vid_info["upload_date"] > subset_end_date[i - 1]]
            subset = subset[subset["upload_date"] <= end_date]
        subsets_df.append(subset)

    subsets_vid_ids = [list(s["video_id"].values) for s in subsets_df]
    anno_filepath = os.path.join(anno_dir, f"{split}.csv")
    for i, video_ids in enumerate(subsets_vid_ids):
        save_path = os.path.join(save_dir, f"{split}-{i}.csv")
        filter_anno(anno_filepath, video_ids, save_path)
