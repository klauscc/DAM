import argparse
import os

import pandas as pd

parser = argparse.ArgumentParser()

parser.add_argument("--video_dir", type=str, required=True, help="The video directory.")
parser.add_argument(
    "--feat_save_dir", type=str, required=True, help="The directory to save features."
)
parser.add_argument(
    "--csv_save_path", type=str, required=True, help="The path to save the csv file."
)

args = parser.parse_args()

vid_names = os.listdir(args.video_dir)

video_paths = []
feat_paths = []

for name in vid_names:
    video_path = os.path.join(args.video_dir, name)

    basename, ext = os.path.splitext(name)
    feat_path = os.path.join(args.feat_save_dir, basename + ".npy")

    video_paths.append(video_path)
    feat_paths.append(feat_path)

df = pd.DataFrame(
    {"video_path": video_paths, "feature_path": feat_paths},
    columns=["video_path", "feature_path"],
)
df.to_csv(args.csv_save_path, index=False)
