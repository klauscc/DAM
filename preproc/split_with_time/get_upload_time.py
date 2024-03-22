import argparse
from multiprocessing import Pool
import os

import pandas as pd
from yt_dlp import YoutubeDL


def get_upload_date(video_id: str):
    """obtain the upload date of the video.

    Args:
        video_id (str): TODO

    Returns: TODO

    """
    try:
        with YoutubeDL() as ydl:
            info_dict = ydl.extract_info(f"https://youtu.be/{video_id}", download=False)
            upload_date = info_dict.get("upload_date")
            return upload_date
    except Exception as e:
        return "0"


""" test

date = get_upload_date("AKiRZgZlYMI")
print(date)

"""

# define arg parser
parser = argparse.ArgumentParser()
parser.add_argument("--anno_path", "-f", type=str, help="path to the annotation file")
parser.add_argument("--save_path", "-s", type=str, help="save_path")
args = parser.parse_args()

anno_path = args.anno_path
save_path = args.save_path

# anno_path = os.path.join(os.environ["CVL_DATA_DIR"], "frozen_bilm/data/iVQA/train.csv")
# save_path = os.path.join(os.environ["CVL_DATA_DIR"], "frozen_bilm/data/iVQA/upload_date.csv")

# create saving dir

save_dir = os.path.dirname(save_path)
os.makedirs(save_dir, exist_ok=True)

# generate upload_date
annos = pd.read_csv(anno_path)
# print(annos["video_id"])
video_ids = list(annos["video_id"].values)
# print(len(video_ids), video_ids[0])


# upload_dates = []
# for video_id in video_ids[:5]:
#     upload_date = get_upload_date(video_id)
#     upload_dates.append(upload_date)

pool = Pool(8)
upload_dates = pool.map(get_upload_date, video_ids)

upload_date_info = list(zip(video_ids, upload_dates))
upload_date_info = sorted(upload_date_info, key=lambda pair: pair[1])

video_ids_sorted = [x for x, _ in upload_date_info]
upload_dates_sorted = [y for _, y in upload_date_info]


df = pd.DataFrame(
    {"video_id": video_ids_sorted, "upload_date": upload_dates_sorted},
    columns=["video_id", "upload_date"],
)
df.to_csv(save_path, index=False)
