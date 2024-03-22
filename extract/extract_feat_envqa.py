import argparse
import json
import math
import os

import clip
import numpy as np
import torch as th
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from args import MODEL_DIR
from extract.preprocessing import Preprocessing
from extract.random_sequence_shuffler import RandomSequenceSampler
from extract.video_loader import VideoLoader
from util import dist


class EnvQAVideoLoader(Dataset):

    """pytorch videodataset for Env-QA dataset."""

    def __init__(
        self,
        anno_filepath: str,
        video_dir: str,
        save_dir: str,
        downsample: int = 4,
        size: int = 224,
        centercrop=False,
    ):
        """

        Args:
            anno_filepath (str): The annotation filepath.
            video_dir (str): The directory to save the videos.
            save_dir (str): The directory to save the features.

        Kwargs:
            downsample (int): The temporal downsample rate.
            size (int): The image spatial size.
            centercrop (TODO): TODO


        """
        super().__init__()

        self.video_dir = video_dir
        self.save_dir = save_dir

        self.downsample = downsample
        self.size = size
        self.centercrop = centercrop

        with open(anno_filepath, "r") as f:
            self.video_info_dict = json.load(f)
        self.video_ids = list(self.video_info_dict.keys())
        self.video_infos = list(self.video_info_dict.values())

    def __len__(self):
        return len(self.video_info_dict)

    def __getitem__(self, idx):
        video_id: str = self.video_ids[idx]
        video_info = self.video_infos[idx]

        frame_ids = sorted(video_info["frame_ids"])

        # downsample video if too long
        if len(frame_ids) > 8 * self.downsample:
            frame_ids = frame_ids[:: self.downsample]

        # video_id: FloorPlan403_physics_25_3Step_06
        parts = video_id.split("_")
        video_name = "_".join(parts[:2])
        video_clip = parts[2]

        # create input output path.
        video_path = os.path.join(self.video_dir, video_name, video_clip)
        output_file = os.path.join(self.save_dir, video_id + ".npy")

        # load frames
        video = []
        for _, frame_id in enumerate(frame_ids):
            frame_path = os.path.join(video_path, frame_id + ".png")
            if os.path.isfile(frame_path):
                img = Image.open(frame_path).resize((self.size, self.size))
                video.append(np.array(img))

        # return if no frame exists
        if len(video) == 0:
            return {"video": th.zeros(1), "input": video_path, "output": output_file}

        video = np.stack(video, axis=0)
        video = th.from_numpy(video).to(th.float32)
        video = video.permute(0, 3, 1, 2)  # BCHW

        return {"video": video, "input": video_path, "output": output_file}


parser = argparse.ArgumentParser(description="Easy video feature extractor")

parser.add_argument(
    "--vid_anno_filepath",
    type=str,
    required=True,
    help="The json annotation file to specify the video segments.",
)
parser.add_argument(
    "--video_dir", type=str, required=True, help="The directory of the video frames"
)
parser.add_argument(
    "--save_dir", type=str, required=True, help="The directory to save the video features"
)

parser.add_argument("--batch_size", type=int, default=128, help="batch size for extraction")
parser.add_argument(
    "--num_decoding_thread",
    type=int,
    default=3,
    help="number of parallel threads for video decoding",
)
parser.add_argument(
    "--half_precision",
    type=int,
    default=1,
    help="whether to output half precision float or not",
)
parser.add_argument(
    "--l2_normalize",
    type=int,
    default=0,
    help="whether to l2 normalize the output feature",
)
parser.add_argument(
    "--feature_dim", type=int, default=768, help="output video feature dimension"
)
parser.add_argument("--framerate", type=int, default=1, help="output video feature dimension")
args = parser.parse_args()

args.dist_url = "env://"

dist.init_distributed_mode(args)

dataset = EnvQAVideoLoader(args.vid_anno_filepath, args.video_dir, args.save_dir, size=224)
n_dataset = len(dataset)
sampler = RandomSequenceSampler(n_dataset, 10)
loader = DataLoader(
    dataset,
    batch_size=1,
    shuffle=False,
    num_workers=args.num_decoding_thread,
    sampler=sampler if n_dataset > 10 else None,
)
preprocess = Preprocessing()
model, _ = clip.load("ViT-L/14", download_root=MODEL_DIR)
model.eval()
model = model.cuda()

with th.no_grad():
    for k, data in enumerate(loader):
        input_file = data["input"][0]
        output_file = data["output"][0]
        if os.path.isfile(output_file):
            print(f"{output_file} already generated. Skip.............")
            continue
        if len(data["video"].shape) > 3:
            print("Computing features of video {}/{}: {}".format(k + 1, n_dataset, input_file))
            video = data["video"].squeeze()
            if len(video.shape) == 4:
                video = preprocess(video)
                n_chunk = len(video)
                features = th.cuda.FloatTensor(n_chunk, args.feature_dim).fill_(0)
                n_iter = int(math.ceil(n_chunk / float(args.batch_size)))
                for i in tqdm(range(n_iter)):
                    min_ind = i * args.batch_size
                    max_ind = (i + 1) * args.batch_size
                    video_batch = video[min_ind:max_ind].cuda()
                    batch_features = model.encode_image(video_batch)
                    if args.l2_normalize:
                        batch_features = F.normalize(batch_features, dim=1)
                    features[min_ind:max_ind] = batch_features
                features = features.cpu().numpy()
                if args.half_precision:
                    features = features.astype("float16")
                np.save(output_file, features)
        else:
            print("Video {} already processed.".format(input_file))
