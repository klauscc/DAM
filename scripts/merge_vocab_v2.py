import argparse
import json
import os
from typing import Dict, Set

from args import name2folder

parser = argparse.ArgumentParser()
parser.add_argument("--datasets", action="store", nargs="+")

args = parser.parse_args()

# datasets = "lsmdc ivqa msrvtt msvd activitynet tgif".split(" ")
datasets = args.datasets

num_dsets = len(datasets)

data_dir = os.environ["CVL_DATA_DIR"]

save_filepath = os.path.join(data_dir, f"frozen_bilm/data/vocab_{num_dsets}db.json")

all_vocab: Set[str] = set()

for dataset in datasets:
    vocab_file = os.path.join(
        data_dir, f"frozen_bilm/data/{name2folder[dataset]}/vocab1000.json"
    )
    with open(vocab_file, "r") as f:
        vocab: Dict[str, int] = json.load(f)
    all_vocab = all_vocab.union(set(vocab.keys()))

all_vocab_dict = dict(zip(all_vocab, range(len(all_vocab))))

print(f"The total number of vocabularies: {len(all_vocab_dict)}")

with open(save_filepath, "w") as f:
    json.dump(all_vocab_dict, f)
