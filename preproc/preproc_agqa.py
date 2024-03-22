import collections
import json
import os

import pandas as pd

from args import DATA_DIR

DATASET_NAME = "AGQA"
ORIG_ANNO_DIR = "/data/shared/datasets/vidqa/AGQA"

# create dataset dir.
os.makedirs(os.path.join(DATA_DIR, DATASET_NAME), exist_ok=True)

splits = ["train", "test"]

for split in splits:

    video_ids = []
    questions = []
    answers = []

    # load annotation
    anno_path = os.path.join(ORIG_ANNO_DIR, f"AGQA_balanced/{split}_balanced.txt")
    with open(anno_path, "r") as f:
        data = json.load(f)

    print(f"number of questions: {len(data)}")

    for k, sample in data.items():
        video_ids.append(sample["video_id"])
        questions.append(sample["question"])
        answers.append(sample["answer"])

    df = pd.DataFrame(
        {"video_id": video_ids, "question": questions, "answer": answers},
        columns=["video_id", "question", "answer"],
    )

    # construct vocabulary
    if split == "train":
        answer_set = set(answers)
        print(f"number of answers: {len(answer_set)}")
        top_answers = collections.Counter(answers).most_common(1000)
        vocab = {x[0]: i for i, x in enumerate(top_answers)}
        print(len(df))
        df = df[df["answer"].isin(vocab)]
        json.dump(vocab, open(f"{DATA_DIR}/{DATASET_NAME}/vocab1000.json", "w"))

    print(len(df))
    df.to_csv(f"{DATA_DIR}/{DATASET_NAME}/{split}.csv", index=False)
