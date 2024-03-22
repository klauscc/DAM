import collections
import json
import os

import pandas as pd

from args import DATA_DIR

DATASET_NAME = "env-qa"
ORIG_ANNO_DIR = f"/data/shared/datasets/vidqa/{DATASET_NAME}/download"

# create dataset dir.
os.makedirs(os.path.join(DATA_DIR, DATASET_NAME), exist_ok=True)

# vid_anno_path = os.path.join(ORIG_ANNO_DIR, "env_qa_video_annotations_v1.json")
#
# with open(vid_anno_path, "r") as f:
#     vid_annos = json.load(f)

splits = ["train", "val", "test"]

for split in splits:

    video_ids = []
    questions = []
    answers = []

    # load annotation
    anno_path = os.path.join(ORIG_ANNO_DIR, f"{split}_full_question.json")
    with open(anno_path, "r") as f:
        data = json.load(f)

    print(f"number of questions: {len(data)}")

    for sample in data:

        if sample["answer"] == "" or sample["question"] == "":
            continue

        if sample["answer"] in ["yes, it is", "yes, it has"]:
            sample["answer"] = "yes"
        if sample["answer"] in ["no, it isn't", "no, it hasn't"]:
            sample["answer"] = "no"

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
        all_vocab = {x: i for i, x in enumerate(answer_set)}
        json.dump(all_vocab, open(f"{DATA_DIR}/{DATASET_NAME}/vocab.json", "w"))

        top_answers = collections.Counter(answers).most_common(2000)
        print(top_answers[1000:2000:100])
        top_answers = collections.Counter(answers).most_common(1000)
        vocab = {x[0]: i for i, x in enumerate(top_answers)}
        print(len(df))
        # df = df[df["answer"].isin(vocab)]
        json.dump(vocab, open(f"{DATA_DIR}/{DATASET_NAME}/vocab1000.json", "w"))

    print(len(df))
    df.to_csv(f"{DATA_DIR}/{DATASET_NAME}/{split}.csv", index=False)
