import collections
import json
import os

import pandas as pd
import random

from args import DATA_DIR

DATASET_NAME = "sutd-TrafficQA"
ORIG_ANNO_DIR = f"/data/shared/datasets/vidqa/{DATASET_NAME}"

# create dataset dir.
os.makedirs(os.path.join(DATA_DIR, DATASET_NAME), exist_ok=True)

splits = ["train", "test"]

random.seed(20232023)

for split in splits:

    video_ids = []
    questions = []
    answers = []

    # load annotation
    # anno_path = os.path.join(ORIG_ANNO_DIR, f"download/annotations/R2_{split}.jsonl")
    anno_path = os.path.join(ORIG_ANNO_DIR, f"download/annotations/output_file_{split}.jsonl")
    with open(anno_path, "r") as f:
        data = f.readlines()

    _header = data.pop(0)

    print(f"number of questions: {len(data)}")

    for sample in data:
        sample = json.loads(sample.strip())
        vid_filename = sample[2][:-4]
        question: str = sample[4]
        options: list = sample[6:10]
        answer_idx: int = sample[10]
        answer_str: str = options[answer_idx]

        # # yes or no questions
        # if "Yes" in options and "No" in options:
        #     new_question = question
        #     new_answer = answer_str.lower()
        #     # append to annotations
        #     video_ids.append(vid_filename)
        #     questions.append(new_question)
        #     answers.append(new_answer)
        #     continue
        
        version = "v2"

        if version == "v1":
            # 4 option questions
            incorrect_idxs = [i for i in range(4) if i != answer_idx and options[i] != ""]
            incorrect_idx = random.choice(incorrect_idxs)

            for idx in [incorrect_idx, answer_idx]:
                new_question = f"{question} Is it '{options[idx]}'?"
                # new_question = f"{question} {options[idx]}"
                new_answer = "yes" if idx == answer_idx else "no"

                # append to annotations
                video_ids.append(vid_filename)
                questions.append(new_question)
                answers.append(new_answer)
        elif version == "v2":
            for idx in range(4):
                if options[idx] != "":
                    new_question = f"{question} Is it '{options[idx]}'?"
                    new_answer = "yes" if idx == answer_idx else "no"

                    # append to annotations
                    video_ids.append(vid_filename)
                    questions.append(new_question)
                    answers.append(new_answer)


    df = pd.DataFrame(
        {"video_id": video_ids, "question": questions, "answer": answers},
        columns=["video_id", "question", "answer"],
    )

    # construct vocabulary
    if split == "train":
        answer_set = set(answers)
        print(f"number of answers: {len(answer_set)}")
        print(f"{answer_set}")
        top_answers = collections.Counter(answers).most_common(1000)
        vocab = {x[0]: i for i, x in enumerate(top_answers)}
        print(len(df))
        df = df[df["answer"].isin(vocab)]
        json.dump(vocab, open(f"{DATA_DIR}/{DATASET_NAME}/vocab1000.json", "w"))

    print(len(df))
    df.to_csv(f"{DATA_DIR}/{DATASET_NAME}/{split}.csv", index=False)
