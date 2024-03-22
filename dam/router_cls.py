import argparse
import os
import pickle as pkl

import numpy as np
import torch
from sklearn import metrics
from torch.nn import functional as F

data_dir = os.path.join(os.environ["CVL_EXP_DIR"], "frozen_bilm")
# datasets = ["lsmdc", "ivqa", "msrvtt", "msvd", "activitynet", "tgif"]
datasets = ["ivqa", "msvd", "msrvtt", "lsmdc", "activitynet", "tgif"]

save_dir = os.path.join(os.environ["CVL_EXP_DIR"], "frozen_bilm/router/kmeans")

parser = argparse.ArgumentParser()
parser.add_argument("--embedding_dir", type=str, default=data_dir)
parser.add_argument("--save_dir", type=str, default=save_dir)
parser.add_argument("--datasets", nargs="*", default=datasets)

args = parser.parse_args()

data_dir = args.embedding_dir
save_dir = args.save_dir
datasets = args.datasets

os.makedirs(save_dir, exist_ok=True)

means = []

# calculate centers
for i, dataset in enumerate(datasets):
    train_pkl = os.path.join(data_dir, f"zs-{dataset}", f"train_embeddings_{dataset}.pkl")
    with open(train_pkl, "rb") as f:
        train_embeddings = pkl.load(f)
    center = np.mean(train_embeddings, 0)
    means.append(torch.from_numpy(center))
means = torch.stack(means, dim=0)
means = F.normalize(means, dim=1)

# do classification
y_preds = []
num_examples = []
for i, dataset in enumerate(datasets):
    print(f"predict for dataset: {dataset}")
    test_pkl = os.path.join(data_dir, f"zs-{dataset}", f"test_embeddings_{dataset}.pkl")
    with open(test_pkl, "rb") as f:
        embeddings = pkl.load(f)  # [B,C]
    embeddings = torch.from_numpy(embeddings)
    embeddings = F.normalize(embeddings, dim=1)
    num_examples.append(len(embeddings))

    logits = torch.matmul(embeddings, means.transpose(0, 1))  # [B, n_datasets]
    y_pred = logits.argmax(dim=1)  # [B]
    y_preds.append(y_pred)

    # save result
    save_path = os.path.join(save_dir, f"predicted-{dataset}.pkl")
    with open(save_path, "wb") as f:
        pkl.dump(y_pred, f)
    logits_save_path = os.path.join(save_dir, f"logits-{dataset}.pkl")
    with open(logits_save_path, "wb") as f:
        pkl.dump(logits, f)


y_preds = torch.cat(y_preds, dim=0).numpy()

y_true = [[i] * num for i, num in enumerate(num_examples)]
y_true = np.concatenate(y_true)

print(num_examples)
confusion_matrix = metrics.confusion_matrix(y_true, y_preds)
percent = confusion_matrix / np.array(num_examples)[:, None]

print(confusion_matrix)
np.set_printoptions(formatter={"float": lambda x: "{0:0.3f}".format(x)})
print(percent)
avg_acc = np.mean(percent.diagonal())
print(f"avg_acc: {avg_acc}")
