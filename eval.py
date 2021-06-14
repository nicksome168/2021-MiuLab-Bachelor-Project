import argparse
from pathlib import Path
import unicodedata

import pandas as pd
from sklearn.metrics import roc_auc_score


def eval_rc(args):
    gt = pd.read_csv(args.gt_path, usecols=["article_id", "label"])
    pred = pd.read_csv(args.pred_path, usecols=["article_id", "probability"])
    df = pred.join(gt.set_index("article_id"), on="article_id")
    df["label"] = df["label"].astype(int)

    print(f"auroc: {roc_auc_score(df['label'], df['probability'])}")


def eval_qa(args):
    gt = pd.read_json(args.gt_path, orient="records")
    pred = pd.read_csv(args.pred_path)
    acc = sum(gt["answer"] == pred["answer"]) / len(gt)

    print(f"acc: {acc}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, required=True, help="either 'rc' or 'qa'")
    parser.add_argument("--gt_path", type=Path, default="data/rc/dev.csv")
    parser.add_argument("--pred_path", type=Path, default="prediction/dev.csv")
    args = parser.parse_args()

    if args.task == "rc":
        eval_rc(args)
    else:
        eval_qa(args)
