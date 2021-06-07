import argparse
from pathlib import Path
import unicodedata

import pandas as pd
from sklearn.metrics import roc_auc_score


def eval(args):
    gt = pd.read_csv(args.gt_path, usecols=["article_id", "label"])
    pred = pd.read_csv(args.pred_path, usecols=["article_id", "probability"])
    df = pred.join(gt.set_index("article_id"), on="article_id")
    df["label"] = df["label"].astype(int)

    print(f"auroc: {roc_auc_score(df['label'], df['probability'])}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt_path", type=Path, default="data/rc/dev.csv")
    parser.add_argument("--pred_path", type=Path, default="prediction/dev.csv")
    args = parser.parse_args()

    eval(args)
