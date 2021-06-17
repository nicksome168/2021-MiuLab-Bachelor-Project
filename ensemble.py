import argparse
from pathlib import Path

import pandas as pd
import numpy as np


def ensemble_rc(args):
    ans_df = None
    n_ens = 0
    # for data_path in args.data_dir.iterdir():
    for data_path in ["rc_1.csv", "rc_3.csv", "rc_4.csv"]:
        data_path = args.data_dir / data_path
        n_ens += 1
        df = pd.read_csv(data_path)
        if ans_df is None:
            ans_df = df
        else:
            ans_df["probability"] += df["probability"]

    ans_df["probability"] /= n_ens
    ans_df.to_csv("prediction/rc_dev.csv", index=False)


def ensemble_qa(args):
    vote = {}
    for data_path in args.data_dir.iterdir():
        df = pd.read_csv(data_path)
        for _, row in df.iterrows():
            if vote.get(row["id"]) is None:
                vote[row["id"]] = np.zeros(3)
            if row["answer"] == "A":
                vote[row["id"]][0] += 1
            elif row["answer"] == "B":
                vote[row["id"]][1] += 1
            else:
                vote[row["id"]][2] += 1
    
    for id, vote_list in vote.items():
        max_idx_list = np.argsort(vote_list)[::-1]
        max_vote = vote_list[max_idx_list[0]]
        cand_list = [max_idx_list[0]]
        for idx in max_idx_list[1:]:
            if vote_list[idx] == max_vote:
                cand_list.append(idx)
            else:
                break
        ans = np.random.choice(cand_list)
        if ans == 0:
            ans = "A"
        elif ans == 1:
            ans = "B"
        else:
            ans = "C"
        df.loc[df["id"] == id, "answer"] = ans
    
    df.to_csv("prediction/qa_dev.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, required=True, help="'qa' or 'rc'")
    parser.add_argument("--data_dir", type=Path, required=True)
    args = parser.parse_args()

    if args.task == "rc":
        ensemble_rc(args)
    else:
        ensemble_qa(args)
