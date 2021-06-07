import argparse
import csv
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import XLNetTokenizerFast
from tqdm import tqdm

from utils import handle_reproducibility
from dataset import RiskClassificationDataset
from model import RiskClassificationModel


@torch.no_grad()
def test(args):
    df = pd.read_csv(args.data_path, usecols=["article_id", "text", "label"])
    # df = preprocess(df)

    # valid_ratio = 0.1
    # valid_df = df.sample(frac=valid_ratio, random_state=0)
    # # train_df = df.drop(valid_df.index).reset_index(drop=True)
    # df = valid_df.reset_index(drop=True)

    tokenizer = XLNetTokenizerFast.from_pretrained(args.model_name)
    dataset = RiskClassificationDataset(df, tokenizer=tokenizer, mode="test")

    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=16,
        pin_memory=True,
    )

    model = RiskClassificationModel(args.model_name)
    model.load_state_dict(torch.load(args.ckpt_path))
    model.to(args.device)
    model.eval()

    pred_score_list = []
    id_list = []
    for pg, id in tqdm(data_loader):
        pg = pg.to(args.device)

        logits = model(pg)
        score = torch.sigmoid(logits)

        pred_score_list.append(score.cpu()[:, 1].view(-1))
        id_list.append(id)

    pred_score_list = torch.cat(pred_score_list)
    id_list = torch.cat(id_list)

    with open(args.pred_path, "w") as file:
        writer = csv.DictWriter(file, fieldnames=["article_id", "probability"])
        writer.writeheader()
        for id, score in zip(id_list, pred_score_list):
            writer.writerow({"article_id": id.item(), "probability": score.item()})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        type=Path,
        default="data/rc/dev.csv",
    )
    parser.add_argument("--ckpt_path", type=Path, required=True)
    parser.add_argument("--pred_path", type=Path, default="prediction/dev.csv")

    parser.add_argument("--device", type=torch.device, default="cuda:0")
    parser.add_argument("--model_name", type=str, default="hfl/chinese-xlnet-base")
    parser.add_argument("--batch_size", type=int, default=2)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    handle_reproducibility(True)

    args = parse_args()

    test(args)
