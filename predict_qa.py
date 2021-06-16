import argparse
import csv
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast
from transformers import BertTokenizer, BertForMultipleChoice
from tqdm import tqdm

from utils import handle_reproducibility, preprocess
from dataset import QADataset


@torch.no_grad()
def predict_qa(args):
    df = pd.read_json(args.data_path, orient="records")
    df = preprocess(df)

    tokenizer = BertTokenizer.from_pretrained(args.model_name)
    dataset = QADataset(df, tokenizer=tokenizer, mode="test")

    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=16,
        pin_memory=True,
    )

    model = BertForMultipleChoice.from_pretrained(args.model_name)
    model.load_state_dict(torch.load(args.ckpt_path))
    model.to(args.device)
    model.eval()

    pred_list = []
    id_list = []
    for input_ids, token_type_ids, attention_mask, id in tqdm(data_loader):
        with autocast():
            input_ids = input_ids.to(args.device)
            token_type_ids = token_type_ids.to(args.device)
            attention_mask = attention_mask.to(args.device)

            logits = model(**{
                "input_ids": input_ids,
                "token_type_ids": token_type_ids,
                "attention_mask": attention_mask,
            }).logits

        pred = torch.argmax(logits, dim=1)
        pred_list.append(pred)
        id_list.append(id)

    pred_list = torch.cat(pred_list)
    pred_ans_list = []
    for pred in pred_list:
        if pred == 0:
            pred_ans_list.append("A")
        elif pred == 1:
            pred_ans_list.append("B")
        else:
            pred_ans_list.append("C")
    id_list = torch.cat(id_list)

    with open(args.pred_path, "w") as file:
        writer = csv.DictWriter(file, fieldnames=["id", "answer"])
        writer.writeheader()
        for id, pred in zip(id_list, pred_ans_list):
            writer.writerow({"id": id.item(), "answer": pred})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=Path, required=True)
    parser.add_argument("--pred_path", type=Path, required=True)

    parser.add_argument("--ckpt_path", type=Path, required=True)

    parser.add_argument("--device", type=torch.device, default="cuda:0")
    parser.add_argument("--model_name", type=str, default="hfl/chinese-macbert-large")
    parser.add_argument("--batch_size", type=int, default=1)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    handle_reproducibility(True)

    args = parse_args()

    predict_qa(args)
