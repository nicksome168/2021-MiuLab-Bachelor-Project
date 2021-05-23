import os
import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import BertTokenizerFast
from tqdm import tqdm

from utils import handle_reproducibility
from dataset import ContextSelectionDataset
from model import ContextSelectionModel


def train(args: argparse.Namespace) -> None:
    with open(args.data_dir / "context.json") as file:
        pg_list = json.load(file)
    with open(args.data_dir / "train.json") as file:
        train_data = json.load(file)
    with open(args.data_dir / "public.json") as file:
        valid_data = json.load(file)

    train_set = ContextSelectionDataset(train_data, pg_list, mode="train")
    valid_set = ContextSelectionDataset(valid_data, pg_list, mode="train")

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=16,
        collate_fn=train_set.collate_fn,
        pin_memory=True,
    )
    valid_loader = DataLoader(
        valid_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=16,
        collate_fn=valid_set.collate_fn,
        pin_memory=True,
    )

    # tokenizer = BertTokenizerFast.from_pretrained("bert-base-chinese")
    tokenizer = BertTokenizerFast.from_pretrained("hfl/chinese-roberta-wwm-ext-large")
    model = ContextSelectionModel()
    model.to(args.device)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)

    if args.wandb_logging:
        import wandb

        wandb.init(project="ADL-hw2-context-selection", entity="mhjuan", name=args.exp_name, config=args)
        wandb.watch(model)
    # print(model)

    best_metric = 0
    for epoch in range(1, args.num_epoch + 1):
        print(f"----- Epoch {epoch} -----")

        model.train()
        optimizer.zero_grad()
        train_loss = 0
        train_corrects = 0
        for batch_idx, (pg, q, label) in enumerate(tqdm(train_loader)):
            x = tokenizer(
                pg,
                q,
                padding="max_length",
                truncation="only_first",
                max_length=512,
                return_tensors="pt",
            )

            x = x.to(args.device)
            label = label.to(args.device)

            loss, logits = model(
                {k: v.view(-1, 7, 512) for k, v in x.items()},
                label,
            )
            
            loss.backward()
            if (batch_idx + 1) % args.n_batch_per_step == 0:
                optimizer.step()
                optimizer.zero_grad()

            train_loss += loss.item()
            pred = torch.argmax(logits, dim=1)
            train_corrects += torch.sum(pred == label)

        train_log = {
            "train_loss": train_loss / len(train_set),
            "train_acc": train_corrects / len(train_set),
        }
        for key, value in train_log.items():
            print(f"{key:30s}: {value:.4}")
        
        # Validation
        with torch.no_grad():
            model.eval()
            valid_loss = 0
            valid_corrects = 0
            for pg, q, label in tqdm(valid_loader):
                x = tokenizer(
                    pg,
                    q,
                    padding="max_length",
                    truncation="only_first",
                    max_length=512,
                    return_tensors="pt",
                )
                x = x.to(args.device)
                label = label.to(args.device)

                loss, logits = model(
                    {k: v.view(-1, 7, 512) for k, v in x.items()},
                    label,
                )

                valid_loss += loss.item()
                pred = torch.argmax(logits, dim=1)
                valid_corrects += torch.sum(pred == label)

            valid_log = {
                "valid_loss": valid_loss / len(valid_set),
                "valid_acc": valid_corrects / len(valid_set),
            }
            for key, value in valid_log.items():
                print(f"{key:30s}: {value:.4}")
            if args.wandb_logging:
                wandb.log({**train_log, **valid_log})

        if valid_log[args.metric_for_best] > best_metric:
            best_metric = valid_log[args.metric_for_best]
            best = True
            if args.wandb_logging:
                wandb.run.summary[f"best_{args.metric_for_best}"] = best_metric
        else:
            best = False

        if best:
            torch.save(model.state_dict(), args.ckpt_dir / f"best_model_{args.exp_name}.pt")
            print(f"{'':30s}*** Best model saved ***")

    if args.wandb_logging:
        wandb.save(str(args.ckpt_dir / f"best_model_{args.exp_name}.pt"))

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=Path,
        help="Directory to the dataset.",
        default="data/",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=Path,
        help="Directory to save model files.",
        default="ckpt/cs/",
    )

    # model

    # optimizer
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--wd", type=float, default=1e-2)

    # data loader
    parser.add_argument("--batch_size", type=int, default=1)

    # training
    parser.add_argument("--device", type=torch.device, default="cuda:0")
    parser.add_argument("--num_epoch", type=int, default=5)
    parser.add_argument("--n_batch_per_step", type=int, default=16)
    parser.add_argument("--metric_for_best", type=str, default="valid_acc")

    # logging
    parser.add_argument("--wandb_logging", type=bool, default=True)
    parser.add_argument("--exp_name", type=str, default="roberta_lr5_bs1_s16")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    handle_reproducibility(True)

    args = parse_args()
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)

    train(args)
