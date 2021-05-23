import os
import argparse
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW
from transformers import XLNetTokenizerFast
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

from utils import handle_reproducibility
from dataset import RiskClassificationDataset
from model import RiskClassificationModel


def train(args: argparse.Namespace) -> None:
    df = pd.read_csv(args.data_dir / "train.csv", usecols=["article_id", "text", "label"])

    valid_ratio = 0.1
    valid_df = df.sample(frac=valid_ratio, random_state=0)
    train_df = df.drop(valid_df.index).reset_index(drop=True)
    valid_df = valid_df.reset_index(drop=True)
    
    train_add_list = []
    for i in range(len(train_df)):
        pg = train_df.loc[i, "text"]
        if len(pg) > 2000:
            train_df.loc[i, "text"] = pg[:2048]
            train_add_list.append({
                "text": pg[-2048:],
                "label": train_df.loc[i, "label"],
            })
    train_df = train_df.append(train_add_list, ignore_index=True)
    # print(train_df)
    # return

    train_set = RiskClassificationDataset(train_df, mode="train")
    valid_set = RiskClassificationDataset(valid_df, mode="valid")

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=16,
        pin_memory=True,
    )
    valid_loader = DataLoader(
        valid_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=16,
        pin_memory=True,
    )

    tokenizer = XLNetTokenizerFast.from_pretrained(args.model_name)
    model = RiskClassificationModel(args.model_name)
    model.to(args.device)

    optimizer = AdamW(model.parameters(), lr=args.lr)

    if args.wandb_logging:
        import wandb

        wandb.init(project="AICUP2020-risk-classification", entity="mhjuan", name=args.exp_name, config=args)
        wandb.watch(model)
    # print(model)

    best_metric = 0
    for epoch in range(1, args.n_epoch + 1):
        print(f"----- Epoch {epoch} -----")

        model.train()
        optimizer.zero_grad()
        train_loss = 0
        train_corrects = 0
        for batch_idx, (seq, label) in enumerate(tqdm(train_loader)):
            seq_tok = tokenizer(
                list(seq),
                padding="max_length",
                truncation=True,
                max_length=2048,
                return_tensors="pt",
            )

            seq_tok = seq_tok.to(args.device)
            label = label.to(args.device)

            loss, logits = model(seq_tok, label)
            
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
            valid_labels = []
            valid_scores = []
            for seq, label in tqdm(valid_loader):
                seq_tok = tokenizer(
                    list(seq),
                    padding="max_length",
                    truncation=True,
                    max_length=4096,
                    return_tensors="pt",
                )

                seq_tok = seq_tok.to(args.device)
                label = label.to(args.device)

                loss, logits = model(seq_tok, label)

                valid_loss += loss.item()
                pred = torch.argmax(logits, dim=1)
                valid_corrects += torch.sum(pred == label)
                valid_labels.append(label.cpu().squeeze())
                valid_scores.append(logits.cpu()[:, 1].squeeze())

            valid_labels = torch.stack(valid_labels)
            valid_scores = torch.stack(valid_scores)
            valid_log = {
                "valid_loss": valid_loss / len(valid_set),
                "valid_acc": valid_corrects / len(valid_set),
                "valid_auroc": roc_auc_score(valid_labels, valid_scores),
            }
            for key, value in valid_log.items():
                print(f"{key:30s}: {value:.4}")
            if args.wandb_logging:
                wandb.log({**train_log, **valid_log, "epoch": epoch})

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
        default="data/risk_classification/",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=Path,
        default="ckpt/risk_classification/",
    )

    # model
    parser.add_argument("--model_name", type=str, default="hfl/chinese-xlnet-base")

    # optimizer
    parser.add_argument("--lr", type=float, default=1e-5)

    # data loader
    parser.add_argument("--batch_size", type=int, default=1)

    # training
    parser.add_argument("--device", type=torch.device, default="cuda:0")
    parser.add_argument("--n_epoch", type=int, default=50)
    parser.add_argument("--n_batch_per_step", type=int, default=16)
    parser.add_argument("--metric_for_best", type=str, default="valid_auroc")

    # logging
    parser.add_argument("--wandb_logging", type=bool, default=True)
    parser.add_argument("--exp_name", type=str, default="xlnet-2048-fb")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    handle_reproducibility(True)

    args = parse_args()
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)

    train(args)
