import os
import argparse
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import XLNetTokenizerFast
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

from utils import handle_reproducibility, preprocess
from dataset import RiskClassificationDataset
from model import RiskClassificationModel


def train(args: argparse.Namespace) -> None:
    df = pd.read_csv(args.data_dir / "train.csv", usecols=["article_id", "text", "label"])
    # df = preprocess(df)

    valid_ratio = 0.1
    valid_df = df.sample(frac=valid_ratio, random_state=0)
    train_df = df.drop(valid_df.index).reset_index(drop=True)
    valid_df = valid_df.reset_index(drop=True)

    tokenizer = XLNetTokenizerFast.from_pretrained(args.model_name)
    train_set = RiskClassificationDataset(train_df, tokenizer, mode="train")
    valid_set = RiskClassificationDataset(valid_df, tokenizer, mode="valid")

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=16,
        pin_memory=True,
    )
    valid_loader = DataLoader(
        valid_set,
        batch_size=args.batch_size*2,
        shuffle=False,
        num_workers=16,
        pin_memory=True,
    )

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
        # torch.cuda.empty_cache()
        print(f"----- Epoch {epoch} -----")

        model.train()
        optimizer.zero_grad()
        train_loss = 0
        train_corrects = 0
        for batch_idx, (input_ids, token_type_ids, attention_mask, label) in enumerate(tqdm(train_loader)):
            input_ids = input_ids.to(args.device)
            token_type_ids = token_type_ids.to(args.device)
            attention_mask = attention_mask.to(args.device)
            label = label.to(args.device)

            loss, logits = model(
                {
                    "input_ids": input_ids,
                    "token_type_ids": token_type_ids,
                    "attention_mask": attention_mask,
                },
                label
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
        torch.cuda.empty_cache()
        with torch.no_grad():
            model.eval()
            valid_loss = 0
            valid_corrects = 0
            valid_labels = []
            valid_scores = []
            for input_ids, token_type_ids, attention_mask, label in tqdm(valid_loader):
                input_ids = input_ids.to(args.device)
                token_type_ids = token_type_ids.to(args.device)
                attention_mask = attention_mask.to(args.device)
                label = label.to(args.device)

                loss, logits = model(
                    {
                        "input_ids": input_ids,
                        "token_type_ids": token_type_ids,
                        "attention_mask": attention_mask,
                    },
                    label
                )

                valid_loss += loss.item()
                pred = torch.argmax(logits, dim=1)
                valid_corrects += torch.sum(pred == label)
                valid_labels.append(label.cpu().view(-1))
                valid_scores.append(logits.cpu()[:, 1].view(-1))

            valid_labels = torch.cat(valid_labels)
            valid_scores = torch.cat(valid_scores)
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
        default="data/rc/",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=Path,
        default="ckpt/rc/",
    )

    # model
    parser.add_argument("--model_name", type=str, default="hfl/chinese-xlnet-base")

    # optimizer
    parser.add_argument("--lr", type=float, default=1e-5)

    # data loader
    parser.add_argument("--batch_size", type=int, default=1)

    # training
    parser.add_argument("--device", type=torch.device, default="cuda:0")
    parser.add_argument("--n_epoch", type=int, default=20)
    parser.add_argument("--n_batch_per_step", type=int, default=16)
    parser.add_argument("--metric_for_best", type=str, default="valid_auroc")

    # logging
    parser.add_argument("--wandb_logging", type=bool, default=True)
    parser.add_argument("--exp_name", type=str, default="xlnet-2048-fb")
    # parser.add_argument("--exp_name", type=str, default="test")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    handle_reproducibility(True)

    args = parse_args()
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)

    train(args)
