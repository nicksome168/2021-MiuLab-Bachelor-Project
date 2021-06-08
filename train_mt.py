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
from dataset import RiskClassificationDataset, QADataset
from model import MultiTaskModel


def train(args: argparse.Namespace) -> None:
    rc_df = pd.read_csv(args.data_dir / "rc" / "train.csv", usecols=["article_id", "text", "label"])
    qa_df = pd.read_json(args.data_dir / "qa" / "processed_train.json", orient="records")
    rc_df = preprocess(rc_df)
    qa_df = preprocess(qa_df)

    valid_ratio = 0.1
    valid_rc_df = rc_df.sample(frac=valid_ratio, random_state=0)
    train_rc_df = rc_df.drop(valid_rc_df.index).reset_index(drop=True)
    valid_rc_df = valid_rc_df.reset_index(drop=True)
    valid_qa_df = qa_df.sample(frac=valid_ratio, random_state=0)
    train_qa_df = qa_df.drop(valid_qa_df.index).reset_index(drop=True)
    valid_qa_df = valid_qa_df.reset_index(drop=True)
    
    tokenizer = XLNetTokenizerFast.from_pretrained(args.model_name)
    train_rc_set = RiskClassificationDataset(train_rc_df, tokenizer, mode="train")
    valid_rc_set = RiskClassificationDataset(valid_rc_df, tokenizer, mode="valid")
    train_qa_set = QADataset(train_qa_df, tokenizer, mode="train")
    valid_qa_set = QADataset(valid_qa_df, tokenizer, mode="valid")

    train_rc_loader = DataLoader(
        train_rc_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=16,
        pin_memory=True,
    )
    valid_rc_loader = DataLoader(
        valid_rc_set,
        batch_size=args.batch_size*2,
        shuffle=False,
        num_workers=16,
        pin_memory=True,
    )
    train_qa_loader = DataLoader(
        train_qa_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=16,
        pin_memory=True,
    )
    valid_qa_loader = DataLoader(
        valid_qa_set,
        batch_size=args.batch_size*2,
        shuffle=False,
        num_workers=16,
        pin_memory=True,
    )

    model = MultiTaskModel(args.model_name)
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
        train_rc_loss = 0
        train_qa_loss = 0
        train_rc_corrects = 0
        train_qa_corrects = 0
        train_qa_iter = iter(train_qa_loader)
        for (
            batch_idx,
            (rc_input_ids, rc_token_type_ids, rc_attention_mask, rc_label),
        ) in enumerate(tqdm(train_rc_loader)):
            rc_input_ids = rc_input_ids.to(args.device)
            rc_token_type_ids = rc_token_type_ids.to(args.device)
            rc_attention_mask = rc_attention_mask.to(args.device)
            rc_label = rc_label.to(args.device)

            rc_loss, rc_logits = model(
                "rc",
                {
                    "input_ids": rc_input_ids,
                    "token_type_ids": rc_token_type_ids,
                    "attention_mask": rc_attention_mask,
                },
                rc_label,
            )

            rc_loss.backward()

            try:
                qa_input_ids, qa_token_type_ids, qa_attention_mask, qa_label = next(train_qa_iter)
            except StopIteration:
                train_qa_iter = iter(train_qa_loader)
                qa_input_ids, qa_token_type_ids, qa_attention_mask, qa_label = next(train_qa_iter)

            qa_input_ids = qa_input_ids.to(args.device)
            qa_token_type_ids = qa_token_type_ids.to(args.device)
            qa_attention_mask = qa_attention_mask.to(args.device)
            qa_label = qa_label.to(args.device)

            qa_loss, qa_logits = model(
                "qa",
                {
                    "input_ids": qa_input_ids,
                    "token_type_ids": qa_token_type_ids,
                    "attention_mask": qa_attention_mask,
                },
                qa_label,
            )
            
            qa_loss.backward()
            # loss = rc_loss + qa_loss
            # loss.backward()
            if (batch_idx + 1) % args.n_batch_per_step == 0:
                optimizer.step()
                optimizer.zero_grad()

            train_rc_loss += rc_loss.item()
            train_qa_loss += qa_loss.item()
            rc_pred = torch.argmax(rc_logits, dim=1)
            train_rc_corrects += torch.sum(rc_pred == rc_label)
            qa_pred = torch.argmax(qa_logits, dim=1)
            train_qa_corrects += torch.sum(qa_pred == qa_label)

        train_log = {
            "train_loss": (train_rc_loss + train_qa_loss) / len(train_rc_loader),
            "train_rc_loss": train_rc_loss / len(train_rc_loader),
            "train_qa_loss": train_qa_loss / len(train_rc_loader),
            "train_rc_acc": train_rc_corrects / len(train_rc_set),
            "train_qa_acc": train_qa_corrects / len(train_rc_set),
        }
        for key, value in train_log.items():
            print(f"{key:30s}: {value:.4}")
        
        # Validation
        torch.cuda.empty_cache()
        with torch.no_grad():
            model.eval()
            valid_rc_loss = 0
            valid_qa_loss = 0
            valid_rc_corrects = 0
            valid_qa_corrects = 0
            valid_rc_labels = []
            valid_rc_scores = []
            for (
                batch_idx,
                (rc_input_ids, rc_token_type_ids, rc_attention_mask, rc_label),
            ) in enumerate(tqdm(valid_rc_loader)):
                rc_input_ids = rc_input_ids.to(args.device)
                rc_token_type_ids = rc_token_type_ids.to(args.device)
                rc_attention_mask = rc_attention_mask.to(args.device)
                rc_label = rc_label.to(args.device)

                rc_loss, rc_logits = model(
                    "rc",
                    {
                        "input_ids": rc_input_ids,
                        "token_type_ids": rc_token_type_ids,
                        "attention_mask": rc_attention_mask,
                    },
                    rc_label,
                )

                valid_rc_loss += rc_loss.item()
                rc_pred = torch.argmax(rc_logits, dim=1)
                valid_rc_corrects += torch.sum(rc_pred == rc_label)
                valid_rc_labels.append(rc_label.cpu().view(-1))
                valid_rc_scores.append(rc_logits.cpu()[:, 1].view(-1))

            for (
                batch_idx,
                (qa_input_ids, qa_token_type_ids, qa_attention_mask, qa_label),
            ) in enumerate(tqdm(valid_qa_loader)):
                qa_input_ids = qa_input_ids.to(args.device)
                qa_token_type_ids = qa_token_type_ids.to(args.device)
                qa_attention_mask = qa_attention_mask.to(args.device)
                qa_label = qa_label.to(args.device)

                qa_loss, qa_logits = model(
                    "qa",
                    {
                        "input_ids": qa_input_ids,
                        "token_type_ids": qa_token_type_ids,
                        "attention_mask": qa_attention_mask,
                    },
                    qa_label,
                )
            
                valid_qa_loss += qa_loss.item()
                qa_pred = torch.argmax(qa_logits, dim=1)
                valid_qa_corrects += torch.sum(qa_pred == qa_label)

            valid_rc_labels = torch.cat(valid_rc_labels)
            valid_rc_scores = torch.cat(valid_rc_scores)
            valid_log = {
                "valid_rc_loss": valid_rc_loss / len(valid_rc_loader),
                "valid_qa_loss": valid_qa_loss / len(valid_qa_loader),
                "valid_rc_acc": valid_rc_corrects / len(valid_rc_set),
                "valid_rc_auroc": roc_auc_score(valid_rc_labels, valid_rc_scores),
                "valid_qa_acc": valid_qa_corrects / len(valid_qa_set),
            }
            for key, value in valid_log.items():
                print(f"{key:30s}: {value:.4}")
            if args.wandb_logging:
                wandb.log({**train_log, **valid_log, "rc_epoch": epoch})

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
        default="data/",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=Path,
        default="ckpt/mt/",
    )

    # model
    parser.add_argument("--model_name", type=str, default="hfl/chinese-xlnet-base")

    # optimizer
    parser.add_argument("--lr", type=float, default=1e-5)

    # data loader
    parser.add_argument("--batch_size", type=int, default=1)

    # training
    parser.add_argument("--device", type=torch.device, default="cuda:0")
    parser.add_argument("--n_epoch", type=int, default=40)
    parser.add_argument("--n_batch_per_step", type=int, default=16)
    parser.add_argument("--metric_for_best", type=str, default="valid_qa_acc")

    # logging
    parser.add_argument("--wandb_logging", type=bool, default=True)
    parser.add_argument("--exp_name", type=str, default="xlnet-mt")
    # parser.add_argument("--exp_name", type=str, default="test")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    handle_reproducibility(True)

    args = parse_args()
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)

    train(args)
