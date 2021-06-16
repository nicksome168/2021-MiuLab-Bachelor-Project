import os
import argparse
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.utils.data.dataset import ConcatDataset
from transformers import BertTokenizer, BertForMultipleChoice
from tqdm import tqdm
from torch.cuda.amp import GradScaler, autocast

from utils import handle_reproducibility, preprocess, read_c3
from dataset import QADataset


def train(args: argparse.Namespace) -> None:
    rc_df = pd.read_csv(args.data_dir / "rc" / "train.csv", usecols=["article_id", "text", "label"])
    qa_df = pd.read_json(args.data_dir / "qa" / "processed_train_150_r2_pg0.json", orient="records")
    train_c3_df = read_c3(args.data_dir / "c3" / "train.json")
    dev_c3_df = read_c3(args.data_dir / "c3" / "dev.json")
    test_c3_df = read_c3(args.data_dir / "c3" / "test.json")
    # train_dream_df = pd.read_json(args.data_dir / "dream" / "train.json", orient="records")
    # dev_dream_df = pd.read_json(args.data_dir / "dream" / "dev.json", orient="records")
    c3_df = train_c3_df.append([dev_c3_df, test_c3_df])
    # dream_df = train_dream_df.append(dev_dream_df)

    rc_df = preprocess(rc_df)
    qa_df = preprocess(qa_df)
    c3_df = preprocess(c3_df)
    # dream_df = preprocess(dream_df)

    valid_ratio = 0.1
    valid_rc_df = rc_df.sample(frac=valid_ratio, random_state=0)

    qa_df = qa_df.set_index("article_id")
    valid_qa_df = qa_df.loc[valid_rc_df["article_id"]]
    train_qa_df = qa_df.drop(valid_qa_df.index).reset_index()
    valid_qa_df = valid_qa_df.reset_index()

    valid_c3_df = c3_df.sample(frac=0.05, random_state=0)
    train_c3_df = c3_df.drop(valid_c3_df.index).reset_index(drop=True)
    valid_c3_df = valid_c3_df.reset_index(drop=True)
    # valid_dream_df = dream_df.sample(frac=0.05, random_state=0)
    # train_dream_df = dream_df.drop(valid_dream_df.index).reset_index(drop=True)
    # valid_dream_df = valid_dream_df.reset_index(drop=True)
    
    tokenizer = BertTokenizer.from_pretrained(args.model_name)
    train_qa_set = QADataset(train_qa_df, tokenizer, mode="train")
    valid_qa_set = QADataset(valid_qa_df, tokenizer, mode="valid")
    train_c3_set = QADataset(train_c3_df, tokenizer, mode="train")
    valid_c3_set = QADataset(valid_c3_df, tokenizer, mode="valid")
    # train_dream_set = QADataset(train_dream_df, tokenizer, mode="train")
    # valid_dream_set = QADataset(valid_dream_df, tokenizer, mode="valid")

    # concat_dataset = ConcatDataset([train_qa_set, train_c3_set, train_dream_set])
    concat_dataset = ConcatDataset([train_qa_set, train_c3_set])
    train_qa_loader = DataLoader(
        # train_qa_set,
        concat_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=16,
        pin_memory=True,
    )
    train_c3_loader = DataLoader(
        train_c3_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=16,
        pin_memory=True,
    )
    valid_qa_loader = DataLoader(
        valid_qa_set,
        batch_size=args.batch_size*8,
        shuffle=False,
        num_workers=16,
        pin_memory=True,
    )
    valid_c3_loader = DataLoader(
        valid_c3_set,
        batch_size=args.batch_size*8,
        shuffle=False,
        num_workers=16,
        pin_memory=True,
    )
    # valid_dream_loader = DataLoader(
    #     valid_dream_set,
    #     batch_size=args.batch_size*8,
    #     shuffle=False,
    #     num_workers=16,
    #     pin_memory=True,
    # )

    model = BertForMultipleChoice.from_pretrained(args.model_name)
    model.to(args.device)

    optimizer = AdamW(model.parameters(), lr=args.lr)
    scaler = GradScaler()

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
        train_qa_loss = 0
        train_c3_loss = 0
        train_qa_corrects = 0
        train_c3_corrects = 0
        train_qa_iter = iter(train_qa_loader)
        train_c3_iter = iter(train_c3_loader)
        for batch_idx in tqdm(range(len(train_qa_loader))):
            # for task in ["c3", "qa"]:
            for task in ["qa"]:
                if task == "c3":
                    try:
                        qa_input_ids, qa_token_type_ids, qa_attention_mask, qa_label = next(train_c3_iter)
                    except StopIteration:
                        train_c3_iter = iter(train_c3_loader)
                        qa_input_ids, qa_token_type_ids, qa_attention_mask, qa_label = next(train_c3_iter)
                else:
                    try:
                        qa_input_ids, qa_token_type_ids, qa_attention_mask, qa_label = next(train_qa_iter)
                    except StopIteration:
                        train_qa_iter = iter(train_qa_loader)
                        qa_input_ids, qa_token_type_ids, qa_attention_mask, qa_label = next(train_qa_iter)

                with autocast():
                    qa_input_ids = qa_input_ids.to(args.device)
                    qa_token_type_ids = qa_token_type_ids.to(args.device)
                    qa_attention_mask = qa_attention_mask.to(args.device)
                    qa_label = qa_label.to(args.device)

                    output = model(
                        **{
                            "input_ids": qa_input_ids,
                            "token_type_ids": qa_token_type_ids,
                            "attention_mask": qa_attention_mask,
                        },
                        labels=qa_label,
                    )
                    qa_loss = output.loss / args.n_batch_per_step
                    qa_logits = output.logits

                scaler.scale(qa_loss).backward()

                if task == "c3":
                    train_c3_loss += qa_loss
                    qa_pred = torch.argmax(qa_logits, dim=1)
                    train_c3_corrects += torch.sum(qa_pred == qa_label)
                else:
                    train_qa_loss += qa_loss
                    qa_pred = torch.argmax(qa_logits, dim=1)
                    train_qa_corrects += torch.sum(qa_pred == qa_label)

            if (batch_idx + 1) % args.n_batch_per_step == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

        n_updates = len(train_qa_loader) / args.n_batch_per_step
        train_qa_loss /= args.n_qa_per_rc
        train_c3_loss /= args.n_qa_per_rc
        train_qa_corrects = train_qa_corrects / args.n_qa_per_rc
        train_c3_corrects = train_c3_corrects / args.n_qa_per_rc
        train_log = {
            "train_loss": (train_qa_loss + train_c3_loss) / n_updates,
            "train_acc": train_qa_corrects / len(concat_dataset),
            # "train_qa_loss": train_qa_loss / n_updates,
            # "train_c3_loss": train_c3_loss / n_updates,
            # "train_qa_acc": train_qa_corrects / len(train_qa_set),
            # "train_c3_acc": train_c3_corrects / len(train_qa_set),
        }
        for key, value in train_log.items():
            print(f"{key:30s}: {value:.4}")
        
        # Validation
        # torch.cuda.empty_cache()
        with torch.no_grad():
            model.eval()
            # for task in ["qa", "c3", "dream"]:
            for task in ["qa", "c3"]:
                valid_loss = 0
                valid_corrects = 0
                if task == "qa":
                    valid_loader = valid_qa_loader
                elif task == "c3":
                    valid_loader = valid_c3_loader
                # else:
                #     valid_loader = valid_dream_loader
                for (
                    batch_idx,
                    (qa_input_ids, qa_token_type_ids, qa_attention_mask, qa_label),
                ) in enumerate(tqdm(valid_loader)):
                    with autocast():
                        qa_input_ids = qa_input_ids.to(args.device)
                        qa_token_type_ids = qa_token_type_ids.to(args.device)
                        qa_attention_mask = qa_attention_mask.to(args.device)
                        qa_label = qa_label.to(args.device)

                        output = model(
                            **{
                                "input_ids": qa_input_ids,
                                "token_type_ids": qa_token_type_ids,
                                "attention_mask": qa_attention_mask,
                            },
                            labels=qa_label,
                        )
                        qa_loss = output.loss
                        qa_logits = output.logits
                
                    valid_loss += qa_loss
                    qa_pred = torch.argmax(qa_logits, dim=1)
                    valid_corrects += torch.sum(qa_pred == qa_label)
                
                if task == "qa":
                    valid_qa_loss = valid_loss
                    valid_qa_corrects = valid_corrects
                elif task == "c3":
                    valid_c3_loss = valid_loss
                    valid_c3_corrects = valid_corrects
                # else:
                #     valid_dream_loss = valid_loss
                #     valid_dream_corrects = valid_corrects
            
            valid_log = {
                "valid_qa_loss": valid_qa_loss / len(valid_qa_loader),
                "valid_c3_loss": valid_c3_loss / len(valid_c3_loader),
                # "valid_dream_loss": valid_dream_loss / len(valid_dream_loader),
                "valid_qa_acc": valid_qa_corrects / len(valid_qa_set),
                "valid_c3_acc": valid_c3_corrects / len(valid_c3_set),
                # "valid_dream_acc": valid_dream_corrects / len(valid_dream_set),
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
        default="data/",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=Path,
        default="ckpt/mt/",
    )

    # model
    parser.add_argument("--model_name", type=str, default="hfl/chinese-macbert-large")

    # optimizer
    parser.add_argument("--lr", type=float, default=3e-5)

    # data loader
    parser.add_argument("--batch_size", type=int, default=1)

    # training
    parser.add_argument("--device", type=torch.device, default="cuda:0")
    parser.add_argument("--n_epoch", type=int, default=50)
    parser.add_argument("--n_batch_per_step", type=int, default=16)
    parser.add_argument("--n_qa_per_rc", type=int, default=1)
    parser.add_argument("--metric_for_best", type=str, default="valid_qa_acc")

    # logging
    parser.add_argument("--wandb_logging", type=bool, default=True)
    parser.add_argument("--exp_name", type=str, default="mac-qa-catc3-150-r2-pg0")
    # parser.add_argument("--exp_name", type=str, default="test")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    handle_reproducibility(True)

    args = parse_args()
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)

    train(args)
