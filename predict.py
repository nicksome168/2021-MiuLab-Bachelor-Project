import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizerFast
from tqdm import tqdm

from utils import handle_reproducibility, se_to_str, Metrics
from dataset import ContextSelectionDataset
from model import ContextSelectionModel, QAModel
from eval import main as eval


@torch.no_grad()
def test(args):
    with open(args.context_path) as file:
        pg_list = json.load(file)
    with open(args.data_path) as file:
        data = json.load(file)

    dataset = ContextSelectionDataset(data, pg_list, mode="test")

    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=16,
        collate_fn=dataset.collate_fn,
        pin_memory=True,
    )

    # tokenizer = BertTokenizerFast.from_pretrained("bert-base-chinese")
    # tokenizer = BertTokenizerFast.from_pretrained("hfl/chinese-roberta-wwm-ext-large")
    tokenizer = BertTokenizerFast.from_pretrained("./ckpt/")
    cs_model = ContextSelectionModel()
    qa_model = QAModel()
    cs_model.load_state_dict(torch.load(args.cs_ckpt_path))
    qa_model.load_state_dict(torch.load(args.qa_ckpt_path))
    cs_model.to(args.device)
    qa_model.to(args.device)
    cs_model.eval()
    qa_model.eval()

    pred_dict = {}
    data_i = 0
    for pg, q in tqdm(data_loader):
        x = tokenizer(
            pg,
            q,
            padding="max_length",
            truncation="only_first",
            max_length=512,
            return_tensors="pt",
        )
        x = x.to(args.device)

        model_input = {k: v.view(-1, 7, 512) for k, v in x.items()}
        logits = cs_model(model_input)

        pred_context = torch.argmax(logits, dim=1)
        
        qa_model_input = {"input_ids": [], "token_type_ids": [], "attention_mask": []}
        for i, p in enumerate(pred_context):
            for k, v in model_input.items():
                qa_model_input[k].append(v[i, p])
        for k, v in model_input.items():
            qa_model_input[k] = torch.stack(qa_model_input[k])

        start_scores, end_scores = qa_model(qa_model_input)

        for i, (ss, es) in enumerate(zip(start_scores, end_scores)):
            batch_idx = i * 7 + pred_context[i]
            ps, pe, _ = Metrics.search_best(x, batch_idx, ss, es)
            pred_str = se_to_str(x, batch_idx, ps, pe, pg[batch_idx])
            # print(pred_str, ps, pe)
            pred_dict.update({data[data_i]["id"]: pred_str})
            data_i += 1

    with open(args.pred_path, "w") as file:
        json.dump(pred_dict, file, ensure_ascii=False)

    if args.eval_output_path:
        eval(args.data_path, args.pred_path, args.eval_output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        type=Path,
        default="data/public.json",
    )
    parser.add_argument(
        "--context_path",
        type=Path,
        default="data/context.json",
    )
    parser.add_argument("--cs_ckpt_path", type=Path, required=True)
    parser.add_argument("--qa_ckpt_path", type=Path, required=True)
    parser.add_argument("--pred_path", type=Path, default="prediction/public.json")
    parser.add_argument("--eval_output_path", type=Path, default=None)


    parser.add_argument("--device", type=torch.device, default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=16)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    handle_reproducibility(True)

    args = parse_args()

    test(args)
