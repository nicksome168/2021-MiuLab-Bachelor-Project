import unicodedata
import json
from pathlib import Path
import random

import pandas as pd
import torch


def handle_reproducibility(is_reproducible: bool = True, rand_seed: int = 0) -> None:
    if is_reproducible:
        print(f'random seed {rand_seed}')
        torch.manual_seed(rand_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    # df["text"] = df["text"].apply(lambda text: text.replace("個管師：", "個："))
    # df["text"] = df["text"].apply(lambda text: text.replace("醫師：", "醫："))
    # df["text"] = df["text"].apply(lambda text: text.replace("民眾：", "民："))
    # df["text"] = df["text"].apply(lambda text: text.replace("家屬：", "家："))
    df["text"] = df["text"].apply(lambda text: text.replace("……", "⋯"))
    df["text"] = df["text"].apply(lambda text: text.replace("⋯⋯", "⋯"))
    df["text"] = df["text"].apply(lambda text: text.replace("......", "⋯"))
    df["text"] = df["text"].apply(lambda text: unicodedata.normalize("NFKC", text))

    return df

def read_c3(path: Path) -> pd.DataFrame:
    random.seed(0)

    with open(path) as file:
        data = json.load(file)

    data_list = []
    for d in data:
        for q in d[1]:
            choice_list = q["choice"]
            if len(choice_list) > 3:
                remove_opt = q["answer"]
                while remove_opt == q["answer"]:
                    remove_opt = random.sample(choice_list, 1)
                choice_list.remove(remove_opt[0])

            code_list = ["A", "B", "C"]
            for idx, opt in enumerate(choice_list):
                if opt == q["answer"]:
                    ans = code_list[idx]

            data_list.append({
                "text": "".join(d[0]),
                "question": {
                    "stem": q["question"],
                    "choices": [{"text": c} for c in choice_list],
                },
                "answer": ans,
            })

    return pd.DataFrame(data_list)
