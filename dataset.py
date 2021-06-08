import unicodedata

from pandas import DataFrame
from torch.utils.data import Dataset
from transformers import XLNetTokenizerFast


class RiskClassificationDataset(Dataset):
    def __init__(self, df: DataFrame, tokenizer: XLNetTokenizerFast, mode: str):
        df["label"] = df["label"].astype(int)
        if mode == "train":
            add_list = []
            for i in range(len(df)):
                pg = df.loc[i, "text"]
                if len(pg) > 2048:
                    df.loc[i, "text"] = pg[:2048]
                    add_list.append({
                        "text": pg[-2048:],
                        "label": df.loc[i, "label"],
                    })
            df = df.append(add_list, ignore_index=True)

        self._df = df
        self._tokenizer = tokenizer
        self._mode = mode
    
    def _tokenize(self, text):
        if self._mode == "train":
            tok = self._tokenizer(
                text,
                padding="max_length",
                truncation=True,
                max_length=2048,
                return_tensors="pt",
            )
            return (
                tok.input_ids.view(-1),
                tok.token_type_ids.view(-1),
                tok.attention_mask.view(-1),
            )
        else:  # mode == "valid" or "test"
            tok = self._tokenizer(
                text,
                padding="max_length",
                truncation=False,
                max_length=4096,
                return_tensors="pt",
            )
            return (
                tok.input_ids.view(-1)[-4096:],
                tok.token_type_ids.view(-1)[-4096:],
                tok.attention_mask.view(-1)[-4096:],
            )

    def __len__(self) -> int:
        return len(self._df)

    def __getitem__(self, idx: int) -> tuple:
        instance = self._df.iloc[idx]
        tok = self._tokenize(instance["text"])

        if self._mode == "train" or self._mode == "valid":
            return *tok, instance["label"]
        else:  # mode == "test"
            return *tok, instance["article_id"]

class QADataset(Dataset):
    def __init__(self, df: DataFrame, tokenizer: XLNetTokenizerFast, mode: str):
        self._df = df
        self._tokenizer = tokenizer
        self._mode = mode
    
    def _tokenize(self, pg, q_opt):
        tok = self._tokenizer(
            pg,
            q_opt,
            padding="max_length",
            truncation=True,
            max_length=1024,
            return_tensors="pt",
        )
        return (
            tok.input_ids,
            tok.token_type_ids,
            tok.attention_mask,
        )

    def __len__(self) -> int:
        return len(self._df)

    def __getitem__(self, idx: int) -> tuple:
        instance = self._df.iloc[idx]
        pg = instance["text"]
        q = unicodedata.normalize("NFKC", instance["question"]["stem"])
        pg_list, q_opt_list = [], []
        for opt in instance["question"]["choices"]:
            opt = unicodedata.normalize("NFKC", opt["text"])
            pg_list.append(pg)
            q_opt_list.append(q + opt)
        
        tok = self._tokenize(pg_list, q_opt_list)

        if self._mode == "train" or self._mode == "valid":
            ans = unicodedata.normalize("NFKC", instance["answer"]).strip()
            if ans == "A":
                ans = 0
            elif ans == "B":
                ans = 1
            elif ans == "C":
                ans = 2
            else:
                print(instance["id"])
                raise Exception
            return *tok, ans
        else:  # mode == "test"
            return *tok, instance["article_id"]
