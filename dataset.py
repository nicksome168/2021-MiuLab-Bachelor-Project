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
            tok_text = self._tokenizer(
                text,
                padding="max_length",
                truncation=True,
                max_length=2048,
                return_tensors="pt",
            ).input_ids.squeeze()
            return tok_text
        else:  # mode == "valid" or "test"
            tok_text = self._tokenizer(
                text,
                padding="max_length",
                truncation=False,
                max_length=4096,
                return_tensors="pt",
            ).input_ids.squeeze()
            # return tok_text[-4096:]
            return tok_text[:4096]

    def __len__(self) -> int:
        return len(self._df)

    def __getitem__(self, idx: int) -> tuple:
        instance = self._df.iloc[idx]
        pg = self._tokenize(instance["text"])

        if self._mode == "train" or self._mode == "valid":
            return pg, instance["label"]
        else:  # mode == "test"
            return pg, instance["article_id"]

class QADataset(Dataset):
    def __init__(self, df: DataFrame, tokenizer: XLNetTokenizerFast, mode: str):
        pg_list, q_opt_list = [], []
        if mode == "train":
            add_list = []
            for i in range(len(df)):
                pg = df.loc[i, "text"]
                q = df.loc[i, "question"]["stem"]
                if len(pg) > 6000:
                    add_list.append({
                        "text": pg[-6000:],
                        "question": df.loc[i, "question"],
                    })
                    df.loc[i, "text"] = pg[:2000 - len(pg)]
                    # if len()todo
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
            tok_text = self._tokenizer(
                text,
                padding="max_length",
                truncation=True,
                max_length=2048,
                return_tensors="pt",
            ).input_ids.squeeze()
            return tok_text
        else:  # mode == "valid" or "test"
            tok_text = self._tokenizer(
                text,
                padding="max_length",
                truncation=False,
                max_length=4096,
                return_tensors="pt",
            ).input_ids.squeeze()
            return tok_text[-4096:]

    def __len__(self) -> int:
        return len(self._df)

    def __getitem__(self, idx: int) -> tuple:
        instance = self._df.iloc[idx]
        input = self._tokenize(instance["text"])

        if self._mode == "train" or self._mode == "valid":
            return input, instance["label"]
        else:  # mode == "test"
            return input
