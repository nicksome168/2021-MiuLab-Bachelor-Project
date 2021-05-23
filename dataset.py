from pandas import DataFrame
from torch.utils.data import Dataset


class RiskClassificationDataset(Dataset):
    def __init__(self, df: DataFrame, mode: str):
        df["label"] = df["label"].astype(int)
        self._df = df
        self._mode = mode

    def __len__(self) -> int:
        return len(self._df)

    def __getitem__(self, idx: int) -> tuple:
        instance = self._df.iloc[idx]

        if self._mode == "train" or self._mode == "valid":
            return instance["text"], instance["label"]
        else:  # mode == "test"
            return instance["text"]
