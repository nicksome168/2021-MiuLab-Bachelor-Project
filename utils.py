import unicodedata

from pandas import DataFrame
import torch


def handle_reproducibility(is_reproducible: bool = True) -> None:
    if is_reproducible:
        torch.manual_seed(0)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def preprocess(df: DataFrame) -> DataFrame:
    # df["text"] = df["text"].apply(lambda text: text.replace("個管師：", "個："))
    # df["text"] = df["text"].apply(lambda text: text.replace("醫師：", "醫："))
    # df["text"] = df["text"].apply(lambda text: text.replace("民眾：", "民："))
    # df["text"] = df["text"].apply(lambda text: text.replace("家屬：", "家："))
    df["text"] = df["text"].apply(lambda text: text.replace("……", "⋯"))
    df["text"] = df["text"].apply(lambda text: text.replace("⋯⋯", "⋯"))
    df["text"] = df["text"].apply(lambda text: text.replace("......", "⋯"))
    df["text"] = df["text"].apply(lambda text: unicodedata.normalize("NFKC", text))

    return df
