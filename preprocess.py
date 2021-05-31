import argparse
from pathlib import Path

import pandas as pd


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        type=Path,
        # default="data/risk_classification/train.csv",
        default="data/qa/train.json",
    )
    parser.add_argument(
        "--output_path",
        type=Path,
        # default="data/risk_classification/processed_train.csv",
        default="data/qa/processed_train.json",
    )
    args = parser.parse_args()

    if args.data_path.suffix == ".csv":
        df = pd.read_csv(args.data_path, usecols=["article_id", "text", "label"])
    elif args.data_path.suffix == ".json":
        df = pd.read_json(args.data_path, orient="records")
    print(df["text"].str.len().mean())
    # print(df["text"].str.len().max())

    df["text"] = df["text"].apply(lambda text: text.replace("個管師：", "個："))
    df["text"] = df["text"].apply(lambda text: text.replace("醫師：", "醫："))
    df["text"] = df["text"].apply(lambda text: text.replace("民眾：", "民："))
    df["text"] = df["text"].apply(lambda text: text.replace("家屬：", "家："))
    df["text"] = df["text"].apply(lambda text: text.replace("……", "⋯"))
    df["text"] = df["text"].apply(lambda text: text.replace("⋯⋯", "⋯"))
    print(df["text"].str.len().mean())
    # print(df["text"].str.len().max())

    if args.output_path.suffix == ".csv":
        df.to_csv(args.output_path)
    elif args.output_path.suffix == ".json":
        df.to_json(args.output_path, orient="records", indent=4, force_ascii=False)
