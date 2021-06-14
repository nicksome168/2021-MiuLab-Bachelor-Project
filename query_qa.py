from pathlib import Path
import math
import pickle
import unicodedata

from ckiptagger import data_utils, construct_dictionary, WS
import pandas as pd
from tqdm import tqdm

from utils import preprocess


def init_ws(ckip_download_dir: Path) -> WS:
    # ckip_download_dir.mkdir(parents=True, exist_ok=True)
    # data_utils.download_data_url(ckip_download_dir)
    return WS(ckip_download_dir / "data")


class Dataset():
    def __init__(self, df: pd.DataFrame = None, ws: WS = None, from_pretrained: bool = False):
        if from_pretrained:
            return
        self._doc = {}  # article_id -> pg_word_list
        self._df = {}  # article_id -> word -> df
        self._avg_dl = {}  # article_id -> dl
        self._ws = ws

        dictionary = construct_dictionary({
            "個管師": 1,
        })

        for _, doc in tqdm(df.iterrows(), total=len(df)):
            article_id = doc["article_id"]
            if article_id in self._doc:
                continue

            article = doc["text"]
            for split_word in ["個管師:", "醫師:"]:
                article = article.replace(split_word, "\n" + split_word)
            pg_list_tmp = [pg for pg in article.split("\n")[1:]]
            pg_list = []
            for pg in pg_list_tmp:
                i = 0
                curr_start_idx = 0
                if len(pg) > 200:
                    for idx in range(len(pg)):
                        if i >= 200 and pg[idx] in [",", "?", "。", "⋯"]:
                            pg_list.append(pg[curr_start_idx : idx + 1])
                            i = 0
                            curr_start_idx = idx + 1
                        elif idx == len(pg) - 1:
                            pg_list.append(pg[curr_start_idx:])
                        else:
                            i += 1
                else:
                    pg_list.append(pg)
            # [print(pg) for pg in pg_list]
            
            self._df[article_id] = {}
            self._avg_dl[article_id] = 0
            pg_word_list = ws(pg_list, coerce_dictionary=dictionary)
            for word_list in pg_word_list:
                self._avg_dl[article_id] += len(word_list)
                for word in word_list:
                    if word in self._df[article_id]:
                        self._df[article_id][word] += 1
                    else:
                        self._df[article_id][word] = 1
                # print(word_list)
            self._doc[article_id] = pg_word_list

            self._avg_dl[article_id] /= len(pg_list)
        # print(self._avg_dl)

    def query(self, article_id, text):
        q = self._ws([text])[0]
        
        k = 1.2
        b = 0.75
        pg_list = self._doc[article_id]
        score_list = []
        for pg in pg_list:
            score = 0
            dl = len(pg)
            for q_word in q:
                tf = pg.count(q_word)
                df = self._df[article_id].get(q_word, 0)
                idf = math.log((len(pg_list) - df + 0.5) / (df + 0.5) + 1)
                score += idf * ((tf * (k + 1)) / (tf + k * (1 - b + b * dl / self._avg_dl[article_id])))
            score_list.append(score)

        sorted_list = sorted(zip(score_list, enumerate(pg_list)), key=lambda k: k[0], reverse=True)
        # print(text)
        # print(q)
        # [print(score, "".join(pg)) for score, (idx, pg) in sorted_list[:5]]

        return [idx for score, (idx, pg) in sorted_list[:3]]

    def get_pg_list(self, article_id):
        return self._doc[article_id]

    def save(self, path: Path):
        with open(path, "wb") as file:
            pickle.dump(
                {
                    "_doc": self._doc,
                    "_df": self._df,
                    "_avg_dl": self._avg_dl,
                },
                file
            )
    
    @classmethod
    def from_pretrained(cls, path: Path, ws):
        dataset = cls(from_pretrained=True)
        with open(path, "rb") as file:
            data_dict = pickle.load(file)
            for attr, data in data_dict.items():
                setattr(dataset, attr, data)
        dataset._ws = ws
        return dataset

if __name__ == "__main__":
    ws = init_ws(ckip_download_dir=Path("ckpt/ckip/"))

    data_dir = Path("data/qa/")
    train_df = pd.read_json(data_dir / "train.json", orient="records")
    # train_df["article_id"] = train_df["article_id"].apply(lambda id: f"train_{id}")
    # dev_df = pd.read_json(data_dir / "dev.json", orient="records")
    # dev_df["article_id"] = dev_df["article_id"].apply(lambda id: f"dev_{id}")
    # df = train_df.append(dev_df)
    df = train_df
    # df = dev_df
    df = preprocess(df)
    
    model_dir = Path("ckpt/bm25/")
    model_name = "model_new_train_300.pkl"

    # Constuct new model
    # dataset = Dataset(df, ws)
    # model_dir.mkdir(parents=True, exist_ok=True)
    # dataset.save(model_dir / model_name)
    # exit()
    # Load existing model
    dataset = Dataset.from_pretrained(model_dir / model_name, ws)

    for row_i, row in tqdm(df.iterrows(), total=len(df)):
        article_id = row["article_id"]
        q = unicodedata.normalize("NFKC", row["question"]["stem"])
        idx_table = {}
        for opt in row["question"]["choices"]:
            if len(opt["text"]) < 1:
                print(row["id"], opt["text"])

            opt = unicodedata.normalize("NFKC", opt["text"])
            query_str = q + opt
            idx_list = dataset.query(article_id, query_str)

            for idx in idx_list:
                # idx_table[idx - 2] = 1
                idx_table[idx - 1] = 2
                idx_table[idx] = 1
                idx_table[idx + 1] = 1
                idx_table[idx + 2] = 3

        ret_pg = ""
        for ord in [1, 2, 3]:
            cand_pg = ""
            for idx, pg in enumerate(dataset.get_pg_list(article_id)):
                if idx in idx_table and idx_table[idx] <= ord:
                    cand_pg += "".join(pg)
            if ord == 1:
                ret_pg = cand_pg
                continue
            if len(cand_pg) > 1000:
                break
            ret_pg = cand_pg
            # print(len(ret_pg))

        df.loc[row_i, "text"] = ret_pg
        if len(ret_pg) > 1000:
            print(row["question"])
            print(len(ret_pg))
            print(ret_pg)

    df.to_json(data_dir / "processed_train_new_300_c3.json", orient="records", force_ascii=False, indent=4)
