from typing import List

import torch
from torch.utils.data import Dataset


class ContextSelectionDataset(Dataset):
    def __init__(self, data: list, pg_list: list, mode: str):
        self._data = data
        self._pg_list = pg_list
        self._mode = mode

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, idx: int) -> tuple:
        instance = self._data[idx]

        pg_list = []
        q_list = []
        if self._mode == "train":
            label = None
            for i in range(7):
                try:
                    pg_id = instance["paragraphs"][i]
                    pg_list.append(self._pg_list[pg_id])
                    if pg_id == instance["relevant"]:
                        label = i
                except IndexError:
                    # Pad to 7 choices
                    pg_list.append("")
                q_list.append(instance["question"])
            return pg_list, q_list, label
        elif self._mode == "test":
            for i in range(7):
                try:
                    pg_id = instance["paragraphs"][i]
                    pg_list.append(self._pg_list[pg_id])
                except IndexError:
                    pg_list.append("")
                q_list.append(instance["question"])
            return pg_list, q_list

    def collate_fn(self, batch: List[tuple]) -> tuple:
        if self._mode == "train":
            pg_batch, q_batch, label_batch = [], [], []
            for pg_list, q_list, label in batch:
                pg_batch += pg_list
                q_batch += q_list
                label_batch.append(label)
            label_batch = torch.tensor(label_batch)
            return pg_batch, q_batch, label_batch
        elif self._mode == "test":
            pg_batch, q_batch = [], []
            for pg_list, q_list in batch:
                pg_batch += pg_list
                q_batch += q_list
            return pg_batch, q_batch


class QADataset(Dataset):
    def __init__(self, data: list, pg_list: list, mode: str):
        self._data = data
        self._pg_list = pg_list
        self._mode = mode

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, idx: int) -> tuple:
        instance = self._data[idx]

        if self._mode == "train":
            pg = self._pg_list[instance["relevant"]]
            q = instance["question"]
            # Sample 1 answer
            ans_sample = torch.randint(len(instance["answers"]), (1,))
            start = instance["answers"][ans_sample]["start"]
            end = start + len(instance["answers"][ans_sample]["text"]) - 1
            return pg, q, start, end
        elif self._mode == "valid":
            pg = self._pg_list[instance["relevant"]]
            q = instance["question"]
            ans = [ans["text"] for ans in instance["answers"]]
            return pg, q, ans

    def collate_fn(self, batch: List[tuple]) -> tuple:
        if self._mode == "train":
            pg_batch, q_batch, s_batch, e_batch = [], [], [], []
            for pg, q, s, e in batch:
                pg_batch.append(pg)
                q_batch.append(q)
                s_batch.append(s)
                e_batch.append(e)
            s_batch = torch.tensor(s_batch)
            e_batch = torch.tensor(e_batch)
            return pg_batch, q_batch, s_batch, e_batch
        elif self._mode == "valid":
            pg_batch, q_batch, ans_batch = [], [], []
            for pg, q, ans in batch:
                pg_batch.append(pg)
                q_batch.append(q)
                ans_batch.append(ans)
            return pg_batch, q_batch, ans_batch
