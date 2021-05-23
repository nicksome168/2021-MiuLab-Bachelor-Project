from typing import List

import torch
from transformers import BatchEncoding

from eval import Tokenizer, compute_metric


def handle_reproducibility(is_reproducible: bool = True) -> None:
    if is_reproducible:
        torch.manual_seed(0)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def se_to_str(
    x: BatchEncoding,
    batch_idx: int,
    start: int,
    end: int,
    pg: str,
):
    if start == 0:
        return ""

    s, _ = x.token_to_chars(batch_idx, start)
    e, _ = x.token_to_chars(batch_idx, end)

    return pg[s:e+1]


class Metrics:
    def __init__(self):
        self._tokenizer = Tokenizer()

    @staticmethod
    def search_best(
        x: BatchEncoding,
        batch_idx: int,
        start_scores: torch.Tensor,
        end_scores: torch.Tensor,
    ) -> tuple:
        beam = 20
        max_score = -1
        max_score_idx = (0, 0)
        for i in torch.topk(start_scores[1:], k=beam)[1]:
            s = i + 1
            try:
                s_span = x.token_to_chars(batch_idx, s)
            except TypeError:
                continue
            if s_span is None:
                continue
            for j in torch.topk(end_scores[s:], k=min(beam, len(end_scores)-s))[1]:
                if j > 64:
                    continue
                e = s + j
                try:
                    e_span = x.token_to_chars(batch_idx, e)
                except TypeError:
                    continue
                if e_span is None:
                    continue
                if start_scores[s] + end_scores[e] > max_score:
                    max_score = start_scores[s] + end_scores[e]
                    max_score_idx = (s, e)
        return max_score_idx[0], max_score_idx[1], max_score

    def compute(
        self,
        x: BatchEncoding,
        y: List,
        start_scores: torch.Tensor,
        end_scores: torch.Tensor,
        pg: List[str],
    ) -> List:
        res_list = []
        for i, (ss, es) in enumerate(zip(start_scores, end_scores)):
            ps, pe, _ = self.search_best(x, i, ss, es)
            pred = se_to_str(x, i, ps, pe, pg[i])
            res_list.append(compute_metric(y[i], pred, self._tokenizer))

        return res_list
