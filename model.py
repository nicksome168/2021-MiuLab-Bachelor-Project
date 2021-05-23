from abc import ABC, abstractmethod

from torch import nn, Tensor
from transformers import (
    BertForMultipleChoice,
    BertForQuestionAnswering,
    BatchEncoding,
    BertConfig,
)


class BaseModel(nn.Module, ABC):
    """Base class for models."""
    @abstractmethod
    def forward(self, *inputs):
        return NotImplemented

    def __str__(self) -> str:
        """For printing the model and the number of trainable parameters."""
        n_params = sum([p.numel() for p in self.parameters() if p.requires_grad])
        separate_line_str = "-" * 50
        return (
            f"{separate_line_str}\n{super().__str__()}\n{separate_line_str}\n"
            f"Trainable parameters: {n_params}\n{separate_line_str}"
        )


class ContextSelectionModel(BaseModel):
    def __init__(self):
        super().__init__()
        # self._model = BertForMultipleChoice.from_pretrained("bert-base-chinese")
        # self._model = BertForMultipleChoice.from_pretrained("hfl/chinese-roberta-wwm-ext-large")
        self._model = BertForMultipleChoice(BertConfig(**{
            "architectures": [
                "BertForMaskedLM"
            ],
            "attention_probs_dropout_prob": 0.1,
            "directionality": "bidi",
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.1,
            "hidden_size": 1024,
            "initializer_range": 0.02,
            "intermediate_size": 4096,
            "layer_norm_eps": 1e-12,
            "max_position_embeddings": 512,
            "model_type": "bert",
            "num_attention_heads": 16,
            "num_hidden_layers": 24,
            "pad_token_id": 0,
            "pooler_fc_size": 768,
            "pooler_num_attention_heads": 12,
            "pooler_num_fc_layers": 3,
            "pooler_size_per_head": 128,
            "pooler_type": "first_token_transform",
            "type_vocab_size": 2,
            "vocab_size": 21128
        }))

    def forward(self, x: BatchEncoding, labels: Tensor = None) -> tuple:
        y = self._model(**x, labels=labels)
        if labels is not None:
            loss, logits = y[:2]
            return loss, logits
        else:
            logits = y[0]
            return logits


class QAModel(BaseModel):
    def __init__(self):
        super().__init__()
        # self._model = BertForQuestionAnswering.from_pretrained("bert-base-chinese")
        # self._model = BertForQuestionAnswering.from_pretrained("hfl/chinese-roberta-wwm-ext-large")
        self._model = BertForQuestionAnswering(BertConfig(**{
            "architectures": [
                "BertForMaskedLM"
            ],
            "attention_probs_dropout_prob": 0.1,
            "directionality": "bidi",
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.1,
            "hidden_size": 1024,
            "initializer_range": 0.02,
            "intermediate_size": 4096,
            "layer_norm_eps": 1e-12,
            "max_position_embeddings": 512,
            "model_type": "bert",
            "num_attention_heads": 16,
            "num_hidden_layers": 24,
            "pad_token_id": 0,
            "pooler_fc_size": 768,
            "pooler_num_attention_heads": 12,
            "pooler_num_fc_layers": 3,
            "pooler_size_per_head": 128,
            "pooler_type": "first_token_transform",
            "type_vocab_size": 2,
            "vocab_size": 21128
        }))

    def forward(self, x: BatchEncoding, start: Tensor = None, end: Tensor = None) -> tuple:
        y = self._model(**x, start_positions=start, end_positions=end)
        if start is not None:
            loss, start_scores, end_scores = y[:3]
            return loss, start_scores, end_scores
        else:
            start_scores, end_scores = y[:2]
            return start_scores, end_scores
