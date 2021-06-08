from abc import ABC, abstractmethod
from typing import Dict

from torch import nn, Tensor
from transformers import AutoModelForSequenceClassification, AutoModelForMultipleChoice


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


class RiskClassificationModel(BaseModel):
    def __init__(self, model_name):
        super().__init__()
        self._model = AutoModelForSequenceClassification.from_pretrained(model_name)

    def forward(self, inputs: Dict, labels: Tensor = None) -> tuple:
        y = self._model(**inputs, labels=labels)
        if labels is not None:
            loss, logits = y[:2]
            return loss, logits
        else:
            logits = y[0]
            return logits


class MultiTaskModel(BaseModel):
    def __init__(self, model_name):
        super().__init__()
        self._model_dict = nn.ModuleDict({
            "rc": AutoModelForSequenceClassification.from_pretrained(model_name),
            "qa": AutoModelForMultipleChoice.from_pretrained(model_name),
        })
        self._model_dict["qa"].transformer = self._model_dict["rc"].transformer
    
    def forward(self, task: str, inputs: Dict, labels: Tensor = None):
        y = self._model_dict[task](**inputs, labels=labels)
        if labels is not None:
            loss, logits = y[:2]
            return loss, logits
        else:
            logits = y[0]
            return logits
