from abc import ABC, abstractmethod
from typing import Dict

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, MultiheadAttention

from transformers import AutoModelForSequenceClassification, AutoModelForMultipleChoice, BertForMultipleChoice, BertConfig, BertModel
from transformers.modeling_outputs import MultipleChoiceModelOutput

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


def separate_seq2(sequence_output, flat_input_ids):
    qa_seq_output = sequence_output.new(sequence_output.size()).zero_()
    qa_mask = torch.ones((sequence_output.shape[0], sequence_output.shape[1]),
                         device=sequence_output.device,
                         dtype=torch.bool)
    p_seq_output = sequence_output.new(sequence_output.size()).zero_()
    p_mask = torch.ones((sequence_output.shape[0], sequence_output.shape[1]),
                        device=sequence_output.device,
                        dtype=torch.bool)
    for i in range(flat_input_ids.size(0)):
        sep_lst = []
        for idx, e in enumerate(flat_input_ids[i]):
            if e == 102:
                sep_lst.append(idx)
        assert len(sep_lst) == 2
        qa_seq_output[i, :sep_lst[0] - 1] = sequence_output[i, 1:sep_lst[0]]
        qa_mask[i, :sep_lst[0] - 1] = 0
        p_seq_output[i, :sep_lst[1] - sep_lst[0] - 1] = sequence_output[i, sep_lst[0] + 1: sep_lst[1]]
        p_mask[i, :sep_lst[1] - sep_lst[0] - 1] = 0
    return qa_seq_output, p_seq_output, qa_mask, p_mask


class DUMALayer(nn.Module):
    def __init__(self, d_model_size, num_heads):
        super(DUMALayer, self).__init__()
        self.attn_qa = MultiheadAttention(d_model_size, num_heads)
        self.attn_p = MultiheadAttention(d_model_size, num_heads)

    def forward(self, qa_seq_representation, p_seq_representation, qa_mask=None, p_mask=None):
        qa_seq_representation = qa_seq_representation.permute([1, 0, 2])
        p_seq_representation = p_seq_representation.permute([1, 0, 2])
        enc_output_qa, _ = self.attn_qa(
            value=qa_seq_representation, key=qa_seq_representation, query=p_seq_representation, key_padding_mask=qa_mask
        )
        enc_output_p, _ = self.attn_p(
            value=p_seq_representation, key=p_seq_representation, query=qa_seq_representation, key_padding_mask=p_mask
        )
        return enc_output_qa.permute([1, 0, 2]), enc_output_p.permute([1, 0, 2])

class QAModel(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.config = BertConfig.from_pretrained(model_name, num_choices=4)
        self.bert = BertModel.from_pretrained(model_name, config=self.config)
        self.duma = DUMALayer(d_model_size=self.config.hidden_size, num_heads=self.config.num_attention_heads)
        self.dropouts = nn.ModuleList([
            nn.Dropout(0.5) for _ in range(1)
        ])
        self.classifier = nn.Linear(self.config.hidden_size, 1)

    def forward(self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        num_choices = input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1]
        
        input_ids = input_ids.view(-1, input_ids.size(-1)) if input_ids is not None else None
        attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None
        inputs_embeds = (
            inputs_embeds.view(-1, inputs_embeds.size(-2), inputs_embeds.size(-1))
            if inputs_embeds is not None
            else None
        )

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        last_output = outputs.last_hidden_state
        qa_seq_output, p_seq_output, qa_mask, p_mask = separate_seq2(last_output, input_ids)
        enc_output_qa, enc_output_p = self.duma(qa_seq_output, p_seq_output, qa_mask, p_mask)
        fused_output = torch.cat([enc_output_qa, enc_output_p], dim=1)
        pooled_output = torch.mean(fused_output, dim=1)
        for i, dropout in enumerate(self.dropouts):
            if i == 0:
                logits = self.classifier(dropout(pooled_output))
            else:
                logits += self.classifier(dropout(pooled_output))
        logits = logits / len(self.dropouts)
        reshaped_logits = F.softmax(logits.view(-1, num_choices), dim=1)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)

        if not return_dict:
            output = (reshaped_logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return MultipleChoiceModelOutput(
            loss=loss,
            logits=reshaped_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )