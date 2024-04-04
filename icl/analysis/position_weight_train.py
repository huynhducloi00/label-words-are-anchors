from enum import Enum
import math
import warnings
from typing import Callable, Optional, List, Union
from functools import wraps, partial
import torch
from torch import nn
from torch.nn import functional as F
from transformers import PreTrainedModel
from transformers.models.gpt2.modeling_gpt2 import GPT2Attention


class AttentionAdapterBase(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.use_flag = True

    def params(self):
        return None

    def zero_grad(self, set_to_none: bool = False) -> None:
        for a in self.params():
            if set_to_none:
                a.grad = None
            else:
                a.grad = torch.zeros_like(a.grad)

    def forward(self, layer_object, attn_weights):
        if self.use_flag:
            return self._forward(layer_object, attn_weights)
        else:
            return attn_weights

    def _forward(self, layer_object, attn_weights):
        raise NotImplementedError

    def register_input_ids(self, input_ids: torch.Tensor):
        self.input_ids = input_ids


def llama_attn(
    self,
    query,
    key,
    value,
    attn_mask=None,
    dropout_p=0.0,
    is_causal=False,
    scale=None,
    attention_adapter=None,
) -> torch.Tensor:
    # Copy paste from https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    attn_bias = torch.zeros(1, 1, L, S, dtype=query.dtype).to(attn_mask.device)
    if is_causal:
        assert attn_mask is None
        temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias.to(query.dtype)

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            attn_bias += attn_mask
    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    attn_weight += attn_bias
    # capture the grad before the softmax
    # softmax will take into account that some cells are being ablated
    attn_weight = torch.softmax(attn_weight, dim=-1)
    if attention_adapter is not None:
        # self here is the layer
        attn_weight = attention_adapter(self, attn_weight)

    attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
    return attn_weight @ value, attn_weight

# Basic grad for all
class AttentionerManagerBase:
    def __init__(
        self, model: PreTrainedModel, n_class, device, n_head
    ):
        self.n_class = n_class
        self.n_head = n_head
        self.device = device
        self.model = model
        self.attention_adapters = self.register_attentioner_to_model()
        self.model.forward = manager_decoractor(self)(self.model.forward)
        self.predictor = predictor

    @property
    def input_ids(self):
        return self._input_ids

    @input_ids.setter
    def input_ids(self, input_ids):
        self._input_ids = input_ids
        for attention_adapter in self.attention_adapters:
            attention_adapter.register_input_ids(input_ids)

    def register_input_ids(self, input_ids):
        self.input_ids = input_ids

    def register_attentioner_to_model(self):
        raise NotImplementedError

    def zero_grad(self, set_to_none=True):
        for attention_adapter in self.attention_adapters:
            attention_adapter.zero_grad(set_to_none)

    def grad_process(self, grad, use_abs=True):
        assert len(grad.shape) == 4
        grad = grad.sum(1)
        if use_abs:
            grad = abs(grad)
        return grad

    def grad(self, *args, **kwargs):
        grads = []
        for attention_adapter in self.attention_adapters:
            grads.append(
                self.grad_process(attention_adapter.params.grad, *args, **kwargs)
            )
        return grads

    def params(self):
        params = []
        for attention_adapter in self.attention_adapters:
            if attention_adapter.params() is None:
                return []
            params.extend(attention_adapter.params())
        return params


class WeightObservingAttentionAdapter(AttentionAdapterBase):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.params = None

    def _forward(self, attn_weights):
        if self.params is None:
            self.params = torch.ones_like(attn_weights, requires_grad=True)
        else:
            self.params.data = torch.ones_like(attn_weights)
        return attn_weights * self.params


def change_attn(self, attn_weights):
    class_poss = self.class_poss
    new_attn = attn_weights.clone()
    for head in range(attn_weights.shape[1]):
        # between answer choice and contents
        for i in range(len(class_poss) - 1):
            for j in range(i + 1, len(class_poss)):
                range_i_start = class_poss[i]
                range_i_end = class_poss[i + 1]
                range_j_start = class_poss[j]
                range_j_end = (
                    class_poss[j + 1] if j < len(class_poss) - 1 else self.answer_pos
                )
                # annil
                new_attn[0, head][
                    range_j_start:range_j_end, range_i_start:range_i_end
                ] = 0
        # Encourage Between answer choice and answer content
        # for i in range(len(class_poss)):
        #     range_i_start = class_poss[i]
        #     range_i_end = (
        #         class_poss[i + 1] if i < len(class_poss) - 1 else self.answer_pos
        #     )
        #     # column, NOT row, max.
        #     max_value = new_attn[0, head][:,range_i_start].max()
        #     new_attn[0, head][
        #         range_i_start + 1 : range_i_end, range_i_start
        #     ] = max_value

        # ablate question and answer choice
        # for i in range(len(class_poss)):
        #     range_i_start = class_poss[i]
        #     new_attn[0, head][
        #         range_i_start : range_i_start + 2, : class_poss[0]
        #     ] = 0  # +2 due to the dot

        # final token and answer content
        # for i in range(len(class_poss)):
        #     range_i_start = class_poss[i] + 1  # because ignore the dot in 'A.'
        #     range_i_end = (
        #         class_poss[i + 1] if i < len(class_poss) - 1 else self.answer_pos
        #     )
        #     new_attn[0, head][self.final_poss, range_i_start:range_i_end] = 0

        # # final token and questions
        # for i in range(len(class_poss)):
        #     range_i_start = class_poss[i]
        #     range_i_end = (
        #         class_poss[i + 1] if i < len(class_poss) - 1 else self.answer_pos
        #     )
        #     new_attn[0, head][self.final_poss, : class_poss[0]] = 0
    return new_attn


def manager_decoractor(manager: AttentionerManagerBase):
    def model_forward_decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            input_ids = kwargs.get("input_ids", None)
            if input_ids is None:
                input_ids = args[0]
            manager.register_input_ids(input_ids)
            return fn(*args, **kwargs)

        return wrapper

    return model_forward_decorator


class T5PositionWeightManager(T5PositionWeightManagerManagerBase):
    def __init__(
        self,
        model: PreTrainedModel,
        n_class,
        predictor: Predictor,
        device,
        kind_of_attention_adapter_initilizer,
        n_head=1,
    ):
        self.kind_of_attention_adapter_initilizer = kind_of_attention_adapter_initilizer
        super().__init__(model, predictor, n_class, device, n_head=n_head)

    def register_attentioner_to_model(self):
        attention_adapters = []
        for _, layer in enumerate(self.model.transformer.h):
            attention_adapter = self.kind_of_attention_adapter_initilizer(
                n_class=self.n_class, device=self.device, n_head=self.n_head
            )
            layer.t5_attention.compute_bias = partial(
                gpt2_attn, layer.attn, attention_adapter=attention_adapter
            )
            attention_adapters.append(attention_adapter)
        return attention_adapters


class ReweightingAttentionAdapter(AttentionAdapterBase):
    def __init__(self, n_class, n_head, device) -> None:
        super().__init__()
        self.n_class = n_class
        self.n_head = n_head
        self.weight = torch.nn.Parameter(
            torch.zeros((n_head, n_class), requires_grad=True, device=device).half()
        )
        self.class_poss = None
        self.final_poss = None

    def _forward(self, layer_object, attn_weights):

        class_poss = self.class_poss
        final_poss = self.final_poss
        weight = self.weight.exp()
        bsz, n_head, seq_len, _ = attn_weights.shape
        assert bsz == 1
        mask_mat = torch.ones(
            (1, n_head, seq_len, seq_len), device=attn_weights.device
        ).to(attn_weights.dtype)
        mask_mat[:, :, final_poss, class_poss] = weight.reshape(
            1, self.n_head, self.n_class
        )
        return attn_weights * mask_mat