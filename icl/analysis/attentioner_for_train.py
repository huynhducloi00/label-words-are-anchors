import math
import warnings
from typing import Callable, Optional, List, Union
from functools import wraps, partial
import torch
from torch import nn
from torch.nn import functional as F
from transformers import PreTrainedModel
from transformers.models.gpt2.modeling_gpt2 import GPT2Attention

from icl.util_classes.predictor_classes import Predictor


class AttentionAdapterBase(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.use_flag = True

    def forward(self, attn_weights):
        if self.use_flag:
            return self._forward(attn_weights)
        else:
            return attn_weights

    def _forward(self, attn_weights):
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
    attn_weight = torch.softmax(attn_weight, dim=-1)

    if attention_adapter is not None:
        attn_weight = attention_adapter(attn_weight)

    attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
    return attn_weight @ value


def gpt2_attn(
    self, query, key, value, attention_mask=None, head_mask=None, attention_adapter=None
):
    attn_weights = torch.matmul(query, key.transpose(-1, -2))

    if self.scale_attn_weights:
        attn_weights = attn_weights / (float(value.size(-1)) ** 0.5)

    if self.scale_attn_by_inverse_layer_idx:
        attn_weights = attn_weights / float(self.layer_idx + 1)

    if not self.is_cross_attention:
        query_length, key_length = query.size(-2), key.size(-2)
        causal_mask = self.bias[
            :, :, key_length - query_length : key_length, :key_length
        ].bool()
        attn_weights = torch.where(
            causal_mask, attn_weights, self.masked_bias.to(attn_weights.dtype)
        )

    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask

    attn_weights = nn.Softmax(dim=-1)(attn_weights)

    if attention_adapter is not None:
        attn_weights = attention_adapter(attn_weights)

    attn_weights = attn_weights.type(value.dtype)
    attn_weights = self.attn_dropout(attn_weights)

    if head_mask is not None:
        attn_weights = attn_weights * head_mask

    attn_output = torch.matmul(attn_weights, value)

    return attn_output, attn_weights

# Basic grad for all
class AttentionerManagerBase:
    def __init__(
        self, model: PreTrainedModel, predictor: Predictor, n_class, device, n_head
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
        class_poss, final_poss = self.predictor.get_pos({"input_ids": input_ids})
        for attention_adapter in self.attention_adapters:
            attention_adapter.register_input_ids(input_ids)
            attention_adapter.class_poss = class_poss
            attention_adapter.final_poss = final_poss

    def register_input_ids(self, input_ids):
        self.input_ids = input_ids

    def register_attentioner_to_model(self):
        raise NotImplementedError

    def zero_grad(self, set_to_none=True):
        if set_to_none:
            for attention_adapter in self.attention_adapters:
                attention_adapter.params = None
        else:
            for attention_adapter in self.attention_adapters:
                attention_adapter.zero_grad(set_to_none=True)

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
            params.append(attention_adapter.weight)
        return params


class WeightObservingAttentionAdapter(AttentionAdapterBase):
    def __init__(self,*args,**kwargs) -> None:
        super().__init__()
        self.params = None

    def _forward(self, attn_weights):
        if self.params is None:
            self.params = torch.ones_like(attn_weights, requires_grad=True)
        else:
            self.params.data = torch.ones_like(attn_weights)
        return attn_weights * self.params

    @property
    def grad(self):
        return self.params.grad

    def zero_grad(self, set_to_none: bool = False) -> None:
        if self.params.grad is not None:
            if set_to_none:
                self.params.grad = None
            else:
                self.params.grad = torch.zeros_like(self.params.grad)

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


class LlamaAttentionerManager(AttentionerManagerBase):
    def __init__(
        self, model: PreTrainedModel, n_class, predictor: Predictor, device, kind_of_attention_adapter_initilizer,n_head=1
    ):
        self.kind_of_attention_adapter_initilizer=kind_of_attention_adapter_initilizer
        super().__init__(model, predictor, n_class, device, n_head=n_head)

    def register_attentioner_to_model(self):
        attention_adapters = []
        for _, layer in enumerate(self.model.model.layers):
            attention_adapter = self.kind_of_attention_adapter_initilizer(
                n_class=self.n_class,
                device=self.device,
                n_head=self.n_head,
                dtype=layer.self_attn.q_proj.quant_storage,
            )
            layer.self_attn._attn = partial(
                llama_attn, layer.self_attn, attention_adapter=attention_adapter
            )
            attention_adapters.append(attention_adapter)
        return attention_adapters


class GPT2AttentionerManager(AttentionerManagerBase):
    def __init__(
        self, model: PreTrainedModel, n_class, predictor: Predictor, device, 
        kind_of_attention_adapter_initilizer, n_head=1,
    ):
        self.kind_of_attention_adapter_initilizer=kind_of_attention_adapter_initilizer
        super().__init__(model, predictor, n_class, device, n_head=n_head)

    def register_attentioner_to_model(self):
        attention_adapters = []
        for _, layer in enumerate(self.model.transformer.h):
            attention_adapter = self.kind_of_attention_adapter_initilizer(
                n_class=self.n_class, device=self.device, n_head=self.n_head
            )
            layer.attn._attn = partial(
                gpt2_attn, layer.attn, attention_adapter=attention_adapter
            )
            attention_adapters.append(attention_adapter)
        return attention_adapters


class ReweightingAttentionAdapter(AttentionAdapterBase):
    def __init__(self, n_class, n_head, device, dtype) -> None:
        super().__init__()
        self.n_class = n_class
        self.n_head = n_head
        self.dtype = dtype
        self.weight = torch.nn.Parameter(
            torch.zeros(
                (n_head, n_class), requires_grad=True, device=device
            ).half()
        )
        self.class_poss = None
        self.final_poss = None

    def _forward(self, attn_weights):
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

    @property
    def grad(self):
        return self.weight.grad

    def zero_grad(self, set_to_none: bool = False) -> None:
        if self.weight.grad is not None:
            if set_to_none:
                self.weight.grad = None
            else:
                self.weight.grad = torch.zeros_like(self.weight.grad)
