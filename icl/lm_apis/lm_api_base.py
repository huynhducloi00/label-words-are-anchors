from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from icl.utils.other import dict_to


class LMForwardAPI(nn.Module):
    def __init__(self, model, model_name, tokenizer, label_id_dict: Dict[int, str]):
        super().__init__()
        self._use_past_key_values = False
        self._past_key_values = None
        self.model = model
        self.model_name = model_name
        self.tokenizer = tokenizer
        self.model.eval()
        self.calibration_probs = None
        self.use_calibration_probs = False
        self.probs_from_results_fn = None
        self.results_args: dict = {}
        self.label_id_dict = label_id_dict
        self.position_offset = 0

    @property
    def device(self):
        return self.model.device

    def cal_logits(self, inputs, **kwargs):
        self.model.eval()
        inputs = dict_to(inputs, self.device)

        if self.use_past_key_values:
            past_key_values = self.get_past_key_values(inputs)
            kwargs["past_key_values"] = past_key_values
            inputs["attention_mask"] = self.get_mask_with_past_key_values(
                inputs["attention_mask"]
            )
            if self.model_name in ["gpt-j-6b", "gpt2-xl"]:
                bsz, sql = inputs["input_ids"].shape
                position_ids = torch.arange(
                    sql, dtype=torch.long, device=self.device
                ).repeat(bsz, 1)
                position_ids = position_ids + self.position_offset
                kwargs["position_ids"] = position_ids

        results = self.model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            **kwargs,
        )
        logits = results["logits"]
        # find last position before pad tokens
        input_ids = inputs["input_ids"]
        eos_token_id: int = self.tokenizer.eos_token_id
        is_not_eos = input_ids != eos_token_id
        prediction_pos = is_not_eos.sum(dim=1) - 1
        is_not_eos = is_not_eos.float()
        # check all eos_tokens are at the end
        assert (is_not_eos[:, :-1] - is_not_eos[:, 1:] >= 0).all()
        # get logits for the last position
        logits = logits[torch.arange(input_ids.shape[0]), prediction_pos, :]
        return logits, results

    def _cal_probs(self, logits):
        interest_indices = np.array(list(self.label_id_dict.values()))
        token_selected=logits.argmax().item()
        category=None
        if token_selected in interest_indices[:,0]:
            print('using category 0, not normal')
            category=0
        elif token_selected in interest_indices[:,1]:
            category=1
        else:
            print('Not in any category')
            category=0
        logits = logits[:, interest_indices[:,category]]
        probs = F.softmax(logits, dim=-1)
        if self.use_calibration_probs:
            assert self.calibration_probs is not None
            probs = probs / self.calibration_probs
        return probs, logits

    def cal_probs(self, inputs, **kwargs):
        logits, results = self.cal_logits(inputs, **kwargs)
        probs, logits = self._cal_probs(logits)
        return probs, logits, results

    def cal_probs_from_results(self, inputs, results):
        # inside predictor class
        return self.probs_from_results_fn(inputs, results)

    @property
    def past_key_values(self):
        return self._past_key_values

    @past_key_values.setter
    def past_key_values(self, past_key_values):
        if past_key_values is not None:
            assert isinstance(past_key_values, tuple)
            assert isinstance(past_key_values[0], tuple)
            assert len(past_key_values[0]) == 2
            assert isinstance(past_key_values[0][0], torch.Tensor)
            assert past_key_values[0][0].shape[0] == 1
            self._past_key_values = tuple(
                tuple(t.to(self.device) for t in tup) for tup in past_key_values
            )
        else:
            self._past_key_values = None

    @property
    def use_past_key_values(self):
        return self._use_past_key_values

    @use_past_key_values.setter
    def use_past_key_values(self, use_past_key_values):
        self._use_past_key_values = use_past_key_values

    def get_mask_with_past_key_values(self, mask):
        if self.past_key_values is None:
            raise ValueError("past_key_values is None, please set it first")
        batch_size = mask.shape[0]
        past_key_values_len = self.past_key_values[0][0].shape[2]
        mask = torch.cat(
            [
                torch.ones(
                    batch_size,
                    past_key_values_len,
                    dtype=torch.bool,
                    device=self.device,
                ),
                mask,
            ],
            dim=1,
        )
        return mask

    def get_past_key_values(self, inputs):
        if self.past_key_values is None:
            raise ValueError("past_key_values is None, please set it first")
        batch_size = inputs["input_ids"].shape[0]
        past_key_values = ()
        for layer_key, layer_value in self.past_key_values:
            past_key_values += (
                (
                    layer_key.expand(batch_size, -1, -1, -1),
                    layer_value.expand(batch_size, -1, -1, -1),
                ),
            )

        return past_key_values

    @torch.no_grad()
    def forward_no_grad(self, inputs):
        ori_logits, results = self.cal_logits(inputs, **self.results_args)
        probs, logits = self._cal_probs(ori_logits)
        probs_from_results = self.cal_probs_from_results(inputs, results)
        probs_from_results["ori_logits"] = ori_logits
        return probs, probs_from_results

    def forward(self, **kwargs):
        # result contains: 'logits', 'past_key_values', 'hidden_states', 'attentions'
        # ori logit is the logit at prediction locations (for all tokens).
        ori_logits, results = self.cal_logits(kwargs, **self.results_args)
        # probs is just softmax(ori_logits) for label dict tokens (filter out all other tokens)
        # logits returned is the raw logits filtered out for label tokens.
        probs, logits = self._cal_probs(ori_logits)

        result = {"probs": probs, "logits": logits, "results": results}
        if self.probs_from_results_fn:
            probs_from_results = self.cal_probs_from_results(kwargs, results)
            # the way 'probs_from_results' structure is [1,96]=stacking the positive (48 values for 48 layers) and negative labels into 96
            result["probs_from_results"] = probs_from_results
        result["ori_logits"] = ori_logits
        return result
