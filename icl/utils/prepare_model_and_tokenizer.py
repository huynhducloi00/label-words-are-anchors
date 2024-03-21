import warnings

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

from .load_local import load_local_model_or_tokenizer
from ..util_classes.arg_classes import DeepArgs


def load_tokenizer(args: DeepArgs):
    tokenizer = load_local_model_or_tokenizer(args.model_name, "tokenizer")
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_model_customize(args: DeepArgs, full=True):
    # model = load_local_model_or_tokenizer(args.model_name, "model")
    # if model is None:
    if full:
        return AutoModelForCausalLM.from_pretrained(args.model_name, device_map="auto")
    return AutoModelForCausalLM.from_pretrained(
        args.model_name,
        device_map="auto",
        quantization_config=BitsAndBytesConfig(
            load_in_8bit_fp32_cpu_offload=True, bnb_4bit_compute_dtype=torch.float16
        ),
        # torch_dtype=torch.float16,
        # quantization_config=GPTQConfig(bits=4, dataset="c4", tokenizer=tokenizer),
    )


def get_label_id_dict_for_args(args: DeepArgs, tokenizer):
    label_id_dict = {
        k: [
            tokenizer.convert_tokens_to_ids(v),  #'char'
            tokenizer.encode(v, add_special_tokens=False)[-1],  #'_char'
            tokenizer.encode(f" {v}", add_special_tokens=False)[-1],  #' char'
        ]
        for k, v in args.label_dict.items()
    }
    return label_id_dict
