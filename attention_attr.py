import os
from icl.utils.experiment_utils import set_gpu
import pickle
import warnings
from dataclasses import dataclass, field
from typing import List

set_gpu(5)
import numpy as np
from torch.optim import Adam, SGD
from tqdm import tqdm
from transformers.hf_argparser import HfArgumentParser
import torch
import torch.nn.functional as F

from icl.analysis.attentioner import GPT2AttentionerManager
from icl.analysis.attentioner_for_train import (
    AlteringAttentionAdapter,
    LlamaAttentionerManager,
    Mode,
    ReweightingAttentionAdapter,
    WeightObservingAttentionAdapter,
    get_attn_adapter_initializer,
)
from icl.lm_apis.lm_api_base import LMForwardAPI
from icl.util_classes.arg_classes import AttrArgs
from icl.util_classes.predictor_classes import Predictor
from icl.utils.data_wrapper import prepare_dataset
from icl.utils.load_huggingface_dataset import load_huggingface_dataset_train_and_test
from icl.utils.load_local import get_model_layer_num
from icl.utils.other import dict_to
from icl.utils.prepare_model_and_tokenizer import (
    get_label_id_dict_for_args,
    load_model_customize,
    load_tokenizer,
)
from transformers import Trainer, TrainingArguments
from datasets import concatenate_datasets
from copy import deepcopy

from random_utils import set_seed


def train(args: AttrArgs):
    # if os.path.exists(args.save_file_name):
    #     return
    if args.sample_from == "test":
        dataset = load_huggingface_dataset_train_and_test(args.task_name)
    else:
        raise NotImplementedError(f"sample_from: {args.sample_from}")

    model_original = load_model_customize(args)

    tokenizer = load_tokenizer(args)
    label_id_dict = get_label_id_dict_for_args(args, tokenizer)

    model = LMForwardAPI(
        model=model_original,
        model_name=args.model_name,
        tokenizer=tokenizer,
        label_id_dict=label_id_dict,
        output_attention=True,
    )

    training_args = TrainingArguments(
        "./output_dir",
        remove_unused_columns=False,
        per_device_eval_batch_size=args.batch_size,
        per_device_train_batch_size=args.batch_size,
    )

    for seed in args.seeds:
        train_dataset = prepare_dataset(seed, dataset["test"], 10, args, tokenizer)
        # train_dataset = train_dataset.select([4])
        training_args = TrainingArguments(
            "./output_dir",
            remove_unused_columns=False,
            per_device_eval_batch_size=1,
            per_device_train_batch_size=1,
        )
        trainer = Trainer(model=model, args=training_args)

        num_layer = get_model_layer_num(model=model.model, model_name=args.model_name)
        predictor = Predictor(
            label_id_dict=label_id_dict,
            pad_token_id=tokenizer.pad_token_id,
            task_name=args.task_name,
            tokenizer=tokenizer,
            layer=num_layer,
        )
        for p in model.parameters():
            p.requires_grad = False
        initialize_adapter = get_attn_adapter_initializer(Mode.MEASURE_GRAD_ONLY)
        if "gpt" in args.model_name:
            attentionermanger = GPT2AttentionerManager(
                model.model,
                4,  # 4 class
                predictor=predictor,
                device=model.device,
                kind_of_attention_adapter_initilizer=initialize_adapter,
                n_head=model_original.transformer.h[0].attn.num_heads,
            )
        else:
            attentionermanger = LlamaAttentionerManager(
                model.model,
                4,  # 4 class
                predictor=predictor,
                device=model.device,
                kind_of_attention_adapter_initilizer=initialize_adapter,
                n_head=model_original.model.layers[0].self_attn.num_heads,
            )

        set_seed(seed)

        train_dataloader = trainer.get_eval_dataloader(train_dataset)
        pbar = tqdm(enumerate(train_dataloader), total=len(train_dataset), leave=False)
        pros_list = []
        correct_answer = 0
        for idx, data in pbar:
            data = dict_to(data, model.device)
            output = model(**data)
            label = data["labels"]
            percent_of_correct_choice = output["probs"][0][label.item()]
            loss = -torch.log(percent_of_correct_choice)
            loss.backward()
            class_poss, final_poss, answer_pos = predictor.get_pos(
                {"input_ids": attentionermanger.input_ids}
            )
            pros = []
            grad_at_criticals_at_layers = torch.cat(
                [
                    attentionermanger.grad(use_abs=False)[i]
                    for i in range(len(attentionermanger.attention_adapters))
                ]
            )
            pros = {
                "grads": grad_at_criticals_at_layers.cpu(),
                "class_pos": class_poss,
                "final_pos": final_poss.cpu(),
                "question": train_dataset[idx]["sentence"],
                "tokens": tokenizer.convert_ids_to_tokens(data["input_ids"][0]),
                "percentage": output["probs"][0].detach().cpu(),
                "correct_choice": label.item(),
                "attentions": torch.cat(
                    [x.detach().cpu() for x in output["results"]["attentions"]]
                ).mean(dim=1),
            }
            correct_answer += pros["percentage"].argmax() == pros["correct_choice"]
            pros_list.append(pros)
            attentionermanger.zero_grad(set_to_none=False)
    print(f"Correct percentage: {correct_answer}/{len(train_dataset)}")
    os.makedirs(os.path.dirname(args.save_file_name), exist_ok=True)
    print(f"File saved: {args.save_file_name}")
    with open(args.save_file_name, "wb") as f:
        pickle.dump(pros_list, f)


hf_parser = HfArgumentParser((AttrArgs,))
args: AttrArgs = hf_parser.parse_args_into_dataclasses()[0]
train(args)
