from enum import Enum
import pickle
import warnings
from dataclasses import dataclass, field
from typing import List
import os
import numpy as np
from peft import LoraConfig, PeftModel
from peft import get_peft_model
from torch.optim import Adam, SGD
from tqdm import tqdm
from transformers.hf_argparser import HfArgumentParser
import torch
import torch.nn.functional as F
from ..lm_apis.lm_api_base import LMForwardAPI
from ..utils.data_wrapper import prepare_dataset, tokenize_dataset
from ..utils.load_huggingface_dataset import load_huggingface_dataset_train_and_test
from ...random_utils import set_seed
from ..utils.other import (
    set_gpu,
    sample_two_set_with_shot_per_class,
    dict_to,
)
from transformers import Trainer, TrainingArguments
from ..utils.load_local import (
    get_model_layer_num,
    load_local_model_or_tokenizer,
)
from ..util_classes.arg_classes import ReweightingArgs
from ..utils.prepare_model_and_tokenizer import (
    load_model_customize,
    get_label_id_dict_for_args,
    load_tokenizer,
)
from ..util_classes.predictor_classes import Predictor
from .attentioner_for_train import (
    AlteringAttentionAdapter,
    GPT2AttentionerManager,
    LlamaAttentionerManager,
    Mode,
    ReweightingAttentionAdapter,
    WeightObservingAttentionAdapter,
    get_attn_adapter_initializer,
)
from datasets import concatenate_datasets
from copy import deepcopy
from transformers import AutoTokenizer


def print_perf(dataset, y):
    truth = dataset["label"] == y[0][0].argmax(axis=1)
    print(f"Accuracy: {truth.sum()/ len(dataset['label'])}")
    print(f'Wrong answer: {np.argwhere(dataset["label"] != y[0][0].argmax(axis=1))}')
    return truth


def quick_prep_input(row):
    return {
        k: torch.tensor(v).view(1, -1)
        for k, v in row.items()
        if k in ["labels", "input_ids", "attention_mask"]
    }


def cal_loss_answers_attn(attn_weights, class_poss, answer_pos):
    total = 0
    stack = torch.cat(attn_weights)
    # between answer choice and contents
    for i in range(len(class_poss) - 1):
        for j in range(i + 1, len(class_poss)):
            range_i_start = class_poss[i]
            range_i_end = class_poss[i + 1]
            range_j_start = class_poss[j]
            range_j_end = class_poss[j + 1] if j < len(class_poss) - 1 else answer_pos
            # annil
            total += (
                stack[:, :, range_j_start:range_j_end, range_i_start:range_i_end]
                .abs()
                .sum()
            )
    return total


def train(args: ReweightingArgs):
    seed = args.seeds[0]
    tokenizer = load_tokenizer(args)

    # if os.path.exists(args.save_file_name):
    #     return
    set_gpu(args.gpu)
    if args.sample_from == "test":
        dataset = load_huggingface_dataset_train_and_test(args.task_name)
    else:
        raise NotImplementedError(f"sample_from: {args.sample_from}")
    test_dataset = prepare_dataset(seed, dataset["test"], 10, args, tokenizer)
    train_dataset = prepare_dataset(seed, dataset["train"], 50, args, tokenizer)
    if False:
        print(f"Example: {test_dataset[2]['sentence']}")
    # Load LoRA configuration
    peft_args = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.1,
        r=64,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model_original_0 = load_model_customize(args)
    use_peft = True
    if use_peft:
        model_original_peft = get_peft_model(model_original_0, peft_args)
    else:
        model_original_peft = model_original_0

    model_original_peft.config.use_cache = False
    model_original_peft.config.pretraining_tp = 1
    label_id_dict = get_label_id_dict_for_args(args, tokenizer)

    model = LMForwardAPI(
        model=model_original_peft,
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

    # for p in model.parameters():
    #     p.requires_grad = False
    initialize_adapter = get_attn_adapter_initializer(Mode.CUSTOM_PATH_ONLY)
    if initialize_adapter != None:
        if "gpt" in args.model_name:
            attentionermanger = GPT2AttentionerManager(
                model_original_0,
                4,  # 4 class
                predictor=predictor,
                device=model.device,
                kind_of_attention_adapter_initilizer=initialize_adapter,
                n_head=model_original_0.transformer.h[0].attn.num_heads,
            )
        else:
            attentionermanger = LlamaAttentionerManager(
                model_original_0,
                4,  # 4 class
                predictor=predictor,
                device=model.device,
                kind_of_attention_adapter_initilizer=initialize_adapter,
                n_head=model_original_0.model.layers[0].self_attn.num_heads,
            )
    params = list(model.parameters()) + list(attentionermanger.params())  # +
    optimizer = SGD(params, lr=1e-1)  # args.lr)
    # # Adam(params, lr=1e-1,betas=(0.1,0.999)) #

    set_seed(seed)
    loss_list = []
    average_loss = 0

    # y = trainer.predict(test_dataset, ignore_keys=["results"])
    # print_perf(test_dataset, y)
    # quit()
    # for name, parameter in model.named_parameters():
    #     parameter.requires_grad = True
    # for _ in attentionermanger.attention_adapters:
    #     _.use_flag = False
    model_original_peft.train()
    loss_item = 0.0
    to_train = True
    if to_train:
        for _ in tqdm(range(args.epoch_num)):
            train_dataset = train_dataset.shuffle()
            train_dataloader = trainer.get_eval_dataloader(train_dataset)
            pbar = tqdm(
                enumerate(train_dataloader), total=len(train_dataset), leave=False
            )
            for _, data in pbar:
                data = dict_to(data, model.device)
                output = model(**data)
                loss_correctness = F.cross_entropy(output["logits"], data["labels"])
                class_poss, final_poss, answer_pos = predictor.get_pos(
                    {"input_ids": data["input_ids"]}
                )
                loss_answer_ignore = cal_loss_answers_attn(
                    output["results"]["attentions"], class_poss, answer_pos
                )
                # -torch.log(
                #     output["probs"].sum()
                # )  #
                total_loss = loss_correctness +1e-3* loss_answer_ignore
                total_loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                loss_item += total_loss.item()
                loss_list.append(loss_item)
                average_loss = average_loss * 0.9 + total_loss.item() * 0.1
                with torch.no_grad():
                    res_moi = model(
                        **dict_to(quick_prep_input(test_dataset[1]), model.device)
                    )["probs"].argmax()
                    pbar.set_postfix_str(
                        f"Correct: {res_moi==test_dataset[1]['labels']} Loss: {average_loss:.2f}"  # , {torch.exp(-loss):.2f}"
                    )
    # data = dict_to(quick_prep_input(test_dataset[2]), model.device)
    # results_bundle = {
    #     "matrix": model(**data)["results"]["attentions"],
    #     "tokens": tokenizer.convert_ids_to_tokens(data["input_ids"][0]),
    # }
    # pickle.dump(results_bundle, open("loi_analysis.pkl", "wb"))
    # y = trainer.predict(test_dataset, ignore_keys=["results"])
    # truth = print_perf(test_dataset, y)
    # pass
    # y2 = trainer.predict(test_dataset, ignore_keys=["results"])

    # ys.append((y, loss_list, params, y2, average_loss))

    # os.makedirs(os.path.dirname(args.save_file_name), exist_ok=True)
    # with open(args.save_file_name, "wb") as f:
    #     pickle.dump(
    #         [
    #             ys,
    #         ],
    #         f,
    #     )
