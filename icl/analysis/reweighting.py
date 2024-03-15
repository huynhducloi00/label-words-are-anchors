import pickle
import warnings
from dataclasses import dataclass, field
from typing import List
import os

from torch.optim import Adam
from tqdm import tqdm
from transformers.hf_argparser import HfArgumentParser
import torch
import torch.nn.functional as F
from ..lm_apis.lm_api_base import LMForwardAPI
from ..utils.data_wrapper import prepare_dataset, wrap_dataset, tokenize_dataset
from ..utils.load_huggingface_dataset import load_huggingface_dataset_train_and_test
from ..utils.random_utils import set_seed
from ..utils.other import (
    set_gpu,
    sample_two_set_with_shot_per_class,
    dict_to,
)
from transformers import Trainer, TrainingArguments
from ..utils.load_local import (
    get_model_layer_num,
)
from ..util_classes.arg_classes import ReweightingArgs
from ..utils.prepare_model_and_tokenizer import (
    load_model_and_tokenizer,
    get_label_id_dict_for_args,
)
from ..util_classes.predictor_classes import Predictor
from .attentioner_for_train import GPT2AttentionerManager, LlamaAttentionerManager
from datasets import concatenate_datasets
from copy import deepcopy


def train(args: ReweightingArgs):
    # if os.path.exists(args.save_file_name):
    #     return
    set_gpu(args.gpu)
    if args.sample_from == "test":
        dataset = load_huggingface_dataset_train_and_test(args.task_name)
    else:
        raise NotImplementedError(f"sample_from: {args.sample_from}")

    model_original, tokenizer = load_model_and_tokenizer(args)
    label_id_dict = get_label_id_dict_for_args(args, tokenizer)

    model = LMForwardAPI(
        model=model_original,
        model_name=args.model_name,
        tokenizer=tokenizer,
        label_id_dict=label_id_dict,
    )

    training_args = TrainingArguments(
        "./output_dir",
        remove_unused_columns=False,
        per_device_eval_batch_size=args.batch_size,
        per_device_train_batch_size=args.batch_size,
    )

    ys = []
    for seed in args.seeds:
        test_dataset = prepare_dataset(seed, dataset["test"], 20, args, tokenizer)
        train_dataset = prepare_dataset(seed, dataset["train"], 100, args, tokenizer)

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
        # if "gpt" in args.model_name:
        #     attentionermanger = GPT2AttentionerManager(
        #         model.model,
        #         4,  # 4 class
        #         predictor=predictor,
        #         device=model.device,
        #         n_head=model_original.transformer.h[0].attn.num_heads,
        #     )
        # else:
        #     attentionermanger = LlamaAttentionerManager(
        #         model.model,
        #         4,  # 4 class
        #         predictor=predictor,
        #         device=model.device,
        #         n_head=model_original.model.layers[0].self_attn.num_heads,
        #     )
        # params = attentionermanger.params() 
        # optimizer = Adam(params, lr=1e-3)  # args.lr)

        set_seed(seed)
        loss_list = []
        average_loss = 0

        def print_perf(dataset, y):
            print(
                f"Accuracy: {(dataset['label'] == y[0][0].argmax(axis=1)).sum()/ len(dataset['label'])}"
            )

        y = trainer.predict(test_dataset, ignore_keys=["results"])
        print_perf(test_dataset, y)
        quit()
        for _ in attentionermanger.attention_adapters:
            _.use_flag = False
        for _ in tqdm(range(args.epoch_num)):
            loss_item = 0.0
            train_dataset = train_dataset.shuffle()
            train_dataloader = trainer.get_eval_dataloader(train_dataset)
            pbar = tqdm(
                enumerate(train_dataloader), total=len(train_dataset), leave=False
            )
            for idx, data in pbar:
                data = dict_to(data, model.device)
                output = model(**data)
                label = data["labels"]
                loss = F.cross_entropy(output["logits"], label)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                loss_item += loss.item()
                loss_list.append(loss_item)
                average_loss = average_loss * 0.9 + loss.item() * 0.1
                pbar.set_postfix_str(f"Loss: {average_loss:.2f}")

        y = trainer.predict(test_dataset, ignore_keys=["results"])
        print_perf(test_dataset, y)

        # y2 = trainer.predict(test_dataset, ignore_keys=["results"])

        # ys.append((y, loss_list, params, y2, average_loss))

    os.makedirs(os.path.dirname(args.save_file_name), exist_ok=True)
    with open(args.save_file_name, "wb") as f:
        pickle.dump(
            [
                ys,
            ],
            f,
        )
