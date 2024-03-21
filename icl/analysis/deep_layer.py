import pickle
import warnings
from dataclasses import dataclass, field
from typing import List
import os
import tqdm
from transformers.hf_argparser import HfArgumentParser
import torch
import torch.nn.functional as F
from ..lm_apis.lm_api_base import LMForwardAPI
from ..utils.data_wrapper import prepare_dataset, wrap_dataset, tokenize_dataset
from ..utils.load_huggingface_dataset import load_huggingface_dataset_train_and_test
from ..utils.random_utils import set_seed
from ..utils.other import load_args, set_gpu, sample_two_set_with_shot_per_class
from transformers import (
    Trainer,
    TrainingArguments,
    PreTrainedModel,
    AutoModelForCausalLM,
    AutoTokenizer,
)
from ..utils.load_local import (
    convert_path_old,
    load_local_model_or_tokenizer,
    get_model_layer_num,
)
from ..util_classes.arg_classes import DeepArgs
from ..utils.prepare_model_and_tokenizer import (
    load_model_customize,
    get_label_id_dict_for_args,
    load_tokenizer,
)
from ..util_classes.predictor_classes import Predictor

def deep_layer(args: DeepArgs):
    # if os.path.exists(args.save_file_name):
    #     return
    set_gpu(args.gpu)
    if args.sample_from == "test":
        dataset = load_huggingface_dataset_train_and_test(args.task_name)
    else:
        raise NotImplementedError(f"sample_from: {args.sample_from}")
    tokenizer=load_tokenizer(args)
    model = load_model_customize(args)
    label_id_dict = get_label_id_dict_for_args(args, tokenizer)
    model = LMForwardAPI(
        model=model,
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

    num_layer = get_model_layer_num(model=model.model, model_name=args.model_name)
    predictor = Predictor(
        label_id_dict=label_id_dict,
        pad_token_id=tokenizer.pad_token_id,
        task_name=args.task_name,
        tokenizer=tokenizer,
        layer=num_layer,
    )

    ys = []
    no_demo_ys = []
    for seed in tqdm.tqdm(args.seeds):
        test_dataset = prepare_dataset(seed, dataset["test"], 1, args, tokenizer)

        model.results_args = {"output_hidden_states": True, "output_attentions": True}
        model.probs_from_results_fn = predictor.cal_all_sim_attn
        trainer = Trainer(model=model, args=training_args)

        y = trainer.predict(test_dataset, ignore_keys=["results"])
        print(
            f"Accuracy: {(test_dataset['label'] == y[0][0].argmax(axis=1)).sum()/ len(test_dataset['label'])}"
        )
        # We will focus on item 2: probs_from_results, which is the attentions value of 4 choices
        ys.append(y)

        # model.results_args = {}
        # model.probs_from_results_fn = None
        # trainer = Trainer(model=model, args=training_args)

        # no_demo_y = trainer.predict(analysis_no_demo_dataset, ignore_keys=["results"])
        # no_demo_ys.append(no_demo_y)

    os.makedirs(os.path.dirname(args.save_file_name), exist_ok=True)
    with open(args.save_file_name, "wb") as f:
        pickle.dump([ys, no_demo_ys], f)
