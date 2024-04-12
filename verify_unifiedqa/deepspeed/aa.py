# %%

# IMPORTANT: RUN THIS CELL IN ORDER TO IMPORT YOUR KAGGLE DATA SOURCES
# TO THE CORRECT LOCATION (/kaggle/input) IN YOUR NOTEBOOK,
# THEN FEEL FREE TO DELETE THIS CELL.
# NOTE: THIS NOTEBOOK ENVIRONMENT DIFFERS FROM KAGGLE'S PYTHON
# ENVIRONMENT SO THERE MAY BE MISSING LIBRARIES USED BY YOUR
# NOTEBOOK.

import os
import pickle
import sys
from tempfile import NamedTemporaryFile
from urllib.request import urlopen
from urllib.parse import unquote, urlparse
from urllib.error import HTTPError
from zipfile import ZipFile
import tarfile
import shutil

CHUNK_SIZE = 40960

# %% [markdown]
# # Introduction
# 
# As we all know , this is a hardware heavy competition and training at a higher MAX_LEN and a larger model does the trick , but all those with limited hardware like me have been struggling to do so . DeepSpeed Library allows one to Train Large models with bigger batch sizes on smaller GPUs , This notebook integrates DeepSpeed with HF Trainer and most of it is a replica of the wonderful Baseline prepared by Darek [Here](https://www.kaggle.com/thedrcat/feedback-prize-huggingface-baseline-training)
# 
# 
# <b>Please Note that in this notebook I show an example to use DeepSpeed and its features to train LongFormer Base with a batch_size of 10 on a 16GB card . We can train even larger batch size or bigger sequences and utilize more juice out of a 16Gb card if we have more CPU memory but since kaggle gives us only 16GB of memory I was only able to demonstrate use of Stage2 ZeRO Optimizer given by DEEPSPEED</b>
# 
# I am able to train longformer large on a bs of 6 on a 24GB RTX 6000 card on [Jarvislabs.ai](https://cloud.jarvislabs.ai/) using DeepSpeed and HF Trainer . Here is an [article](https://jarvislabs.ai/blogs/deepspeed) which describes in detail to make full use of DeepSpeed
# 
# 
# Hope you all are able to use your GPU's more efficiently , thanks for reading

# %%
SAMPLE = False # set True for debugging

# %%
import os
os.environ["WANDB_DISABLED"] = "true"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# %%
# CONFIG

EXP_NUM = 4
task = "ner"
model_checkpoint = "allenai/longformer-base-4096"
max_length = 1024
stride = 128
min_tokens = 6
model_path = f'{model_checkpoint.split("/")[-1]}-{EXP_NUM}'

# TRAINING HYPERPARAMS
BS = 10
GRAD_ACC = 2
LR = 5e-5
WD = 0.01
WARMUP = 0.1
N_EPOCHS = 5

# %% [markdown]
# ## Data Preprocessing

# %%
import pandas as pd

# read train data
train = pd.read_csv('kaggle/input/feedback-prize-2021/train.csv')
train.head(1)

# %%
# check unique classes
classes = train.discourse_type.unique().tolist()
classes

# %%
# setup label indices

from collections import defaultdict
tags = defaultdict()

for i, c in enumerate(classes):
    tags[f'B-{c}'] = i
    tags[f'I-{c}'] = i + len(classes)
tags[f'O'] = len(classes) * 2
tags[f'Special'] = -100

l2i = dict(tags)

i2l = defaultdict()
for k, v in l2i.items():
    i2l[v] = k
i2l[-100] = 'Special'

i2l = dict(i2l)

N_LABELS = len(i2l) - 1 # not accounting for -100

# %%
# some helper functions

from pathlib import Path

path = Path('kaggle/input/feedback-prize-2021/train')

def get_raw_text(ids):
    with open(path/f'{ids}.txt', 'r') as file: data = file.read()
    return data

# %%
# group training labels by text file

df1 = train.groupby('id')['discourse_type'].apply(list).reset_index(name='classlist')
df2 = train.groupby('id')['discourse_start'].apply(list).reset_index(name='starts')
df3 = train.groupby('id')['discourse_end'].apply(list).reset_index(name='ends')
df4 = train.groupby('id')['predictionstring'].apply(list).reset_index(name='predictionstrings')

df = pd.merge(df1, df2, how='inner', on='id')
df = pd.merge(df, df3, how='inner', on='id')
df = pd.merge(df, df4, how='inner', on='id')
df['text'] = df['id'].apply(get_raw_text)

df.head()

# %%
# debugging
if SAMPLE: df = df.sample(n=100).reset_index(drop=True)

# %%
# we will use HuggingFace datasets
from datasets import Dataset, load_metric

ds = Dataset.from_pandas(df)
datasets = ds.train_test_split(test_size=0.1, shuffle=True, seed=42)
datasets

# %%
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, add_prefix_space=True)

# %%
# Not sure if this is needed, but in case we create a span with certain class without starting token of that class,
# let's convert the first token to be the starting token.

e = [0,7,7,7,1,1,8,8,8,9,9,9,14,4,4,4]

def fix_beginnings(labels):
    for i in range(1,len(labels)):
        curr_lab = labels[i]
        prev_lab = labels[i-1]
        if curr_lab in range(7,14):
            if prev_lab != curr_lab and prev_lab != curr_lab - 7:
                labels[i] = curr_lab -7
    return labels

fix_beginnings(e)

# %%
# tokenize and add labels
def tokenize_and_align_labels(examples):

    o = tokenizer(examples['text'], truncation=True, padding=True, return_offsets_mapping=True, max_length=max_length, stride=stride, return_overflowing_tokens=True)

    # Since one example might give us several features if it has a long context, we need a map from a feature to
    # its corresponding example. This key gives us just that.
    sample_mapping = o["overflow_to_sample_mapping"]
    # The offset mappings will give us a map from token to character position in the original context. This will
    # help us compute the start_positions and end_positions.
    offset_mapping = o["offset_mapping"]

    o["labels"] = []

    for i in range(len(offset_mapping)):

        sample_index = sample_mapping[i]

        labels = [l2i['O'] for i in range(len(o['input_ids'][i]))]

        for label_start, label_end, label in \
        list(zip(examples['starts'][sample_index], examples['ends'][sample_index], examples['classlist'][sample_index])):
            for j in range(len(labels)):
                token_start = offset_mapping[i][j][0]
                token_end = offset_mapping[i][j][1]
                if token_start == label_start:
                    labels[j] = l2i[f'B-{label}']
                if token_start > label_start and token_end <= label_end:
                    labels[j] = l2i[f'I-{label}']

        for k, input_id in enumerate(o['input_ids'][i]):
            if input_id in [0,1,2]:
                labels[k] = -100

        labels = fix_beginnings(labels)

        o["labels"].append(labels)

    return o

# %%
tokenized_datasets = datasets.map(tokenize_and_align_labels, batched=True,batch_size=20000, remove_columns=datasets["train"].column_names)
pickle.dump(tokenized_datasets,open('tokenized_datasets.pkl','wb'))
# %%
tokenized_datasets

# %% [markdown]
# ## Model and Training

# %%
# we will use auto model for token classification
from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer

model = AutoModelForTokenClassification.from_pretrained(model_checkpoint, num_labels=N_LABELS)

# %%
###### DECLARING DEEPSPEED CONDFIG ##################

ds_config_dict = {
    "fp16": {
        "enabled": "auto",
        "loss_scale": 0,
        "loss_scale_window": 1000,
        "initial_scale_power": 16,
        "hysteresis": 2,
        "min_loss_scale": 1
    },

    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": "auto",
            "betas": "auto",
            "eps": "auto",
            "weight_decay": "auto"
        }
    },

    "scheduler": {
        "type": "WarmupLR",
        "params": {
            "warmup_min_lr": "auto",
            "warmup_max_lr": "auto",
            "warmup_num_steps": "auto"
        }
    },

    "zero_optimization": {
        "stage": 2,
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": True
        },
        "allgather_partitions": True,
        "allgather_bucket_size": 2e8,
        "overlap_comm": True,
        "reduce_scatter": True,
        "reduce_bucket_size": 5e8,
        "contiguous_gradients": True
    },

    "gradient_accumulation_steps": "auto",
    "gradient_clipping": "auto",
    "steps_per_print": 2000,
    "train_batch_size": "auto",
    "train_micro_batch_size_per_gpu": "auto",
    "wall_clock_breakdown": False
}

# %%
model_name = model_checkpoint.split("/")[-1]
args = TrainingArguments(
    f"{model_name}-finetuned-{task}",
    evaluation_strategy = "epoch",
    logging_strategy = "epoch",
    save_strategy = "epoch",
    learning_rate=LR,
    per_device_train_batch_size=BS,
    per_device_eval_batch_size=BS,
    num_train_epochs=N_EPOCHS,
    weight_decay=WD,
   # report_to='wandb',
    gradient_accumulation_steps=GRAD_ACC,
    warmup_ratio=WARMUP,
    fp16 = True,

    #### THE ONLY CHANGE YOU NEED TO MAKE TO USE DEEPSPEED ########
    deepspeed=ds_config_dict
)

# %%
from transformers import DataCollatorForTokenClassification

data_collator = DataCollatorForTokenClassification(tokenizer)

# %%
# this is not the competition metric, but for now this will be better than nothing...

metric = load_metric("seqeval")

# %%
import numpy as np

def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    # Remove ignored index (special tokens)
    true_predictions = [
        [i2l[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [i2l[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = metric.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }

# %%
trainer = Trainer(
    model,
    args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# %%
trainer.train()

# %%
trainer.save_model(model_path)

# %% [markdown]
# ## End
# 
# I'll appreciate every upvote or comment!


