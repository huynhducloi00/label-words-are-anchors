    # %%
import torch
import pickle
from torch.utils.data import Dataset, DataLoader
from tqdm import trange
from transformers import T5Tokenizer, T5ForConditionalGeneration, Adafactor
from functools import wraps, partial
from torch.nn.modules.sparse import Embedding
from torch.optim import Adam, SGD
import torch.nn as nn
from tqdm import tqdm
from random_utils import set_seed
import pandas as pd

# %%
model_name = "google-t5/t5-large"
# model_name = (
#     "allenai/unifiedqa-v2-t5-large-1363200"  # you can specify the model size here
# )
tokenizer = T5Tokenizer.from_pretrained(model_name)
DEVICE = 3
model_original = T5ForConditionalGeneration.from_pretrained(
    model_name, device_map=f"cuda:{DEVICE}")  #'auto')

# %%
model = model_original


def DEFAULT_COMPUTE_BIAS(self, query_length, key_length, device=None):
    """Compute binned relative position bias"""
    if device is None:
        device = self.relative_attention_bias.weight.device
    context_position = torch.arange(query_length, dtype=torch.long, device=device)[
        :, None
    ]
    memory_position = torch.arange(key_length, dtype=torch.long, device=device)[None, :]
    relative_position = (
        memory_position - context_position
    )  # shape (query_length, key_length)
    relative_position_bucket = self._relative_position_bucket(
        relative_position,  # shape (query_length, key_length)
        bidirectional=(not self.is_decoder),
        num_buckets=self.relative_attention_num_buckets,
        max_distance=self.relative_attention_max_distance,
    )
    values = self.relative_attention_bias(
        relative_position_bucket
    )  # shape (query_length, key_length, num_heads)
    values = values.permute([2, 0, 1]).unsqueeze(
        0
    )  # shape (1, num_heads, query_length, key_length)
    return values


DATABASE_NAME = "obqa_fact"
dataset_test = pickle.load(
    open(f"multiple_choice_datasets/{DATABASE_NAME}_test.pkl", "rb")
)
dataset_train = pickle.load(
    open(f"multiple_choice_datasets/{DATABASE_NAME}_test.pkl", "rb")
)
MODE = "new"  #'old'

model.hf_device_map

# %%
import textwrap


def measure_unalike(arr, print_arr=False):
    n = len(arr)
    arr = pd.Series(arr).value_counts()
    if print_arr:
        print(arr)
    return 1 - ((arr / n) ** 2).sum()


question_to_do = 5
per_question = 20
def get_model_forward(input_tokens, model=model):
    encoder_attentions = None
    last_hidden = None
    with torch.no_grad():
        start = [0]
        for k in range(MAX_ANSWER_LENGTH):
            result = model(
                input_ids=input_tokens.to(DEVICE),
                decoder_input_ids=torch.tensor([start]).to(DEVICE),
                output_attentions=True,
            )
            encoder_attentions = result.encoder_attentions
            last_hidden = result.encoder_last_hidden_state
            item = result.logits.argmax(dim=2)[0][-1].item()
            start.append(item)
            if item == 1:
                break
            # print(start)
    return (
        tokenizer.decode(start, skip_special_tokens=True),
        tokenizer.convert_ids_to_tokens(start),
        last_hidden,
        encoder_attentions,
    )
def run_model(input_str):
    input_ids=tokenizer.encode(input_str, return_tensors="pt")
    answer,_,_,_=get_model_forward(input_ids)
    return answer


# %%
DEFAULT_MODEL_FORWARD = model.forward

# %%
# %%
# %%
from typing import Optional, Tuple


QUESTION_MAX_LENGTH = 76
MAX_ANSWER_LENGTH = 40

def check_encoded(input_ids):
    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    print(tokens)
    original = input_ids.tolist()
    anchor = []
    for i in range(len(tokens)):
        if (
            i < len(tokens) - 2
            and tokens[i] == "▁("
            and tokens[i + 1] == "▁"
            and tokens[i + 2] == ")"
        ) or original[i] == 1:
            anchor.append(i)
    # 0 1 2 3 4
    for x in reversed(range(1, 5)):
        if anchor[x] - anchor[x - 1] < MAX_ANSWER_LENGTH:
            [
                original.insert(anchor[x], 0)
                for _ in range(MAX_ANSWER_LENGTH - (anchor[x] - anchor[x - 1]))
            ]
        else:
            print(f"Wrong size ANSWER: {anchor[x] - anchor[x - 1] }")
            return None
    if anchor[0] < QUESTION_MAX_LENGTH:
        [original.insert(anchor[0], 0) for _ in range(QUESTION_MAX_LENGTH - anchor[0])]
    else:
        print(f"Wrong size QUESTION: {anchor[0]}")
        return None
    return torch.tensor(original).view(1, -1)

# %%
def new_compute_bias(self, query_length, key_length, device=None):
    """Compute binned relative position bias"""
    if device is None:
        device = self.relative_attention_bias.weight.device
    context_position = torch.arange(query_length, dtype=torch.long, device=device)[
        :, None
    ]
    memory_position = torch.arange(key_length, dtype=torch.long, device=device)[None, :]

    relative_position = (
        memory_position - context_position
    )  # shape (query_length, key_length)
    # implementation='simple'
    if self.is_decoder:
        pass
    else:
        check_encoded(self.set_input_ids)
        # a,b,c,d,
        # mot=[a,b,c,d]
        context_position_new = context_position.clone()
        context_position_new[b[0] : b[1]] = context_position_new[a[0] : a[1]]
        context_position_new[c[0] : c[1]] = context_position_new[a[0] : a[1]]
        context_position_new[d[0] : d[1]] = context_position_new[a[0] : a[1]]
        context_position_new[-1] = context_position_new[a[0]] + leng
        memory_position_new = context_position_new.clone().view(1, -1)
        relative_position = (
            memory_position_new - context_position_new
        )  # shape (query_length, key_length)
        for i in range(len(mot)):
            for j in range(len(mot)):
                if i != j:
                    x = mot[i]
                    y = mot[j]
                    relative_position[x[0] : x[1], y[0] : y[1]] += MAX_ANSWER_LENGTH
    relative_position_bucket = self._relative_position_bucket(
        relative_position,  # shape (query_length, key_length)
        bidirectional=(not self.is_decoder),
        num_buckets=self.relative_attention_num_buckets,
        max_distance=self.relative_attention_max_distance,
    )
    implementation = "complicated1"  # "change_32"  # "complicated"
    values = self.relative_attention_bias(relative_position_bucket)
    values = values.permute([2, 0, 1]).unsqueeze(
        0
    )  # shape (1, num_heads, query_length, key_length)
    return values


extra_dim_learning = []


def model_forward(self, input_ids, **kwargs):
    model.encoder.block[0].layer[0].SelfAttention.set_input_ids = input_ids
    DEFAULT_MODEL_FORWARD(input_ids, **kwargs)
    print('here ',model.encoder.block[0].layer[0].SelfAttention)

def set_mode(MODE):
    itself = model.encoder.block[0].layer[0].SelfAttention
    model.forward = partial(model_forward, model)
    if MODE == "new":
        itself.compute_bias = partial(new_compute_bias, itself)
    else:
        itself.compute_bias = partial(DEFAULT_COMPUTE_BIAS, itself)


print(textwrap.fill(dataset_train[0][0]))
# set_mode("old")
# print("old ", run_tokens(check(dataset_train[0][0]).to(DEVICE)))
set_mode("new")
print("new ", run_model(dataset_train[0][0]))

# %%
kk = [(index, x, y) for index, (x, y) in enumerate(model.named_parameters())
      if y.requires_grad == True]
[(index, x) for index, x, y in kk if "decoder" in x]
len(kk)
all_position_weight = [
    y for index, x, y in kk if ("extra_dimension_embedding" in x) or (
        ("encoder" in x) and ("relative_attention_bias" in x))
]

# %%
not_train = [
    "shared.weight",
    "encoder.block.0.layer.0.SelfAttention.q.weight",
    "encoder.block.0.layer.0.SelfAttention.k.weight",
    "encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight",
]
to_train_model = [(x, y) for index, x, y in kk]  # [:196]]
# to_train_model=to_train_model+
# to_train_model=[(x, y) for x, y in model.named_parameters() if x=="encoder.block.0.layer.0.SelfAttention.extra_dimension_embedding.weight"]
# to_train=[]
to_train = [
    "encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight",
    "encoder.block.0.layer.0.layer_norm.weight",
    "encoder.block.0.layer.1.DenseReluDense.wi.weight",
    "encoder.block.0.layer.1.DenseReluDense.wo.weight",
    "encoder.block.0.layer.1.layer_norm.weight",
    "encoder.block.0.layer.0.SelfAttention.extra_dimension_embedding.weight",
]
# to_train_model=[(x,y) for x, y in model.named_parameters() if x in to_train]
# to_train_model = [(x, y) for x, y in model.named_parameters()
#                   if not x in not_train]

for y in model.parameters():
    y.requires_grad = False
for x, y in to_train_model:
    y.requires_grad = True
[x for x, y in to_train_model]

# %%
# for param in model.parameters():
#     param.requires_grad = False
# to_train_model=[y for x, y in model.named_parameters() if x in train_name_list ]
# for y in to_train_model:
#     y.requires_grad=True

# %%
from random_utils import set_seed

set_seed(42)


def shape(input):
    return input.view(1, -1)

# %%
# from transformers import AdamW, get_linear_schedule_with_warmup
# no_decay = ['layer_norm.weight']
# optimizer_grouped_parameters = [
#     {'params': [p for n, p in to_train_model], 'weight_decay': 0.0},
#     ]
# optimizer = AdamW(optimizer_grouped_parameters, lr=1e-5, eps=1e-8)
# scheduler =  get_linear_schedule_with_warmup(optimizer,
#                                 num_warmup_steps=0,
#                                 num_training_steps=100000)
to_train_model = [y for x, y in to_train_model]
# optimizer = SGD(to_train_model, lr=1e-2)

# %%
import numpy as np
from torch.nn import CrossEntropyLoss

loss_fn = CrossEntropyLoss(ignore_index=-100)


def get_loss(logits, labels):
    loss = torch.tensor(0)
    found = False
    for i in range(len(labels[0])):
        current_loss = loss_fn(logits[0][i], labels[0][i].to(DEVICE))

        current_certainly = torch.exp(-current_loss)
        if current_certainly < 0.9:
            loss = current_loss
            found = True
            break
    if not found:
        loss = loss_fn(logits[0], labels[0].to(DEVICE))
    return loss

# %%
if False:
    # pbar = trange(0, len(dataset_train), 24)
    # loss_score = 0
    # count = 0
    # extra_info = ""
    # step=0
    # # if count>20:
    # #     break
    # # print(textwrap.fill(dataset_train[0][0]))
    step = 0
    pbar = trange(200)
    for re in pbar:
        input_tokens = check(dataset_train[step][0])
        labels = tokenizer.encode(dataset_train[step][1], return_tensors="pt")
        result = model(input_ids=input_tokens.to(DEVICE),
                       labels=shape(labels).to(DEVICE))
        loss = get_loss(result.logits, labels)
        # print(result.logits.argmax(dim=2), labels)
        optimizer.zero_grad()
        # loss = result.loss
        # print(result.logits, labels, loss)
        if loss.item() != 0:
            loss_score = loss.item()  # loss_score * 0.9 + loss.item() * 0.1
            loss.backward()
        optimizer.step()
        # scheduler.step()
        # with torch.no_grad():
        #     mong= model(input_ids=check(dataset_train[0][0]).to(DEVICE), decoder_input_ids=torch.tensor([[0]]).to(DEVICE))
        #     print(mong.logits.argmax(dim=2).shape)
        # print(tokenizer.decode())

        extra_info = get_model_forward(
            check(dataset_train[step][0]).to(DEVICE))
        pbar.set_postfix_str(f"Loss: {loss_score:.10f}:{extra_info}")

# %%
data_array = [(k, v, l.split(" ( ) ")[1:])
              for l, k, v in [(dataset_train[x][0], check(dataset_train[x][0]),
                               dataset_train[x][1])
                              for x in range(0, len(dataset_train), 24)]
              if k is not None]

# %%
class CheckTransform(object):

    def __call__(self, sample):
        # print(f"'{sample[1]}'")
        return {
            "input_ids": sample[0][0],
            "label_index": sample[2].index(sample[1]),
            "all_labels": sample[2],
        }


class CustomDataset(Dataset):

    def __init__(self, dataset_array, transform=None):
        self.dataset = dataset_array
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.transform(self.dataset[idx])


def collate(datas):
    label_ids = tokenizer(sum([x["all_labels"] for x in datas], []),
                          padding=True)
    wrapper = label_ids
    wrapper["all_label_ids"] = torch.tensor(wrapper.pop("input_ids"))
    # wrapper["label_index"] = torch.tensor([x["label_index"] for x in datas])
    for k in wrapper["all_label_ids"]:
        k[k == tokenizer.pad_token_id] = -100
    wrapper["all_decoder_attention_masks"] = torch.tensor(
        wrapper.pop("attention_mask"))
    wrapper["input_ids"] = torch.stack([x["input_ids"] for x in datas])
    wrapper["label_index"] = torch.tensor([x["label_index"] for x in datas])
    return wrapper


loi_dataloader = DataLoader(
    CustomDataset(
        data_array,
        CheckTransform(),
    ),
    batch_size=10,
    shuffle=True,
    collate_fn=collate,
)
# for k in loi_dataloader:
#     print(k["all_label_ids"])
#     break

# %%
# attention 898704
# hidden state 242688
# classification_layer = nn.Linear(242688, 4).to(DEVICE)
optimizer = Adafactor(
    to_train_model,  # + [x for x in classification_layer.parameters()],
    relative_step=True,
    warmup_init=True,
    lr=None,
)

# %%
def turn_position_learning(on):
    for x in all_position_weight:
        x.requires_grad = on


loss_running_score = 0
correct_running_score = 0
conform_running_score = 0
count = 0
extra_info = ""
res_tokens = []
accumulate = 10
optimizer.zero_grad()
set_seed(42)
turn_position = False
turn_position_learning(False)
for learn_pos in range(6):
    pbar = tqdm(loi_dataloader)
    for wrapper in pbar:
        count += 1
        # if count%20==0:
        #     turn_position=not turn_position
        #     turn_position_learning(turn_position)
        # if count>20:
        #     break
        # print(textwrap.fill(dataset_train[0][0]))
        only_correct_label_ids = torch.stack([
            wrapper["all_label_ids"][batch_index * 4 + x]
            for batch_index, x in enumerate(wrapper["label_index"])
        ])
        only_correct_decoder_attention_mask = torch.stack([
            wrapper["all_decoder_attention_masks"][batch_index * 4 + x]
            for batch_index, x in enumerate(wrapper["label_index"])
        ])
        result = model(
            input_ids=wrapper["input_ids"].to(DEVICE),
            labels=only_correct_label_ids.to(DEVICE),
            decoder_attention_mask=only_correct_decoder_attention_mask.to(
                DEVICE),  # output_attentions=True
        )
        # conform_loss = 0
        # for batch in range(wrapper["input_ids"].shape[0]):
        #     selected_answer = result.logits[batch].argmax(dim=1)
        #     found = False
        #     conform_losses = [0, 0, 0, 0]
        #     for each_answer in range(4):
        #         tui_batch = wrapper["all_label_ids"][batch * 4 + each_answer]
        #         conform_losses[each_answer] += loss_fn(
        #                     result.logits[batch], tui_batch.to(DEVICE)
        #                 )
        #         # for m in range(len(tui_batch)):
        #         #     if selected_answer[m] != tui_batch[m] and tui_batch[m] != -100:
        #         #         conform_losses[each_answer] += loss_fn(
        #         #             result.logits[batch][m], tui_batch[m].to(DEVICE)
        #         #         )
        #         # conform_min_index = torch.argmin(conform_losses)
        #         # print(conform_min_index)
        #     conform_loss += min(conform_losses)  # conform_losses[conform_min_index]
        # conform_loss = conform_loss / wrapper["input_ids"].shape[0]
        # kk1=result.encoder_attentions
        # break
        # final_logits = classification_layer(
        #     torch.flatten(result.encoder_last_hidden_state, start_dim=1)
        # )
        # loss = loss_fn(final_logits, wrapper["label_index"].to(DEVICE))
        loss = result.loss
        loss_running_score = loss_running_score * 0.9 + loss.item() * 0.1
        if loss != 0:
            loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        # scheduler.step()
        with torch.no_grad():
            if count % 10 == 0:
                extra_info, res_tokens, _, _ = get_model_forward(
                    check(dataset_test[0][0]).to(DEVICE))
                # final_logits = classification_layer(torch.flatten(hidden, start_dim=1))
                # extra_info = str(final_logits.argmax())
            pbar.set_description_str(f"Loss: {loss_running_score:.3f}")
            pbar.set_postfix_str(extra_info)
pass

# %%
model.save_pretrained("loi_best_model.pkl", from_pt=True)

# %% [markdown]
# ### Measure accuracy and answer coverage

# %%
for data in [dataset_test, dataset_train]:
    print(f"test {data==dataset_test}")
    count = 0
    count1 = 0
    count2 = 0
    count10 = 0
    total = 0
    pbar1 = trange(int(len(data) / 24))
    for ques in pbar1:
        question = data[24 * ques][0]
        key = data[24 * ques][1]
        question_convert = check(question)
        if question_convert is None:
            continue
        total += 1
        answer, _, _, _ = get_model_forward(question_convert.to(DEVICE))
        if key == answer:
            count += 1
        if key[0] == answer[0]:
            count1 += 1
        if key[:2] == answer[:2]:
            count2 += 1
        if answer in question:
            count10 += 1
        pbar1.set_postfix_str(
            f"{count}, {count1}, {count2}, {count10},{total}")

# %% [markdown]
# ### Measure resilient

# %%
def measure_unalike(arr):
    n = len(arr)
    arr = pd.Series(arr).value_counts()
    return 1 - ((arr / n)**2).sum()


measure_unalike(["a", "a", "a"])

# %%
for data in [dataset_test]:
    count = 0
    count1 = 0
    count2 = 0
    count10 = 0
    total = 0
    question_index = range(5)
    pbar1 = tqdm(question_index)
    unalike = []
    for ques1 in pbar1:
        answer_set = []
        for m in trange(24):
            ques = ques1 * 24 + m
            question = data[ques][0]
            key = data[ques][1]
            question_convert = check(question)
            if question_convert is None:
                continue
            total += 1
            answer, _, _, _ = get_model_forward(question_convert.to(DEVICE),
                                                model=model2)
            answer_set.append(answer)
        unalike.append(measure_unalike(answer_set))
print(f"Mean unalikeability: {sum(unalike)/len(unalike)}")

# %%
# pbar = trange(0, len(dataset_train), 24)
# loss_score = 0
# count = 0
# extra_info = ""
# set_seed(42)
# res_tokens=[]
# for learn_pos in range(10):
#     for step in pbar:
#         count += 1
#         # if count>20:
#         #     break
#         # print(textwrap.fill(dataset_train[0][0]))
#         input_tokens = check(dataset_train[step][0])
#         if input_tokens is None:
#             continue
#         labels = tokenizer.encode(dataset_train[step][1], return_tensors="pt")
#         result = model(input_ids=input_tokens.to(DEVICE), labels=shape(labels).to(DEVICE))

#         optimizer.zero_grad()
#         loss =loss_fn(result.logits[0][learn_pos],labels[0][learn_pos].to(DEVICE))
#         loss_score = loss_score * 0.9 + loss.item() * 0.1
#         if loss.item()!=0:
#             loss.backward()
#         optimizer.step()
#         # scheduler.step()
#         with torch.no_grad():
#             if count % 10 == 0:
#                 extra_info, res_tokens = get_model_forward(check(dataset_test[0][0]).to(DEVICE))
#             pbar.set_description_str(f"Loss: {loss_score:.2f}")
#             pbar.set_postfix_str(res_tokens[:learn_pos+2])
# pass

# %%
model1 = T5ForConditionalGeneration.from_pretrained(
    "loi_vanilla.pkl", device_map=f"cuda:{DEVICE}")

# %%
model2 = T5ForConditionalGeneration.from_pretrained(
    "loi_best_model.pkl", device_map=f"cuda:{DEVICE}"
)


