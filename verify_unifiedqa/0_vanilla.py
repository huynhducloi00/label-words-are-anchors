# %%
import torch
from tqdm import trange
from transformers import T5Tokenizer, T5ForConditionalGeneration, Adafactor
from functools import wraps, partial
from torch.nn.modules.sparse import Embedding
from torch.optim import Adam, SGD
import torch.nn as nn

model_name = "google-t5/t5-large"
# model_name = (
#     "allenai/unifiedqa-v2-t5-large-1363200"  # you can specify the model size here
# )
tokenizer = T5Tokenizer.from_pretrained(model_name)
DEVICE = 0
model_original = T5ForConditionalGeneration.from_pretrained(
    model_name, device_map=f"cuda:{DEVICE}")  #'auto')

# %%
import copy

# del model
model = model_original  # copy.deepcopy(model_original)


# %%
def DEFAULT_COMPUTE_BIAS(self, query_length, key_length, device=None):
    """Compute binned relative position bias"""
    if device is None:
        device = self.relative_attention_bias.weight.device
    context_position = torch.arange(query_length,
                                    dtype=torch.long,
                                    device=device)[:, None]
    memory_position = torch.arange(key_length, dtype=torch.long,
                                   device=device)[None, :]
    relative_position = (memory_position - context_position
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
        0)  # shape (1, num_heads, query_length, key_length)
    return values


# %%
import pickle


DATABASE_NAME = "obqa_fact"
dataset_test = pickle.load(
    open(f"multiple_choice_datasets/{DATABASE_NAME}_test.pkl", "rb")
)
dataset_train = pickle.load(
    open(f"multiple_choice_datasets/{DATABASE_NAME}_test.pkl", "rb")
)
# %%
MODE = "new"  #'old'

# if hasattr(layer, 'EncDecAttention'):
#     layer.EncDecAttention.compute_bias = partial(
#         new_compute_bias, layer.EncDecAttention)

# %%
model.hf_device_map

# %%
import textwrap


def measure_unalike(arr, print_arr=False):
    n = len(arr)
    arr = pd.Series(arr).value_counts()
    if print_arr:
        print(arr)
    return 1 - ((arr / n)**2).sum()


question_to_do = 5
per_question = 20


def run_tokens(tokens):
    res = model.generate(tokens, max_new_tokens=MAX_ANSWER_LENGTH)
    return tokenizer.batch_decode(res, skip_special_tokens=True)


def run_model(input_string, **generator_args):
    input_ids = tokenizer.encode(input_string, return_tensors="pt")
    # print(torch.argwhere(input_ids[0]==2)[0,0]+2)
    res = model.generate(input_ids.to(DEVICE),
                         **generator_args,
                         max_new_tokens=MAX_ANSWER_LENGTH)
    return tokenizer.batch_decode(res, skip_special_tokens=True)


# %%
# %%
# %%
QUESTION_MAX_LENGTH = 76
MAX_ANSWER_LENGTH = 40


# %%
def check(input_text):
    input_ids = tokenizer.encode(input_text, return_tensors="pt")[0]
    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    original = input_ids.tolist()
    anchor = []
    for i in range(len(tokens)):
        if (i < len(tokens) - 2 and tokens[i] == "▁(" and tokens[i + 1] == "▁"
                and tokens[i + 2] == ")") or original[i] == 1:
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
        [
            original.insert(anchor[0], 0)
            for _ in range(QUESTION_MAX_LENGTH - anchor[0])
        ]
    else:
        print(f"Wrong size QUESTION: {anchor[0]}")
        return None
    return torch.tensor(original).view(1, -1)


start_pos = QUESTION_MAX_LENGTH
leng = MAX_ANSWER_LENGTH
a = torch.arange(start_pos + leng * 0, start_pos + leng * 1, dtype=int)
b = torch.arange(start_pos + leng * 1, start_pos + leng * 2, dtype=int)
c = torch.arange(start_pos + leng * 2, start_pos + leng * 3, dtype=int)
d = torch.arange(start_pos + leng * 3, start_pos + leng * 4, dtype=int)
question_portion = torch.arange(0, start_pos)
question_ending = start_pos + leng * 4
DEC = {"01": 0, "02": 1, "03": 2, "12": 3, "13": 4, "23": 5}
mot = [a, b, c, d]
six_mask_turn_off = torch.ones((237, 237, 16))
six_mask_turn_on = torch.zeros((6, 237, 237, 16))

for i in range(len(mot) - 1):
    for j in range(i + 1, len(mot)):
        x = mot[i]
        y = mot[j]
        # print(mask_turn_off_hyper_dimension[x][:, y][:].shape)
        # cal index in 6
        comb_index = DEC[f"{i}{j}"]
        # no distance, a very special distance
        six_mask_turn_on[comb_index][x][:, y][:] = 2
        six_mask_turn_on[comb_index][y][:, x][:] = 2
        six_mask_turn_off[x][:, y][:] = 0
        six_mask_turn_off[y][:, x][:] = 0


# %%
def new_compute_bias(self, query_length, key_length, device=None):
    """Compute binned relative position bias"""
    if device is None:
        device = self.relative_attention_bias.weight.device
    context_position = torch.arange(query_length,
                                    dtype=torch.long,
                                    device=device)[:, None]
    memory_position = torch.arange(key_length, dtype=torch.long,
                                   device=device)[None, :]

    relative_position = (memory_position - context_position
                         )  # shape (query_length, key_length)
    # implementation='simple'
    if self.is_decoder:
        pass
    else:
        context_position_new = context_position.clone()
        context_position_new[b] = context_position_new[a]
        context_position_new[c] = context_position_new[a]
        context_position_new[d] = context_position_new[a]
        context_position_new[-1] = context_position_new[a[0]] + leng
        memory_position_new = context_position_new.clone().view(1, -1)
        relative_position = (memory_position_new - context_position_new
                             )  # shape (query_length, key_length)

    relative_position_bucket = self._relative_position_bucket(
        relative_position,  # shape (query_length, key_length)
        bidirectional=(not self.is_decoder),
        num_buckets=self.relative_attention_num_buckets,
        max_distance=self.relative_attention_max_distance,
    )

    values = self.relative_attention_bias(relative_position_bucket)
    values = values.permute([2, 0, 1]).unsqueeze(
        0)  # shape (1, num_heads, query_length, key_length)
    return values


extra_dim_learning = []


def set_mode(MODE):
    for part in ["encoder"]:  # , 'decoder']:
        for block in getattr(model, part).block:
            for layer in block.layer:
                if hasattr(layer, "SelfAttention"):
                    # ALL block
                    itself = layer.SelfAttention
                    if layer.SelfAttention.has_relative_attention_bias:
                        pass
                        # # block 0 ONLY
                        # if MODE == "new":
                        #     itself.compute_bias = partial(
                        #         new_compute_bias, layer.SelfAttention)
                        # else:
                        #     itself.compute_bias = partial(
                        #         DEFAULT_COMPUTE_BIAS, layer.SelfAttention)


print(textwrap.fill(dataset_train[0][0]))
set_mode("old")
# print("old ", run_tokens(check(dataset_train[0][0]).to(DEVICE)))
set_mode("new")
# print("new ", run_tokens(check(dataset_train[0][0]).to(DEVICE)))

# %%
kk = [(index, x, y) for index, (x, y) in enumerate(model.named_parameters())
      if y.requires_grad == True]
to_train_model = [y for index, x, y in kk]  # [:196]]

# %%
from random_utils import set_seed

set_seed(42)


def shape(input):
    return input.view(1, -1)


# %%
def get_model_forward(input_tokens):
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
from torch.utils.data import Dataset, DataLoader


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


from tqdm import tqdm
from random_utils import set_seed

loss_running_score = 0
correct_running_score = 0
conform_running_score = 0
count = 0
extra_info = ""
res_tokens = []
accumulate = 10
optimizer.zero_grad()
set_seed(42)
for learn_pos in range(5):
    pbar = tqdm(loi_dataloader)
    for wrapper in pbar:
        count += 1
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
            pbar.set_description_str(f"Loss: {loss_running_score:.3f}")
            pbar.set_postfix_str(extra_info)
pass

# %%
# len_to_find=len(wrapper["label_ids"][0])-1
#         for batch in range(wrapper["input_ids"].shape[0]):
#             keys= result.logits[batch].argmax(dim=1)
#             found=False
#             tui_batch=wrapper["label_ids"][batch]
#             for m in range(len(tui_batch)):
#                 if keys[m]!=tui_batch[m]:
#                     if len_to_find>m:
#                         len_to_find=m
#                     break
#         for batch in range(wrapper["input_ids"].shape[0]):
#             loss+=loss_fn(result.logits[batch][:len_to_find+1],tui_batch[:len_to_find+1].to(DEVICE))
#         loss=loss/wrapper["input_ids"].shape[0]


model.save_pretrained("loi_vanilla.pkl", from_pt=True)
for data in [dataset_test]:
    print(f'test {data==dataset_test}')
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
print('=============')
for data in [dataset_test]:
    count = 0
    count1 = 0
    count2 = 0
    count10 = 0
    total = 0
    pbar1 = tqdm(range(24*3,24*4))
    for ques in pbar1:
        question = data[ques][0]
        key = data[ques][1]
        question_convert = check(question)
        if question_convert is None:
            continue
        total += 1
        answer, _, _, _ = get_model_forward(question_convert.to(DEVICE), model=model1)
        # print(ques, question)
        # print(answer,',', key)
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
