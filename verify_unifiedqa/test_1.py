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
DEVICE = 2
model_original = T5ForConditionalGeneration.from_pretrained(
    model_name, device_map=f"cuda:{DEVICE}")  #'auto')

# %%
model = model_original
DATABASE_NAME = "obqa_fact"
dataset_test = pickle.load(
    open(f"multiple_choice_datasets/{DATABASE_NAME}_test.pkl", "rb"))
dataset_train = pickle.load(
    open(f"multiple_choice_datasets/{DATABASE_NAME}_train.pkl", "rb"))
MODE = "new"  #'old'

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


def get_model_forward(input_ids,attention_mask, model=model):
    encoder_attentions = None
    last_hidden = None
    print(input_ids)
    print(attention_mask)
    with torch.no_grad():
        start = [0]
        for k in range(50):
            result = model(
                input_ids=input_ids.to(DEVICE),
                attention_mask=attention_mask.to(DEVICE),
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


def run_model(input_strs):
    if input_strs is str:
        input_strs=[input_strs]
    input_ids_wrapper = tokenizer(input_strs, padding=True, return_tensors='pt')
    
    answer, _, _, _ = get_model_forward(input_ids_wrapper['input_ids'],
                                        input_ids_wrapper['attention_mask'])
    return answer


# %%
DEFAULT_MODEL_FORWARD = T5ForConditionalGeneration.forward

# %%
def hook(hook_before, oldfunc, hook_after):

    def foo(*args, **kwargs):
        hook_before(*args, **kwargs)
        aa= oldfunc(*args, **kwargs)
        hook_after(*args, **kwargs)
        return aa

    return foo


def input_before_hooker(*args, **kwargs):
    model.encoder.block[0].layer[0].SelfAttention.set_input_ids = kwargs[
        "input_ids"]
def input_after_hooker(*args, **kwargs):
    model.encoder.block[0].layer[0].SelfAttention.set_input_ids = None

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
    if self.is_decoder:
        values = values.permute([2, 0, 1]).unsqueeze(
            0
        )  # shape (1, num_heads, query_length, key_length)
        return values

    anchors = check_encoded(self.set_input_ids)
    values = []
    for anchor in anchors:
        a = [anchor[0], anchor[1]]
        b = [anchor[1], anchor[2]]
        c = [anchor[2], anchor[3]]
        d = [anchor[3], anchor[4]]
        mot = [a, b, c, d]
        max_answer_length = max([x[1] - x[0] for x in mot])
        # print(a, b, c, d, max_answer_length)
        context_position_new = context_position.clone()
        context_position_new[b[0] : b[1]] = context_position_new[b[0] : b[1]] - a[0]
        context_position_new[c[0] : c[1]] = context_position_new[c[0] : c[1]] - a[0]
        context_position_new[d[0] : d[1]] = context_position_new[d[0] : d[1]] - a[0]
        context_position_new[-1] = context_position_new[a[0]] + 2 * max_answer_length
        memory_position_new = context_position_new.clone().view(1, -1)
        relative_position = (
            memory_position_new - context_position_new
        )  # shape (query_length, key_length)
        for i in range(len(mot)):
            for j in range(len(mot)):
                if i != j:
                    x = mot[i]
                    y = mot[j]
                    relative_position[x[0] : x[1], y[0] : y[1]] += max_answer_length
        relative_position_bucket = self._relative_position_bucket(
            relative_position,  # shape (query_length, key_length)
            bidirectional=(not self.is_decoder),
            num_buckets=self.relative_attention_num_buckets,
            max_distance=self.relative_attention_max_distance,
        )
        value = self.relative_attention_bias(relative_position_bucket)
        values.append(value)
    values = torch.stack(values)  # shape [1, 91, 91, 16]
    values = values.permute(
        [0, 3, 1, 2]
    )  # shape (batch size, num_heads, query_length, key_length)
    return values


def modified_self_attention_forward(
        self,
        hidden_states,
        mask=None,
        key_value_states=None,
        position_bias=None,
        past_key_value=None,
        layer_head_mask=None,
        query_length=None,
        use_cache=False,
        output_attentions=False,
    ):
        """
        Self-attention (if key_value_states is None) or attention over source sentence (provided by key_value_states).
        """
        # Input is (batch_size, seq_length, dim)
        # Mask is (batch_size, key_length) (non-causal) or (batch_size, key_length, key_length)
        # past_key_value[0] is (batch_size, n_heads, q_len - 1, dim_per_head)
        batch_size, seq_length = hidden_states.shape[:2]

        real_seq_length = seq_length

        if past_key_value is not None:
            if len(past_key_value) != 2:
                raise ValueError(
                    f"past_key_value should have 2 past states: keys and values. Got { len(past_key_value)} past states"
                )
            real_seq_length += past_key_value[0].shape[2] if query_length is None else query_length

        key_length = real_seq_length if key_value_states is None else key_value_states.shape[1]

        def shape(states):
            """projection"""
            return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)

        def unshape(states):
            """reshape"""
            return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)

        def project(hidden_states, proj_layer, key_value_states, past_key_value):
            """projects hidden states correctly to key/query states"""
            if key_value_states is None:
                # self-attn
                # (batch_size, n_heads, seq_length, dim_per_head)
                hidden_states = shape(proj_layer(hidden_states))
            elif past_key_value is None:
                # cross-attn
                # (batch_size, n_heads, seq_length, dim_per_head)
                hidden_states = shape(proj_layer(key_value_states))

            if past_key_value is not None:
                if key_value_states is None:
                    # self-attn
                    # (batch_size, n_heads, key_length, dim_per_head)
                    hidden_states = torch.cat([past_key_value, hidden_states], dim=2)
                elif past_key_value.shape[2] != key_value_states.shape[1]:
                    # checking that the `sequence_length` of the `past_key_value` is the same as
                    # the provided `key_value_states` to support prefix tuning
                    # cross-attn
                    # (batch_size, n_heads, seq_length, dim_per_head)
                    hidden_states = shape(proj_layer(key_value_states))
                else:
                    # cross-attn
                    hidden_states = past_key_value
            return hidden_states

        # get query states
        query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)

        # get key/value states
        key_states = project(
            hidden_states, self.k, key_value_states, past_key_value[0] if past_key_value is not None else None
        )
        value_states = project(
            hidden_states, self.v, key_value_states, past_key_value[1] if past_key_value is not None else None
        )

        # compute scores
        scores = torch.matmul(
            query_states, key_states.transpose(3, 2)
        )  # equivalent of torch.einsum("bnqd,bnkd->bnqk", query_states, key_states), compatible with onnx op>9

        if position_bias is None:
            if not self.has_relative_attention_bias:
                position_bias = torch.zeros(
                    (1, self.n_heads, real_seq_length, key_length), device=scores.device, dtype=scores.dtype
                )
                if self.gradient_checkpointing and self.training:
                    position_bias.requires_grad = True
            else:
                position_bias = new_compute_bias(self, real_seq_length, key_length, device=scores.device)

            # if key and values are already calculated
            # we want only the last query position bias
            if past_key_value is not None:
                position_bias = position_bias[:, :, -hidden_states.size(1) :, :]
            print('loi ', mask.shape, torch.allclose(mask,torch.tensor(0)))
            if mask is not None:
                position_bias = position_bias + mask  # (batch_size, n_heads, seq_length, key_length)

        if self.pruned_heads:
            mask = torch.ones(position_bias.shape[1])
            mask[list(self.pruned_heads)] = 0
            position_bias_masked = position_bias[:, mask.bool()]
        else:
            position_bias_masked = position_bias

        scores += position_bias_masked
        attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
            scores
        )  # (batch_size, n_heads, seq_length, key_length)
        attn_weights = nn.functional.dropout(
            attn_weights, p=self.dropout, training=self.training
        )  # (batch_size, n_heads, seq_length, key_length)

        # Mask heads if we want to
        if layer_head_mask is not None:
            attn_weights = attn_weights * layer_head_mask

        attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
        attn_output = self.o(attn_output)

        present_key_value_state = (key_states, value_states) if (self.is_decoder and use_cache) else None
        outputs = (attn_output,) + (present_key_value_state,) + (position_bias,)

        if output_attentions:
            outputs = outputs + (attn_weights,)
        return outputs

# %%

extra_dim_learning = []


def set_mode(MODE):
    itself = model.encoder.block[0].layer[0].SelfAttention
    if MODE == "new":
        itself.forward = partial(modified_self_attention_forward, itself)
    else:
        itself.forward = partial(
            model.encoder.block[0].layer[0].SelfAttention.__class__.forward, itself
        )


def check_encoded(all_input_ids):
    anchors = []
    for input_ids in all_input_ids:
        # print('\n'.join([f'{x.item()},{y}' for x,y in zip(input_ids, tokens)][50:]))
        original = input_ids.tolist()
        anchor = []
        for i in range(len(input_ids)):
            if (
                i < len(input_ids) - 2
                and input_ids[i] == 41
                and input_ids[i + 1] == 3
                and input_ids[i + 2] == 61
            ) or original[i] == 1:
                anchor.append(i)
        anchors.append(anchor)
    return anchors


model.forward = hook(input_before_hooker, partial(DEFAULT_MODEL_FORWARD, model), input_after_hooker)

print(textwrap.fill(dataset_train[0][0]))
print(textwrap.fill(dataset_train[1][0]))

# set_mode("old")
# print("old ", run_model(crazy_text))
#                         # dataset_train[0][0]))
set_mode('old')
run_model(["""A person wants to start""",'mot hai ba'])
set_mode("new")
print("new ", run_model([dataset_train[0][0],dataset_train[0][0]]))
# set_mode("old")
# print("old ", run_model(dataset_train[0][0]))
# set_mode("new")
# print("new ", run_model(dataset_train[0][0]))

# %%
kk = [(index, x, y) for index, (x, y) in enumerate(model.named_parameters())
      if y.requires_grad == True]
[(index, x) for index, x, y in kk if "decoder" in x]
len(kk)
all_position_weight = [
    y for index, x, y in kk if ("extra_dimension_embedding" in x) or (
        ("encoder" in x) and ("relative_attention_bias" in x))
]
to_train_model = [y for index, x, y in kk]

# %%
data_array = [
    (ques,answer, ques.split(" ( ) ")[1:])
    for ques, answer in [
        (
            dataset_train[x][0],
            dataset_train[x][1],
        )
        for x in range(len(dataset_train))
    ]
]

# %%
class CheckTransform(object):

    def __call__(self, sample):
        # print(f"'{sample[1]}'")
        return {
            "input_ids": sample[0],
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
    wrapper = tokenizer(sum([x["all_labels"] for x in datas], []),
                          padding=True)
    wrapper["all_label_ids"] = torch.tensor(wrapper.pop("input_ids"))
    # wrapper["label_index"] = torch.tensor([x["label_index"] for x in datas])
    for k in wrapper["all_label_ids"]:
        k[k == tokenizer.pad_token_id] = -100
    wrapper["all_decoder_attention_masks"] = torch.tensor(
        wrapper.pop("attention_mask"))
    
    for_input = tokenizer([x["input_ids"] for x in datas],padding=True)
    wrapper['input_ids']=torch.tensor(for_input.pop('input_ids'))
    wrapper['attention_mask']=torch.tensor(for_input.pop('attention_mask'))
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
len(data_array)

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
            attention_mask=wrapper['attention_mask'].to(DEVICE),
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
                extra_info= run_model(dataset_test[0][0])
                # final_logits = classification_layer(torch.flatten(hidden, start_dim=1))
                # extra_info = str(final_logits.argmax())
            pbar.set_description_str(f"Loss: {loss_running_score:.3f}")
            pbar.set_postfix_str(extra_info)
pass

# %%
# model.save_pretrained("loi_best_model.pkl", from_pt=True)

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


