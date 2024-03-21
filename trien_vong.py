from matplotlib.colors import LinearSegmentedColormap
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
from icl.utils.experiment_utils import set_gpu
import copy
from icl.lm_apis.lm_api_base import LMForwardAPI
from icl.util_classes.arg_classes import AttrArgs
from icl.analysis.attentioner_for_train import (
    LlamaAttentionerManager,
    Mode,
    get_attn_adapter_initializer,
)
from icl.util_classes.predictor_classes import Predictor
from icl.utils.load_local import get_model_layer_num
import numpy as np
from icl.util_classes.arg_classes import AttrArgs
import matplotlib.pyplot as plt
import torch

from icl.utils.prepare_model_and_tokenizer import (
    get_label_id_dict_for_args,
    load_model_customize,
    load_tokenizer,
)
from icl.utils.load_huggingface_dataset import load_huggingface_dataset_train_and_test
from icl.utils.other import dict_to
from icl.analysis.reweighting import quick_prep_input

seed = 42
neglect_args = AttrArgs(
    version="normal2",
    task_name="obqa",
    sample_size=1000,
    seeds=[seed],
    demonstration_shot=0,
)

# %%
from icl.utils.load_huggingface_dataset import load_huggingface_dataset_train_and_test

dataset = load_huggingface_dataset_train_and_test(neglect_args.task_name)
tokenizer = load_tokenizer(neglect_args)

# %%
model_original = load_model_customize(neglect_args)
model_copy = copy.deepcopy(model_original)
label_id_dict = get_label_id_dict_for_args(neglect_args, tokenizer)

# %%
tokenizer.model_max_length

# %%
# import imptools  # pip3 install imptools

# my_module = imptools.import_path(
#     '/home/ldh0033@auburn.edu/learning_nlp/SocialSense/testing/llm_unlearn_loi_version/label-words-are-anchors/icl/util_classes/predictor_classes',  # Path to a module directory or single file.
#     notfound='error',        # Raise 'error' or 'ignore' if not found.
#     reload=True,            # Whether to import if already available.
# )
# from icl.util_classes.predictor_classes import Predictor

# %%
from icl.utils.data_wrapper import prepare_dataset


def sen_gen_inverted(input_sample):
    choices = input_sample["choices"]["text"]
    inputs_1 = f"""Question: {input_sample['question_stem']}? {choices[0]} A; {choices[1]} B; {choices[2]} C; {choices[3]} D; Answer:"""
    return inputs_1


test_dataset_inverted = prepare_dataset(
    seed, dataset["test"], -1, neglect_args, tokenizer, sen_gen_inverted
)
train_dataset_inverted = prepare_dataset(
    seed, dataset["train"], -1, neglect_args, tokenizer, sen_gen_inverted
)

# %%
### Remove 4151
def remove_point(dataset,point):
    select_ = list(range(len(dataset)))
    select_.remove(point)
    return dataset.select(select_)

# %%
a_label, b_label, c_label, d_label = (319, 350, 315, 360)
star_fake = tokenizer.convert_tokens_to_ids("*")
question_label1 = tokenizer.convert_tokens_to_ids("?")
question_label2 = tokenizer.convert_tokens_to_ids("??")
question_label3 = tokenizer.convert_tokens_to_ids("▁??")
question_label4 = tokenizer.convert_tokens_to_ids("▁???")
answer_label = tokenizer.convert_tokens_to_ids(":")
question_label0 = tokenizer.convert_tokens_to_ids(":")
gap = []


def where_am(inputs, *token_ids):
    aa = [torch.argwhere(inputs == x) for x in token_ids]
    aa = [x for x in aa if x.shape[0] > 0]
    if len(aa) == 1:
        return aa[0][-1]
    return aa
from tqdm import trange

marker = tokenizer.convert_tokens_to_ids("-")


def convert_to_fixed_position(dataset):
    converted_data = []
    for j in trange(len(dataset)):
        # print(i, train_dataset_inverted[i]['sentence'])
        inputs = torch.tensor(dataset[j]["input_ids"])
        a = where_am(inputs, a_label)
        b = where_am(inputs, b_label)
        c = where_am(inputs, c_label)
        d = where_am(inputs, d_label)
        question_mark = where_am(
            inputs, question_label1, question_label2, question_label3
        )
        inputs = inputs.numpy()
        len_a, len_b, len_c, len_d = (a - question_mark, b - a, c - b, d - c)
        for i in range(35 - len_d):
            inputs = np.insert(inputs, c + 2, star_fake)
        for i in range(35 - len_c):
            inputs = np.insert(inputs, b + 2, star_fake)
        for i in range(35 - len_b):
            inputs = np.insert(inputs, a + 2, star_fake)
        for i in range(35 - len_a):
            inputs = np.insert(inputs, question_mark + 1, star_fake)
        for i in range(90 - question_mark):
            inputs = np.insert(inputs, question_mark + 1, star_fake)
        converted_data.append(
            {"input_ids": torch.tensor(inputs), "labels": dataset[j]["labels"]}
        )
    return converted_data

    # print(tokenizer.decode(inputs))

# %%
test_dataset_inverted=remove_point(test_dataset_inverted,185)

# %%
converted_test = convert_to_fixed_position(test_dataset_inverted)

# %%
modified_prompt_model = LMForwardAPI(
    model=model_copy,
    model_name=neglect_args.model_name,
    tokenizer=tokenizer,
    label_id_dict=label_id_dict,
    output_attention=True,
)
predictor = Predictor(
    label_id_dict=get_label_id_dict_for_args(neglect_args, tokenizer),
    pad_token_id=tokenizer.pad_token_id,
    task_name=neglect_args.task_name,
    tokenizer=tokenizer,
    layer=get_model_layer_num(
        model=modified_prompt_model.model, model_name=neglect_args.model_name
    ),
)
for p in modified_prompt_model.parameters():
    p.requires_grad = False

# %%
cmap = LinearSegmentedColormap.from_list("", ["red", "white", "blue"])


def get_df(
    bundle, question_id, layer_id, show_info=False, show_attention=False, start_index=0
):
    ques = bundle[question_id]
    viewing = (
        ques["attentions"][layer_id] if show_attention else -ques["grads"][layer_id]
    )
    ss = viewing.shape[-1]  # ques["final_pos"] + 1
    tokens = ques["tokens"][start_index:ss]
    viewing = viewing[start_index:ss, start_index:ss].clone()
    if show_attention:
        viewing[[np.arange(viewing.shape[0])] * 2] = 0
    df_cm = pd.DataFrame(viewing, index=tokens, columns=tokens)
    return df_cm


def show_question(
    bundle, question_id, layer_id, show_info=False, show_attention=False, start_index=0
):
    ques = bundle[question_id]
    if show_info:
        print(ques["question"])
        print(
            f"percent: {[f'{x:.2f}' for x in ques['percentage']]}, correct choice: {ques['correct_choice']}"
        )

    plt.figure(figsize=(20, 15))
    df_cm = get_df(
        bundle, question_id, layer_id, show_info, show_attention, start_index
    )
    if show_attention:
        sn.heatmap(df_cm, annot=False, cmap="Blues")
    else:
        scale_max = abs(np.max(df_cm.values))
        sn.heatmap(
            df_cm,
            annot=False,
            vmin=-scale_max,
            vmax=scale_max,
            cmap=LinearSegmentedColormap.from_list("", ["red", "white", "blue"]),
        )
    # return viewing

# %%
from icl.analysis.attentioner_for_train import CustomPathOnlyAttentionAdapter

a_label, b_label, c_label, d_label = (319, 350, 315, 360)
answer_label = tokenizer.convert_tokens_to_ids(":")
question_label = [
    tokenizer.convert_tokens_to_ids("?"),
    tokenizer.convert_tokens_to_ids("??"),
]


def finding_pos(input_ids):
    a = torch.argwhere(input_ids == a_label)[-1].item()
    b = torch.argwhere(input_ids == b_label)[-1].item()
    c = torch.argwhere(input_ids == c_label)[-1].item()
    d = torch.argwhere(input_ids == d_label)[-1].item()
    answer_mark = torch.argwhere(input_ids == answer_label)[-1].item()
    question_mark1 = torch.argwhere((input_ids == question_label[0]))
    question_mark2 = torch.argwhere((input_ids == question_label[1]))
    question_mark = question_mark1 if question_mark1.shape[0] > 0 else question_mark2
    return a, b, c, d, answer_mark, question_mark


def masking_attn(layer_id, input_ids):
    viewing = torch.ones((input_ids.shape[0], input_ids.shape[0]))
    if layer_id < 39:
        return viewing
    a, b, c, d, answer_mark, question_mark = finding_pos(input_ids)
    viewing[answer_mark, question_mark[-1].item() + 1 : a] = 0
    viewing[answer_mark, a + 1 : b] = 0
    viewing[answer_mark, b + 1 : c] = 0
    viewing[answer_mark, c + 1 : d] = 0
    return viewing


class LoiReweightingFixedPositionAdapter(CustomPathOnlyAttentionAdapter):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(
            torch.ones(
                (kwargs["n_head"], 4), requires_grad=True, device=kwargs["device"]
            )
        )

    def _forward(self, layer_object, attn_weights):
        a, b, c, d, answer_mark, question_mark = finding_pos(self.input_ids[0])
        masking = torch.ones_like(attn_weights)
        for i in range(attn_weights.shape[1]):
            masking[0, i, answer_mark, a] = self.weight[i, 0]
            masking[0, i, answer_mark, b] = self.weight[i, 1]
            masking[0, i, answer_mark, c] = self.weight[i, 2]
            masking[0, i, answer_mark, d] = self.weight[i, 3]
        return attn_weights * masking


initialize_adapter = LoiReweightingFixedPositionAdapter

attentionermanger = LlamaAttentionerManager(
    modified_prompt_model.model,
    4,  # 4 class
    predictor=predictor,
    device=modified_prompt_model.device,
    kind_of_attention_adapter_initilizer=initialize_adapter,
    n_head=model_copy.model.layers[0].self_attn.num_heads,
)

# %%
all_labels = [a_label, b_label, c_label, d_label]

# %%
from icl.analysis.attentioner_for_train import MeasureGradOnlyAttentionAdapter
from icl.utils.random_utils import set_seed
from torch.optim import Adam, SGD
from tqdm import trange

# %%
def measure_effective(dataset, use_flag, data_size=10):
    for _ in attentionermanger.attention_adapters:
        _.use_flag = use_flag
    correct = 0
    pbar = trange(data_size)  # len(test_dataset_v2))
    for point in pbar:
        test_points = dict_to(
            quick_prep_input(dataset[point]), modified_prompt_model.device
        )
        results = modified_prompt_model(**test_points)
        correct += results["probs"].argmax() == test_dataset_inverted[point]["labels"]
        pbar.set_postfix_str(f"Acc: {correct/(point+1)*100:.2f}")

# %%
measure_effective(converted_test, False, 10)
