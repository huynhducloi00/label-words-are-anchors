# %%
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt

# %%
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

model_original = load_model_customize(neglect_args)
model_copy=copy.deepcopy(model_original)
label_id_dict = get_label_id_dict_for_args(neglect_args, tokenizer)


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


def sen_gen(input_sample):
    choices = input_sample["choices"]["text"]
    # for vicinity
    # inputs = f"Question: {input_sample['question_stem']}: A. {choices[0]} B. {choices[1]} C. {choices[2]} D. {choices[3]} Answer:"
    inputs_1 = f"""Question: {input_sample['question_stem']}? "{choices[0]}" is A; "{choices[1]}" is B; "{choices[2]}" is C; "{choices[3]}" is D; Answer:"""
    # inputs_2 = f"""Question: {input_sample['question_stem']}: "{choices[0]}" is True or False: ;"{choices[1]}" is True or False: ;"{choices[2]}" is True or False: ;"{choices[3]}" is True or False: ; Answer: """
    # inputs = f"Question: {input_sample['question_stem']}:\n'{choices[0]}' is A.\n'{choices[1]}' is B.\n'{choices[2]}' is C.\n'{choices[3]}' is D.\nAnswer:"
    # inputs = f"Question: {input_sample['question_stem']}\n A. {choices[0]}\n B. {choices[1]}\n C. {choices[2]}\n D. {choices[3]}\nAnswer:"
    # 27.6 inputs = f"Question: {input_sample['question_stem']}\n A. {choices[0]}\n B. {choices[1]}\n C. {choices[2]}\n D. {choices[3]}\nSelect either A, B, C, or D:"
    # 27.6 inputs = f"Question: 1+1=\n A. 0 B. 1 C. 2 D. 3. Answer: C. Question: {input_sample['question_stem']}\n A. {choices[0]}\n B. {choices[1]}\n C. {choices[2]}\n D. {choices[3]}\n Answer:"
    return inputs_1


test_dataset = prepare_dataset(
    42, dataset["test"], 10, neglect_args, tokenizer, sen_gen
)

# %%
test_dataset[0]["sentence"]

# %%
modified_prompt_model = LMForwardAPI(
    model=model_copy,
    model_name=neglect_args.model_name,
    tokenizer=tokenizer,
    label_id_dict=label_id_dict,
    output_attention=True,
)

# %%
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
# for point in range(10):
#     test_points = dict_to(quick_prep_input(test_dataset[point]), modified_prompt_model.device)
#     results = modified_prompt_model(**test_points)
#     print(
#         f"{results['probs'].detach().cpu()}: {results['probs'].argmax()} - {test_dataset[point]['labels']}: {results['probs'].argmax() == test_dataset[point]['labels']}"
#     )

# %%
a_label, b_label, c_label, d_label = (319, 350, 315, 360)
answer_label = tokenizer.convert_tokens_to_ids(":")
question_label = tokenizer.convert_tokens_to_ids("?")


def masking_attn(input_ids):
    viewing = np.ones((input_ids.shape[0], input_ids.shape[0]))
    a = np.argwhere(input_ids == a_label)[-1].item()
    b = np.argwhere(input_ids == b_label)[-1].item()
    c = np.argwhere(input_ids == c_label)[-1].item()
    d = np.argwhere(input_ids == d_label)[-1].item()
    answer_mark = np.argwhere(input_ids == answer_label)[-1].item()
    question_mark = np.argwhere(input_ids == question_label)[-1].item()
    viewing[answer_mark, question_mark + 1 : a] = 0
    viewing[answer_mark, a + 1 : b] = 0
    viewing[answer_mark, b + 1 : c] = 0
    viewing[answer_mark, c + 1 : d] = 0
    return viewing


# viewing=masking_attn(input_ids)
input_ids = np.array(test_dataset[0]["input_ids"])
tokens = tokenizer.convert_ids_to_tokens(test_dataset[0]["input_ids"])
df_cm = pd.DataFrame(masking_attn(input_ids), index=tokens, columns=tokens)
sn.set_theme(rc={"figure.figsize": (20, 1)})
# sn.heatmap(df_cm.iloc[answer_mark : answer_mark + 1, :], annot=False, cmap="Blues")

# %%
df_cm.shape

# %%
np.ones((input_ids.shape[0], input_ids.shape[0]))

# %%
question_mark = "?"
dot = 29936

# %%
from icl.analysis.attentioner_for_train import CustomPathOnlyAttentionAdapter


class LoiExpAdapter(CustomPathOnlyAttentionAdapter):
    def _forward(self, attn_weights):
        if self.params is None:
            self.params = torch.ones_like(attn_weights, requires_grad=True)
        else:
            self.params.data = torch.ones_like(attn_weights)
        return attn_weights * masking_attn(attn_weights) * self.params

# %%
initialize_adapter = LoiExpAdapter
attentionermanger = LlamaAttentionerManager(
    modified_prompt_model.model,
    4,  # 4 class
    predictor=predictor,
    device=modified_prompt_model.device,
    kind_of_attention_adapter_initilizer=initialize_adapter,
    n_head=model_copy.model.layers[0].self_attn.num_heads,
)

# %%
from random_utils import set_seed


pros = []
attentionermanger.zero_grad(set_to_none=True)
set_seed(seed)

for point in range(1):
    data = dict_to(quick_prep_input(test_dataset[point]), modified_prompt_model.device)
    output = modified_prompt_model(**data)
    label = data["labels"]
    percent_of_correct_choice = output["probs"][0][label.item()]
    loss = -torch.log(percent_of_correct_choice)
    loss.backward()
    class_poss, final_poss, answer_pos = predictor.get_pos(
        {"input_ids": data["input_ids"]}
    )
    pros = []
    grad_at_criticals_at_layers = torch.cat(
        [
            attentionermanger.grad(use_abs=False)[i].cpu()
            for i in range(len(attentionermanger.attention_adapters))
        ]
    )
    attentionermanger.zero_grad(set_to_none=False)
    pro = {
        "grads": grad_at_criticals_at_layers.cpu(),
        "class_pos": class_poss,
        "final_pos": final_poss.cpu(),
        "question": test_dataset[point]["sentence"],
        "tokens": tokenizer.convert_ids_to_tokens(data["input_ids"][0]),
        "percentage": output["probs"][0].detach().cpu(),
        "correct_choice": label.item(),
        "attentions": torch.cat(
            [x.detach().cpu() for x in output["results"]["attentions"]]
        ).mean(dim=1),
    }
    pros.append(pro)

# %%
print(len(pros[0]["tokens"]), pros[0]["final_pos"])
print(pros[0]["percentage"])

# %%
import numpy as np

show_question(pros, 0, -1, True, show_attention=False, start_index=1)

# %%
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt

cmap = LinearSegmentedColormap.from_list("", ["red", "white", "blue"])


def get_df(
    bundle, question_id, layer_id, show_info=False, show_attention=False, start_index=0
):
    ques = bundle[question_id]
    viewing = (
        ques["attentions"][layer_id] if show_attention else -ques["grads"][layer_id]
    )
    ss = ques["final_pos"] + 1
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
            cmap=LinearSegmentedColormap.from_list("", ["blue", "white", "red"]),
        )
    # return viewing


show_question(pros, 0, -1, True, show_attention=False, start_index=1)

# %%
# show_question(results_neglect, 1, -2, True)
show_question(results_original, 1, -2, True)

# %%
kk.index

# %%
# show_question(results_neglect, 1, -1, True)
kk = get_df(results_original, 1, -1, show_attention=True, start_index=1)
# show_question(results_original, 1,-1,False, show_attention= True, start_index=1)
type(kk["▁desert"])
kk["▁desert"].sort_values(ascending=False)[:30]

# %%
get_df(results_original, 1, -1, show_attention=False)["▁desert"].sort_values(
    key=lambda x: abs(x), ascending=False
)

# %%
kk["▁green"].sort_values(ascending=False)[:20]

# %%
def plot_ndarray(data, args):
    names = [r"$S_{wp}$", r"$S_{pq}$", r"$S_{ww}$"]
    type_num = len(data)

    fig, ax = plt.subplots()

    for i in range(type_num):
        ax.plot(data[i], label=names[i])

    ax.legend()
    ax.set_ylabel("S")
    ax.set_xlabel("Layer")

    fig.savefig(f"attn_attr_{args.task_name}_{args.demonstration_shot}.pdf")
    plt.show()

# %%
task = "obqa"  # 'agnews'
demonstration_shot = 0
proportions_list = []
seed = 42
ana_args = AttrArgs(
    version="original",
    task_name=task,
    sample_size=1000,
    seeds=[seed],
    demonstration_shot=demonstration_shot,
)
print(ana_args.save_file_name)
results_original = ana_args.load_result()

# %%
show_question(results_original, 1, -1, True)

# %%
show_question(results_original, 1, -1, True)

# %%
[
    (
        i,
        results_original[i]["percentage"].argmax()
        == results_original[i]["correct_choice"],
    )
    for i in range(10)
]
# sum([results_original[i]['percentage'].argmax()==results_original[i]['correct_choice'] for i in range(10)])

# %%
results_original[4]["question"]
results_original[4]["percentage"]

# %%
results_neglect = neglect_args.load_result()

# %%
def measure_correctness(dataset):
    [
        print(
            i,
            dataset[i]["percentage"].argmax() == dataset[i]["correct_choice"],
        )
        for i in range(len(dataset))
    ]
    return sum(
        [
            dataset[i]["percentage"].argmax() == dataset[i]["correct_choice"]
            for i in range(len(dataset))
        ]
    )

# %%
measure_correctness(results_neglect)

# %%
sum(
    [
        results_original[i]["percentage"].argmax()
        == results_original[i]["correct_choice"]
        for i in range(10)
    ]
)

# %%
print(results_original[7]["question"], results_original[7]["percentage"])
print(results_neglect[7]["percentage"])

# %%
def get_proportion(grads_one_layer, class_poss, final_poss):
    grads_one_layer = grads_one_layer.detach().clone().cpu()
    class_poss = torch.hstack(class_poss).detach().clone().cpu()
    final_poss = final_poss.detach().clone().cpu()
    grads_one_layer = grads_one_layer.numpy()
    np.fill_diagonal(grads_one_layer, 0)
    proportion1 = grads_one_layer[class_poss, :].sum()
    proportion2 = grads_one_layer[final_poss, class_poss].sum()
    proportion3 = grads_one_layer.sum() - proportion1 - proportion2

    N = int(final_poss)
    sum3 = (N + 1) * N / 2 - sum(class_poss) - len(class_poss)
    proportion1 = proportion1 / sum(class_poss)
    proportion2 = proportion2 / len(class_poss)
    proportion3 = proportion3 / sum3
    proportions = np.array([proportion1, proportion2, proportion3])
    return proportions

# %%
plot_thu = [
    get_proportion(grads[0][layer], results[0]["class_pos"], results[0]["final_pos"])[1]
    for layer in range(40)
]

# %%
plt.plot(plot_thu)

# %%
cmap = LinearSegmentedColormap.from_list("", ["red", "white", "blue"])


def get_df(bundle, question_id, layer_id, show_attention=False, start_index=0):
    ques = bundle[question_id]
    viewing = (
        ques["attentions"][layer_id] if show_attention else -ques["grads"][layer_id]
    )
    ss = ques["final_pos"] + 1
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

    plt.figure(figsize=(15, 15))
    df_cm = get_df(bundle, question_id, layer_id, show_attention, start_index)
    if show_attention:
        sn.heatmap(df_cm, annot=False, cmap="Blues")
    else:
        scale_max = abs(np.max(df_cm.values))
        sn.heatmap(
            df_cm,
            annot=False,
            vmin=-scale_max,
            vmax=scale_max,
            cmap=LinearSegmentedColormap.from_list("", ["blue", "white", "red"]),
        )
    # return viewing


show_question(results_original, 1, -1, True, show_attention=False, start_index=1)

# %%
viewing[[np.arange(viewing.shape[0])] * 2] = 0
viewing

# %%


# %%
show_question(results_neglect, 1, -1, False, show_attention=True, start_index=1)

# %%
show_question(results_neglect, 1, -6)

# %%
results_neglect[7]["class_pos"]

# %%
results_neglect[7]["attentions"][0][26].sum()

# %%
show_question(results_neglect, 1, -1, show_attention=True, start_index=1)

# %%
show_question(results_original, 7, 0)

# %%
results[layer_id]["class_pos"]

# %%
tokens[31]


