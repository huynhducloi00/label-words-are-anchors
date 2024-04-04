from torch.optim import Adam, SGD
from transformers import (
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

model_name = "allenai/unifiedqa-t5-large"  # you can specify the model size here
tokenizer = T5Tokenizer.from_pretrained(model_name)


model = T5ForConditionalGeneration.from_pretrained(model_name, device_map="auto")


# model = None
def run_model(input_string, labels):
    input_ids = tokenizer.encode(input_string, return_tensors="pt")
    labels = tokenizer.encode(labels, return_tensors="pt")
    # print(torch.argwhere(input_ids[0]==2)[0,0]+2)
    res = model(
        input_ids=input_ids.to(0), labels=labels
    )
    return res


from icl.utils.data_wrapper import prepare_dataset

from icl.utils.load_huggingface_dataset import load_huggingface_dataset_train_and_test

dataset = load_huggingface_dataset_train_and_test("obqa")

QUESTION_PAD = 45
ANSWER_PAD = 20
# def permuted(input_sample, index):
#     perm = np.random.permutation(4)
#     choices = input_sample["choices"]["text"]
#     inputs = f"""Question: {input_sample['question_stem']}? {choices[perm[0]]} A; {choices[perm[1]]} B; {choices[perm[2]]} C; {choices[perm[3]]} D; Answer:"""
#     # f"Question: {input_sample['question_stem']}? A. {choices[perm[0]]} B. {choices[perm[1]]} C. {choices[perm[2]]} D. {choices[perm[3]]} Answer:"
#     return inputs, perm, np.argwhere(np.array(perm) == [input_sample["label"]])[0, 0]


# def volume(num_question, num_rep):
#     return sum([[i] * num_rep for i in range(num_question)], [])


import itertools


perm_order = list(itertools.permutations([0, 1, 2, 3], 4))


import textwrap
import numpy as np
from tqdm import trange


def get_prompt(index, perm):
    json_line = dataset["test"][index]
    question = json_line["question_stem"]
    choices = json_line["choices"]
    choice_texts = choices["text"]
    choice_texts = [choice_texts[perm[i]] for i in range(len(choice_texts))]

    def change(text, leng=20, is_question=False):
        pad_x = [0] * leng
        encoded = tokenizer.encode(text)[:-1]
        if len(encoded) > leng:
            print("too long ", len(encoded), leng, is_question)
        pad_x[: len(encoded)] = encoded
        return tokenizer.decode(pad_x)

    candidates = " ".join(
        [
            change(f"( ) {text}", ANSWER_PAD)
            for text, label in zip(choice_texts, choices["label"])
        ]
    ).replace("\n", " ")
    # print(json_line)
    answer_key = json_line["answerKey"]
    answer_key_idx = ord(answer_key[0]) - ord("A")
    # answer_text = choice_texts[answer_key_idx]
    # id = "OBQA_" + json_line['id']
    question_pad = f"{question} \\n "
    prompt = f"{change(question_pad,QUESTION_PAD, is_question=True)}{candidates}"
    MAP = {"A": 0, "B": 1, "C": 2, "D": 3}
    return prompt, choice_texts[MAP[json_line["answerKey"]]], choice_texts


import pandas as pd


def measure_unalike(arr, print_arr=False):
    n = len(arr)
    arr = pd.Series(arr).value_counts()
    if print_arr:
        print(arr)
    return 1 - ((arr / n) ** 2).sum()


question_to_do = 5
per_question = 20

for i in range(question_to_do):
    choice0 = None
    answer = []
    for k in trange(per_question):
        prompt, labels, choice = get_prompt(i, perm_order[k % 24])
        if choice0 is None:
            choice0 = choice
        # print(perm, textwrap.fill(prompt))
        answer.append(run_model(prompt, labels)[0])
    print("question ", i, choice0, measure_unalike(answer, print_arr=True))

# trainer = Seq2SeqTrainer(
#     model_init=model_init,
#     args=args,
#     train_dataset=tokenized_datasets["train"],
#     eval_dataset=tokenized_datasets["validation"],
#     data_collator=data_collator,
#     tokenizer=tokenizer,
#     compute_metrics=compute_metrics,
# )
# trainer.train()

position_weight_manger = Seq2SeqPositionWeightManager(
    model,
    device=model.device,
    kind_of_weight_adapter_initilizer=initialize_adapter,
    n_position_dim=16, #model_copy.model.layers[0].self_attn.num_heads,
)
params = position_weight_manger.params()  # list(model.parameters()) +
optimizer = SGD(params, lr=1e-1)  # args.lr)
for epoch in range(2):
    for i in range(question_to_do):
        for k in trange(per_question):
            prompt, labels, choice = get_prompt(i, perm_order[k % 24])
            # print(perm, textwrap.fill(prompt))
            res=run_model(prompt, labels)
            res.loss.backward()
    # tokenizer.batch_decode(res, skip_special_tokens=True)