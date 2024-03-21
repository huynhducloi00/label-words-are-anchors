# Commented out IPython magic to ensure Python compatibility.
# %%capture
# %pip install accelerate peft bitsandbytes transformers trl

import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    pipeline,
    logging,
)
from peft import LoraConfig, PeftModel
from trl import SFTTrainer

"""## Model Config"""

# Model from Hugging Face hub
base_model = "huggyllama/llama-7b"#"meta-llama/Llama-2-7b-hf"

# New instruction dataset
guanaco_dataset = "mlabonne/guanaco-llama2-1k"

dataset = load_dataset(guanaco_dataset, split="train")

compute_dtype = getattr(torch, "float16")

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=False,
)

# Load base model
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    quantization_config=quant_config,
    device_map='auto'#{"": 0}
)
model.config.use_cache = False
model.config.pretraining_tp = 1

# Load LLaMA tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# Load LoRA configuration
peft_args = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=64,
    bias="none",
    task_type="CAUSAL_LM",
)

# Set training parameters
training_params = TrainingArguments(
    output_dir="./results",
    num_train_epochs=1,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=1,
    optim="paged_adamw_32bit",
    save_steps=25,
    logging_steps=25,
    learning_rate=2e-4,
    weight_decay=0.001,
    fp16=False,
    bf16=False,
    max_grad_norm=0.3,
    max_steps=-1,
    warmup_ratio=0.03,
    group_by_length=True,
    lr_scheduler_type="constant",
)

# Set supervised fine-tuning parameters
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_args,
    dataset_text_field="text",
    max_seq_length=None,
    tokenizer=tokenizer,
    args=training_params,
    packing=False,
)

# Train model
trainer.train()

# # Save trained model
# trainer.model.save_pretrained(new_model)

# !kill 26235

# from tensorboard import notebook
# log_dir = "results/runs"
# notebook.start("--logdir {} --port 4000".format(log_dir))

# # Ignore warnings
# logging.set_verbosity(logging.CRITICAL)

# # Run text generation pipeline with our next model
# prompt = "Who is Leonardo Da Vinci?"
# pipe = pipeline(task="text-generation", model=new_model, tokenizer=new_model, max_length=200)
# result = pipe(f"<s>[INST] {prompt} [/INST]")
# print(result[0]['generated_text'])

# prompt = "What is Datacamp Career track?"
# result = pipe(f"<s>[INST] {prompt} [/INST]")
# print(result[0]['generated_text'])

# !huggingface-cli login

# # Reload model in FP16 and merge it with LoRA weights
# load_model = AutoModelForCausalLM.from_pretrained(
#     base_model,
#     low_cpu_mem_usage=True,
#     return_dict=True,
#     torch_dtype=torch.float16,
#     device_map={"": 0},
# )

# model = PeftModel.from_pretrained(load_model, new_model)
# model = model.merge_and_unload()

# # Reload tokenizer to save it
# tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
# tokenizer.add_special_tokens({'pad_token': '[PAD]'})
# tokenizer.pad_token = tokenizer.eos_token
# tokenizer.padding_side = "right"

# model.push_to_hub(new_model, use_temp_dir=False)
# tokenizer.push_to_hub(new_model, use_temp_dir=False)