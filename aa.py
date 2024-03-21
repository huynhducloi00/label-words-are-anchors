import torch
import transformers
from transformers import AutoTokenizer

model_name = "meta-llama/llama-2-13b-chat-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name)
pipeline = transformers.pipeline(
    "text-generation",
    model=model_name,
    tokenizer=tokenizer,
    torch_dtype=torch.float16,
    device_map='auto',
)
sequences = pipeline(
    "Live birth is exemplified in snakes slithering out of eggs. True or False? ",
    num_return_sequences=1,
    temperature=0,
    top_p=1.0,
    top_k=0,
    eos_token_id=tokenizer.eos_token_id,
    max_new_tokens=15,
)
print(sequences)
