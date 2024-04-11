import argparse
import torch
from tqdm import trange
from transformers import T5Tokenizer, T5ForConditionalGeneration, Adafactor
from functools import wraps, partial
from torch.nn.modules.sparse import Embedding
from torch.optim import Adam, SGD
import torch.nn as nn
parser = argparse.ArgumentParser(
                    prog='ProgramName',
                    description='What the program does',
                    epilog='Text at the bottom of help')
parser.add_argument('model_name')  
args = parser.parse_args()
model_name =args.model_name #"allenai/unifiedqa-v2-t5-3b-1363200"
# model_name = (
#     "allenai/unifiedqa-v2-t5-large-1363200"  # you can specify the model size here
# )
tokenizer = T5Tokenizer.from_pretrained(model_name)
model_original = T5ForConditionalGeneration.from_pretrained(
    model_name, device_map='auto')
print('successfull!')
