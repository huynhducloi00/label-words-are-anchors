import os
LOI_DEVICE = 0
os.environ["CUDA_VISIBLE_DEVICES"] = f"{LOI_DEVICE}"
from icl.analysis.deep_layer import deep_layer, DeepArgs
from transformers.hf_argparser import HfArgumentParser

parser = HfArgumentParser((DeepArgs,))
args, = parser.parse_args_into_dataclasses()
deep_layer(args)