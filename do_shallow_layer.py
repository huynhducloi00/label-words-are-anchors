import os

LOI_DEVICE = 0
os.environ["CUDA_VISIBLE_DEVICES"] = f"{LOI_DEVICE}"

from icl.util_classes.arg_classes import ShallowArgs
from icl.analysis.shallow_layer import shallow_layer
from transformers.hf_argparser import HfArgumentParser

parser = HfArgumentParser((ShallowArgs,))
args, = parser.parse_args_into_dataclasses()
shallow_layer(args)