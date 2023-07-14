import os
import math
import json
import random
import argparse
import numpy as np

import time

import torch
from torch.profiler import profile, record_function, ProfilerActivity
import torch.distributed as dist
import pytorch_lightning as pl
from pytorch_lightning import LightningModule, LightningDataModule
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.strategies.ddp import DDPStrategy
from transformers import get_scheduler

from dataset import NERDataset

#from rxnscribe.model import Encoder, Decoder
#from rxnscribe.pix2seq import build_pix2seq_model
#from rxnscribe.loss import Criterion
#from rxnscribe.tokenizer import get_tokenizer
#from rxnscribe.dataset import ReactionDataset, get_collate_fn
#from rxnscribe.data import postprocess_reactions, postprocess_bboxes
#from rxnscribe.evaluate import CocoEvaluator, ReactionEvaluator, CorefEvaluator
#import rxnscribe.utils as utils
#from rxnscribe.pix2seq.transformer import build_transformer


def get_args(notebook=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--do_train', action='store_true')
    parser.add_argument('--do_valid', action='store_true')
    parser.add_argument('--do_test', action='store_true')
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--print_freq', type=int, default=200)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--no_eval', action='store_true')
    # Model
    parser.add_argument('--encoder', type=str, default='resnet34')
    parser.add_argument('--decoder', type=str, default='lstm')
    parser.add_argument('--trunc_encoder', action='store_true')  # use the hidden states before downsample
    parser.add_argument('--no_pretrained', action='store_true')
    parser.add_argument('--use_checkpoint', action='store_true')
    parser.add_argument('--lstm_dropout', type=float, default=0.5)
    parser.add_argument('--embed_dim', type=int, default=256)
    parser.add_argument('--enc_pos_emb', action='store_true')
    group = parser.add_argument_group("lstm_options")
    group.add_argument('--decoder_dim', type=int, default=512)
    group.add_argument('--decoder_layer', type=int, default=1)
    group.add_argument('--attention_dim', type=int, default=256)
    group = parser.add_argument_group("transformer_options")
    group.add_argument("--dec_num_layers", help="No. of layers in transformer decoder", type=int, default=6)
    group.add_argument("--dec_hidden_size", help="Decoder hidden size", type=int, default=256)
    group.add_argument("--dec_attn_heads", help="Decoder no. of attention heads", type=int, default=8)
    group.add_argument("--dec_num_queries", type=int, default=128)
    group.add_argument("--hidden_dropout", help="Hidden dropout", type=float, default=0.1)
    group.add_argument("--attn_dropout", help="Attention dropout", type=float, default=0.1)
    group.add_argument("--max_relative_positions", help="Max relative positions", type=int, default=0)

    parser.add_argument('--is_coco', action = 'store_true')
    
    # Pix2Seq
    parser.add_argument('--pix2seq', action='store_true', help="specify the model from playground")
    parser.add_argument('--pix2seq_ckpt', type=str, default=None)
    parser.add_argument('--large_scale_jitter', action='store_true', help='large scale jitter')
    parser.add_argument('--pred_eos', action='store_true', help='use eos token instead of predicting 100 objects')
    parser.add_argument('--use_hf_transformer', action='store_true', help='use huggingface transformer for transformer')
    parser.add_argument('--linear_head', action = 'store_true')
    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str, help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")
    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int, help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int, help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=1024, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float, help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--pre_norm', action='store_true')
    # Data
    parser.add_argument('--data_path', type=str, default=None)
    parser.add_argument('--image_path', type=str, default=None)
    parser.add_argument('--train_file', type=str, default=None)
    parser.add_argument('--valid_file', type=str, default=None)
    parser.add_argument('--test_file', type=str, default=None)
    parser.add_argument('--vocab_file', type=str, default=None)
    parser.add_argument('--format', type=str, default='reaction')
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--input_size', type=int, default=224)
    parser.add_argument('--augment', action='store_true')
    parser.add_argument('--composite_augment', action='store_true')
    parser.add_argument('--coord_bins', type=int, default=100)
    parser.add_argument('--sep_xy', action='store_true')
    parser.add_argument('--rand_order', action='store_true', help="randomly permute the sequence of input targets")
    parser.add_argument('--split_heuristic', action = 'store_true', help="make the sequence of tokens follow a heuristic")
    parser.add_argument('--add_noise', action='store_true')
    parser.add_argument('--mix_noise', action='store_true')
    parser.add_argument('--shuffle_bbox', action='store_true')
    parser.add_argument('--images', type=str, default='')
    # Training
    parser.add_argument('--epochs', type=int, default=8)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--max_grad_norm', type=float, default=5.)
    parser.add_argument('--scheduler', type=str, choices=['cosine', 'constant'], default='cosine')
    parser.add_argument('--warmup_ratio', type=float, default=0)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--load_path', type=str, default=None)
    parser.add_argument('--load_encoder_only', action='store_true')
    parser.add_argument('--train_steps_per_epoch', type=int, default=-1)
    parser.add_argument('--eval_per_epoch', type=int, default=10)
    parser.add_argument('--save_path', type=str, default='output/')
    parser.add_argument('--save_mode', type=str, default='best', choices=['best', 'all', 'last'])
    parser.add_argument('--load_ckpt', type=str, default='best')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--num_train_example', type=int, default=None)
    parser.add_argument('--label_smoothing', type=float, default=0.0)
    parser.add_argument('--save_image', action='store_true')
    # Inference
    parser.add_argument('--beam_size', type=int, default=1)
    parser.add_argument('--n_best', type=int, default=1)
    parser.add_argument('--molscribe', action='store_true')

    parser.add_argument('--punish_first', action='store_true')

    args = parser.parse_args([]) if notebook else parser.parse_args()

    args.images = args.images.split(',')



    return args


class NERDataModule(LightningDataModule):

    def __init__(self, args):
        super().__init__()
        self.args = args

    def prepare_data(self):
        args = self.args
        if args.do_train:
            self.train_dataset = NERDataset(args, args.train_file, split='train')
        if self.args.do_train or self.args.do_valid:
            self.val_dataset = NERDataset(args, args.valid_file, split='valid')
        
    def print_stats(self):
        if self.args.do_train:
            print(f'Train dataset: {len(self.train_dataset)}')
        if self.args.do_train or self.args.do_valid:
            print(f'Valid dataset: {len(self.val_dataset)}')

def main():

    args = get_args()

    pl.seed_everything(args.seed, workers = True)

    