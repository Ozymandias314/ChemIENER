import torch
from torch import nn

from TorchCRF import CRF

from transformers import BertForTokenClassification, RobertaForTokenClassification


'''
class RobertaCRF(nn.Module):

    def __init__(self, args):
        self.roberta = RobertaForTokenClassification.from_pretrained(args.roberta_checkpoint, num_labels = 21, cache_dir = '/Mounts/rbg-storage1/users/urop/vincentf/.local/bin', return_dict = False, max_position_embeddings = 2048, ignore_mismatched_sizes = True)
        self.crf = CRF(num_tags = 21, batch)
'''

def build_model(args):
    #if args.crf:


    if args.corpus == "chemu":
        return RobertaForTokenClassification.from_pretrained(args.roberta_checkpoint, num_labels = 21, cache_dir = '/Mounts/rbg-storage1/users/urop/vincentf/.local/bin', return_dict = False)
    elif args.corpus == "chemdner":
        return RobertaForTokenClassification.from_pretrained(args.roberta_checkpoint, num_labels = 17, cache_dir = '/Mounts/rbg-storage1/users/urop/vincentf/.local/bin', return_dict = False) 
