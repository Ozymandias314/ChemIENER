import os
import cv2
import copy
import random
import json
import contextlib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence

from transformers import BertTokenizerFast



class NERDataset(Dataset):
    def __init__(self, args, data_file, split='train'):
        super().__init__()
        self.args = args
        data_path = os.path.join(args.data_path, data_file)
        with open(data_path) as f:
            self.data = json.load(f)
        self.split = split
        self.is_train = (split == 'train')
        self.tokenizer = BertTokenizerFast.from_pretrained('allenai/scibert_scivocab_uncased')
        self.class_to_index = {'EXAMPLE_LABEL': 1, 'REACTION_PRODUCT': 2, 'STARTING_MATERIAL': 3, 'REAGENT_CATALYST': 4, 'SOLVENT': 5, 'OTHER_COMPOUND': 6, 'TIME': 7, 'TEMPERATURE': 8, 'YIELD_OTHER': 9, 'YIELD_PERCENT': 10, 'NONE': 0}
        self.index_to_class = {class_to_index[key]: key for key in class_to_index}


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text_tokenized = self.tokenizer(self.data[idx]['text']) 
        return text_tokenized, self.align_labels(text_tokenized, self.data[idx]['entities'])

    def align_labels(self, text_tokenized, entities):
        char_to_class = {}

        for entity in entities: 
            for span in entities[entity].span:
                for i in range(span[0], span[1]):
                    char_to_class[i] = class_to_index[entities[entity].type]
            for i in range(len(file_content)):
                if i not in char_to_class:
                    char_to_class[i] = 0
        
        for i in range(len(text_tokenized[0])):
            span = text_tokenized.token_to_chars(i)
            if span is not None:
                classes.append(char_to_class[span.start])
            else:
                classes.append(-100)

        return torch.Tensor(classes)


