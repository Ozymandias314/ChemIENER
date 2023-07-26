import os
import argparse
from typing import List
import torch
import numpy as np

from model import build_model

from dataset import NERDataset, get_collate_fn

from huggingface_hub import hf_hub_download

class ChemNER:

    def __init__(self, model_path, device = None):

        args = self._get_args()

        states = torch.load(model_path, map_location = torch.device('cpu'))

        if device is None:
            device = torch.device('cpu')

        self.device = device

        self.model = self.get_model(args)

        self.collate = get_collate_fn()

        self.dataset = NERDataset(args)

        if self.args.corpus == "chemu":
            self.class_to_index = {'B-EXAMPLE_LABEL': 1, 'B-REACTION_PRODUCT': 2, 'B-STARTING_MATERIAL': 3, 'B-REAGENT_CATALYST': 4, 'B-SOLVENT': 5, 'B-OTHER_COMPOUND': 6, 'B-TIME': 7, 'B-TEMPERATURE': 8, 'B-YIELD_OTHER': 9, 'B-YIELD_PERCENT': 10, 'O': 0,
                 'I-EXAMPLE_LABEL': 11, 'I-REACTION_PRODUCT': 12, 'I-STARTING_MATERIAL': 13, 'I-REAGENT_CATALYST': 14, 'I-SOLVENT': 15, 'I-OTHER_COMPOUND': 16, 'I-TIME': 17, 'I-TEMPERATURE': 18, 'I-YIELD_OTHER': 19, 'I-YIELD_PERCENT': 20}
        elif self.args.corpus == "chemdner":
            self.class_to_index = self.class_to_index = {'O': 0, 'B-ABBREVIATION': 1, 'B-FAMILY': 2,  'B-FORMULA': 3, 'B-IDENTIFIER': 4, 'B-MULTIPLE': 5, 'B-SYSTEMATIC': 6, 'B-TRIVIAL': 7, 'B-NO CLASS': 8, 'I-ABBREVIATION': 9, 'I-FAMILY': 10,  'I-FORMULA': 11, 'I-IDENTIFIER': 12, 'I-MULTIPLE': 13, 'I-SYSTEMATIC': 14, 'I-TRIVIAL': 15, 'I-NO CLASS': 16}

        self.index_to_class = {self.class_to_index[key]: key for key in self.class_to_index}

    def _get_args(self):
        parser = argparse.ArgumentParser()

        parser.add_argument('roberta_checkpoint', default = 'roberta-base', type=str, help='which roberta config to use')

        parser.add_argument('corpus', default = "chemu", type=str, help="which corpus should the tags be from")

    def get_model(self, args, device, model_states):
        model = build_model(args)

        model.load_state_dict(model_states, strict = False)

        model.to(device)

        model.eval()

        return model

    def predict_strings(self, strings: List, batch_size = 8):
        device = self.device

        predictions = []

        output = {"sentences": [], "predictions": []}
        for idx in range(0, len(strings), batch_size):
            batch_strings = strings[idx:idx+batch_size]
            batch_strings_tokenized = [(self.dataset.tokenizer(s, truncation = True, max_length = 512), None ) for s in batch_strings]

            sentences, masks, refs = self.collate(batch_strings_tokenized)

            predictions = self.model(input_ids = sentences, attention_mask = masks)[0].argmax(dim = 2).to('cpu')

            sentences_list = list(sentences)

            refs_list = list(refs)

            predictions_list = list(predictions)

            output["sentences"]+=[ [self.tokenizer.decode(int(word.item())) for (word, label) in zip(sentence_w, sentence_l) if label!= -100] for (sentence_w, sentence_l) in zip(sentences_list, labels_list)],
            output["predictions"]+=[[self.index_to_class[int(pred.item())] for (pred, label) in zip(sentence_p, sentence_l) if label!=-100] for (sentence_p, sentence_l) in zip(predictions_list, labels_list)]
        
        return output



            

