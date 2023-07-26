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

from dataset import NERDataset, get_collate_fn

from model import build_model

from utils import get_class_to_index

import evaluate

from seqeval.metrics import accuracy_score
from seqeval.metrics import classification_report
from seqeval.metrics import f1_score
from seqeval.scheme import IOB2

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
    '''
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
    '''

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
    #parser.add_argument('--augment', action='store_true')
    #parser.add_argument('--composite_augment', action='store_true')
    #parser.add_argument('--coord_bins', type=int, default=100)
    #parser.add_argument('--sep_xy', action='store_true')
    #parser.add_argument('--rand_order', action='store_true', help="randomly permute the sequence of input targets")
    #parser.add_argument('--split_heuristic', action = 'store_true', help="make the sequence of tokens follow a heuristic")
    #parser.add_argument('--add_noise', action='store_true')
    #parser.add_argument('--mix_noise', action='store_true')
    #parser.add_argument('--shuffle_bbox', action='store_true')
    #parser.add_argument('--images', type=str, default='')
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
    #parser.add_argument('--label_smoothing', type=float, default=0.0)
    #parser.add_argument('--save_image', action='store_true')
    # Inference
    #parser.add_argument('--beam_size', type=int, default=1)
    #parser.add_argument('--n_best', type=int, default=1)
    #parser.add_argument('--molscribe', action='store_true')

    #parser.add_argument('--punish_first', action='store_true')

    parser.add_argument('--roberta_checkpoint', type=str, default = "roberta-base")

    parser.add_argument('--corpus', type=str, default = "chemu")

    parser.add_argument('--cache_dir')

    args = parser.parse_args([]) if notebook else parser.parse_args()

    args.images = args.images.split(',')





    return args


class ChemIENERecognizer(LightningModule):

    def __init__(self, args):
        super().__init__()

        self.args = args

        self.model = build_model(args)

        self.validation_step_outputs = []

    def training_step(self, batch, batch_idx):
        



        sentences, masks, refs = batch
        '''
        print("sentences " + str(sentences))
        print("sentence shape " + str(sentences.shape))
        print("masks " + str(masks))
        print("masks shape " + str(masks.shape))
        print("refs " + str(refs))
        print("refs shape " + str(refs.shape))
        '''

        

        loss, logits = self.model(input_ids=sentences, attention_mask=masks, labels=refs)
        self.log('train/loss', loss)
        self.log('lr', self.lr_schedulers().get_lr()[0], prog_bar=True, logger=False)
        return loss
    
    def validation_step(self, batch, batch_idx):
        
        sentences, masks, refs = batch
        '''
        print("sentences " + str(sentences))
        print("sentence shape " + str(sentences.shape))
        print("masks " + str(masks))
        print("masks shape " + str(masks.shape))
        print("refs " + str(refs))
        print("refs shape " + str(refs.shape))
        '''

        logits = self.model(input_ids = sentences, attention_mask=masks)[0]
        '''
        print("logits " + str(logits))
        print(sentences.shape)
        print(logits.shape)
        print(torch.eq(logits.argmax(dim = 2), refs).sum())
        '''
        self.validation_step_outputs.append((sentences.to("cpu"), logits.argmax(dim = 2).to("cpu"), refs.to('cpu')))


    def on_validation_epoch_end(self):
        if self.trainer.num_devices > 1:
            gathered_outputs = [None for i in range(self.trainer.num_devices)]
            dist.all_gather_object(gathered_outputs, self.validation_step_outputs)
            gathered_outputs = sum(gathered_outputs, [])
        else:
            gathered_outputs = self.validation_step_outputs
        
        sentences = [list(output[0]) for output in gathered_outputs]  

        class_to_index = get_class_to_index(self.args.corpus)



        index_to_class = {class_to_index[key]: key for key in class_to_index}
        predictions = [list(output[1]) for output in gathered_outputs] 
        labels = [list(output[2]) for output in gathered_outputs]


        output = {"sentences": [[int(word.item()) for (word, label) in zip(sentence_w, sentence_l) if label != -100] for (batched_w, batched_l) in zip(sentences, labels) for (sentence_w, sentence_l) in zip(batched_w, batched_l) ], 
                  "predictions": [[index_to_class[int(pred.item())] for (pred, label) in zip(sentence_p, sentence_l) if label!=-100] for (batched_p, batched_l) in zip(predictions, labels) for (sentence_p, sentence_l) in zip(batched_p, batched_l)  ],
                  "groundtruth": [[index_to_class[int(label.item())] for label in sentence if label != -100] for batched in labels for sentence in batched]}


        #true_labels = [str(label.item()) for batched in labels for sentence in batched for label in sentence if label != -100]
        #true_predictions = [str(pred.item()) for (batched_p, batched_l) in zip(predictions, labels) for (sentence_p, sentence_l) in zip(batched_p, batched_l) for (pred, label) in zip(sentence_p, sentence_l) if label!=-100 ]



        #print("true_label " + str(len(true_labels)) + " true_predictions "+str(len(true_predictions)))


        #predictions = utils.merge_predictions(gathered_outputs)
        name = self.eval_dataset.name
        scores = [0]

        #print(predictions)
        #print(predictions[0].shape)

        if self.trainer.is_global_zero:
            if not self.args.no_eval:
                epoch = self.trainer.current_epoch

                metric = evaluate.load("seqeval", cache_dir = self.args.cache_dir)

                all_metrics = metric.compute(predictions = output['predictions'], references = output['groundtruth'])

                #accuracy = sum([1 if p == l else 0 for (p, l) in zip(true_predictions, true_labels)])/len(true_labels)

                #precision = torch.eq(self.eval_dataset.data, predictions.argmax(dim = 1)).sum().float()/self.eval_dataset.data.numel()
                #self.print("Epoch: "+str(epoch)+" accuracy: "+str(accuracy))
                report = classification_report(output['groundtruth'], output['predictions'], mode = 'strict', scheme = IOB2, output_dict = True)
                self.print(report)
                scores = [report['weighted avg']['f1-score']]
            with open(os.path.join(self.trainer.default_root_dir, f'prediction_{name}.json'), 'w') as f:
                    json.dump(output, f)
        
        dist.broadcast_object_list(scores)

        self.log('val/score', scores[0], prog_bar=True, rank_zero_only=True)
        self.validation_step_outputs.clear()
        
        

        self.validation_step_outputs.clear()

    def configure_optimizers(self):
        num_training_steps = self.trainer.num_training_steps
        
        self.print(f'Num training steps: {num_training_steps}')
        num_warmup_steps = int(num_training_steps * self.args.warmup_ratio)
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
        scheduler = get_scheduler(self.args.scheduler, optimizer, num_warmup_steps, num_training_steps)
        return {'optimizer': optimizer, 'lr_scheduler': {'scheduler': scheduler, 'interval': 'step'}}

class NERDataModule(LightningDataModule):

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.collate_fn = get_collate_fn()

    def prepare_data(self):
        args = self.args
        if args.do_train:
            self.train_dataset = NERDataset(args, args.train_file, split='train')
        if self.args.do_train or self.args.do_valid:
            self.val_dataset = NERDataset(args, args.valid_file, split='valid')
        if self.args.do_test:
            self.test_dataset = NERDataset(args, args.test_file, split='valid')
        
    def print_stats(self):
        if self.args.do_train:
            print(f'Train dataset: {len(self.train_dataset)}')
        if self.args.do_train or self.args.do_valid:
            print(f'Valid dataset: {len(self.val_dataset)}')
        if self.args.do_test:
            print(f'Test dataset: {len(self.test_dataset)}')

    
    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset, batch_size=self.args.batch_size, num_workers=self.args.num_workers,
            collate_fn=self.collate_fn)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset, batch_size=self.args.batch_size, num_workers=self.args.num_workers,
            collate_fn=self.collate_fn)
    

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_dataset, batch_size=self.args.batch_size, num_workers=self.args.num_workers,
            collate_fn=self.collate_fn)



class ModelCheckpoint(pl.callbacks.ModelCheckpoint):
    def _get_metric_interpolated_filepath_name(self, monitor_candidates, trainer, del_filepath=None) -> str:
        filepath = self.format_checkpoint_name(monitor_candidates)
        return filepath

def main():

    args = get_args()

    pl.seed_everything(args.seed, workers = True)

    if args.do_train:
        model = ChemIENERecognizer(args)
    else:
        model = ChemIENERecognizer.load_from_checkpoint(os.path.join(args.save_path, 'checkpoints/best.ckpt'), strict=False,
                                        args=args)

    dm = NERDataModule(args)
    dm.prepare_data()
    dm.print_stats()

    checkpoint = ModelCheckpoint(monitor='val/score', mode='max', save_top_k=1, filename='best', save_last=True)
    # checkpoint = ModelCheckpoint(monitor=None, save_top_k=0, save_last=True)
    lr_monitor = LearningRateMonitor(logging_interval='step')
    logger = pl.loggers.TensorBoardLogger(args.save_path, name='', version='')

    trainer = pl.Trainer(
        strategy=DDPStrategy(find_unused_parameters=False),
        accelerator='gpu',
        precision = 16,
        devices=args.gpus,
        logger=logger,
        default_root_dir=args.save_path,
        callbacks=[checkpoint, lr_monitor],
        max_epochs=args.epochs,
        gradient_clip_val=args.max_grad_norm,
        accumulate_grad_batches=args.gradient_accumulation_steps,
        check_val_every_n_epoch=args.eval_per_epoch,
        log_every_n_steps=10,
        deterministic='warn')

    if args.do_train:
        trainer.num_training_steps = math.ceil(
            len(dm.train_dataset) / (args.batch_size * args.gpus * args.gradient_accumulation_steps)) * args.epochs
        model.eval_dataset = dm.val_dataset
        ckpt_path = os.path.join(args.save_path, 'checkpoints/last.ckpt') if args.resume else None
        trainer.fit(model, datamodule=dm, ckpt_path=ckpt_path)
        model = ChemIENERecognizer.load_from_checkpoint(checkpoint.best_model_path, args=args)

    if args.do_valid:

        model.eval_dataset = dm.val_dataset

        trainer.validate(model, datamodule=dm)

    if args.do_test:

        model.test_dataset = dm.test_dataset

        trainer.test(model, datamodule=dm)


if __name__ == "__main__":
    main()
    