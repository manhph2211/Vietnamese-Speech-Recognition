import os
import json
import time
import argparse
from tqdm import tqdm
from collections import defaultdict
import math
import argparse
import importlib

from datasets.dataset import get_dataset
from datasets.collate import collate_fn, gpu_collate, no_pad_collate
from datasets.transforms import (
        Compose, AddLengths, AudioSqueeze, TextPreprocess,
        MaskSpectrogram, ToNumpy, BPEtexts, MelSpectrogram,
        ToGpu, Pad, NormalizedMelSpectrogram
)
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset, ConcatDataset
# from tensorboardX import SummaryWriter
import numpy as np

from functools import partial

# model:
from models.quartznet.model import QuartzNet
from models.quartznet.base.decoder import GreedyDecoder, BeamCTCDecoder

# utils:
import yaml
import wandb
from easydict import EasyDict as edict
from utils.utils import fix_seeds, remove_from_dict, prepare_bpe
from utils.utils import fix_seeds, remove_from_dict, prepare_bpe


def evaluate(config):
    fix_seeds(seed=config.train.get('seed', 42))
    bpe = prepare_bpe(config)

    transforms_val = Compose([
            TextPreprocess(),
            ToNumpy(),
            BPEtexts(bpe=bpe),
            AudioSqueeze()
    ])

    batch_transforms_val = Compose([
            ToGpu('cuda' if torch.cuda.is_available() else 'cpu'),
            NormalizedMelSpectrogram(
                sample_rate=config.dataset.get('sample_rate', 16000), # for LJspeech
                n_mels=config.model.feat_in,
                normalize=config.dataset.get('normalize', None)
            ).to('cuda' if torch.cuda.is_available() else 'cpu'),
            AddLengths(),
            Pad()
    ])

    val_dataset = get_dataset(config, transforms=transforms_val, part='val')
    val_dataloader = DataLoader(val_dataset, num_workers=config.train.get('num_workers', 4),
                batch_size=1, collate_fn=no_pad_collate)

    model = QuartzNet(**remove_from_dict(config.model, ['name']))

    if config.train.get('from_checkpoint', None) is not None:
        print("LOADING CHECKPOINT...")
        model.load_weights(config.train.from_checkpoint)

    if torch.cuda.is_available():
        model = model.cuda()

    criterion = nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True)
    decoder = GreedyDecoder(bpe=bpe)

    model.eval()
    val_stats = defaultdict(list)
    for batch_idx, batch in enumerate(val_dataloader):
        batch = batch_transforms_val(batch)
        with torch.no_grad():
            logits = model(batch['audio'])
            output_length = torch.ceil(batch['input_lengths'].float() / model.stride).int()
            loss = criterion(logits.permute(2, 0, 1).log_softmax(dim=2), batch['text'], output_length, batch['target_lengths'])

        target_strings = decoder.convert_to_strings(batch['text'])
        decoded_output = decoder.decode(logits.permute(0, 2, 1).softmax(dim=2))
        wer = np.mean([decoder.wer(true, pred) for true, pred in zip(target_strings, decoded_output)])
        cer = np.mean([decoder.cer(true, pred) for true, pred in zip(target_strings, decoded_output)])
        val_stats['val_loss'].append(loss.item())
        val_stats['wer'].append(wer)
        val_stats['cer'].append(cer)
    for k, v in val_stats.items():
        val_stats[k] = np.mean(v)
    val_stats['val_samples'] = wandb.Table(columns=['gt_text', 'pred_text'], data=list(zip(target_strings, decoded_output)))
    print(val_stats)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluation model.')
    parser.add_argument('--config', default='configs/config.yml',
                        help='path to config file')
    args = parser.parse_args()
    with open(args.config, 'r') as f:
        config = edict(yaml.safe_load(f))
    evaluate(config)