from torch.utils import data
import torchaudio
import os
from torch.utils.data import Subset


class LJSpeechDataset(torchaudio.datasets.LJSPEECH):
    def __init__(self, transforms, *args, **kwargs):
        # if kwargs.get('download', False):
        #     os.makedirs(kwargs['root'], exist_ok=True)
        super(LJSpeechDataset, self).__init__(*args, **kwargs)
        self.transforms = transforms

    def __getitem__(self, idx):
        audio, sample_rate, _, norm_text = super().__getitem__(idx)
        return self.transforms({'audio' : audio, 'text': norm_text.upper(), 'sample_rate': sample_rate})

    def get_text(self, n):
        line = self._flist[n]
        fileid, transcript, normalized_transcript = line
        return self.transforms({'text' : normalized_transcript.upper()})['text']


def get_dataset(config, transforms=lambda x: x, part='train'):
    if part == 'train':
        dataset = LJSpeechDataset(root=config.dataset.root, download=False, transforms=transforms)
        indices = list(range(len(dataset)))
        dataset = Subset(dataset, indices[:int(config.dataset.get('train_part', 0.95) * len(dataset))])
        return dataset
    elif part == 'val':
        dataset = LJSpeechDataset(root=config.dataset.root, download=False, transforms=transforms)
        indices = list(range(len(dataset)))
        dataset = Subset(dataset, indices[int(config.dataset.get('train_part', 0.95) * len(dataset)):])
        return dataset
    elif part == 'bpe':
        dataset = LJSpeechDataset(root=config.dataset.root, download=False, transforms=transforms)
        indices = list(range(len(dataset)))[:int(config.dataset.get('train_part', 0.95) * len(dataset))]
        return dataset, indices
    else:
        raise ValueError('Unknown')