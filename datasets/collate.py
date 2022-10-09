import torch
import numpy as np
from torch.utils.data.dataloader import default_collate


def no_pad_collate(batch):
    keys = batch[0].keys()
    collated_batch = {key: [] for key in keys}
    for key in keys:
        items = [item[key] for item in batch]
        collated_batch[key] = items
    return collated_batch


def gpu_collate(batch):
    '''
    Padds batch of variable length
    note: it converts things ToTensor manually here since the ToTensor transform
    assume it takes in images rather than arbitrary tensors.
    '''
    keys = batch[0].keys()
    collated_batch = {key: [] for key in keys}
    for key in keys:
        items = [item[key] for item in batch]
        if len(items[0]) < 2:
            items = [item[None] for item in items]
        items = torch.nn.utils.rnn.pad_sequence(items)
        collated_batch[key] = items
    return collated_batch


def collate_fn(batch):
    keys = batch[0].keys()
    max_lengths = {key: 0 for key in keys}
    collated_batch = {key: [] for key in keys}

    # find out the max lengths
    for row in batch:
        for key in keys:
            if not np.isscalar(row[key]):
                max_lengths[key] = max(max_lengths[key], row[key].shape[-1])

    # pad to the max lengths
    for row in batch:
        for key in keys:
            if not np.isscalar(row[key]):
                array = row[key]
                dim = len(array.shape)
                assert dim == 1 or dim == 2
                if dim == 1:
                    padded_array = np.pad(array, (0, max_lengths[key] - array.shape[-1]), mode='constant')
                else:
                    # padded_array = np.pad(array, ((0, max_lengths[key] - array.shape[0]), (0, 0)), mode='constant')
                    padded_array = np.pad(array, ((0, 0), (0, max_lengths[key] - array.shape[-1])), mode='constant')
                collated_batch[key].append(padded_array)
            else:
                collated_batch[key].append(row[key])

    # use the default_collate to convert to tensors
    for key in keys:
        collated_batch[key] = default_collate(collated_batch[key])
    return collated_batch