import numpy as np

import torch
from torch.utils.data import DataLoader, Sampler

def zero_pad_concat(inputs):
    max_t = max(inp.shape[0] for inp in inputs)
    shape = (len(inputs), max_t)
    input_mat = np.zeros(shape, dtype=np.float32)
    for e, inp in enumerate(inputs):
        input_mat[e, :inp.shape[0]] = inp
    return input_mat

def _collate_fn(inputs):
    cleans, transcript, mixeds, name = zip(*inputs)
    seq_lens = torch.IntTensor([i.shape[0] for i in mixeds])

    x = torch.FloatTensor(zero_pad_concat(mixeds))
    y = torch.FloatTensor(zero_pad_concat(cleans))

    batch = [x, y, seq_lens, name]

    return batch

class AudioDataLoader(DataLoader):

    def __init__(self, *args, **kwargs):
        super(AudioDataLoader, self).__init__(*args, **kwargs)
        self.collate_fn = _collate_fn
