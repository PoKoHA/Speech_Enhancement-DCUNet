import math
import numpy as np

import torch
import torch.nn as nn

import matplotlib.pyplot as plt

class PositionalEncoding(nn.Module):

    def __init__(self, d_model=512, max_len=1600):
        super(PositionalEncoding, self).__init__()

        PE = torch.zeros(max_len, d_model, requires_grad=False)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        PE[:, 0::2] = torch.sin(pos * div_term)
        PE[:, 1::2] = torch.cos(pos * div_term)
        # fig, ax = plt.subplots(figsize=(20,20))
        # cax = ax.matshow(PE)
        # plt.colorbar(cax)
        # plt.show()
        PE = PE.unsqueeze(0)
        self.register_buffer('PE', PE)

    def forward(self, length):
        # todo input.size(1) 아니면 length
        return self.PE[:, :length]

class Embedding(nn.Module):
    """
    Decoder Input 은 음성이 아닌 Transcript 이니까 그 때 사용
    """
    def __init__(self, num_embeddings, d_model=512):
        super(Embedding, self).__init__()
        self.sqrt_dim = math.sqrt(d_model)
        self.embedding = nn.Embedding(num_embeddings, d_model)

    def forward(self, inputs):
        # Transformer Paper) we multiply those weights by sqrt(d_model)
        return self.embedding(inputs) * self.sqrt_dim

a = PositionalEncoding()