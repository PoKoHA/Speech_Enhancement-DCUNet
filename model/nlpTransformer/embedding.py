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


if __name__ == "__main__":
    a = torch.randn(2, 165974).cuda()
    b = PositionalEncoding(max_len=165975, d_model=512).cuda()
    print(b(a.size(1)).size())
    a = a.unsqueeze(dim=-1)
    print("a", a.size())
    c = nn.Linear(1, 512).cuda()
    print(c(a).size())
    print(c.weight.transpose(0,1).size())

    # b = create_position_vector(a).cuda()
    # print(b.size())
    # c = create_position_encoding(165974, 512).cuda()
    # print(c.size())
