import torch
import torch.nn as nn


def init_weight(m):
    nn.init.xavier_normal_(m.weight)
    if m.bias is not None:
        nn.init.constant_(m.bias, 0)


class LayerNorm(nn.Module):

    def __init__(self, dim, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, inputs):
        mean = inputs.mean(dim=-1, keepdim=True)
        # keepdim = dim 유지하지만 1로 바뀜
        # e.g) Tensor[3, 3, 4] --> Tensor[3, 3, 1]
        std = inputs.std(dim=-1, keepdim=True)

        output = (inputs - mean) / (std + self.eps)
        output = self.gamma * output + self.beta

        return output


class AddNorm(nn.Module):

    def __init__(self, sublayer, d_model=512):
        super(AddNorm, self).__init__()
        self.sublayer = sublayer
        self.layer_norm = LayerNorm(d_model)

    def forward(self, *args):
        residual = args[0]
        output = self.sublayer(*args)

        if isinstance(output, tuple):
            # e.g) MulitHeadAttention 하면 return 값으로 (output, attn_map)
            return self.layer_norm(output[0] + residual), output[1]

        return self.layer_norm(output + residual)


class PositionWiseFeedForwardNet(nn.Module):

    def __init__(self, d_model=512, d_ff=2048):
        super(PositionWiseFeedForwardNet, self).__init__()

        # Transormer paper) 원래 Linear 2번이지만 Conv1d kernel=1 로하는 것랑 일치
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)

    def forward(self, inputs):
        conv1 = self.conv1(inputs.transpose(1, 2))
        relu = self.relu(conv1)
        conv2 = self.conv2(relu)
        output = conv2.transpose(1, 2)

        return output

