import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class Self_Attn(nn.Module):

    def __init__(self, in_channels=1):
        super(Self_Attn, self).__init__()

        self.in_channels = in_channels

        self.conv_q = nn.Conv2d(in_channels=in_channels, out_channels=in_channels // 8, kernel_size=1)
        self.conv_k = nn.Conv2d(in_channels=in_channels, out_channels=in_channels // 8, kernel_size=1)
        self.conv_v = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
        input: Spectrogram[batch, Channel=1, 1539, 214]
        output: self Attn Value + input
        attn = B X N X N (N: width * height)
        """
        # print("x", x.size())
        batch, channel, freq, time = x.size()
        Q = self.conv_q(x).view(batch, -1, freq*time).permute(0, 2, 1) # [batch, freq*time, channel]
        K = self.conv_k(x).view(batch, -1, freq*time)
        V = self.conv_v(x).view(batch, -1, freq*time)

        energy = torch.bmm(Q, K) # [B, freq*time, freq*time]
        attn = self.softmax(energy)

        out = torch.bmm(V, attn.permute(0, 2, 1))
        # print("V*attm: ", out.size())
        out = out.view(batch, channel, freq, time)

        out = self.gamma * out + x

        return out, attn


if __name__ == "__main__":
    attn = Self_Attn(in_channels=1).cuda()
    test = torch.randn(2, 1, 770, 107).cuda()

    print(attn(test)[0].size())
