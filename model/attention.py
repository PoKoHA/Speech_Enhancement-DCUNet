import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

"""
Transformer에서 사용하는 MultiheadSelfAttention 적용
"""
"""
Linear 차원이 너무 커서 X
"""
def init_weight(m):
    nn.init.xavier_uniform_(m.weight)
    if m.bias is not None:
        nn.init.constant_(m.bias, 0)

class SelfAttention(nn.Module):

    def __init__(self, args):
        super(SelfAttention, self).__init__()
        self.hidden_dim = 6099
        self.attention_dim = 6099 // args.n_head

        self.linear_Q = nn.Linear(6099, self.attention_dim)
        self.linear_K = nn.Linear(6099, self.attention_dim)
        self.linear_V = nn.Linear(6099, self.attention_dim)

        self.dropout = nn.Dropout(args.dropout)
        self.scale_factor = torch.sqrt(torch.FloatTensor([self.attention_dim])).cuda(args.gpu)

    def forward(self, query, key, value, mask=None):
        q = self.linear_Q(query)
        k = self.linear_K(key)
        v = self.linear_V(value)

        self_attention = torch.bmm(q, k.permute(0, 2, 1))
        self_attention = self_attention / self.scale_factor

        if mask is not None:
            self_attention = self_attention.masked_fill(mask, -np.inf)

        attn_score = F.softmax(self_attention, dim=-1)
        dropout = self.dropout(attn_score)

        weighted_v = torch.bmm(dropout, v)

        return self.dropout(weighted_v), attn_score

class MultiHeadAttention(nn.Module):

    def __init__(self, args):
        super(MultiHeadAttention, self).__init__()
        self.attentions = nn.ModuleList([SelfAttention(args).cuda(args.gpu)
                                         for _ in range(args.n_head)]).cuda(args.gpu)

        self.o_w = nn.Linear(6099, 6099, bias=False).cuda(args.gpu)
        init_weight(self.o_w)
        self.dropout = nn.Dropout(args.dropout)

    def forward(self, query, key, value, mask=None):
        self_attentions = [attention(query, key, value, mask) for attention in self.attentions]
        weight_vs = [weighted_v[0] for weighted_v in self_attentions]
        attentions = [weighted_v[1] for weighted_v in self_attentions]
        weighted_v = torch.cat(weight_vs, dim=-1)
        output = self.dropout(self.o_w(weighted_v))

        return output, attentions

########
# CBAM
########
class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)


class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = F.sigmoid(x_out) # broadcasting
        return x * scale


if __name__ == "__main__":
    test = torch.randn(1, 1, 32, 16)
    C = SpatialGate()
    print(test.size())
    print(C(test).size())
