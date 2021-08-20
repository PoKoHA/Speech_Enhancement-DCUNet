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

class ScaledDotProductAttention(nn.Module):

    def __init__(self, dim, dropout_p=0.1):
        super(ScaledDotProductAttention, self).__init__()
        self.sqrt_dim = np.sqrt(dim)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, query, key, value, mask=None):
        score = torch.bmm(query, key.transpose(1, 2)) / self.sqrt_dim

        if mask is not None:
            # print("score: ", score.size())
            # print("mask: ", mask.size())
            score.masked_fill_(mask, -np.inf)

        attn = F.softmax(score, -1)
        attn = self.dropout(attn)
        context = torch.bmm(attn, value)

        return context, attn


class MultiHeadAttention(nn.Module):

    def __init__(self, d_model=512, n_heads=8):
        super(MultiHeadAttention, self).__init__()
        assert d_model % n_heads == 0, "attention dim = d_model / heads 해야 하기 때문에"

        self.attn_dim = int(d_model / n_heads) # default:64
        self.n_heads = n_heads

        # todo 뒤 사이즈가 조금 다름 원래 attn_dim만 들어갔는데
        # Projection
        self.Linear_Q = nn.Linear(d_model, self.attn_dim * n_heads, bias=True)
        self.Linear_K = nn.Linear(d_model, self.attn_dim * n_heads, bias=True)
        self.Linear_V = nn.Linear(d_model, self.attn_dim * n_heads, bias=True)
        init_weight(self.Linear_Q)
        init_weight(self.Linear_K)
        init_weight(self.Linear_V)

        self.scaled_dot_attn = ScaledDotProductAttention(self.attn_dim) # sqrt(d_k)

    def forward(self, q, k, v, mask=None):
        batch_size = v.size(0)

        # [Batch, Length, N, D] = [Batch, Length, 8, 64]
        query = self.Linear_Q(q).view(batch_size, -1, self.n_heads, self.attn_dim)
        key = self.Linear_K(k).view(batch_size, -1, self.n_heads, self.attn_dim)
        value = self.Linear_V(v).view(batch_size, -1, self.n_heads, self.attn_dim)

        # [Batch * N, Length, Dim]
        query = query.permute(2, 0, 1, 3).contiguous().view(batch_size * self.n_heads, -1, self.attn_dim)
        key = key.permute(2, 0, 1, 3).contiguous().view(batch_size * self.n_heads, -1, self.attn_dim)
        value = value.permute(2, 0, 1, 3).contiguous().view(batch_size * self.n_heads, -1, self.attn_dim)

        # mask
        if mask is not None:
            mask = mask.repeat(self.n_heads, 1, 1)

        context, attn = self.scaled_dot_attn(query, key, value, mask)
        context = context.view(self.n_heads, batch_size, -1, self.attn_dim)
        context = context.permute(1, 2, 0, 3).contiguous().view(batch_size, -1, self.n_heads * self.attn_dim)

        return context, attn


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
