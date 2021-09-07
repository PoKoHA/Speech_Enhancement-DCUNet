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

    def __init__(self, dim, n_heads, dropout_p=0.1):
        super(SelfAttention, self).__init__()
        self.d_model = dim
        self.n_heads = n_heads

        self.sqrt_dim = np.sqrt(dim)
        self.attn_dim = dim // n_heads # 512 // 8 = 64
        self.dropout = nn.Dropout(dropout_p)

        self.Linear_Q = nn.Linear(dim, self.attn_dim, bias=True)
        self.Linear_K = nn.Linear(dim, self.attn_dim, bias=True)
        self.Linear_V = nn.Linear(dim, self.attn_dim, bias=True)
        init_weight(self.Linear_Q)
        init_weight(self.Linear_K)
        init_weight(self.Linear_V)

    def forward(self, query, key, value, mask=None):
        # print("B")
        query = self.Linear_Q(query)
        key = self.Linear_K(key)
        value = self.Linear_V(value)

        # mask
        if mask is not None:
            mask = mask.repeat(self.n_heads, 1, 1)

        score = torch.bmm(query, key.transpose(1, 2)) / self.sqrt_dim

        if mask is not None:
            # # print("score: ", score.size())
            # # print("mask: ", mask.size())
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
        self.attentions = nn.ModuleList([SelfAttention(d_model, n_heads)
                                         for _ in range(n_heads)]) # sqrt(d_k)

        self.fc = nn.Linear(d_model, d_model, bias=False)
        init_weight(self.fc)

    def forward(self, q, k, v, mask=None):
        # print("  [Attention]  ")
        # print("   q", q.size())
        # print("   k", k.size())
        # print("   v", v.size())

        # context, attn = self.attentions(q, k, v, mask) # [1, 1539, 64]
        self_attentions = [attention(q, k, v, mask) for attention in self.attentions]

        weight_vs = [weighted_v[0] for weighted_v in self_attentions]
        attentions = [weighted_v[1] for weighted_v in self_attentions]
        weighted_v = torch.cat(weight_vs, dim=-1)
        # print("  torch.cat", weighted_v.size())
        # context = weighted_v.permute(1, 2, 0, 3).contiguous().view(batch_size, -1, self.n_heads * self.attn_dim)
        context = self.fc(weighted_v)
        # print("  Linear: ", context.size())

        return context, attentions


#####################
# CBAM
#####################
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

# Spatial Attention
class ChannelPool(nn.Module):
    def forward(self, x):
        # x [batch, chaanel, freq, time]
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)


class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)

    def forward(self, x):
        # x [batch, chaanel, freq, time]
        x_compress = self.compress(x) # [batch, max+mean=2, freq, time]
        x_out = self.spatial(x_compress) # [batch, 1, freq, time]]
        scale = F.sigmoid(x_out) # [batch, 1, freq, time]
        return x * scale


# Channel Attention
class Flatten(nn.Module):
    def forward(self, x):
        # x [batch, channel, 1, 1]
        return x.view(x.size(0), -1) # [batch, channel]


class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
            )
        self.pool_types = pool_types

    def forward(self, x):
        # x [batch, channel, freq, time]
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type =='avg':
                avg_pool = F.avg_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                # [batch, channel, 1, 1]
                channel_att_raw = self.mlp( avg_pool )
                # [batch, channel]
                # print(channel_att_raw.size())
            elif pool_type =='max':
                max_pool = F.max_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                # [batch, channel, 1, 1]
                channel_att_raw = self.mlp( max_pool )
                # [batch, channel]
                # print(channel_att_raw.size())
            elif pool_type =='lp':
                lp_pool = F.lp_pool2d( x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( lp_pool )
            elif pool_type =='lse':
                # LSE pool only
                lse_pool = logsumexp_2d(x)
                channel_att_raw = self.mlp( lse_pool )

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = F.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).expand_as(x)
        # [batch, channel, freq, time]

        return x * scale


def logsumexp_2d(tensor):
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs


class CCBAM(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False):
        super(CCBAM, self).__init__()
        self.ChannelGate_real = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.ChannelGate_imag = ChannelGate(gate_channels, reduction_ratio, pool_types)

        self.no_spatial = no_spatial
        if not no_spatial:
            self.SpatialGate_real = SpatialGate()
            self.SpatialGate_imag = SpatialGate()

    def forward(self, x):
        real = x[..., 0]
        imag = x[..., 1]

        real_out = self.ChannelGate_real(real)
        if not self.no_spatial:
            real_out = self.SpatialGate_real(real_out)

        imag_out = self.ChannelGate_imag(imag)
        if not self.no_spatial:
            imag_out = self.SpatialGate_imag(imag_out)

        x_out = torch.stack([real_out, imag_out], dim=-1)

        return x_out


#################################
# Self-Attn(SAGAN)
#################################
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

###############################
# Freq, Temporal Attention
###############################
class FreqGate(nn.Module):

    def __init__(self):
        super(FreqGate, self).__init__()

        self.conv = BasicConv(2, 1, kernel_size=(7, 1), stride=1, padding=(3, 0), relu=False)
        self.compress = ChannelPool()

    def forward(self, x):
        # x [Batch, C, Freq, Time]
        x_avg = F.avg_pool2d(x, (1, x.size(3)), stride=(1, x.size(3))) # [batch, C, Freq, 1]
        compress = self.compress(x_avg) # [batch, 2, Freq, 1]
        conv = self.conv(compress) # [batch, 1, Freq, 1]
        scale = F.sigmoid(conv)

        return x * scale


class TemporalGate(nn.Module):

    def __init__(self):
        super(TemporalGate, self).__init__()

        self.conv = BasicConv(2, 1, kernel_size=(1, 7), stride=1, padding=(0, 3), relu=False)
        self.compress = ChannelPool()

    def forward(self, x):
        # x [Batch, C, Freq, Time]
        x_avg = F.avg_pool2d(x, (x.size(2), 1), stride=(x.size(2), 1)) # [batch, C, 1, Time]
        compress = self.compress(x_avg) # [batch, 2, 1, Time]
        conv = self.conv(compress) # [batch, 1, 1, TIme]
        scale = F.sigmoid(conv)

        return x * scale


if __name__ == "__main__":
    test = torch.randn(2, 32, 259, 120)
    c = FreqGate()
    t = TemporalGate()

