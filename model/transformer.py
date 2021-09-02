import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.sublayer import *
from model.embedding import *
from model.attention import *
from model.VGGExtractor import VGGExtractor

class EncoderLayer(nn.Module):

    def __init__(self, d_model=512, n_heads=8, d_ff=2048):
        super(EncoderLayer, self).__init__()
        self.self_attention = AddNorm(MultiHeadAttention(d_model, n_heads), d_model)
        self.feed_forward = AddNorm(PositionWiseFeedForwardNet(d_model, d_ff), d_model)

    def forward(self, inputs, mask=None):
        attn_output, attn_map = self.self_attention(inputs, inputs, inputs, mask)
        output = self.feed_forward(attn_output)

        return output, attn_map

class Encoder(nn.Module):

    def __init__(self, args, d_model=512, input_dim=1539, d_ff=2048, n_layers=6, n_heads=8, dropout_p=0.3):
        super(Encoder, self).__init__()

        super(Encoder, self).__init__()

        self.args = args
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads

        self.conv = VGGExtractor(args=args, input_dim=input_dim)
        self.linear = nn.Linear(3392, d_model)  # README 참고 Linear 한번 해주고 나서 Encoder layer 실행
        init_weight(self.linear)

        self.dropout = nn.Dropout(dropout_p)
        self.positional_encoding = PositionalEncoding(d_model)

        self.layers = nn.ModuleList([EncoderLayer(d_model, n_heads, d_ff) for _ in range(n_layers)])

    def forward(self, inputs):
        # print('--[Encoder]--')
        # print("  Input(mask)", inputs.size())
        conv_outputs = self.conv(inputs)
        # print("p", conv_outputs.size())
        # print(self.linear)
        outputs = self.linear(conv_outputs)
        # print("  Linear:", outputs.size())
        # print("pos", self.positional_encoding(outputs.size(1)).size())
        outputs += self.positional_encoding(outputs.size(1))
        # print("  output+Pos_Encoding:", outputs.size())
        outputs = self.dropout(outputs)

        for layer in self.layers:
            outputs, attn = layer(outputs, None)
        # print("En out:", outputs.size())
        return outputs, attn

############################################
class DecoderLayer(nn.Module):

    def __init__(self, d_model=512, n_heads=8, d_ff=2048):
        super(DecoderLayer, self).__init__()

        self.self_attention = AddNorm(MultiHeadAttention(d_model, n_heads), d_model)
        self.cross_attention = AddNorm(MultiHeadAttention(d_model, n_heads), d_model)
        self.feed_forward = AddNorm(PositionWiseFeedForwardNet(d_model, d_ff), d_model)

    def forward(self, inputs, encoder_outputs, mask=None, cross_mask=None):
        output, self_attn = self.self_attention(inputs, inputs, inputs, mask)
        output, cross_attn = self.cross_attention(output, encoder_outputs, encoder_outputs, cross_mask)
        output = self.feed_forward(output)
        # print("output: ", output.size())
        # print("cross_attn: ", cross_attn.size())

        return output, self_attn, cross_attn


class Decoder(nn.Module):

    def __init__(
            self,
            args,
            input_dim,
            d_model=512,
            d_ff=2048,
            n_layers=6,
            n_heads=8,
            dropout_p=0.3,
    ):
        super(Decoder, self).__init__()

        self.args = args
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads

        self.conv = VGGExtractor(args=args, input_dim=input_dim)
        self.linear = nn.Linear(3392, d_model)  # README 참고 Linear 한번 해주고 나서 Encoder layer 실행
        init_weight(self.linear)

        self.dropout = nn.Dropout(dropout_p)
        self.positional_encoding = PositionalEncoding(d_model)

        self.layers = nn.ModuleList([
            DecoderLayer(d_model, n_heads, d_ff) for _ in range(n_layers)
        ])

        self.layerNorm = LayerNorm(d_model)
        self.fc = nn.Linear(d_model, input_dim, bias=False)

    def forward(self, inputs, encoder_outputs, decoder_mask=None, encoder_pad_mask=None):

        conv_outputs = self.conv(inputs)
        outputs = self.linear(conv_outputs)
        outputs += self.positional_encoding(outputs.size(1))
        outputs = self.dropout(outputs)

        for layer in self.layers:
            outputs, self_attn, cross_attn = layer(
                outputs, encoder_outputs, decoder_mask, encoder_pad_mask
            )

        # print("  Multi-Head-Attn-Output: ", outputs.size())
        outputs = self.fc(outputs)
        # print("  Linear: ", outputs.size())
        # print("  Expand: ", outputs.unsqueeze(1).size())

        return outputs, self_attn, cross_attn


class SpeechTransformer(nn.Module):

    def __init__(self, args, input_dim=214, d_model=512, d_ff=2048, n_heads=8, n_encoder_layers=3,
                 n_decoder_layers=3, dropout_p=0.3, max_length=7000):
        super(SpeechTransformer, self).__init__()
        assert d_model % n_heads == 0

        self.encoder = Encoder(
            args=args,
            d_model=d_model,
            input_dim=input_dim,
            d_ff=d_ff,
            n_heads=n_heads,
            n_layers=n_encoder_layers,
            dropout_p=dropout_p,
        )

        self.fc = nn.Linear(d_model, input_dim)
        # self.decoder = Decoder(
        #     args=args,
        #     input_dim=input_dim,
        #     d_model=d_model,
        #     d_ff=d_ff,
        #     n_layers=n_decoder_layers,
        #     n_heads=n_heads,
        #     dropout_p=dropout_p,
        # )

    def forward(self, mask):
        encoder_outputs, encoder_attn = self.encoder(mask)
        encoder_outputs = self.fc(encoder_outputs)
        # print("encoder", encoder_outputs.size())
        # decoder_outputs, target_self_attn, cross_attn = self.decoder(target, encoder_outputs)

        return encoder_outputs


if __name__ == "__main__":
    a = torch.randn(2, 1, 1539, 214).cuda()
    b = torch.randn(2, 1539, 512).cuda()
    c = SpeechTransformer(args="GOOD").cuda()
    print(c.decoder(a, b)[0].size())

