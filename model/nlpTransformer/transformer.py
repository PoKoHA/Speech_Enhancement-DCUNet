import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.nlpTransformer.embedding import *
from model.nlpTransformer.FFN import *
from model.nlpTransformer.attention import *


class EncoderLayer(nn.Module):

    def __init__(self, args):
        super(EncoderLayer, self).__init__()

        self.layer_norm = nn.LayerNorm(args.hidden_dim, eps=1e-6)
        self.self_attention = MultiHeadAttention(args)
        self.position_wise_ffn = PositionWiseFeedForward(args)

    def forward(self, inputs, source_mask=None):

        normalized_source = self.layer_norm(inputs)

        print("Q", normalized_source.size())

        self_attention = inputs + self.self_attention(
            normalized_source, normalized_source, normalized_source, source_mask)[0]

        normalized_self_attention = self.layer_norm(self_attention)

        output = self_attention + self.position_wise_ffn(normalized_self_attention)

        return output


class Encoder(nn.Module):

    def __init__(self, args):
        super(Encoder, self).__init__()

        self.args = args

        self.input_linear = nn.Linear(1, 32)
        nn.init.normal_(self.input_linear.weight, mean=0, std=args.hidden_dim ** -0.5)

        self.embedding_scale = args.hidden_dim ** 0.5

        self.positional_encoding = PositionalEncoding(args.hidden_dim, max_len=1000)

        self.encoder_layers = nn.ModuleList([EncoderLayer(args) for _ in range(args.n_layer)])

        self.dropout = nn.Dropout(args.dropout)
        self.layer_norm = nn.LayerNorm(args.hidden_dim, eps=1e-6)

    def forward(self, inputs):
        # print("P")
        embedding_scaled = self.input_linear(inputs) * self.embedding_scale
        # print("ASS")
        embedding = self.dropout(embedding_scaled + self.positional_encoding(inputs.size(1)))

        for encoder_layer in self.encoder_layers:
            embedding = encoder_layer(embedding)

        return self.layer_norm(embedding)

##################################################
# DECODER
##################################################
class DecoderLayer(nn.Module):

    def __init__(self, args):
        super(DecoderLayer, self).__init__()

        self.args = args
        self.layer_norm = nn.LayerNorm(args.hidden_dim, eps=1e-6)
        self.self_attention = MultiHeadAttention(args)
        self.encoder_attention = MultiHeadAttention(args)
        self.position_wise_ffn = PositionWiseFeedForward(args)

    def forward(self, target, encoder_output, target_mask=None, dec_enc_mask=None):
        norm_target = self.layer_norm(target)
        self_attention = target + self.self_attention(norm_target, norm_target, norm_target, target_mask)[0]
        normalized_self_attention = self.layer_norm(self_attention)
        sub_layer, attn_map = self.encoder_attention(normalized_self_attention,
                                                     encoder_output, encoder_output, dec_enc_mask)

        sub_layer_output = self_attention + sub_layer

        norm_sub_layer_norm = self.layer_norm(sub_layer_output)
        # print("layer_norm: ", norm_sub_layer_norm.size())
        output = sub_layer_output + self.position_wise_ffn(norm_sub_layer_norm)
        # output [batch_size, target_length, hidden_dim]

        return output, attn_map


class Decoder(nn.Module):

    def __init__(self, args):
        super(Decoder, self).__init__()
        self.args = args
        # hidden_dim 512 / output_dim 19545 [len(eng.vocab)]
        self.input_linear = nn.Linear(1, 32)
        nn.init.normal_(self.input_linear.weight, mean=0, std=args.hidden_dim ** -0.5)

        self.embedding_scale = args.hidden_dim ** 0.5
        self.positional_encoding = PositionalEncoding(args.hidden_dim, max_len=1000)

        self.decoder_layers = nn.ModuleList([DecoderLayer(args) for _ in range(args.n_layer)])
        self.dropout = nn.Dropout(args.dropout)
        self.layer_norm = nn.LayerNorm(args.hidden_dim, eps=1e-6)

    def forward(self, target, encoder_output):

        embedding_scaled = self.input_linear(target) * self.embedding_scale
        embedding = self.dropout(embedding_scaled + self.positional_encoding(target.size(1)))

        for decoder_layer in self.decoder_layers:
            target, attn_map = decoder_layer(embedding, encoder_output)

        target_norm = self.layer_norm(target)
        output = torch.matmul(target_norm, self.input_linear.weight.transpose(0, 1))

        return output, attn_map


class Transformer(nn.Module):

    def __init__(self, args):
        super(Transformer, self).__init__()
        self.encoder = Encoder(args)
        self.decoder = Decoder(args)

    def forward(self, inputs, target):
        # source [batch_size, source_length] = [128, 6]
        # target [batch_size, target_length] = [128, 24]

        encoder_output = self.encoder(inputs)
        # [batch_size, source_length, hidden_dim]
        output, attn_map = self.decoder(target, encoder_output)
        # [batch_size, target_length, output_dim]

        return output, attn_map
