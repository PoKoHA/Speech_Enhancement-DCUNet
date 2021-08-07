import torch
import torch.nn as nn

from model.complex_nn import CConv2d, CConvTranspose2d, CBatchNorm2d
from model.ISTFT import ISTFT

class EncoderBlock(nn.Module):

    def __init__(self, in_channels=1, out_channels=45, kernel_size=(7, 5), stride=(2, 2),
                 padding=(0, 0), bias=False):
        super(EncoderBlock, self).__init__()

        self.cConv = CConv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                             padding=padding, bias=bias)
        self.cBN = CBatchNorm2d(out_channels)
        self.leaky_relu = nn.LeakyReLU(inplace=True, negative_slope=0.1)

    def forward(self, x):
        cConv = self.cConv(x)
        cBN = self.cBN(cConv)
        output = self.leaky_relu(cBN)

        return output


class DecoderBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=(0, 0), output_padding=(0, 0),
                 last=False, bias=False):

        super(DecoderBlock, self).__init__()
        self.last = last

        self.Trans_cConv = CConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                                            padding=padding, output_padding=output_padding,bias=bias)
        self.cBN = CBatchNorm2d(num_features=out_channels)
        self.leaky_relu = nn.LeakyReLU(inplace=True, negative_slope=0.1)

    def forward(self, x):

        Trans_cConv = self.Trans_cConv(x)
        # Paper) last_decoder_layer 에서는 BN과 Activation 을 사용하지 않음
        if self.last:
            mask_phase = Trans_cConv / (torch.abs(Trans_cConv) + 1e-8)
            mask_mag = torch.tanh(torch.abs(Trans_cConv))
            output = mask_phase * mask_mag
        else:
            normed = self.cBN(Trans_cConv)
            output = self.leaky_relu(normed)

        return output


class DCUNet10(nn.Module):

    def __init__(self, args, n_fft=64, hop_length=16):
        super(DCUNet10, self).__init__()

        # ISTFT hyperparam
        self.args = args
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.istft = ISTFT(hop_length=hop_length, n_fft=n_fft).cuda(args.gpu)

        # Encoder(downsampling)
        self.downsample0 = EncoderBlock(kernel_size=(7, 5), stride=(2, 2), in_channels=1, out_channels=45)
        self.downsample1 = EncoderBlock(kernel_size=(7, 5), stride=(2, 2), in_channels=45, out_channels=90)
        self.downsample2 = EncoderBlock(kernel_size=(5, 3), stride=(2, 2), in_channels=90, out_channels=90)
        self.downsample3 = EncoderBlock(kernel_size=(5, 3), stride=(2, 2), in_channels=90, out_channels=90)
        self.downsample4 = EncoderBlock(kernel_size=(5, 3), stride=(2, 1), in_channels=90, out_channels=90)

        # Decoder(Upsampling)
        self.upsample0 = DecoderBlock(kernel_size=(5, 3), stride=(2, 1), in_channels=90, out_channels=90)
        self.upsample1 = DecoderBlock(kernel_size=(5, 3), stride=(2, 2), in_channels=180, out_channels=90)
        self.upsample2 = DecoderBlock(kernel_size=(5, 3), stride=(2, 2), in_channels=180, out_channels=90)
        self.upsample3 = DecoderBlock(kernel_size=(7, 5), stride=(2, 2), in_channels=180, out_channels=45)
        self.upsample4 = DecoderBlock(kernel_size=(7, 5), stride=(2, 2), in_channels=90, out_channels=1,
                                      output_padding=(0, 1), bias=True, last=True)

    def forward(self, x, is_istft=True):
        # downsampling/encoding
        # print("input: ", x.size())
        d0 = self.downsample0(x)
        d1 = self.downsample1(d0)
        d2 = self.downsample2(d1)
        d3 = self.downsample3(d2)
        d4 = self.downsample4(d3)

        # bridge 첫번째 Decoder에 skip connection X
        u0 = self.upsample0(d4)

        # skip-connection
        # print("d3: ", d3.size())
        # print("u0: ", u0.size())
        c0 = torch.cat((u0, d3), dim=1)

        u1 = self.upsample1(c0)
        # print("d2: ", d2.size())
        # print("u1: ", u1.size())
        c1 = torch.cat((u1, d2), dim=1)

        u2 = self.upsample2(c1)
        # print("d1: ", d1.size())
        # print("u2: ", u2.size())
        c2 = torch.cat((u2, d1), dim=1)

        u3 = self.upsample3(c2)
        # print("d0: ", d0.size())
        # print("u3: ", u3.size())
        c3 = torch.cat((u3, d0), dim=1)

        u4 = self.upsample4(c3)

        # print("x: ", x.size())
        # print("mask_hat: ", mask_hat.size())

        output = x * u4
        if is_istft:
            output = self.istft(output)
            output = torch.squeeze(output, 1)

        return output

class DCUNet16(nn.Module):

    def __init__(self, args, n_fft=64, hop_length=16):
        super(DCUNet16, self).__init__()

        # ISTFT hyperparam
        self.args = args
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.istft = ISTFT(hop_length=hop_length, n_fft=n_fft).cuda(args.gpu)

        # Encoder(downsampling)
        self.downsample0 = EncoderBlock(kernel_size=(7, 5), stride=(2, 2), padding=(3, 2), in_channels=1, out_channels=32)
        self.downsample1 = EncoderBlock(kernel_size=(7, 5), stride=(2, 1), padding=(3, 2), in_channels=32, out_channels=32)
        self.downsample2 = EncoderBlock(kernel_size=(7, 5), stride=(2, 2), padding=(3, 2), in_channels=32, out_channels=64)
        self.downsample3 = EncoderBlock(kernel_size=(5, 3), stride=(2, 1), padding=(2, 1), in_channels=64, out_channels=64)
        self.downsample4 = EncoderBlock(kernel_size=(5, 3), stride=(2, 2), padding=(2, 1), in_channels=64, out_channels=64)
        self.downsample5 = EncoderBlock(kernel_size=(5, 3), stride=(2, 1), padding=(2, 1), in_channels=64, out_channels=64)
        self.downsample6 = EncoderBlock(kernel_size=(5, 3), stride=(2, 2), padding=(2, 1), in_channels=64, out_channels=64)
        self.downsample7 = EncoderBlock(kernel_size=(5, 3), stride=(2, 1), padding=(2, 1), in_channels=64, out_channels=64)

        # Decoder(Upsampling)
        self.upsample0 = DecoderBlock(kernel_size=(5, 3), stride=(2, 1), padding=(2, 1), in_channels=64, out_channels=64)
        self.upsample1 = DecoderBlock(kernel_size=(5, 3), stride=(2, 2), padding=(2, 1), in_channels=128, out_channels=64)
        self.upsample2 = DecoderBlock(kernel_size=(5, 3), stride=(2, 1), padding=(2, 1), in_channels=128, out_channels=64)
        self.upsample3 = DecoderBlock(kernel_size=(5, 3), stride=(2, 2), padding=(2, 1), in_channels=128, out_channels=64, output_padding=(0, 1))
        self.upsample4 = DecoderBlock(kernel_size=(5, 3), stride=(2, 1), padding=(2, 1), in_channels=128, out_channels=64)
        self.upsample5 = DecoderBlock(kernel_size=(7, 5), stride=(2, 2), padding=(3, 2), in_channels=128, out_channels=32)
        self.upsample6 = DecoderBlock(kernel_size=(7, 5), stride=(2, 1), padding=(3, 2), in_channels=64, out_channels=32, output_padding=(1, 0))
        self.upsample7 = DecoderBlock(kernel_size=(7, 5), stride=(2, 2), padding=(3, 2), in_channels=64, out_channels=1,
                                      output_padding=(0, 1), bias=True, last=True)


    def forward(self, x, is_istft=True):
        # downsampling/encoding
        # print("input: ", x.size())
        d0 = self.downsample0(x)
        d1 = self.downsample1(d0)
        d2 = self.downsample2(d1)
        d3 = self.downsample3(d2)
        d4 = self.downsample4(d3)
        d5 = self.downsample5(d4)
        d6 = self.downsample6(d5)
        d7 = self.downsample7(d6)

        # bridge 첫번째 Decoder에 skip connection X
        u0 = self.upsample0(d7)

        # skip-connection
        # print("d3: ", d3.size())
        # print("u0: ", u0.size())
        c0 = torch.cat((u0, d6), dim=1)

        u1 = self.upsample1(c0)
        # print("d2: ", d2.size())
        # print("u1: ", u1.size())
        c1 = torch.cat((u1, d5), dim=1)

        u2 = self.upsample2(c1)
        # print("d1: ", d1.size())
        # print("u2: ", u2.size())
        c2 = torch.cat((u2, d4), dim=1)

        u3 = self.upsample3(c2)
        c3 = torch.cat((u3, d3), dim=1)

        u4 = self.upsample4(c3)

        c4 = torch.cat((u4, d2), dim=1)

        u5 = self.upsample5(c4)

        c5 = torch.cat((u5, d1), dim=1)

        u6 = self.upsample6(c5)

        c6 = torch.cat((u6, d0), dim=1)

        u7 = self.upsample7(c6)
        output = x * u7

        if is_istft:
            # output = torch.squeeze(output, 1)
            # output = torch.istft(output, n_fft=self.n_fft, hop_length=self.hop_length, normalized=True)
            output = self.istft(output)
            output = torch.squeeze(output, 1)

        return output