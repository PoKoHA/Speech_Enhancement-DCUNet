import torch
import torch.nn as nn

from model.complex_nn import CConv2d, CConvTranspose2d, CBatchNorm2d

class EncoderBlock(nn.Module):

    def __init__(self, filter_size=(7, 5), stride_size=(2, 2), in_channels=1, out_channels=45, padding=(0, 0)):
        super(EncoderBlock, self).__init__()

        self.filter_size = filter_size
        self.stride_size = stride_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.padding = padding

        self.cConv = CConv2d(in_channels=self.in_channels, out_channels=self.out_channels,
                             kernel_size=self.filter_size, stride=self.stride_size, padding=self.padding)
        self.cBN = CBatchNorm2d(num_features=self.out_channels)
        self.leaky_relu = nn.LeakyReLU()

    def forward(self, x):
        cConv = self.cConv(x)
        cBN = self.cBN(cConv)
        output = self.leaky_relu(cBN)

        return output


class DecoderBlock(nn.Module):

    def __init__(self, filter_size=(7, 5), stride_size=(2, 2), in_channels=1, out_channels=45,
                 output_padding=(0, 0), padding=(0, 0), last_layer=False):
        super(DecoderBlock, self).__init__()

        self.filter_size = filter_size
        self.stride_size = stride_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.output_padding = output_padding
        self.padding = padding

        self.last_layer = last_layer

        self.Trans_cConv = CConvTranspose2d(in_channels=self.in_channels, out_channels=self.out_channels,
                                       kernel_size=self.filter_size, stride=self.stride_size,
                                       output_padding=self.output_padding, padding=self.padding)

        self.cBN = CBatchNorm2d(num_features=self.out_channels)

        self.leaky_relu = nn.LeakyReLU()

    def forward(self, x):

        Trans_cConv = self.Trans_cConv(x)

        # Paper) last_decoder_layer 에서는 BN과 Activation 을 사용하지 않음
        if not self.last_layer:
            normed = self.cBN(Trans_cConv)
            output = self.leaky_relu(normed)
        else:
            mask_phase = Trans_cConv / (torch.abs(Trans_cConv) + 1e-8)
            mask_mag = torch.tanh(torch.abs(Trans_cConv))
            output = mask_phase * mask_mag

        return output


class DCUNet(nn.Module):

    def __init__(self, n_fft=64, hop_length=16):
        super(DCUNet, self).__init__()

        # ISTFT hyperparam
        self.n_fft = n_fft
        self.hop_length = hop_length

        # Encoder(downsampling)
        self.downsample0 = EncoderBlock(filter_size=(7, 5), stride_size=(2, 2), in_channels=1, out_channels=45)
        self.downsample1 = EncoderBlock(filter_size=(7, 5), stride_size=(2, 2), in_channels=45, out_channels=90)
        self.downsample2 = EncoderBlock(filter_size=(5, 3), stride_size=(2, 2), in_channels=90, out_channels=90)
        self.downsample3 = EncoderBlock(filter_size=(5, 3), stride_size=(2, 2), in_channels=90, out_channels=90)
        self.downsample4 = EncoderBlock(filter_size=(5, 3), stride_size=(2, 1), in_channels=90, out_channels=90)

        # Decoder(Upsampling)
        self.upsample0 = DecoderBlock(filter_size=(5, 3), stride_size=(2, 1), in_channels=90, out_channels=90)
        self.upsample1 = DecoderBlock(filter_size=(5, 3), stride_size=(2, 2), in_channels=180, out_channels=90)
        self.upsample2 = DecoderBlock(filter_size=(5, 3), stride_size=(2, 2), in_channels=180, out_channels=90)
        self.upsample3 = DecoderBlock(filter_size=(7, 5), stride_size=(2, 2), in_channels=180, out_channels=45)
        self.upsample4 = DecoderBlock(filter_size=(7, 5), stride_size=(2, 2), in_channels=90, output_padding=(0, 1),
                                 out_channels=1, last_layer=True)

    def forward(self, x, is_istft=True):
        # downsampling/encoding
        d0 = self.downsample0(x)
        d1 = self.downsample1(d0)
        d2 = self.downsample2(d1)
        d3 = self.downsample3(d2)
        d4 = self.downsample4(d3)

        # upsampling/decoding
        u0 = self.upsample0(d4)
        # skip-connection
        c0 = torch.cat((u0, d3), dim=1)

        u1 = self.upsample1(c0)
        c1 = torch.cat((u1, d2), dim=1)

        u2 = self.upsample2(c1)
        c2 = torch.cat((u2, d1), dim=1)

        u3 = self.upsample3(c2)
        c3 = torch.cat((u3, d0), dim=1)

        u4 = self.upsample4(c3)

        # u4 - the mask (apply mask)
        output = u4 * x
        if is_istft:
            output = torch.squeeze(output, 1)
            output = torch.istft(output, n_fft=self.n_fft, hop_length=self.hop_length, normalized=True)

        return output

