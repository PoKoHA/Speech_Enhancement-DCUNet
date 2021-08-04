import numpy
import math

import torch
import torch.nn as nn
import torch.functional as F

from model.complex_module import *

# X1을 X2 크기와 맞게 Pad
def pad2d_as(x1, x2):
    diffH = x2.size()[2] - x1.size()[2]
    diffW = x2.size()[3] - x1.size()[3]

    return F.pad(x1, (0, diffW, 0, diffH)) # (L, R, Top , Bottom)

def padded_cat(x1, x2, dim):
    x1 = pad2d_as(x1, x2)
    x1 = torch.cat([x1, x2], dim=dim)

    return x1


class Encoder(nn.Module):

    def __init__(self, config, leaky_slope):
        super(Encoder, self).__init__()

        self.conv = CConvWrapper(nn.Conv2d, *config, bias=False)
        self.bn = CBatchNorm(config[1])
        self.leaky_relu = CLeakyReLU(leaky_slope, inplace=True)

    def forward(self, input_real, input_img):
        output_real, output_img = self.leaky_relu(*self.bn(*self.conv(input_real, input_img)))
        return output_real, output_img


class Decoder(nn.Module):
    def __init__(self, config, leaky_slope):
        super(Decoder, self).__init__()

        self.deconv = CConvWrapper(nn.ConvTranspose2d, *config, bias=False)
        self.bn = CBatchNorm(config[1])
        self.leaky_relu = CLeakyReLU(leaky_slope, inplace=True)

    def forward(self, input_real, input_img, skip=None):
        if skip is not None:
            input_real = padded_cat(input_real, skip[0], dim=1)
            input_img = padded_cat(input_img, skip[1], dim=1)

        output_real, output_img = self.leaky_relu(*self.bn(*self.deconv(input_real, input_img)))

        return output_real, output_img


class DCUnet(nn.Module):

    def __init__(self, config):
        super(DCUnet, self).__init__()

        self.encoders = nn.ModuleList()
        for encoder_config in config['encoders']:
            self.encoders.append(Encoder(encoder_config, config['leaky_slope']))

        self.decoders = nn.ModuleList()
        for decoder_config in config['decoders'][:-1]:
            self.decoders.append(Decoder(decoder_config, config['leaky_slope']))

        # Last Decoder는 BN & RLEU 사용 X. Use Bias
        self.last_decoder = CConvWrapper(nn.ConvTranspose2d, *config['decoders'][-1], bias=True)

        self.ratio_mask_tpye = config['ratio_mask']

    def get_ratio_mask(self, out_real, out_img):
        def inner_fn(real, img):
            if self.ratio_mask_tpye == 'BDSS':
                # Paper 2page)
                return torch.sigmoid(out_real) * real, torch.sigmoid(out_img) * img

            else:
                mag_mask = torch.sqrt(out_real ** 2 + out_img ** 2) # magnitude 공식
                # 아크 탄젠트(input, other): input/other
                phase_rotate = torch.atan2(out_img, out_real)
                # 각도(phase) = arctan(Y(img) / X(real))


                if self.ratio_mask_tpye == 'BDT':
                    mag_mas = torch.tanh(mag_mask)

                # or UBD(UnBounded)
                mag = mag_mask * torch.sqrt(real ** 2 + img ** 2)
                phase = phase_rotate + torch.atan2(img, real)

                # return Real, Img
                return mag * torch.cos(phase), mag * torch.sin(phase) # 오일러 공식

        return inner_fn

    def forward(self, xr, xi):
        input_real, input_img = xr, xi # residual을 위햐서 따로 선언
        skips = list()

        for encoder in self.encoders:
            xr, xi = encoder(xr, xi)
            skips.append((xr, xi))

        skip = skips.pop()
        skip = None # 첫번째거 어차피 decoder input이므로 삭제

        for decoder in self.decoders:
            xr, xi = decoder(xr, xi, skip)
            skip = skips.pop()

        xr = padded_cat(xr, skip[0], dim=1)
        xi = padded_cat(xi, skip[1], dim=1)

        xr, xi = self.last_decoder(xr, xi)

        # residual 전 같은 사이즈로 되게 padd
        xr = pad2d_as(xr, input_real)
        xi = pad2d_as(xi, input_img)

        ratio_mask_fn = self.get_ratio_mask(xr, xi) # 사실상 inner_fn 객체

        out_real, out_img = ratio_mask_fn(input_real, input_img) # residual

        return out_real, out_img































