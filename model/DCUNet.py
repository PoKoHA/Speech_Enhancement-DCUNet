"""
complex-valued convolutional filter W = A+iB
complex vector h = x + iy

W ∗h = (A ∗ x − B ∗ y) + i(B ∗ x+ A ∗ y)
"""
import matplotlib.pyplot as plt
import librosa

import torch
import torch.nn as nn

from utils.utils import display_feature

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
            # display_feature(Trans_cConv[..., 0], "Decoder_8_real")
            # display_feature(Trans_cConv[..., 1], "Decoder_8_imag")

            mask_phase = Trans_cConv / (torch.abs(Trans_cConv) + 1e-8)
            # print("mask_ph: ", mask_phase.size())
            mask_mag = torch.tanh(torch.abs(Trans_cConv))
            # print("mask_mag: ", mask_mag.size())
            output = mask_phase * mask_mag # [batch, channel, 1539, 214, 2 ]
            # real = output[..., 0]
            # imag = output[..., 1]
            # mag = torch.abs(torch.sqrt(real ** 2 + imag ** 2))
            # phase = torch.atan2(imag, real)
            #
            # real_db = librosa.amplitude_to_db(real.cpu().detach().numpy())
            # imag_db = librosa.amplitude_to_db(imag.cpu().detach().numpy())
            # phase_db = librosa.amplitude_to_db(phase.cpu().detach().numpy())
            # mag_db = librosa.amplitude_to_db(mag.cpu().detach().numpy())

            #display_spectrogram(real_db, "mask_Real")
            #display_spectrogram(imag_db, "mask_Imag")
            #display_spectrogram(mag_db, "mask_mag")
            #display_spectrogram(phase_db, "mask_phase")

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
        # self.attn = MultiHeadAttention(args=args)

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
        # # print("input: ", x.size())
        d0 = self.downsample0(x)
        d1 = self.downsample1(d0)
        d2 = self.downsample2(d1)
        d3 = self.downsample3(d2)
        d4 = self.downsample4(d3)

        # bridge 첫번째 Decoder에 skip connection X
        u0 = self.upsample0(d4)

        # skip-connection
        # # print("d3: ", d3.size())
        # # print("u0: ", u0.size())
        c0 = torch.cat((u0, d3), dim=1)

        u1 = self.upsample1(c0)
        # # print("d2: ", d2.size())
        # # print("u1: ", u1.size())
        c1 = torch.cat((u1, d2), dim=1)

        u2 = self.upsample2(c1)
        # # print("d1: ", d1.size())
        # # print("u2: ", u2.size())
        c2 = torch.cat((u2, d1), dim=1)

        u3 = self.upsample3(c2)
        # # print("d0: ", d0.size())
        # # print("u3: ", u3.size())
        c3 = torch.cat((u3, d0), dim=1)

        u4 = self.upsample4(c3)

        # # print("x: ", x.size())
        # # print("mask_hat: ", mask_hat.size())

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
        # self.transformer = Transformer(args=args)

        self.linear_q = nn.Linear(1, 128)
        self.linear_k = nn.Linear(1, 128)
        self.linear_v = nn.Linear(1, 128)
        # self.real_attn = MultiHeadAttention(args=args)
        #
        # self.imag_attn = MultiHeadAttention(args=args)
        #
        # self.target_real_attn = MultiHeadAttention(args=args)
        # self.target_imag_attn = MultiHeadAttention(args=args)
        #
        # self.real_cross_attn = MultiHeadAttention(args=args)
        # self.target_cross_attn = MultiHeadAttention(args=args)
        # self.real_attn = SpatialGate()
        # self.imag_attn = SpatialGate()

        # self.real_transformer = SpeechTransformer(args=args).cuda(args.gpu)
        # self.imag_transformer = SpeechTransformer(args=args).cuda(args.gpu)
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


    def forward(self, x, target, is_istft=True):
        # downsampling/encoding
        # print("   --[Encoder]-- ")
        # print("       Input(spec): ", x.size())
        # display_feature(x[..., 0], "input_real")
        # display_feature(x[..., 1], "input_imag")
        d0 = self.downsample0(x)
        # display_feature(d0[..., 0], "Encoder_1_real")
        # display_feature(d0[..., 1], "Encoder_1_imag")
        # print("   d0: ", d0.size())
        d1 = self.downsample1(d0)
        # display_feature(d1[..., 0], "Encoder_2_real")
        # display_feature(d1[..., 1], "Encoder_2_imag")
        # print("   d1: ", d1.size())
        d2 = self.downsample2(d1)
        # display_feature(d2[..., 0], "Encoder_3_real")
        # display_feature(d2[..., 1], "Encoder_3_imag")
        # print("   d2: ", d2.size())
        d3 = self.downsample3(d2)
        # display_feature(d3[..., 0], "Encoder_4_real")
        # display_feature(d3[..., 1], "Encoder_4_imag")
        # print("   d3: ", d3.size())
        d4 = self.downsample4(d3)
        # display_feature(d4[..., 0], "Encoder_5_real")
        # display_feature(d4[..., 1], "Encoder_5_imag")
        # print("   d4: ", d4.size())
        d5 = self.downsample5(d4)
        # display_feature(d5[..., 0], "Encoder_6_real")
        # display_feature(d5[..., 1], "Encoder_6_imag")
        # print("   d5: ", d5.size())
        d6 = self.downsample6(d5)
        # display_feature(d6[..., 0], "Encoder_7_real")
        # display_feature(d6[..., 1], "Encoder_7_imag")
        # print("   d6: ", d6.size())
        d7 = self.downsample7(d6)
        # display_feature(d7[..., 0], "Encoder_8_real")
        # display_feature(d7[..., 1], "Encoder_8_imag")
        # print("   d7: ", d7.size())

        # print("   --[Decoder]-- ")
        # bridge 첫번째 Decoder에 skip connection X
        u0 = self.upsample0(d7)
        # display_feature(u0[..., 0], "Decoder_1_real")
        # display_feature(u0[..., 1], "Decoder_1_imag")

        # skip-connection
        c0 = torch.cat((u0, d6), dim=1)
        # print("   u0: ", u0.size())
        # print("   concat(u0,d6): ", c0.size())

        u1 = self.upsample1(c0)
        # display_feature(u1[..., 0], "Decoder_2_real")
        # display_feature(u1[..., 1], "Decoder_2_imag")
        c1 = torch.cat((u1, d5), dim=1)
        # print("   u1: ", u1.size())
        # print("   concat(u1,d5): ", c1.size())

        u2 = self.upsample2(c1)
        # display_feature(u2[..., 0], "Decoder_3_real")
        # display_feature(u2[..., 1], "Decoder_3_imag")
        c2 = torch.cat((u2, d4), dim=1)
        # print("   u2: ", u2.size())
        # print("   concat(u2,d4): ", c2.size())

        u3 = self.upsample3(c2)
        # display_feature(u3[..., 0], "Decoder_4_real")
        # display_feature(u3[..., 1], "Decoder_4_imag")
        c3 = torch.cat((u3, d3), dim=1)
        # print("   u3: ", u3.size())
        # print("   concat(u3,d3): ", c3.size())

        u4 = self.upsample4(c3)
        # display_feature(u4[..., 0], "Decoder_5_real")
        # display_feature(u4[..., 1], "Decoder_5_imag")
        c4 = torch.cat((u4, d2), dim=1)
        # print("   u4: ", u4.size())
        # print("   concat(u4,d2): ", c4.size())

        u5 = self.upsample5(c4)
        # display_feature(u5[..., 0], "Decoder_6_real")
        # display_feature(u5[..., 1], "Decoder_6_imag")
        c5 = torch.cat((u5, d1), dim=1)
        # print("   u5: ", u5.size())
        # print("   concat(u5,d1): ", c5.size())

        u6 = self.upsample6(c5)
        # display_feature(u6[..., 0], "Decoder_7_real")
        # display_feature(u6[..., 1], "Decoder_7_imag")
        c6 = torch.cat((u6, d0), dim=1)
        # print("   u6: ", u6.size())
        # print("   concat(u6,d0): ", c6.size())

        u7 = self.upsample7(c6)
        # print("   u7: ", u7.size()) # [1, 1, 1539, 214, 2]
        # print("real", real_attn.size())
        # mask = torch.stack([real_attn, imag_attn], dim=-1)
        # print(mask.size())
        # print("mask", mask.size())

        # m_db = librosa.amplitude_to_db(u7[..., 0].cpu().detach().numpy())
        # display_spectrogram(m_db, "u7")
        # m2_db = librosa.amplitude_to_db(mask[..., 0].cpu().detach().numpy())
        # display_spectrogram(m2_db, "mask")
        # mask = mask * u7
        output_spec = x * u7
        # print("pass", output.size())

        # print("x", x)
        # real = output[..., 0]
        # imag = output[..., 1]
        # print(real.size())
        # mag = torch.abs(torch.sqrt(real ** 2 + imag ** 2))
        # phase = torch.atan2(imag, real)

        # real_db = librosa.amplitude_to_db(real.cpu().detach().numpy())
        # real_mean_db = librosa.amplitude_to_db(real_pool.cpu().detach().numpy())
        # imag_db = librosa.amplitude_to_db(imag.cpu().detach().numpy())
        # phase_db = librosa.amplitude_to_db(phase.cpu().detach().numpy())
        # mag_db = librosa.amplitude_to_db(mag.cpu().detach().numpy())

        # display_spectrogram(real_db, "denoising_real")
        # display_spectrogram(real_mean_db, "denoising_real_mean")
        #display_spectrogram(imag_db, "denoising_Imag")
        #display_spectrogram(mag_db, "denoising_mag")
        #display_spectrogram(phase_db, "denoising_phase")
        # print("\n", "  Apply Mask(input * u7): ", output.size())
        if is_istft:
            # output = torch.squeeze(output, 1)
            # output = torch.istft(output, n_fft=self.n_fft, hop_length=self.hop_length, normalized=True)
            output = self.istft(output_spec) # [batch ,1 , 165974]
            output = torch.squeeze(output, 1)
            # output = output.transpose(1, 2) # [batch, 165974, 1]

            # print(output.size())

            # target = self.istft(target)
            # target = torch.squeeze(target, 1)
            # target = target.transpose(1, 2)# [batch, 165974, 1]

            # Q = self.linear_q(target) # [batch, 165974, 128]
            # print(Q.size())
            # K = self.linear_k(output)
            # V = self.linear_v(output)

            # attention = torch.bmm(Q, K.permute(0, 2, 1))


            # plt.figure(figsize=(15, 5))
            # plt.plot(output.squeeze(0).cpu().detach().numpy())
            # plt.title("denoising")
            # plt.show()

        return output, output_spec

    def gram_matrix(self, y):
        (b, ch, h, w) = y.size()
        features = y.view(b * ch, h * w)
        # print("1",features.size())
        gram = torch.mm(features, features.t())
        # print("2",gram.size())
        # mm -> 행렬곱 (비슷한 것: element-wise/아다마르곱==> element끼리 곱)
        gram = gram.div(b * ch * h * w)
        # print("3",gram.size())
        return gram

def display_spectrogram(x, title):
    plt.figure(figsize=(15, 10))
    plt.pcolormesh(x[0][0], cmap='hot') # 여기서는 Batch shape가 추가
    plt.colorbar(format="%+2.f dB")
    plt.title(title)
    plt.show()
