"""
PreTrained된 model을 이용하여 Denoising 된 파일들을 Generate
"""
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import os

import torch
import torchaudio
from model.ISTFT import ISTFT
from data.STFT import STFT

def generate_wav(model, mixed, max_len, n_fft, hop_length, args):
    model.eval()
    stft = STFT(fft_length=n_fft, hop_length=hop_length, normalized=True).cuda(args.gpu)
    file_list = os.listdir(args.denoising_file)
    for idx in range(len(file_list)):
        name = file_list[idx]
        mixed = os.path.join(args.denoising_file, name)
        waveform, _ = torchaudio.load(mixed)
        # plt.figure(figsize=(15, 5))
        # plt.plot(waveform.squeeze(0).cpu().numpy())
        # plt.title("Origin")
        # plt.show()

        waveform = waveform.numpy()

        current_len = waveform.shape[1]
        pad = np.zeros((1, max_len), dtype='float32')
        pad[0, -current_len:] = waveform[0, :max_len]

        input = torch.from_numpy(pad).cuda(args.gpu)

        # plt.figure(figsize=(15, 5))
        # plt.plot(input.squeeze(0).cpu().numpy())
        # plt.show()
        input_stft = stft(input)

        input_stft = input_stft.unsqueeze(0) # batch
        with torch.no_grad():
            d0 = model.downsample0(input_stft)
            d0_attn = model.ccbam_0(d0)

            d1 = model.downsample1(d0)
            d1_attn = model.ccbam_1(d1)

            d2 = model.downsample2(d1)
            d2_attn = model.ccbam_2(d2)

            d3 = model.downsample3(d2)
            d3_attn = model.ccbam_3(d3)

            d4 = model.downsample4(d3)
            d4_attn = model.ccbam_4(d4)

            d5 = model.downsample5(d4)
            d5_attn = model.ccbam_5(d5)

            d6 = model.downsample6(d5)
            d6_attn = model.ccbam_6(d6)

            d7 = model.downsample7(d6)
            d7_attn = model.ccbam_7(d7)
            # bridge 첫번째 Decoder에 skip connection X
            # skip-connection
            u0 = model.upsample0(d7_attn)
            c0 = torch.cat((u0, d6_attn), dim=1)

            u1 = model.upsample1(c0)
            c1 = torch.cat((u1, d5_attn), dim=1)

            u2 = model.upsample2(c1)
            c2 = torch.cat((u2, d4_attn), dim=1)

            u3 = model.upsample3(c2)
            c3 = torch.cat((u3, d3_attn), dim=1)

            u4 = model.upsample4(c3)
            c4 = torch.cat((u4, d2_attn), dim=1)

            u5 = model.upsample5(c4)
            c5 = torch.cat((u5, d1_attn), dim=1)

            u6 = model.upsample6(c5)
            c6 = torch.cat((u6, d0_attn), dim=1)

            u7 = model.upsample7(c6)

            decoder_input = u7 * input_stft

            pred, _ = model(input_stft, decoder_input)

        name = "predict_" + name
        output = os.path.join("output", name)
        sf.write(output, pred.squeeze(0)[-current_len:].cpu().numpy(), samplerate=48000, format='WAV', subtype='PCM_16')


##############################################
def display_feature(input, title):
    x = input[0]
    size = x.size(0) # Channel
    plt.figure(figsize=(50, 10))
    plt.title(title)
    x = x.cpu().detach().numpy()
    x = x[:, ::-1, :]
    a = 0
    for i in range(size):
        a += x[i]
    # plt.subplot(16, 32, i+1)
    plt.imshow(a, cmap='gray')
    plt.axis("off")

    plt.show()
    plt.close()


##################################################
# Discriminator Scheduler

class LambdaLR:
    def __init__(self, epochs, offset, decay_start_epoch):
        assert (epochs - decay_start_epoch) > 0, "전체 epoch가 decay_start_epoch보다 커야함"

        self.num_epochs = epochs # 설정한 총 epoch
        self.offset = offset # (저장했었던) start epoch
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch): # epoch : 현재 epoch
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch) / (self.num_epochs - self.decay_start_epoch)


####################################################
# Transformer ScheduleAdam
####################################################
class ScheduleAdam():

    def __init__(self, optimizer, hidden_dim, warm_steps):
        self.init_lr = np.power(hidden_dim, -0.5) # 1 / sqrt(hidden_dim)
        self.optimizer = optimizer
        self.current_steps = 0
        self.warm_steps = warm_steps

    def step(self):
        # current_step 정보를 이용해서 lr Update
        self.current_steps += 1
        lr = self.init_lr * self.get_scale()

        for p in self.optimizer.param_groups:
            p['lr'] = lr # lr Update

        self.optimizer.step()

    def zero_grad(self):
        self.optimizer.zero_grad()

    def get_scale(self):
        return np.min([np.power(self.current_steps, -0.5),
                       self.current_steps * np.power(self.warm_steps, -1.5)
                       ])


