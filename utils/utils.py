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
        # print(waveform.size())
        waveform = waveform.numpy()

        current_len = waveform.shape[1]
        pad = np.zeros((1, max_len), dtype='float32')
        pad[0, -current_len:] = waveform[0, :max_len]

        input = torch.from_numpy(pad).cuda(args.gpu)
        # print(input.size())
        # plt.figure(figsize=(15, 5))
        # plt.plot(input.squeeze(0).cpu().numpy())
        # plt.show()
        input_stft = stft(input)
        # print(input_stft.size())
        input_stft = input_stft.unsqueeze(0) # batch
        with torch.no_grad():
            pred = model(input_stft)

        # plt.figure(figsize=(15,5))
        # plt.plot(pred.squeeze(0)[-current_len:].cpu().numpy())
        # plt.title("Denoising")
        # plt.show()
        # print("A", pred.size())
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


