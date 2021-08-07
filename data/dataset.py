import numpy as np
import matplotlib.pyplot as plt

import torch
import torchaudio
from torch.utils.data import Dataset

from data.STFT import STFT

class SpeechDataset(Dataset):

    def __init__(self, args, noisy_files, clean_files, max_len, n_fft=64, hop_length=16):
        super(SpeechDataset, self).__init__()
        self.args = args

        # STFT
        self.stft = STFT(fft_length=n_fft, hop_length=hop_length, normalized=True)
        # default 로 window 는 hanning 이고 length 는 n_fft와 동일하게

        # list of files
        self.noisy_files = sorted(noisy_files)
        self.clean_files = sorted(clean_files)

        # fixed len
        self.max_len = max_len

        # stft parameters
        self.n_fft = n_fft
        self.hop_length = hop_length

        self.datasize = len(self.noisy_files)

    def __len__(self):
        return self.datasize

    def load_sample(self, file):
        waveform, sr = torchaudio.load(file)

        return waveform

    def _prepare_sample(self, waveform):
        waveform = waveform.numpy()
        current_len = waveform.shape[1]

        output = np.zeros((1, self.max_len), dtype='float32')
        output[0, -current_len:] = waveform[0, :self.max_len]
        output = torch.from_numpy(output)

        return output

    def __getitem__(self, idx):
        x_clean = self.load_sample(self.clean_files[idx])
        x_noisy = self.load_sample(self.noisy_files[idx])

        # padding / cutting
        x_clean = self._prepare_sample(x_clean)
        x_noisy = self._prepare_sample(x_noisy)

        # STFT
        x_noisy_stft = self.stft(x_noisy)
        x_clean_stft = self.stft(x_clean)

        # real = x_noisy_stft[:, :, :, 0]
        # imag = x_noisy_stft[:, :, :, 1]

        # mag = torch.abs(real)
        # plt.figure(figsize=(15, 10))
        # plt.pcolormesh(mag[0])
        # plt.colorbar(format="%+2.f dB")
        # plt.title(self.noisy_files[idx])
        # plt.show()
        # print("A: ", x_noisy_stft.size())
        # print("B: ", x_clean_stft.size())

        return x_noisy_stft, x_clean_stft
