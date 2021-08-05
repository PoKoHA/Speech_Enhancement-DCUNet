import numpy as np

import torch
import torchaudio
from torch.utils.data import Dataset

class SpeechDataset(Dataset):

    def __init__(self, args, noisy_files, clean_files, max_len, n_fft=64, hop_length=16):
        super(SpeechDataset, self).__init__()
        self.args = args

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
        x_noisy_stft = torch.stft(input=x_noisy, n_fft=self.n_fft, hop_length=self.hop_length, normalized=True)
        x_clean_stft = torch.stft(input=x_clean, n_fft=self.n_fft, hop_length=self.hop_length, normalized=True)

        # print("A: ", x_noisy_stft.size())
        # print("B: ", x_clean_stft.size())

        return x_noisy_stft, x_clean_stft
