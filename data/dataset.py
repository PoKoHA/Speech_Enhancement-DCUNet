import numpy as np
import matplotlib.pyplot as plt
import librosa

import torch
import torchaudio
import torchaudio.functional as F

from torch.utils.data import Dataset

from data.STFT import STFT

class SpeechDataset(Dataset):

    def __init__(self, args, noisy_files, clean_files, max_len, n_fft=64, hop_length=16):
        super(SpeechDataset, self).__init__()
        self.args = args
        self.n_fft = n_fft
        self.hop_length = hop_length

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
        # x_clean = self.load_sample(self.clean_files[idx])
        x_clean = self.load_sample("test/10_GT.wav")
        # x_noisy = self.load_sample(self.noisy_files[idx])
        x_noisy = self.load_sample("test/10.wav")

        # print(self.clean_files[idx], x_clean.size())
        # print(self.noisy_files[idx], x_noisy.size())
        # plt.figure(figsize=(15, 5))
        # plt.plot(x_clean.squeeze(0).cpu().numpy())
        # plt.title(self.clean_files[idx])
        # plt.show()
        # print(x_noisy.size())

        # padding / cutting
        x_clean = self._prepare_sample(x_clean)
        x_noisy = self._prepare_sample(x_noisy)
        # plt.figure(figsize=(15, 5))
        # plt.plot(x_clean.squeeze(0).cpu().numpy())
        # plt.title(self.clean_files[idx])
        # plt.show()
        # print(x_noisy.size())

        # STFT
        x_noisy_stft = self.stft(x_noisy)
        x_clean_stft = self.stft(x_clean)
        # print("target: ", x_clean_stft)
        # print("noisy: ", x_noisy_stft)

        real = x_noisy_stft[:, :, :, 0]
        real_gt = x_clean_stft[:, :, :, 0]
        imag = x_noisy_stft[:, :, :, 1]
        imag_gt = x_clean_stft[:, :, :, 1]
        # print(real.size())
        real_db = librosa.amplitude_to_db(real)
        real_gt_db = librosa.amplitude_to_db(real_gt)
        #display_spectrogram(real_db, "Real")
        #display_spectrogram(real_gt_db, "Real_GT")

        imag_db = librosa.amplitude_to_db(imag)
        imag_gt_db = librosa.amplitude_to_db(imag_gt)
        #display_spectrogram(imag_db, "imag")
        #display_spectrogram(imag_gt_db, "imag_GT")


        """
        비교: librosa가 소리가 없는곳에는 확실히 더 큰 -db찍히고
        비교적 transform은 비슷하게? 찍힘
        librosa 사용예정
        """
        mag = torch.abs(torch.sqrt(real**2+imag**2))
        mag_gt = torch.abs(torch.sqrt(real_gt**2+imag_gt**2))
        # transform = torchaudio.transforms.AmplitudeToDB()
        # mag_db = transform(mag)

        mag_db = librosa.amplitude_to_db(mag)
        mag_gt_db = librosa.amplitude_to_db(mag_gt)
        #display_spectrogram(mag_db, "mag")
        #display_spectrogram(mag_gt_db, "mag_GT")

        phase = torch.atan2(imag, real)
        phase_gt = torch.atan2(imag_gt, real_gt)
        phase_db = librosa.amplitude_to_db(phase)
        phase_gt_db = librosa.amplitude_to_db(phase_gt)
        #display_spectrogram(phase_db, "phase")
        #display_spectrogram(phase_gt_db, "phase_GT")

        # transform = torchaudio.transforms.AmplitudeToDB()
        # phase_db = transform(phase)

        # print("A: ", x_noisy_stft.size())
        # print("B: ", x_clean_stft.size())

        return x_noisy_stft, x_clean_stft


def display_spectrogram(x, title):
    plt.figure(figsize=(15, 10))
    plt.pcolormesh(x[0], cmap='hot')
    plt.colorbar(format="%+2.f dB")
    plt.title(title)
    plt.show()
