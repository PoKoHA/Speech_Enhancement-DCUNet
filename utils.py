"""
PreTrained된 model을 이용하여 Denoising 된 파일들을 Generate
"""
import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt

import torch
import torchaudio
from model.ISTFT import ISTFT
from data.STFT import STFT

def generate_wav(model, mixed, max_len, n_fft, hop_length, args):
    model.eval()
    stft = STFT(fft_length=n_fft, hop_length=hop_length, normalized=True).cuda(args.gpu)

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

    sf.write("output/test.wav", pred.squeeze(0)[-current_len:].cpu().numpy(), samplerate=48000, format='WAV', subtype='PCM_16')


