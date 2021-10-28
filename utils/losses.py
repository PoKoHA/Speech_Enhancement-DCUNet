# from model.ISTFT import *

# Time-domain waveform 축에서 수행
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.ISTFT import *

# Time-domain waveform 축에서 수행
def wSDR(args, n_fft, hop_length, mixed, y_pred, y_GT, eps=1e-8):
    isft = ISTFT(n_fft=n_fft, hop_length=hop_length).cuda(args.gpu)

    time_y_GT = isft(y_GT)
    time_mixed = isft(mixed)
    # print("time_y_GT: ", time_y_GT.size())
    # print("time_y_mixed: ", time_mixed.size())

    time_y_GT = torch.squeeze(time_y_GT, 1)
    time_mixed = torch.squeeze(time_mixed, 1)
    # print("reshape: ", time_y_GT.size())
    # print("reshape: ", time_mixed.size())


    time_y_pred = y_pred.flatten(1) # size는 변함없음
    time_y_GT = time_y_GT.flatten(1)
    time_mixed = time_mixed.flatten(1)
    # print("A: ", time_y_GT.size())
    # print("B: ", time_mixed.size())
    # print("c: ", time_y_pred.size())


    def SDR(GT, pred, eps=1e-8):
        num = torch.sum(GT * pred, dim=1)
        energy = torch.norm(GT, p=2, dim=1) * torch.norm(pred, p=2, dim=1)

        return -(num / (energy + eps))

    noise_GT = time_mixed - time_y_GT # noise Ground Truth
    noise_pred = time_mixed - time_y_pred

    alpha = torch.sum(time_y_GT**2, dim=1) / (torch.sum(time_y_GT**2, dim=1) + torch.sum(noise_GT**2, dim=1))

    wsdr = alpha * SDR(time_y_GT, time_y_pred, eps) + (1 - alpha) * SDR(noise_GT, noise_pred, eps)

    return torch.mean(wsdr)


def l2_norm(s1, s2):
    # norm = torch.sqrt(torch.sum(s1*s2, 1, keepdim=True))
    # norm = torch.norm(s1*s2, 1, keepdim=True)

    norm = torch.sum(s1 * s2, -1, keepdim=True)
    return norm


def si_snr(s1, s2, eps=1e-8):
    # s1: estimated time domain waveform( mask 입혀진 waveform )
    # s2: clean time domain waveform( Ground Truth 느낌 )
    s1_s2_norm = l2_norm(s1, s2)
    s2_s2_norm = l2_norm(s2, s2)
    s_target = s1_s2_norm / (s2_s2_norm + eps) * s2
    e_noise = s1 - s_target

    target_norm = l2_norm(s_target, s_target)
    noise_norm = l2_norm(e_noise, e_noise)

    snr = 10 * torch.log10((target_norm) / (noise_norm + eps) + eps)
    return torch.mean(snr)


class SISNRLoss(nn.Module):

    def __init__(self, eps=1e-8):
        super(SISNRLoss, self).__init__()
        self.eps = eps

    def forward(self, s1, s2):
        # s1: estimated time domain waveform( mask 입혀진 waveform )
        # s2: clean time domain waveform( Ground Truth 느낌 )
        s2 = torch.squeeze(s2, 1)
        # print(s2.size())
        # print("A",s1.size())
        return -(si_snr(s1, s2, eps=self.eps))

