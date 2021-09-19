import torch
# from model.ISTFT import *

# Time-domain waveform 축에서 수행
import torch
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


