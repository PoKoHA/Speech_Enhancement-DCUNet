from tqdm import tqdm
import numpy as np

import torch
from torch.autograd import Variable
import torchaudio

from pypesq import pesq
from model.ISTFT import ISTFT

Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor

def pesq_score(model, dataloader, criterion, args, N_FFT, HOP_LENGTH,
               D_real, D_imag, criterion_D):

    model.eval()
    D_real.eval()
    D_imag.eval()

    test_pesq = 0.
    total_loss = 0
    istft = ISTFT(hop_length=HOP_LENGTH, n_fft=N_FFT).cuda(args.gpu)

    with torch.no_grad():
        total_nan = 0
        for i, (mixed, target) in tqdm(enumerate(dataloader)):
            mixed = mixed.cuda(args.gpu)
            target = target.cuda(args.gpu)

            valid = Variable(Tensor(np.ones((mixed.size(0), *(1, 96, 13)))), requires_grad=False)
            fake = Variable(Tensor(np.zeros((mixed.size(0), *(1, 96, 13)))), requires_grad=False)

            # test loss 구하기WW
            pred_x, pred_spec = model(mixed) # time domain

            # Real Discriminator
            dis_fake_loss = criterion_D(D_real(pred_spec[..., 0].detach()), fake)
            dis_real_loss = criterion_D(D_real(target[..., 0]), valid)

            loss_D_R = (dis_real_loss + dis_fake_loss) / 2

            # Imag Discriminator
            dis_fake_loss = criterion_D(D_imag(pred_spec[..., 1].detach()), fake)
            dis_real_loss = criterion_D(D_imag(target[..., 1]), valid)

            loss_D_I = (dis_real_loss + dis_fake_loss) / 2

            # SDR
            sdr_loss = criterion(args, N_FFT, HOP_LENGTH, mixed, pred_x, target)
            # print(pred_x)
            loss = sdr_loss + loss_D_I + loss_D_R
            total_loss += loss.item()

            # PESQ score 구하기
            target = istft(target)
            target = torch.squeeze(target, 1) # [batch, 1539, 214, 2]

            psq = 0.
            nan = 0

            for idx in range(len(target)):
                clean_x_16 = torchaudio.transforms.Resample(48000, 16000).cuda(args.gpu)(target[idx, :].view(1, -1))
                pred_x_16 = torchaudio.transforms.Resample(48000, 16000).cuda(args.gpu)(pred_x[idx, :].view(1, -1))

                clean_x_16 = clean_x_16.cpu().numpy()
                pred_x_16 = pred_x_16.detach().cpu().numpy()

                # print("A", clean_x_16.flatten().size)
                # print("B", pred_x_16.flatten().shape)
                # print("i", i, "idx:", idx)
                score = pesq(clean_x_16.flatten(), pred_x_16.flatten(), 16000)

                if np.isnan(score):
                    nan += 1
                    total_nan += 1
                else:
                    psq += score
                # print("A", psq)

            # print("B", len(target) - nan)
            psq /= (len(target) - nan)
            test_pesq += psq
            # print(test_pesq)

        # print(total_nan)
        # print(len(dataloader) - total_nan)
        test_pesq /= (len(dataloader) - total_nan)
        loss_avg = total_loss / len(dataloader)

    return test_pesq, loss_avg
