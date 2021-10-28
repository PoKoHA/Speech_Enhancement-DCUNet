from tqdm import tqdm
import numpy as np

import torch
import torchaudio
from pypesq import pesq
from model.ISTFT import ISTFT


def pesq_score(model, dataloader, criterion, args, N_FFT, HOP_LENGTH, summary, epoch):
    model.eval()
    test_pesq = 0.
    total_loss = 0
    # istft = ISTFT(hop_length=HOP_LENGTH, n_fft=N_FFT).cuda(args.gpu)
    with torch.no_grad():
        total_nan = 0
        for i, (mixed, target) in tqdm(enumerate(dataloader)):
            mixed = mixed.cuda(args.gpu)
            target = target.cuda(args.gpu)

            # test loss 구하기WW
            pred = model(mixed) # time domain
            loss = criterion(pred, target)
            # total_loss += loss.item()
            niter = epoch * len(dataloader) + i
            summary.add_scalar('Valid/loss', loss.item(), niter)
            # PESQ score 구하기
            # target = istft(target)
            target = torch.squeeze(target, 1) # [batch, 1539, 214, 2]
            pred = torch.squeeze(pred, 1) # [batch, 1539, 214, 2]

            psq = 0.
            nan = 0

            for idx in range(len(target)):
                clean_x_16 = torchaudio.transforms.Resample(48000, 16000).cuda(args.gpu)(target[idx, :].view(1, -1))
                pred_x_16 = torchaudio.transforms.Resample(48000, 16000).cuda(args.gpu)(pred[idx, :].view(1, -1))

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
        summary.add_scalar('Valid/pesq', test_pesq, epoch)
    return test_pesq, loss_avg
