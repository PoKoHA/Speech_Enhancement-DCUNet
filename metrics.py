from tqdm import tqdm

import torch
import torchaudio
from pypesq import pesq
from model.ISTFT import ISTFT

def pesq_score(model, dataloader, criterion, args, N_FFT, HOP_LENGTH):
    model.eval()
    test_pesq = 0.
    total_loss = 0
    istft = ISTFT(hop_length=HOP_LENGTH, n_fft=N_FFT).cuda(args.gpu)

    with torch.no_grad():
        for i, (mixed, target) in tqdm(enumerate(dataloader)):
            mixed = mixed.cuda(args.gpu)
            target = target.cuda(args.gpu)

            # test loss 구하기WW
            pred_x = model(mixed) # time domain
            loss = criterion(args, N_FFT, HOP_LENGTH, mixed, pred_x, target)
            total_loss += loss.item()

            # PESQ score 구하기
            target = istft(target)
            target = torch.squeeze(target, 1) # [batch, 1539, 214, 2]

            psq = 0.
            for idx in range(len(target)):
                clean_x_16 = torchaudio.transforms.Resample(48000, 16000).cuda(args.gpu)(target[idx, :].view(1, -1))
                pred_x_16 = torchaudio.transforms.Resample(48000, 16000).cuda(args.gpu)(pred_x[idx, :].view(1, -1))

                clean_x_16 = clean_x_16.cpu().numpy()
                pred_x_16 = pred_x_16.detach().cpu().numpy()

                psq += pesq(clean_x_16.flatten(), pred_x_16.flatten(), 16000)

            psq /= len(target)
            test_pesq += psq

        test_pesq /= len(dataloader)
        loss_avg = total_loss / len(dataloader)

    return test_pesq, loss_avg
