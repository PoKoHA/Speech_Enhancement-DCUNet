import os
import json
import random
import argparse
import numpy as np
import warnings
import soundfile as sf
import os
import time
import datetime

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ExponentialLR


import Levenshtein as Lev
# Levenshtein distance: 두 시퀀스 간의 차이를 측정하기 위한 문자열 메트릭

from data.dataset import *
from data.data_loader import *
from data.sampler import *

from model.DCUnet import DCUnet
from losses import *
from ISTFT import ISTFT


parser = argparse.ArgumentParser()
# Datasets
parser.add_argument('--train-file', type=str,
                    help='data list about train dataset', default='dataset/train.json')
parser.add_argument('--test-file-list',
                    help='data list about test dataset', default=['dataset/test.json'])
parser.add_argument('--dataset-path', default='dataset', help='Target dataset path')

# Hyperparameters
parser.add_argument('--epochs', type=int, default=100, help='Number of max epochs in training (default: 100)')
parser.add_argument('--start-epoch', type=int, default=0)
parser.add_argument('--batch-size', type=int, default=32, help='Batch size in training (default: 32)')
parser.add_argument('--num-workers', type=int, default=4, help='Number of workers in dataset loader (default: 4)')
parser.add_argument('--gpu', type=int, default=None)
parser.add_argument('--num-gpu', type=int, default=1, help='Number of gpus (default: 1)')
parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate (default: 3e-4)')
# Audio Config
parser.add_argument('--sample-rate', default=16000, type=int, help='Sampling Rate')
parser.add_argument('--window-size', default=.02, type=float, help='Window size for spectrogram')
parser.add_argument('--window-stride', default=.01, type=float, help='Window stride for spectrogram')
# System
parser.add_argument('--print-freq', default=1, type=int)
parser.add_argument('--resume', default=None, type=str, metavar='PATH')
parser.add_argument('--seed', type=int, default=None, help='random seed (default: None)')
parser.add_argument('--mode', type=str, default='train', help='Train or Test')


CONFIG = {
        "leaky_slope" : 0.1,
        "ratio_mask" : "BDT",
        "encoders" : [
            [1, 32, [7, 5], [2, 2], [3, 2]],
            [32, 32, [7, 5], [2, 1], [3, 2]],
            [32, 64, [7, 5], [2, 2], [3, 2]],
            [64, 64, [5, 3], [2, 1], [2, 1]],
            [64, 64, [5, 3], [2, 2], [2, 1]],
            [64, 64, [5, 3], [2, 1], [2, 1]],
            [64, 64, [5, 3], [2, 2], [2, 1]],
            [64, 64, [5, 3], [2, 1], [2, 1]]
        ],
        "decoders" : [
            [64, 64, [5, 3], [2, 1], [2, 1]],
            [128, 64, [5, 3], [2, 2], [2, 1]],
            [128, 64, [5, 3], [2, 1], [2, 1]],
            [128, 64, [5, 3], [2, 2], [2, 1]],
            [128, 64, [5, 3], [2, 1], [2, 1]],
            [128, 32, [7, 5], [2, 2], [3, 2]],
            [64, 32, [7, 5], [2, 1], [3, 2]],
            [64, 1, [7, 5], [2, 2], [3, 2]]
        ]
}

def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    ngpus_per_node = torch.cuda.device_count()

    main_worker(args.gpu, ngpus_per_node, args)

def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu

    # model
    model = DCUnet(config=CONFIG)

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if not torch.cuda.is_available():  # GPU가 없을 시
        print('using CPU, this will be slow')

    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel은 사용가능한 gpu에다가 batchsize을 나누고 할당
        model = nn.DataParallel(model).cuda(args.gpu)

    if args.resume:
        model.load_state_dict(torch.load(args.resume))

    cudnn.benchmark = True

    # audio 설정
    audio_conf = dict(sample_rate=args.sample_rate,  # 16,000
                      window_size=args.window_size,  # .02
                      window_stride=args.window_stride)  # .01

    n_fft = int(audio_conf['sample_rate'] * audio_conf['window_size']) # 320
    window_size = n_fft
    stride_size = int(audio_conf['sample_rate'] * audio_conf['window_stride']) # 160
    window = torch.hann_window(n_fft).cuda(args.gpu)

    stft = lambda x: torch.stft(x, n_fft=n_fft, hop_length=stride_size, win_length=window_size,
                                window=window)
    Istft = ISTFT(n_fft, stride_size, window='hanning').cuda(args.gpu)

    # Train dataset/ loader

    batch_size = args.batch_size * args.num_gpu # 32 * 1

    trainData_list = []
    with open(args.train_file, 'r', encoding='utf-8') as f: # train_file: data/train.json
        trainData_list = json.load(f)

    train_dataset = MelFilterBankDataset(data_list=trainData_list, mode='train')
    train_sampler = BucketingSampler(data_source=train_dataset, batch_size=batch_size)
    train_loader = AudioDataLoader(train_dataset, num_workers=args.num_workers, batch_sampler=train_sampler, pin_memory=True)

    # Test dataset/ loader
    testLoader_dict = {}
    for test_file in args.test_file_list:  # ['data/test.json']
        testData_list = []
        with open(test_file, 'r', encoding='utf-8') as f:
            testData_list = json.load(f)
            # print(testData_list)
            # [{"wav": "....", "text":, ...., speaker_id: ....}]

    test_dataset = MelFilterBankDataset(data_list=testData_list, mode="test")
    testLoader_dict[test_file] = AudioDataLoader(test_dataset, batch_size=1, num_workers=args.num_workers,
                                                 pin_memory=True)

    # optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = ExponentialLR(optimizer, 0.95)

    # train

    for epoch in range(args.start_epoch, args.epochs):
        train(args, train_loader, train_sampler, model, optimizer, scheduler, stft, Istft, epoch)

        torch.save(model.state_dict(), "saved_models_%d.pth" % (epoch + 1))


def train(args, train_data_loader, train_sampler, model, optimizer, scheduler, stft, Istft, epoch):

    model.train()
    iter = 0
    for i, (input) in enumerate(train_data_loader):
        optimizer.zero_grad()

        train_mixed, train_clean, seq_len, name = input
        train_mixed = train_mixed.cuda(args.gpu, non_blocking=True)
        train_clean = train_clean.cuda(args.gpu, non_blocking=True)
        seq_len = seq_len.cuda(args.gpu, non_blocking=True)

        mixed = stft(train_mixed).unsqueeze(dim=1)

        real, img = mixed[..., 0], mixed[..., 1]
        out_real, out_img = model(real, img)
        out_real = torch.squeeze(out_real, 1)
        out_img = torch.squeeze(out_img, 1)

        out_audio = Istft(out_real, out_img, train_mixed.size(1))
        out_audio = torch.squeeze(out_audio, dim=1)

        for i, len in enumerate(seq_len):
            out_audio[i, len:] = 0

        sf.write('output/%s' % name[0], out_audio[0].cpu().data.numpy()[:seq_len[0].cpu().data.numpy()], 16000)
        loss = wSDRLoss(train_mixed, train_clean, out_audio)

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}]\t'
                  'Loss {loss:.4f}\t'.format((epoch + 1), (iter + 1), loss=loss))

        loss.backward()
        optimizer.step()
        iter += 1

    scheduler.step()


if __name__ == '__main__':
    main()
















