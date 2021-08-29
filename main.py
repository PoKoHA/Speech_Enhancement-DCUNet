import argparse
import warnings
import os
import random
import numpy as np
import time
import datetime
from pathlib import Path
from tqdm import tqdm
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim as optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed

from torch.autograd import Variable

from data.dataset import SpeechDataset
from model.DCUNet import *
from model.Discriminator import Discriminator
from utils.losses import *
from utils.metrics import pesq_score
from utils.utils import *

warnings.filterwarnings(action='ignore')

parser = argparse.ArgumentParser()

parser.add_argument('--epochs', type=int, default=100, help='Number of max epochs in training (default: 100)')
parser.add_argument('--start-epoch', type=int, default=0)
parser.add_argument("--decay_epoch", type=int, default=50, help="epoch from which to start lr decay")
parser.add_argument('--workers', type=int, default=4, help='Number of workers in dataset loader (default: 4)')
parser.add_argument('--batch-size', type=int, default=32, help='Batch size in training (default: 32)')
parser.add_argument('--lr', default=1e-4)
parser.add_argument('--lr-decay', default=0.1)
parser.add_argument('--arch', type=str, default="DCUnet10")

parser.add_argument('--clean-train-dir', type=str, default="dataset/56spk/clean_trainset_56spk_wav")
parser.add_argument('--noisy-train-dir', type=str, default="dataset/56spk/noisy_trainset_56spk_wav")
parser.add_argument('--clean-valid-dir', type=str, default="dataset/56spk/clean_validset_56spk_wav")
parser.add_argument('--noisy-valid-dir', type=str, default="dataset/56spk/noisy_validset_56spk_wav")
parser.add_argument('--clean-test-dir', type=str, default="dataset/clean_testset_wav")
parser.add_argument('--noisy-test-dir', type=str, default="dataset/noisy_testset_wav")

# Transformer
parser.add_argument('--sample-rate', type=int, default=48000, help="STFT hyperparam")
parser.add_argument('--max-len', type=int, default=165000)
parser.add_argument('--hidden-dim', type=int, default=512)
parser.add_argument('--n-head', type=int, default=3)
parser.add_argument('--dropout', default=0.1, type=float)
parser.add_argument('--warm-steps', default=4000, type=int)

parser.add_argument('--gpu', type=int, default=None)
parser.add_argument('--print-freq', type=int, default=1)
parser.add_argument('--seed', type=int, default=None, help='random seed (default: None)')
parser.add_argument('--resume', default=None, type=str, metavar='PATH', help="model_args.resume")
parser.add_argument('--resume-D-R', default=None, type=str, metavar='PATH', help="model_args.resume")
parser.add_argument('--resume-D-I', default=None, type=str, metavar='PATH', help="model_args.resume")
parser.add_argument('--evaluate', '-e', default=False, action='store_true')

# generate
parser.add_argument('--generate', '-g', default=False, action='store_true')
parser.add_argument('--denoising-file', type=str, help="denoising 하고 싶은 파일경로")

# Distributed
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor

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

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu

    # STFT 인자
    sampling_rate = args.sample_rate
    N_FFT = sampling_rate * 64 // 1000 + 4
    # N_FFT = int(.02 * args.sample_rate)

    HOP_LENGTH = sampling_rate * 16 // 1000 + 4
    # HOP_LENGTH = int(.01 * args.sample_rate)
    # print(HOP_LENGTH)

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)

    # Model
    # if args.arch == 'DCUnet10': # 거의 사용 X
    #     model = DCUNet10(args=args, n_fft=N_FFT, hop_length=HOP_LENGTH)
    #     print("DCUNET10")
    # else:
    model = DCUNet16(args=args, n_fft=N_FFT, hop_length=HOP_LENGTH)
    D_real = Discriminator(input_shape=(1, 1539, 214))
    D_imag = Discriminator(input_shape=(1, 1539, 214))
    print("DCUNET16")

    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
    elif args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            print("this")
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            D_real.cuda(args.gpu)
            D_imag.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
            D_real = torch.nn.parallel.DistributedDataParallel(D_real, device_ids=[args.gpu], find_unused_parameters=True)
            D_imag = torch.nn.parallel.DistributedDataParallel(D_imag, device_ids=[args.gpu], find_unused_parameters=True)

        else:
            model.cuda()
            D_real.cuda()
            D_imag.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
            D_real = torch.nn.parallel.DistributedDataParallel(D_real)
            D_imag = torch.nn.parallel.DistributedDataParallel(D_imag)

    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
        D_real = D_real.cuda(args.gpu)
        D_imag = D_imag.cuda(args.gpu)

    else:
        model = torch.nn.DataParallel(model).cuda()
        D_real = torch.nn.DataParallel(D_real).cuda()
        D_imag = torch.nn.DataParallel(D_imag).cuda()

    # Optimizer / criterion(wSDR)
    criterion = SISNRLoss(args=args, n_fft=N_FFT, hop_length=HOP_LENGTH).cuda(args.gpu)
    criterion_D = nn.MSELoss().cuda(args.gpu)

    optimizer = ScheduleAdam(
        torch.optim.Adam(model.parameters(), betas=(0.9, 0.98), eps=1e-9),
        hidden_dim=args.hidden_dim,
        warm_steps=args.warm_steps)
    optimizer_D_real = optim.Adam(D_real.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_D_imag = optim.Adam(D_imag.parameters(), lr=0.0002, betas=(0.5, 0.999))

    scheduler_D_real = optim.lr_scheduler.LambdaLR(
        optimizer_D_real, lr_lambda=LambdaLR(args.epochs, args.start_epoch, args.decay_epoch).step)
    scheduler_D_imag = optim.lr_scheduler.LambdaLR(
        optimizer_D_imag, lr_lambda=LambdaLR(args.epochs, args.start_epoch, args.decay_epoch).step)

    # Resume
    if args.resume:
        assert args.resume_D_R and args.resume_D_I is not None, "resume-D도 설정"
        model.load_state_dict(torch.load(args.resume, map_location="cuda:0"))
        D_real.load_state_dict(torch.load(args.resume_D_R, map_location="cuda:0"))
        D_imag.load_state_dict(torch.load(args.resume_D_I, map_location="cuda:0"))
        # 만약 Dataparallel 으로 저장했을 시 이 코드 사용
        # stat_dict = torch.load(args.resume, map_location="cuda:0")
        #
        # new_state_dict = OrderedDict()
        # for k, v in stat_dict.items():
        #     name = k[7:] # remove 'module'
        #     new_state_dict[name] = v
        #
        # model.load_state_dict(new_state_dict)

    # generate wav file
    if args.generate:
        generate_wav(model, args.denoising_file, args.max_len, N_FFT, HOP_LENGTH, args)
        print("Generate Denoising File")
        return

    # Dataset path
    mixed_train_dir = Path(args.noisy_train_dir)
    clean_train_dir = Path(args.clean_train_dir)

    mixed_valid_dir = Path(args.noisy_valid_dir)
    clean_valid_dir = Path(args.clean_valid_dir)

    mixed_test_dir = Path(args.noisy_test_dir)
    clean_test_dir = Path(args.clean_test_dir)

    # 파일 리스트
    mixed_train_files = sorted(list(mixed_train_dir.rglob('*.wav')))
    clean_train_files = sorted(list(clean_train_dir.rglob('*.wav')))

    mixed_valid_files = sorted(list(mixed_valid_dir.rglob('*.wav')))
    clean_valid_files = sorted(list(clean_valid_dir.rglob('*.wav')))

    mixed_test_files = sorted(list(mixed_test_dir.rglob('*.wav')))
    clean_test_files = sorted(list(clean_test_dir.rglob('*.wav')))

    # Dataset
    train_dataset = SpeechDataset(args, mixed_train_files, clean_train_files, args.max_len, N_FFT, HOP_LENGTH)
    valid_dataset = SpeechDataset(args, mixed_valid_files, clean_valid_files, args.max_len, N_FFT, HOP_LENGTH)
    test_dataset = SpeechDataset(args, mixed_test_files, clean_test_files, args.max_len, N_FFT, HOP_LENGTH)

    # Sampler
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        print("Sampler")
    else:
        train_sampler = None

    # DataLoader
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                               shuffle=(train_sampler is None),
                                               num_workers=args.workers, pin_memory=True, sampler=train_sampler)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False,
                                               num_workers=args.workers, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False,
                                              num_workers=args.workers, pin_memory=True)

    # Evaluate
    if args.evaluate:
        PESQ, loss = validate(test_loader, model, criterion, N_FFT, HOP_LENGTH, args,
                              D_real, D_imag, criterion_D)

        print(f"loss: {loss:.4f} | PESQ: {PESQ:.4f}".format(
            loss=loss, PESQ=PESQ
        ))
        return

    # Train
    best_PESQ = -1e10

    for epoch in range(args.start_epoch, args.epochs):

        if args.distributed:
            train_sampler.set_epoch(epoch)

        train(train_loader, model, criterion, optimizer, epoch, N_FFT, HOP_LENGTH, args,
              D_real, D_imag, criterion_D, optimizer_D_real, optimizer_D_imag, scheduler_D_imag, scheduler_D_real)

        print("--validate--")
        PESQ, loss = validate(valid_loader, model, criterion, N_FFT, HOP_LENGTH, args,
                              D_real, D_imag, criterion_D)

        print(f"loss: {loss:.4f} | PESQ: {PESQ:.4f}".format(
            loss=loss, PESQ=PESQ
        ))

        if best_PESQ < PESQ: # 현재 PESQ 더 클시
            print("Found better validated model")
            torch.save(model.state_dict(), "saved_models/model_%d.pth" % (epoch + 1))
            torch.save(D_real.state_dict(), "saved_models/D_R_%d.pth" % (epoch + 1))
            torch.save(D_imag.state_dict(), "saved_models/D_I_%d.pth" % (epoch + 1))

            best_PESQ = PESQ


def train(train_loader, model, criterion, optimizer, epoch, n_fft, hop_length, args,
          D_real, D_imag, criterion_D, optimizer_D_real, optimizer_D_imag, scheduler_D_imag, scheduler_D_real):

    model.train()
    D_real.train()
    D_imag.train()

    end = time.time()
    # Dataset return x_noisy_stft, x_clean_stft

    for i, (mixed, target) in enumerate(train_loader):

        mixed = mixed.cuda(args.gpu) # noisy
        target = target.cuda(args.gpu)# Clean

        valid = Variable(Tensor(np.ones((mixed.size(0), *(1, 96, 13)))), requires_grad=False)
        fake = Variable(Tensor(np.zeros((mixed.size(0), *(1, 96, 13)))), requires_grad=False)

        # DCUnet Loss
        est_wav, est_spec = model(mixed, target=target) # DCUnet
        si_snr_loss = criterion(est_wav, target)


        optimizer.zero_grad()
        si_snr_loss.backward()
        optimizer.step()

        # Discriminator_Real
        # print("GT", target.size())
        # print("pred", pred_spec.size())
        optimizer_D_real.zero_grad()
        dis_fake_loss = criterion_D(D_real(est_spec[..., 0].detach()), fake)
        dis_real_loss = criterion_D(D_real(target[..., 0]), valid)

        loss_D_R = (dis_real_loss + dis_fake_loss) / 2

        loss_D_R.backward()
        optimizer_D_real.step()

        # Discriminator_Imag
        optimizer_D_imag.zero_grad()
        dis_fake_loss = criterion_D(D_imag(est_spec[..., 1].detach()), fake)
        dis_real_loss = criterion_D(D_imag(target[..., 1]), valid)

        loss_D_I = (dis_real_loss + dis_fake_loss) / 2

        loss_D_I.backward()
        optimizer_D_imag.step()

        loss = si_snr_loss + loss_D_R + loss_D_I

        if i % args.print_freq == 0:
            print(" Epoch [%d][%d/%d] | D_loss_R: %f | D_loss_I: %f | si_snr_loss: %f | Loss: %f"
                  % (epoch + 1, i, len(train_loader), loss_D_R, loss_D_I, si_snr_loss, loss))

    scheduler_D_real.step()
    scheduler_D_imag.step()

    elapse = datetime.timedelta(seconds=time.time() - end)
    print(f"걸린 시간: ", elapse)


def validate(dataloader, model, criterion, n_fft, hop_length, args,
             D_real, D_imag, criterion_D):

    model.eval()
    D_real.eval()
    D_imag.eval()
    # loss와 score를 동시에 구하는 함수로 대체하였음
    score, loss_avg = pesq_score(model, dataloader, criterion, args, n_fft, hop_length,
                                 D_real, D_imag, criterion_D)

    return score, loss_avg


if __name__ == "__main__":
    main()














