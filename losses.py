import torch

def wSDRLoss(mixed, clean, clean_hat, eps=2e-7):
    # Time domain에서 실행
    # shape (N x T)
    bsum = lambda x: torch.sum(x, dim=1)

    def mSDRLoss(orig, est):
        # Original SDR Loss: <x, x`>**2 / <x`, x`> (== ||x`||**2)
        # Modified SDR loss, <x, x`> / (||x|| * ||x`||) : L2 Norm.
        correlation = bsum(orig * est)
        energies = torch.norm(orig, p=2, dim=1) * torch.norm(est, p=2, dim=1)

        return -(correlation / (energies + eps))

    noise = mixed - clean
    noise_est = mixed - clean_hat

    a = bsum(clean ** 2) / (bsum(clean ** 2) + bsum(noise ** 2) + eps)
    wSDR = a * mSDRLoss(clean, clean_hat) + (1 - a) * mSDRLoss(noise, noise_est)

    return torch.mean(wSDR)


