import torch
import torch.nn as nn
import torch.nn.functional as F


def istft(stft_matrix, hop_length=None, win_length=None, window='hann',
          center=True, normalized=True, onesided=True, length=None):
    # keunwoochoi's implementation
    # https://gist.github.com/keunwoochoi/2f349e72cc941f6f10d4adf9b0d3f37e

    """stft_matrix = (batch, freq, time, complex)
    All based on librosa
        - http://librosa.github.io/librosa/_modules/librosa/core/spectrum.html#istft
    What's missing?
        - normalize by sum of squared window --> do we need it here?
        Actually the result is ok by simply dividing y by 2.
    """
    # assert normalized == False
    assert onesided == True
    assert window == "hann"
    assert center == True

    device = stft_matrix.device
    n_fft = 2 * (stft_matrix.shape[-3] - 1)

    batch = stft_matrix.shape[0]

    # By default, use the entire frame
    if win_length is None:
        win_length = n_fft

    if hop_length is None:
        hop_length = int(win_length // 4)

    istft_window = torch.hann_window(n_fft).to(device).view(1, -1)  # (batch, freq)

    n_frames = stft_matrix.shape[-2]
    expected_signal_len = n_fft + hop_length * (n_frames - 1)

    y = torch.zeros(batch, expected_signal_len, device=device)
    for i in range(n_frames):
        sample = i * hop_length
        spec = stft_matrix[:, :, i]
        iffted = torch.irfft(spec, signal_ndim=1, signal_sizes=(win_length,), normalized=normalized)

        ytmp = istft_window * iffted
        y[:, sample:(sample + n_fft)] += ytmp

    y = y[:, n_fft // 2:]

    if length is not None:
        if y.shape[1] > length:
            y = y[:, :length]
        elif y.shape[1] < length:
            y = F.pad(y, (0, length - y.shape[1]))
            # y = torch.cat((y[:, :length], torch.zeros(y.shape[0], length - y.shape[1])))

    coeff = n_fft / float(
        hop_length) / 2.0  # -> this might go wrong if curretnly asserted values (especially, `normalized`) changes.
    return y / coeff


class ISTFT(nn.Module):
    def __init__(self, n_fft, hop_length):
        super(ISTFT, self).__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length

    def forward(self, x):
        batch = x.size(0)
        channel = x.size(1)

        reshape = x.view(x.size(0) * x.size(1), x.size(2), x.size(3), x.size(4))
        # ISTFT input [batch, freq, time, complex] 여야함
        transform = istft(reshape, hop_length=self.hop_length, win_length=self.n_fft)
        y_hat = transform.view(batch, channel, -1)

        return y_hat
