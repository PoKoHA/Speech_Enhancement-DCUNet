"""
https://github.com/keunwoochoi/torchaudio-contrib/blob/master/torchaudio_contrib/layers.py
torch audio STFT 그대로
"""
import torch
import torch.nn as nn

def stft(waveforms, fft_length, hop_length=None, win_length=None, window=None,
         center=True, pad_mode='reflect', normalized=False, onesided=True):
    """Compute a short-time Fourier transform of the input waveform(s).
    It wraps `torch.stft` but after reshaping the input audio
    to allow for `waveforms` that `.dim()` >= 3.
    It follows most of the `torch.stft` default value, but for `window`,
    if it's not specified (`None`), it uses hann window.
    Args:
        waveforms (Tensor): Tensor of audio signal
            of size `(*, channel, time)`
        fft_length (int): FFT size [sample]
        hop_length (int): Hop size [sample] between STFT frames.
            Defaults to `fft_length // 4` (75%-overlapping windows)
            by `torch.stft`.
        win_length (int): Size of STFT window.
            Defaults to `fft_length` by `torch.stft`.
        window (Tensor): 1-D Tensor.
            Defaults to Hann Window of size `win_length`
            *unlike* `torch.stft`.
        center (bool): Whether to pad `waveforms` on both sides so that the
            `t`-th frame is centered at time `t * hop_length`.
            Defaults to `True` by `torch.stft`.
        pad_mode (str): padding method (see `torch.nn.functional.pad`).
            Defaults to `'reflect'` by `torch.stft`.
        normalized (bool): Whether the results are normalized.
            Defaults to `False` by `torch.stft`.
        onesided (bool): Whether the half + 1 frequency bins
            are returned to removethe symmetric part of STFT
            of real-valued signal. Defaults to `True`
            by `torch.stft`.
    Returns:
        complex_specgrams (Tensor): `(*, channel, num_freqs, time, complex=2)`
    Example:
        >>> waveforms = torch.randn(16, 2, 10000)  # (batch, channel, time)
        >>> x = stft(waveforms, 2048, 512)
        >>> x.shape
        torch.Size([16, 2, 1025, 20])
    """
    leading_dims = waveforms.shape[:-1]

    waveforms = waveforms.reshape(-1, waveforms.size(-1))

    if window is None:
        if win_length is None:
            window = torch.hann_window(fft_length)
        else:
            window = torch.hann_window(win_length)

    complex_specgrams = torch.stft(waveforms,
                                   n_fft=fft_length,
                                   hop_length=hop_length,
                                   win_length=win_length,
                                   window=window,
                                   center=center,
                                   pad_mode=pad_mode,
                                   normalized=normalized,
                                   onesided=onesided)

    complex_specgrams = complex_specgrams.reshape(
        leading_dims +
        complex_specgrams.shape[1:])

    return complex_specgrams

class _ModuleNoStateBuffers(nn.Module):
    """
    Extension of nn.Module that removes buffers
    from state_dict.
    """

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        ret = super(_ModuleNoStateBuffers, self).state_dict(
            destination, prefix, keep_vars)
        for k in self._buffers:
            del ret[prefix + k]
        return ret

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        # temporarily hide the buffers; we do not want to restore them

        buffers = self._buffers
        self._buffers = {}
        result = super(_ModuleNoStateBuffers, self)._load_from_state_dict(
            state_dict, prefix, *args, **kwargs)
        self._buffers = buffers
        return result



class STFT(_ModuleNoStateBuffers):
    """Compute a short-time Fourier transform of the input waveform(s).
    It essentially wraps `torch.stft` but after reshaping the input audio
    to allow for `waveforms` that `.dim()` >= 3.
    It follows most of the `torch.stft` default value, but for `window`,
    if it's not specified (`None`), it uses hann window.
    Args:
        fft_length (int): FFT size [sample]
        hop_length (int): Hop size [sample] between STFT frames.
            Defaults to `fft_length // 4` (75%-overlapping windows) by `torch.stft`.
        win_length (int): Size of STFT window.
            Defaults to `fft_length` by `torch.stft`.
        window (Tensor): 1-D Tensor.
            Defaults to Hann Window of size `win_length` *unlike* `torch.stft`.
        center (bool): Whether to pad `waveforms` on both sides so that the
            `t`-th frame is centered at time `t * hop_length`.
            Defaults to `True` by `torch.stft`.
        pad_mode (str): padding method (see `torch.nn.functional.pad`).
            Defaults to `'reflect'` by `torch.stft`.
        normalized (bool): Whether the results are normalized.
            Defaults to `False` by `torch.stft`.
        onesided (bool): Whether the half + 1 frequency bins are returned to remove
            the symmetric part of STFT of real-valued signal.
            Defaults to `True` by `torch.stft`.
    """

    def __init__(self, fft_length, hop_length=None, win_length=None,
                 window=None, center=True, pad_mode='reflect',
                 normalized=False, onesided=True):
        super(STFT, self).__init__()

        self.fft_length = fft_length
        self.hop_length = hop_length
        self.win_length = win_length

        self.center = center
        self.pad_mode = pad_mode
        self.normalized = normalized
        self.onesided = onesided

        if window is None:
            if win_length is None:
                window = torch.hann_window(fft_length)
            else:
                window = torch.hann_window(win_length)

        self.register_buffer('window', window)

    def forward(self, waveforms):
        """
        Args:
            waveforms (Tensor): Tensor of audio signal of size `(*, channel, time)`
        Returns:
            complex_specgrams (Tensor): `(*, channel, num_freqs, time, complex=2)`
        """

        complex_specgrams = stft(waveforms, self.fft_length,
                                 hop_length=self.hop_length,
                                 win_length=self.win_length,
                                 window=self.window,
                                 center=self.center,
                                 pad_mode=self.pad_mode,
                                 normalized=self.normalized,
                                 onesided=self.onesided)

        return complex_specgrams

    def __repr__(self):
        param_str1 = '(fft_length={}, hop_length={}, win_length={})'.format(
            self.fft_length, self.hop_length, self.win_length)
        param_str2 = '(center={}, pad_mode={}, normalized={}, onesided={})'.format(
            self.center, self.pad_mode, self.normalized, self.onesided)
        return self.__class__.__name__ + param_str1 + param_str2
