import torch
import torch.nn as nn

class CConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=0):
        super(CConv2d, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride

        self.real_conv = nn.Conv2d(in_channels=self.in_channels,
                                   out_channels=self.out_channels,
                                   kernel_size=self.kernel_size,
                                   padding=self.padding,
                                   stride=self.stride)

        self.imag_conv = nn.Conv2d(in_channels=self.in_channels,
                                 out_channels=self.out_channels,
                                 kernel_size=self.kernel_size,
                                 padding=self.padding,
                                 stride=self.stride)

        nn.init.xavier_uniform_(self.real_conv.weight)
        nn.init.xavier_uniform_(self.imag_conv.weight)

    def forward(self, x):
        x_real = x[..., 0]
        x_imag = x[..., 1]

        complex_real = self.real_conv(x_real) - self.imag_conv(x_imag)
        complex_imag = self.imag_conv(x_real) + self.real_conv(x_imag)

        output = torch.stack([complex_real, complex_imag], dim=-1)

        return output

class CConvTranspose2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, output_padding=0, padding=0):
        super(CConvTranspose2d, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.output_padding = output_padding
        self.padding = padding
        self.stride = stride

        self.real_Transconv = nn.ConvTranspose2d(in_channels=self.in_channels,
                                             out_channels=self.out_channels,
                                             kernel_size=self.kernel_size,
                                             output_padding=self.output_padding,
                                             padding=self.padding,
                                             stride=self.stride)

        self.imag_Transconv = nn.ConvTranspose2d(in_channels=self.in_channels,
                                           out_channels=self.out_channels,
                                           kernel_size=self.kernel_size,
                                           output_padding=self.output_padding,
                                           padding=self.padding,
                                           stride=self.stride)

        # Glorot initialization.
        nn.init.xavier_uniform_(self.real_Transconv.weight)
        nn.init.xavier_uniform_(self.imag_Transconv.weight)


    def forward(self, x):
        x_real = x[..., 0]
        x_imag = x[..., 1]

        complex_real = self.real_Transconv(x_real) - self.imag_Transconv(x_imag)
        complex_imag = self.imag_Transconv(x_real) + self.real_Transconv(x_imag)

        output = torch.stack([complex_real, complex_imag], dim=-1)

        return output


class CBatchNorm2d(nn.Module):

    def __init__(self, num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True):
        super(CBatchNorm2d, self).__init__()

        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats

        self.real_BNorm = nn.BatchNorm2d(num_features=self.num_features, eps=self.eps, momentum=self.momentum,
                                     affine=self.affine, track_running_stats=self.track_running_stats)
        self.imag_BNorm = nn.BatchNorm2d(num_features=self.num_features, eps=self.eps, momentum=self.momentum,
                                   affine=self.affine, track_running_stats=self.track_running_stats)

    def forward(self, x):
        x_real = x[..., 0]
        x_imag = x[..., 1]

        real_BNorm = self.real_BNorm(x_real)
        imag_BNorm = self.imag_BNorm(x_imag)

        output = torch.stack([real_BNorm, imag_BNorm], dim=-1)

        return output












