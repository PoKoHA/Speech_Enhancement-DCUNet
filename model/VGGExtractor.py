import numpy as np

import torch
import torch.nn as nn

###############
# MaskCNN(Deep Speech)
###############
#
# class MaskCNN(nn.Module):
#
#     def __init__(self, sequential, args):
#         super(MaskCNN, self).__init__()
#         self.args = args
#         self.sequential = sequential
#
#     def forward(self, inputs): # todo print
#         output = None
#
#         for module in self.sequential:
#             output = module(inputs)
#             mask = torch.BoolTensor(output.size()).fill_(0)
#             mask = mask.cuda(self.args.gpu)
#
#             output = output.masked_fill(mask, 0)
#             inputs = output
#
#         return output,
#
#     def _get_sequence_lengths(self, module, seq_lengths):
#         # 아래 식은 그저 원래 CNN 걸친 결과 식
#         if isinstance(module, nn.Conv2d):
#             numerator = seq_lengths + 2 * module.padding[1] - module.dilation[1] * (module.kernel_size[1] - 1) - 1
#             seq_lengths = numerator.float() / float(module.stride[1])
#             seq_lengths = seq_lengths.int() + 1
#
#         elif isinstance(module, nn.MaxPool2d):
#             seq_lengths >>= 1 # todo >> 아닌지 확인
#
#         return seq_lengths.int()

############################
# VGG Extractor
############################

class VGGExtractor(nn.Module):

    def __init__(self, args, input_dim=80, in_channels=1, out_channels=(32, 64)):
        self.args = args
        super(VGGExtractor, self).__init__()
        self.input_dim = input_dim # Frame 수가 됨
        self.in_channels = in_channels # 1
        self.out_channels = out_channels # 64, 128

        # todo 얼마나 작아는지 확인
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels[0], kernel_size=(3, 2) , stride=(1, 1), padding=(1,1), bias=False),
            nn.BatchNorm2d(num_features=out_channels[0]),
            nn.ReLU(),
            nn.Conv2d(out_channels[0], out_channels[0], kernel_size=(3, 2), stride=(1,1), padding=(1, 0), bias=False),
            nn.BatchNorm2d(num_features=out_channels[0]),
            nn.ReLU(),
            nn.MaxPool2d((2, 1), stride=(2, 1)),
            nn.Conv2d(out_channels[0], out_channels[1], kernel_size=(3, 2), stride=(1,1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(num_features=out_channels[1]),
            nn.ReLU(),
            nn.Conv2d(out_channels[1], out_channels[1], kernel_size=(3, 2), stride=(1,1), padding=(1, 0), bias=False),
            nn.BatchNorm2d(num_features=out_channels[1]),
            nn.ReLU(),
            nn.MaxPool2d((2, 1), stride=(2, 1)),
            # nn.MaxPool2d(2, stride=2),
        )

    # todo 의미
    def get_output_dim(self):
        return (self.input_dim - 1) << 5 if self.input_dim % 2 else self.input_dim << 5

    def forward(self, inputs):
        print("  [VGGExtractor]  ")
        # [batch, channel=1, freq=1539, time=214]
        # print(inputs)
        # print("C", inputs.transpose(2,3).size())
        outputs = self.conv(inputs.transpose(2, 3)) # [batch, cnannel, 214t, 1539freq]
        print("    Mask.Transpose(2,3): ", inputs.transpose(2,3).size())
        print("    VGG: ", outputs.size())
        # print("g", outputs.size())
        batch_size, channels, dimension, seq_lengths = outputs.size()
        outputs = outputs.permute(0, 3, 1, 2) # [batch, freq, channel,time]
        print("    Permute[batch, freq, channel, time]: ", outputs.size())
        outputs = outputs.view(batch_size, seq_lengths, channels * dimension)
        print("    reshape[batch, freq, channel * time]: ", outputs.size())

        return outputs