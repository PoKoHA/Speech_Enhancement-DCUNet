import torch
import torch.nn as nn

from model.attention import Self_Attn

"""
본 Discriminator는 CycleGan에서 사용하였던 Basic_Discriminator을 그대로 가져왔음
https://github.com/park-cheol/Pytorch-CycleGan_2/blob/28632c98d15f2dea75aa26bb0195d290203724d0/model.py#L166
"""

class Discriminator(nn.Module):
    # 논문에서 70 x 70 patchGan 사용
    # leakyrelu with a slope of 0.2
    # 1-dimensional output
    def __init__(self, input_shape):
        super(Discriminator, self).__init__()
        # input_shape (1, 1539, 214)
        channels, height, width = input_shape
        self.output_shape = (1, height // 3 ** 4, width // 2)
        # [1, 19, 107]

        # initial conv
        self.leakyrelu = nn.LeakyReLU(0.1, inplace=True)
        self.zeropad = nn.ZeroPad2d((1, 0, 1, 0))

        self.convLayer_1 = nn.Sequential(nn.Conv2d(channels, 64, 4, stride=2, padding=1))

        self.convLayer_2 = nn.Sequential(nn.Conv2d(64, 128, 4, stride=2, padding=1),
                                         nn.InstanceNorm2d(128))

        self.convLayer_3 = nn.Sequential(nn.Conv2d(128, 256, 4, stride=2, padding=1),
                                         nn.InstanceNorm2d(256))

        self.convLayer_4 = nn.Sequential(nn.Conv2d(256, 512, 4, stride=2, padding=1),
                                         nn.InstanceNorm2d(512))

        self.convLayer_5 = nn.Conv2d(512, 1, 4, padding=1)

        self.attn1 = Self_Attn(256)
        self.attn2 = Self_Attn(512)

    def forward(self, input): # input[1, 1, 1539, 214]
        # print("I:", input.size())
        convLayer_1 = self.convLayer_1(input) # [B, 64, 769, 107]
        # print("1:", convLayer_1.size())
        convLayer_1_relu = self.leakyrelu(convLayer_1)

        convLayer_2 = self.convLayer_2(convLayer_1_relu) # [2, 128, 384, 53]
        # print("2:", convLayer_2.size())
        convLayer_2_relu = self.leakyrelu(convLayer_2)

        convLayer_3 = self.convLayer_3(convLayer_2_relu) # [B, 256, 192, 26]
        # print("3:", convLayer_3.size())
        convLayer_3_relu = self.leakyrelu(convLayer_3)
        # print("PASS")
        attn_1, map_1 = self.attn1(convLayer_3_relu)
        # print(attn_1.size())

        convLayer_4 = self.convLayer_4(attn_1) # [B, 512, 96, 13]
        # print("4:", convLayer_4.size())
        convLayer_4_relu = self.leakyrelu(convLayer_4)
        # print("pass")
        attn_2, map_2 = self.attn2(convLayer_4_relu)

        zeropad = self.zeropad(attn_2) # [B, 512, 97, 14]
        # print("5:", zeropad.size())
        output = self.convLayer_5(zeropad) # [B, 512, 96, 13]
        # print("output:", output.size())

        return output # [B, 1, 96, 13]


if __name__ == '__main__':
    test = torch.randn(2, 1, 1539, 214).cuda()
    D = Discriminator(input_shape=(1, 1539, 214)).cuda()
    D(test)
