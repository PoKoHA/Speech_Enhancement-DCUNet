import torch
import torch.nn as nn
import torch.nn.functional as F

def gram_matrix(y):
    (b, ch, h, w) = y.size()
    features = y.view(b * ch, h * w)
    # print("1",features.size())
    gram = torch.mm(features, features.t())
    # print("2",gram.size())
    # mm -> 행렬곱 (비슷한 것: element-wise/아다마르곱==> element끼리 곱)
    gram = gram.div(b * ch * h * w)
    # print("3",gram.size())
    return gram


class StyleLoss(nn.Module):
    def __init__(self, style_image):
        super(StyleLoss, self).__init__()
        self.style = gram_matrix(style_image).detach()
        # print(self.style.size())

    def forward(self, input):
        gram = gram_matrix(input)
        self.loss = F.mse_loss(gram, self.style)

        return self.loss


if __name__ == '__main__':
    tensor = torch.randn(8, 1, 1539, 214)
    target = torch.randn(8, 1, 1539, 214)
    s = StyleLoss(target)
