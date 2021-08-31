import torch
from torch import nn
import torch.nn.functional as F

from torch.autograd import Variable
from torch import Tensor
from torch.nn import Parameter


def l2normalize(v, eps=1e-12):
    # L2 Norm값으로 Normalize
    # L2: sqrt(sum(each component ** 2))
    return v / (v.norm() + eps)


class SpectralNorm(nn.Module):

    def __init__(self, module, name='weight', power_iterations=1):
        super(SpectralNorm, self).__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        if not self._made_params():
            self._make_params()

    def _update_u_v(self):
        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")
        w = getattr(self.module, self.name + "_bar")
        # print(u.size())
        # print(v.size())
        # print(w.size())
        # print(self.module)
        # print("pp")

        height = w.data.shape[0]
        for _ in range(self.power_iterations):
            v.data = l2normalize(torch.mv(torch.t(w.view(height, -1).data), u.data)) # [out_channel size같음]
            u.data = l2normalize(torch.mv(w.view(height, -1).data, v.data))

        # sigma = torch.dot(u.data, torch.mv(w.view(height,-1).data, v.data))
        # dot은 1D tensor만 가능
        sigma = u.dot(w.view(height, -1).mv(v))
        setattr(self.module, self.name, w / sigma.expand_as(w))

    def _made_params(self):
        try:
            u = getattr(self.module, self.name + "_u")
            v = getattr(self.module, self.name + "_v")
            w = getattr(self.module, self.name + "_bar")
            # print(self.module)
            # print(u)
            # print(v)
            # print(w)
            # print("th")

            return True
        except AttributeError:
            return False


    def _make_params(self):
        w = getattr(self.module, self.name)

        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]

        u = Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
        v = Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)
        u.data = l2normalize(u.data)
        v.data = l2normalize(v.data)
        w_bar = Parameter(w.data)

        del self.module._parameters[self.name]

        self.module.register_parameter(self.name + "_u", u)
        self.module.register_parameter(self.name + "_v", v)
        self.module.register_parameter(self.name + "_bar", w_bar)


    def forward(self, *args):
        self._update_u_v()
        return self.module.forward(*args)

if __name__ == "__main__":
    t = torch.randn(1, 2, 3, 6).cuda()
    print("input", t.size())
    a = nn.Conv2d(2, 8, 3).cuda()
    b = SpectralNorm(nn.Conv2d(2, 8 , 3)).cuda()
    print(b)
    # print("A", b(t))
    print("B", b(t).size())
