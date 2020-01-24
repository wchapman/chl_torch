import torch
from torch.nn.functional import relu
from torch import sigmoid, tanh


class Layer(torch.nn.Module):
    def __init__(self, n):
        super().__init__()
        self.n = n
        self.x = torch.zeros(1, n)
        self.b = (torch.rand(1, n)-0.5)*0.2
        self.f = sigmoid

        self.inputs = list()
        self.outputs = list()
        self.clamp = False

    def forward(self, inp=None):
        if self.clamp:
            self.x = inp.clone()
        else:
            x = torch.zeros_like(self.x)
            for i in self.inputs:
                x += i.compute_forward()
            for o in self.outputs:
                x += o.compute_feedback()
            dx = -self.x + self.f(x+self.b)
            self.x += dx*0.001
