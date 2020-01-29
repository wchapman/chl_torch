import torch
from torch.nn.functional import relu
from torch import sigmoid, tanh


class Layer(torch.nn.Module):
    def __init__(self, n, f=torch.sigmoid, name='None'):
        super().__init__()
        self.n = n
        self.x = torch.zeros(1, 1, n)
        self.b = torch.zeros(1,1,n)#torch.rand(1, 1, n)-0.5
        self.f = f

        self.batch_size = 1
        self.inputs = list()
        self.outputs = list()
        self.clamp = False
        self.name = name

        self.plus = torch.zeros(1, 1, n)
        self.minus = torch.zeros(1, 1, n)

    def reset(self):
        self.x = torch.zeros(self.batch_size, 1, self.n)

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size
        self.x = torch.zeros(batch_size, self.n)

    def forward(self, inp=None):
        if self.clamp and not (inp is None):
            self.x = inp.clone()
        else:
            x = torch.zeros_like(self.x)
            for i in self.inputs:
                x += i.compute_forward()
            for o in self.outputs:
                x += o.compute_feedback()
            if self.f is None:
                x = -self.x + x + self.b
            else:
                x = -self.x + self.f(x+self.b)
            self.x += x*0.05

    def end_minus(self):
        self.minus = self.x.clone()

    def end_plus(self):
        self.plus = self.x.clone()

    def update(self):
        if self.train:
            #self.b += 0.01*(self.plus.mean(0).unsqueeze(0) - self.minus.mean(0).unsqueeze(0))
            pass
