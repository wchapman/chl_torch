import torch
import numpy as np
import matplotlib.pyplot as plt

class Network(torch.nn.Module):
    def __init__(self, layers=None, connections=None,
                 batch_size=1):
        super().__init__()

        if connections is None:
            connections = dict()
        if layers is None:
            layers = dict()

        self.connections = connections
        self.layers = layers

        self.batch_size = 1
        self.set_batch(batch_size)

        self.settle_time_plus = 25
        self.settle_time_minus = 25

    def trial(self, stims):
        self.reset()
        self.minus(stims)
        self.reset()
        self.plus(stims)

        # update weights:
        for c in self.connections.keys():
            self.connections[c].update()
        for l in self.layers.keys():
            self.layers[l].update()

    def forward(self, stims):
        for ln in self.layers.keys():
            if self.layers[ln].clamp:
                self.layers[ln].forward(stims[ln][1])
            else:
                self.layers[ln].forward()

    def train(self, mode=True):
        for c in self.connections.values():
            c.train(mode)
        for l in self.layers.values():
            l.train(mode)
        return super().train(mode)

    def set_batch(self, batch_size):
        if self.batch_size == batch_size:
            pass
        else:
            for l in self.layers.values():
                l.set_batch_size(batch_size)
            self.batch_size = batch_size

    def plus(self, stims):
        for s in stims.keys():
            if (stims[s][0] == 'input') or (stims[s][0] == 'target'):
                self.layers[s].clamp = True
            else:
                self.layers[s].clamp = False

        for k in range(0, self.settle_time_plus):
            self.forward(stims)
        for l in self.layers.keys():
            self.layers[l].end_plus()

    def minus(self, stims):
        for s in stims.keys():
            if stims[s][0] == 'input':
                self.layers[s].clamp = True
            else:
                self.layers[s].clamp = False

        for k in range(0, self.settle_time_minus):
            self.forward(stims)
        for l in self.layers.keys():
            self.layers[l].end_minus()

    def reset(self):
        for l in self.layers.values():
            l.reset()
