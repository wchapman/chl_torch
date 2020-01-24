import torch


class Network(torch.nn.Module):
    def __init__(self, layers=None, connections=None):
        super().__init__()

        if connections is None:
            connections = dict()
        if layers is None:
            layers = dict()

        self.connections = connections
        self.layers = layers

    def trial(self, stims):

        # minus phase
        self.layers['inp'].clamp = True
        self.layers['out'].clamp = True
        for k in range(0, 375):
            self.forward(stims)
        for c in self.connections.keys():
            self.connections[c].end_plus()

        # plus phase
        self.layers['inp'].clamp = True
        self.layers['out'].clamp = False
        for k in range(0, 375):
            self.forward(stims)
        for c in self.connections.keys():
            self.connections[c].end_minus()

        # update weights:
        for c in self.connections.keys():
            self.connections[c].update()

    def forward(self, stims):
        for ln in self.layers.keys():
            if self.layers[ln].clamp:
                self.layers[ln].forward(stims[ln])
            else:
                self.layers[ln].forward()