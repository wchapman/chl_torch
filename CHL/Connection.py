import torch


class Connection(torch.nn.Module):
    def __init__(self,
                 source,
                 target,
                 mu=0.1,      # learning rate
                 gamma=0.05,  # feedback gain
                 gain=0):     # Gain for learning in this connection (gamma ** (k-L))
        super().__init__()

        self.source = source
        self.target = target
        self.mu = mu
        self.gamma = gamma
        self.gain = gain

        self.W = torch.rand(self.target.n, self.source.n) - 0.5

        self.source.outputs.append(self)
        self.target.inputs.append(self)

    def compute_forward(self):
        return self.source.x @ self.W.t()

    def compute_feedback(self):
        return self.gamma * (self.target.x @ self.W)

    def update(self):
        if self.training:
            t0 = self.mu * (self.gamma ** -self.gain)
            t1 = self.target.plus.transpose(1, 2) @ self.source.plus
            t2 = self.target.minus.transpose(1, 2) @ self.source.minus
            self.W += t0 * (t1 - t2).mean(0)
        else:
            pass

    def train(self, mode=True):
        return super().train(mode)


class RandomConnection(torch.nn.Module):
    def __init__(self,
                 source,
                 target,
                 mu=0.01,  # learning rate
                 gamma=0.05,  # feedback gain
                 gain=0):
        super().__init__()

        self.source = source
        self.target = target
        self.mu = mu
        self.gamma = gamma
        self.gain = gain

        self.W = torch.randn(self.target.n, self.source.n)
        self.G = torch.randn(self.source.n, self.target.n)

        self.source.outputs.append(self)
        self.target.inputs.append(self)

    def compute_forward(self):
        return self.source.x @ self.W.t()

    def compute_feedback(self):
        return self.gamma * (self.target.x @ self.G.t())

    def update(self):
        if self.training:
            t0 = self.mu * (self.gamma ** -self.gain)
            t1 = self.target.plus.transpose(1, 2) @ self.source.plus
            t2 = self.target.minus.transpose(1, 2) @ self.source.minus
            self.W += t0 * (t1 - t2).mean(0)
        else:
            pass

    def train(self, mode=True):
        return super().train(mode)

