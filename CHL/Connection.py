import torch


class Connection(torch.nn.Module):
    def __init__(self,
                 source,
                 target,
                 mu=0.01,  # learning rate
                 gamma=0.5):  # feedback gain
        super().__init__()

        self.source = source
        self.target = target
        self.mu = mu
        self.gamma = gamma

        self.W = torch.randn(self.source.n, self.target.n)  # normal(0,1)

        self.pre_clamp = None
        self.post_clamp = None
        self.pre_free = None
        self.post_free = None

        self.source.outputs.append(self)
        self.target.inputs.append(self)

    def compute_forward(self):
        return self.source.x @ self.W

    def compute_feedback(self):
        return self.gamma * self.target.x @ self.W.transpose(0, 1)

    def update(self):
        # self.W += (
        #         (self.mu * self.gamma) * (
        #          (self.pre_clamp.transpose(0, 1) @ self.post_clamp) -
        #          (self.pre_free.transpose(0, 1) @ self.post_free))
        # )
        self.W += (
            (self.mu * self.gamma) * (
             (self.post_clamp * self.pre_clamp.transpose(0, 1)) -
             (self.post_free * self.pre_free.transpose(0, 1)))
        )

    def end_plus(self):
        self.pre_clamp = self.source.x.clone()
        self.post_clamp = self.target.x.clone()

    def end_minus(self):
        self.pre_free = self.source.x.clone()
        self.post_free = self.target.x.clone()


class RandomConnection(torch.nn.Module):
    def __init__(self,
                 source,
                 target,
                 mu=0.05,  # learning rate
                 gamma=0.5):  # feedback gain
        super().__init__()

        self.source = source
        self.target = target
        self.mu = mu
        self.gamma = gamma

        self.W = torch.randn(self.source.n, self.target.n)  # normal(0,1)
        self.G = torch.randn(self.target.n, self.source.n)  # seperate normal(0,1) for feedback

        self.pre_clamp = None
        self.post_clamp = None
        self.pre_free = None
        self.post_free = None

        self.source.outputs.append(self)
        self.target.inputs.append(self)

    def compute_forward(self):
        return self.source.x @ self.W

    def compute_feedback(self):
        return self.gamma * self.target.x @ self.G

    def update(self):
        self.W += (
            (self.mu * self.gamma) * (
             (self.post_clamp * self.pre_clamp.transpose(0, 1)) -
             (self.post_free * self.pre_free.transpose(0, 1)))
        )

    def end_plus(self):
        self.pre_clamp = self.source.x.clone()
        self.post_clamp = self.target.x.clone()

    def end_minus(self):
        self.pre_free = self.source.x.clone()
        self.post_free = self.target.x.clone()