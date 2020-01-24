# %%
import torch
from torch.nn.functional import relu
from torch import tanh
import torchvision
from tqdm import trange

# %%
data = torchvision.datasets.MNIST('D:\Datasets',
                                  transform=torchvision.transforms.ToTensor())
data_loader = torch.utils.data.DataLoader(data,
                                          batch_size=1,
                                          shuffle=False,
                                          num_workers=0)
data_iter = iter(data_loader)


# %%
class SimpleRelu(torch.nn.Module):
    def __init__(self):
        super(SimpleRelu, self).__init__()

        self.inp = torch.nn.Linear(784, 128)
        self.h1 = torch.nn.Linear(128, 64)
        self.out = torch.nn.Linear(64, 10)

    def forward(self, x):
        x = relu(self.inp(x))
        x = relu(self.h1(x))
        x = tanh(x)

        return x


# %%
model = SimpleRelu()
loss_ = []
num_epochs = 1
criterion = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)

