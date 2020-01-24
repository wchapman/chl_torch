# %%
from CHL.Layer import Layer
from CHL.Network import Network
from CHL.Connection import Connection, RandomConnection
import torch
import matplotlib.pyplot as plt
import numpy as np
import torchvision
torch.set_default_tensor_type("torch.DoubleTensor")
torch.multiprocessing.set_sharing_strategy('file_system')
from tqdm import trange


# # %%
# inp = np.zeros((6, 5, 5))
# inp[0] = [[1, 0, 0, 0, 1],
#           [0, 0, 0, 0, 0],
#           [0, 0, 1, 0, 0],
#           [0, 0, 0, 0, 0],
#           [1, 0, 0, 0, 1]]
#
# inp[1] = [[0, 0, 1, 0, 0],
#           [0, 0, 1, 0, 0],
#           [0, 0, 1, 0, 0],
#           [0, 0, 1, 0, 0],
#           [0, 0, 1, 0, 0]]
#
# inp[2] = [[0, 0, 0, 0, 0],
#           [0, 0, 0, 0, 0],
#           [1, 1, 1, 1, 1],
#           [0, 0, 0, 0, 0],
#           [0, 0, 0, 0, 0]]
#
# inp[3] = [[0, 0, 0, 0, 1],
#           [0, 0, 0, 1, 0],
#           [0, 0, 1, 0, 0],
#           [0, 1, 0, 0, 0],
#           [1, 0, 0, 0, 0]]
#
# inp[4] = [[1, 0, 0, 0, 0],
#           [0, 1, 0, 0, 0],
#           [0, 0, 1, 0, 0],
#           [0, 0, 0, 1, 0],
#           [0, 0, 0, 0, 1]]
#
# inp[5] = [[1, 0, 0, 0, 0],
#           [0, 0, 0, 1, 0],
#           [0, 1, 0, 0, 0],
#           [1, 0, 0, 0, 1],
#           [0, 0, 0, 0, 0]]
#
# ots = np.zeros((6, 5, 5))
# ots[0] = [[1, 0, 0, 0, 1],
#           [0, 0, 0, 0, 0],
#           [0, 0, 1, 0, 0],
#           [0, 0, 0, 0, 0],
#           [1, 0, 0, 0, 1]]
#
# ots[1] = [[0, 0, 1, 0, 0],
#           [0, 0, 1, 0, 0],
#           [0, 0, 1, 0, 0],
#           [0, 0, 1, 0, 0],
#           [0, 0, 1, 0, 0]]
#
# ots[2] = [[0, 0, 0, 0, 0],
#           [0, 0, 0, 0, 0],
#           [1, 1, 1, 1, 1],
#           [0, 0, 0, 0, 0],
#           [0, 0, 0, 0, 0]]
#
# ots[3] = [[0, 0, 0, 0, 1],
#           [0, 0, 0, 1, 0],
#           [0, 0, 1, 0, 0],
#           [0, 1, 0, 0, 0],
#           [1, 0, 0, 0, 0]]
#
# ots[4] = [[1, 0, 0, 0, 0],
#           [0, 1, 0, 0, 0],
#           [0, 0, 1, 0, 0],
#           [0, 0, 0, 1, 0],
#           [0, 0, 0, 0, 1]]
#
# ots[5] = [[1, 0, 0, 0, 0],
#           [0, 0, 0, 1, 0],
#           [0, 1, 0, 0, 1],
#           [1, 0, 0, 0, 0],
#           [0, 0, 0, 0, 0]]
#
# inp = torch.from_numpy(inp).flatten(1)
# ots = torch.from_numpy(ots).flatten(1)
#
# task = list()
# for i in range(0, 6):
#     task.append({'inp': inp[i].unsqueeze(0),
#                'out': ots[i].unsqueeze(0)})
#
# # %%
# layers = {
#     'inp': Layer(25),
#     'h1': Layer(50),
#     'h2': Layer(50),
#     'out': Layer(25)
# }
#
# conns = {
#     'inp_h1': RandomConnection(layers['inp'], layers['h1']),
#     'inp_h2': RandomConnection(layers['h1'], layers['h2']),
#     'h1_out': RandomConnection(layers['h1'], layers['out'])
# }
#
# model = Network(layers=layers, connections=conns)
#
# ne = 1
# err = np.zeros((ne, 6))
# for e in np.arange(0, ne):
#     print(e)
#     for i in np.arange(0, 6):
#         model.trial(task[i])
#         err[e, i] = torch.mean(torch.abs(model.layers['out'].x - task[i]['out'])).detach().numpy()

# %%
# plt.figure()
# plt.plot(err.mean(1))
# plt.show()
#
# for i in range(0, 6):
#     model.trial(task[i])
#     plt.imshow(model.layers['out'].x.reshape((5, 5)))
#     plt.show()
#
#     plt.imshow(task[i]['out'].reshape((5, 5)))
#     plt.show()


# %%

data = torchvision.datasets.MNIST('D:\Datasets',
                                  transform=torchvision.transforms.ToTensor())
data_loader = torch.utils.data.DataLoader(data,
                                          batch_size=1,
                                          shuffle=False,
                                          num_workers=0)
data_iter = iter(data_loader)

# %%
layers = {
    'inp': Layer(784),
    'h1': Layer(128),
    'h2': Layer(64),
    'out': Layer(10)
}

conns = {
    'inp_h1': RandomConnection(layers['inp'], layers['h1']),
    'inp_h2': RandomConnection(layers['h1'], layers['h2']),
    'h1_out': RandomConnection(layers['h2'], layers['out'])
}

model = Network(layers=layers, connections=conns)

# %%
ne = 1000
err = np.zeros(ne)
epoch = trange(ne)

for e in epoch:
    inp, o = data_iter.next()
    out = torch.zeros((1, 10))
    out[0, o] = 1.

    task = {'inp': inp.flatten().unsqueeze(0).double(),
            'out': out.double()}

    model.trial(task)
    err[e] = torch.mean(torch.abs(model.layers['out'].x - task['out'])).detach().numpy()

print('done')

plt.figure()
plt.plot(err)