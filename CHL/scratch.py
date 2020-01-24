# %%
from CHL.Layer import Layer
from CHL.Network import Network
from CHL.Connection import Connection, RandomConnection
import torch
import matplotlib.pyplot as plt
import numpy as np

torch.set_default_tensor_type("torch.DoubleTensor")

# %%
inp = np.zeros((6, 5, 5))
inp[0] = [[1, 0, 0, 0, 1],
          [0, 0, 0, 0, 0],
          [0, 0, 1, 0, 0],
          [0, 0, 0, 0, 0],
          [1, 0, 0, 0, 1]]

inp[1] = [[0, 0, 1, 0, 0],
          [0, 0, 1, 0, 0],
          [0, 0, 1, 0, 0],
          [0, 0, 1, 0, 0],
          [0, 0, 1, 0, 0]]

inp[2] = [[0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0],
          [1, 1, 1, 1, 1],
          [0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0]]

inp[3] = [[0, 0, 0, 0, 1],
          [0, 0, 0, 1, 0],
          [0, 0, 1, 0, 0],
          [0, 1, 0, 0, 0],
          [1, 0, 0, 0, 0]]

inp[4] = [[1, 0, 0, 0, 0],
          [0, 1, 0, 0, 0],
          [0, 0, 1, 0, 0],
          [0, 0, 0, 1, 0],
          [0, 0, 0, 0, 1]]

inp[5] = [[1, 0, 0, 0, 0],
          [0, 0, 0, 1, 0],
          [0, 1, 0, 0, 0],
          [1, 0, 0, 0, 1],
          [0, 0, 0, 0, 0]]

ots = np.zeros((6, 5, 5))
ots[0] = [[1, 0, 0, 0, 1],
          [0, 0, 0, 0, 0],
          [0, 0, 1, 0, 0],
          [0, 0, 0, 0, 0],
          [1, 0, 0, 0, 1]]

ots[1] = [[0, 0, 1, 0, 0],
          [0, 0, 1, 0, 0],
          [0, 0, 1, 0, 0],
          [0, 0, 1, 0, 0],
          [0, 0, 1, 0, 0]]

ots[2] = [[0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0],
          [1, 1, 1, 1, 1],
          [0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0]]

ots[3] = [[0, 0, 0, 0, 1],
          [0, 0, 0, 1, 0],
          [0, 0, 1, 0, 0],
          [0, 1, 0, 0, 0],
          [1, 0, 0, 0, 0]]

ots[4] = [[1, 0, 0, 0, 0],
          [0, 1, 0, 0, 0],
          [0, 0, 1, 0, 0],
          [0, 0, 0, 1, 0],
          [0, 0, 0, 0, 1]]

ots[5] = [[1, 0, 0, 0, 0],
          [0, 0, 0, 1, 0],
          [0, 1, 0, 0, 1],
          [1, 0, 0, 0, 0],
          [0, 0, 0, 0, 0]]

inp = torch.from_numpy(inp).flatten(1)
ots = torch.from_numpy(ots).flatten(1)

task = list()
for i in range(0, 6):
    task.append({'inp': inp[i].unsqueeze(0),
               'out': ots[i].unsqueeze(0)})

# %%
layers = {
    'inp': Layer(25),
    'h1': Layer(50),
    'h2': Layer(50),
    'out': Layer(25)
}

conns = {
    'inp_h1': RandomConnection(layers['inp'], layers['h1']),
    'inp_h2': RandomConnection(layers['h1'], layers['h2']),
    'h1_out': RandomConnection(layers['h1'], layers['out'])
}

model = Network(layers=layers, connections=conns)

ne = 50
err = np.zeros((ne, 6))
for e in np.arange(0, ne):
    print(e)
    for i in np.arange(0, 6):
        model.trial(task[i])
        err[e, i] = torch.mean(torch.abs(model.layers['out'].x - task[i]['out'])).detach().numpy()

# %%
plt.figure()
plt.plot(err.mean(1))
plt.show()

for i in range(0, 6):
    model.trial(task[i])
    plt.imshow(model.layers['out'].x.reshape((5, 5)))
    plt.show()

    plt.imshow(task[i]['out'].reshape((5, 5)))
    plt.show()