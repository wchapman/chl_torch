# %%
import torch
from torch.nn.functional import relu
import torch.nn.functional as F
from torch import tanh
import torchvision
from tqdm import trange
from CHL.Network import Network
from CHL.Connection import RandomConnection, Connection
from CHL.Layer import Layer
import matplotlib.pyplot as plt
import numpy as np

gpu = True

if gpu:
    device = torch.device("cuda:0")
    torch.set_default_tensor_type(torch.cuda.DoubleTensor)
else:
    device = "cpu"
    torch.set_default_dtype(torch.double)


# %%
bs = 1000
bst = 1000


data_train = torchvision.datasets.MNIST('D:\Datasets',
                                        transform=torchvision.transforms.ToTensor(),
                                        train=True)

data_train_loader = torch.utils.data.DataLoader(data_train,
                                                batch_size=bs,
                                                shuffle=True,
                                                num_workers=0)
data_train_iter = iter(data_train_loader)

# test set
data_test = torchvision.datasets.MNIST('D:\Datasets',
                                       transform=torchvision.transforms.ToTensor(),
                                       train=False)

data_test_loader = torch.utils.data.DataLoader(data_test,
                                               batch_size=bst,
                                               shuffle=True,
                                               num_workers=0)
data_test_iter = iter(data_test_loader)
num_epochs = 500


# %% Rate-based CHL Approach
layers = {
    'inp': Layer(784, name='inp'),
    'h1': Layer(128, name='h1'),
    'h2': Layer(64, name='h2'),
    'out': Layer(10, name='out')
}

conns = {
    'inp_h1': RandomConnection(layers['inp'], layers['h1'], gain=2.),
    'inp_h2': RandomConnection(layers['h1'], layers['h2'], gain=1.),
    'h1_out': RandomConnection(layers['h1'], layers['out'], gain=0.)
}

model_chl = Network(layers=layers, connections=conns, batch_size=bs)

# %%
criterion = torch.nn.MSELoss()
err_train = np.zeros(num_epochs)
err_test = np.zeros(num_epochs)
n_test = np.zeros(num_epochs)
n_train = np.zeros(num_epochs)
t = trange(num_epochs)

for epoch in t:

    # test before training
    model_chl.train(False)
    model_chl.set_batch(bst)
    loss_e = 0
    for inp, label in data_test_loader:
        model_chl.reset()
        inp = inp.double().to(device)
        out = torch.zeros((bst, 1, 10)).to(device)

        for i in range(0, bst):
            out[i, 0, label[i]] = 1

        task = {'inp': ['input', inp.flatten(2)],
                'out': ['target', None]
                }

        model_chl.trial(task)
        pred = model_chl.layers['out'].minus

        loss = criterion(pred, out)
        loss_e += loss.detach().cpu().numpy()
        n_test[epoch] += (pred.max(2)[1] == label.unsqueeze(1)).sum()

    err_test[epoch] = loss_e

    # train
    model_chl.train(True)
    model_chl.set_batch(bs)
    loss_e = 0
    for inp, label in data_train_loader:
        model_chl.reset()
        inp = inp.double().to(device)
        out = torch.zeros((bs, 1, 10)).to(device)

        for i in range(0, bs):
            out[i, 0, label[i]] = 1

        task = {'inp': ['input', inp.flatten(2)],
                'out': ['target', out]
                }

        model_chl.trial(task)
        pred = model_chl.layers['out'].minus

        loss = criterion(pred, out)
        loss_e += loss.detach().cpu().numpy()
        n_train[epoch] += (pred.max(2)[1] == label.unsqueeze(1)).sum()

    err_train[epoch] = loss_e

    k = {'err_test': err_test[epoch],
         'err_train': err_train[epoch],
         'acc_test': n_test[epoch] / data_test.__len__(),
         'acc_train': n_train[epoch] / data_train.__len__()}
    t.set_postfix(k)

# # %%
plt.plot(100 * n_test / data_test.__len__(), label='test')
plt.plot(100 * n_train / data_train.__len__(), label='train')
plt.legend()
plt.show()

plt.figure()
plt.plot(100 * err_test / data_test.__len__(), label='test')
plt.plot(100 * err_train / data_train.__len__(), label='train')
plt.show()

# # %%
# bs = 100
# bst = 100
# data_train = torchvision.datasets.MNIST('D:\Datasets',
#                                         transform=torchvision.transforms.ToTensor(),
#                                         train=True)
# data_train.data = data_train.data[0:1000]
#
# data_train_loader = torch.utils.data.DataLoader(data_train,
#                                                 batch_size=bs,
#                                                 shuffle=True,
#                                                 num_workers=0)
# data_train_iter = iter(data_train_loader)
#
# # test set
# data_test = torchvision.datasets.MNIST('D:\Datasets',
#                                        transform=torchvision.transforms.ToTensor(),
#                                        train=False)
# data_test.data = data_test.data[0:100]
# data_test_loader = torch.utils.data.DataLoader(data_test,
#                                                batch_size=bst,
#                                                shuffle=True,
#                                                num_workers=0)
# data_test_iter = iter(data_test_loader)
#
# # %% Evaluate on training set
# model_chl.train(False)
# preds = torch.zeros((data_train.__len__(), 10))
# outs = torch.zeros((data_train.__len__(), 10))
# n = 0
# model_chl.set_batch(bs)
# for inp, label in data_train_loader:
#     model_chl.reset()
#     inp = inp.double().to(device)
#     out = torch.zeros((bs, 1, 10)).to(device)
#
#     for i in range(0, bs):
#         out[i, 0, label[i]] = 1
#
#     task = {'inp': ['input', inp.flatten(2)],
#             'out': ['target', None]
#             }
#
#     model_chl.trial(task)
#     pred = model_chl.layers['out'].minus.squeeze(1)
#
#     preds[(n*bs):((n+1)*bs), :] = pred.detach().cpu()
#     outs[(n*bs):((n+1)*bs), :] = out.detach().cpu().squeeze(1)
#
#     n+=1
#
# # %%
# preds_bin = torch.zeros_like(preds).cpu()
# for i in range(0, preds.shape[0]):
#     preds_bin[i, preds[i,:].max(0)[1]] = 1
#
# plt.figure()
# plt.subplot(121)
# plt.imshow(preds_bin[0:50])
#
# plt.subplot(122)
# plt.imshow(outs[0:50].cpu())
#
# acc_train = (preds.max(1)[1] == outs.max(1)[1]).sum().float() / data_train.__len__()
# print(str(acc_train))
# plt.show()
#
# # %%
# model_chl.train(False)
# preds = torch.zeros((data_test.__len__(), 10))
# outs = torch.zeros((data_test.__len__(), 10))
# n = 0
# bs = 1
# model_chl.set_batch(bs)
# for inp, label in data_test_loader:
#     model_chl.reset()
#     inp = inp.double().to(device)
#     out = torch.zeros((bs, 1, 10)).to(device)
#
#     for i in range(0, bs):
#         out[i, 0, label[i]] = 1
#
#     task = {'inp': ['input', inp.flatten(2)],
#             'out': ['target', None]
#             }
#
#     model_chl.trial(task)
#     pred = model_chl.layers['out'].minus
#
#     preds[n:] = pred.detach().cpu()
#     outs[n:] = out.detach().cpu()
#
#     n+=1
#
#
# # %%
# plt.figure()
# plt.subplot(211)
# plt.imshow(preds)
#
# plt.subplot(212)
# plt.imshow(outs)
#
# acc_test = (preds.max(1)[1] == outs.max(1)[1]).sum().float() / data_test.__len__()
# print(acc_test)
#
# plt.show()
#
