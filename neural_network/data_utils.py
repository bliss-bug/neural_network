import numpy as np
import torch
from torch.utils.data import *


def data_pre(mode='train'):
    if mode == 'train':
        file_path = 'data/pendigits.tra' 
    else:
        file_path = 'data/pendigits.tes'

    data = np.loadtxt(file_path, dtype=np.int, delimiter=',')
    x,y = data[:, :-1],data[:,-1]
    dataset=TensorDataset(torch.Tensor(x), torch.Tensor(y))
    return dataset


def get_data(mode):
    dataset = data_pre(mode)
    dataloader = DataLoader(dataset, batch_size=100, shuffle=(mode=='train'))
    return dataloader
