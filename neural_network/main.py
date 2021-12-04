import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from model_build import *
from data_utils import get_data

config = {
        'model': 'base',
        'optimizer': 'Adam',
        'lr': 0.01,
        'weight_decay': 0.001,
        'lr_end': 0.0001,
        'epoch': 20,
        'dropout_rt': 0.01
    }

device='cpu'

def train(train_set, dev_set, model):
    op = getattr(torch.optim, config['optimizer'])
    optimizer = op(model.parameters(), lr=config['lr'],weight_decay=config['weight_decay'])

    epoch = config['epoch']
    lr, lr_end =config['lr'], config['lr_end']

    gamma = (lr_end / lr) ** (1 / epoch)
    lr_sch = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epoch):
        model.train()
        print(epoch+1, end=' ')
        # total, correct = 0, 0
        for x, y in train_set:
            optimizer.zero_grad()
            x, y = x.to(device), y.to(device).long()
            y_p = model(x)
            loss = criterion(y_p, y)
            loss.backward()
            optimizer.step()
        
        dev(dev_set, model, criterion)
        lr_sch.step()



def dev(dev_set, model, criterion):
    correct,total = 0,0
    model.eval()
    for x, y in dev_set:
        x, y = x.to(device), y.to(device).long()
        with torch.no_grad():
            y_p = model(x)
            loss = criterion(y_p, y)
            correct += np.sum((torch.argmax(F.softmax(y_p, dim=1), dim=1) == y).numpy())
            total += y.shape[0]

    print('dev acc: %.2f%%' % (correct / total * 100))


def test(test_set, model):
    correct,total = 0,0
    model.eval()
    for x, y in test_set:
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            y_p = torch.argmax(F.softmax(model(x), dim=1), dim=1)
            correct += np.sum((y_p == y).numpy())
            total += y.shape[0]

    print('test acc: %.2f%%' % (correct / total * 100))


def main():
    # load data
    train_set = get_data('train')
    test_set = get_data('test')

    dr = config['dropout_rt']
    if config['model'] == 'base':
        model = Base(dr)
    elif config['model'] == 'naive':
        model = Naive(dr)
    elif config['model'] == 'deep':
        model = Deep(dr)
    
    model.apply(weight_init)

    tstart = time.time()
    train(train_set, test_set, model)
    tend = time.time()
    print(f'Training takes %.2fms per epoch.' % ((tend - tstart) / config['epoch'] * 1000))

    test(test_set, model)


if __name__ == '__main__':
    main()
