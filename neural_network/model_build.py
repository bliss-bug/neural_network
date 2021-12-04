import numpy as np
import torch
import torch.nn as nn

def Naive(drop_rt=0.01):
    model = nn.Sequential(
            nn.Linear(16, 32),
            nn.BatchNorm1d(32),
            nn.Dropout(drop_rt),
            nn.ReLU(),
            nn.Linear(32, 10)
        )
    return model



def Base(drop_rt=0.01):
    model = nn.Sequential(
            nn.Linear(16, 32),
            nn.BatchNorm1d(32),
            #nn.ReLU(),
            #nn.Sigmoid(),
            nn.Tanh(),
            nn.Dropout(drop_rt),
            
            nn.Linear(32, 64),
            nn.BatchNorm1d(64),
            #nn.ReLU(),
            #nn.Sigmoid(),
            nn.Tanh(),
            nn.Dropout(drop_rt),

            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            #nn.ReLU(),
            #nn.Sigmoid(),
            nn.Tanh(),
            nn.Dropout(drop_rt),
            nn.Linear(32, 10)
        )
    return model



def Deep(drop_rt=0.01):
    model = nn.Sequential(
            nn.Linear(16, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(drop_rt),
            
            nn.Linear(32, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(drop_rt),

            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(drop_rt),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(drop_rt),

            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(drop_rt),
            nn.Linear(32, 10)
        )
    return model


def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0)
