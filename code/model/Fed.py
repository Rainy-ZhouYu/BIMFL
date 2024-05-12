#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import torch
from torch import nn
from DataSetting import DatasizeUE
import numpy as np

# def FedAvg(w):
#     w_avg = copy.deepcopy(w[0])
#     for k in w_avg.keys():
#         for i in range(1, len(w)):
#             w_avg[k] += w[i][k]
#         w_avg[k] = torch.div(w_avg[k], len(w))
#     return w_avg

def FedAvg(w, idxs_users):
    w_avg = copy.deepcopy(w[0])
    AlldataSize = 0
    for idx in idxs_users:
        AlldataSize += DatasizeUE[idx]
    for k in w_avg.keys(): # parameters
        w_avg[k] = w_avg[k] * DatasizeUE[idxs_users[0]]
        for i in range(1, len(w)): # number of users
            w_avg[k] = w_avg[k] + w[i][k] * DatasizeUE[idxs_users[i]]
        w_avg[k] = torch.div(w_avg[k], AlldataSize)
    return w_avg
