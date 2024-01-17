#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import torch
from torch import nn


def FedAvg(w,dict_data_ratio_list):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(len(w)):
            if i == 0:
                w_avg[k] *= dict_data_ratio_list[0]
            else:
                w_avg[k] += w[i][k] * dict_data_ratio_list[i]
    return w_avg
