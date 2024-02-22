import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data

import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

from pathlib import Path

import json

import matplotlib.pyplot as plt
import numpy as np

import copy
import random
import time
import math



import torch.backends.cudnn as cudnn

def logsumexp(x):
        a = torch.max(x)
        return a+torch.log(torch.sum(torch.exp(x-a)))

class logsumexp_loss(torch.nn.Module):

    def __init__(self):
        super(logsumexp_loss,self).__init__()

    def logsumexp(x): #logsumexp trick
        a = torch.max(x)
        return a+torch.log(torch.sum(torch.exp(x-a)))

    def forward(self, pers_model, glob_models, sigma_list, probs, PERS=True, beta=1e-5, alpha = -1):

        tmp = 0
       
        delta = 0 #1e-3 2.6 used 1e-3 or 1e-2 for linear regression
        if(PERS):
            for sigma in sigma_list:
                tmp +=  torch.sum( torch.log( sigma + delta ) )
            a = tmp.item()
            #scaling the effect:
            tmp = tmp * (2*alpha+3)


            i = 0
            b = 0
            for p_p, p in zip(pers_model.parameters(), glob_models.parameters()):
                max_i = torch.max(sigma_list[i])
                coef = 1
                b += (1/2) * torch.norm( torch.div(torch.sqrt(2*beta+coef*(p-p_p)**2), sigma_list[i]+delta) )**2 #gaussian

                i += 1
            tmp += b
            b = b.item()
            tmp = tmp 
            return tmp, a, b

        else:
            for p_p, p in zip(pers_model.parameters(), glob_models.parameters()):
                tmp += torch.norm(p-p_p)**2
                
        return tmp

    