import os
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data

from torch.utils.data.distributed import DistributedSampler

from random import Random
from torch.multiprocessing import Process
from torch.autograd import Variable

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
import partition_dataset as partition_dataset
""" Dataset partitioning helper """
class Partition(object):

    def __init__(self, data, index):
        self.data = data
        self.index = index

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        data_idx = self.index[index]
        return self.data[data_idx]



def partition_dataset_mnist(homogen,classorder, class_dist, cl_worker, data_ratio= 1, fmnist = False, tau=20, new_het=False):
    normalize = transforms.Normalize(mean=[0.1307]
                                ,std=[0.3081])

    if fmnist:
        train_data = datasets.FashionMNIST(root='./data', train=True, transform=transforms.Compose([
            transforms.ToTensor(),
        ]), download=True)

        test_data = datasets.FashionMNIST(root='./data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
        ]), download=True)
    else:
        train_data = datasets.MNIST(root='./data', train=True, transform=transforms.Compose([
            transforms.ToTensor(),
        ]), download=True)


        test_data = datasets.MNIST(root='./data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
        ]), download=True)
    size = dist.get_world_size()
    
    partition_sizes = [1.0 / size for _ in range(size)]


    indexes_toadd = np.arange(int(size)).reshape(int(size),1)
                

    train_partition = DataPartitioner_mnist(train_data, partition_sizes, cl_worker, new_het, homogen, classorder, indexes_toadd , False , data_ratio)

    indexes_toadd = train_partition.indexes_toadd
    train_partition = train_partition.use(dist.get_rank())

    test_partition = DataPartitioner_mnist(test_data, partition_sizes, cl_worker, new_het, homogen, classorder, indexes_toadd,  False, 1)
    test_partition = test_partition.use(dist.get_rank())

    n_train_examples = int(len(train_partition))
    n_test_examples = int(len(test_partition))
    bsz =  np.floor(n_train_examples/tau) #128 / float(size)



    train_iterator = torch.utils.data.DataLoader(train_partition,
                                          batch_size=int(bsz),
                                          shuffle=True)
    
    valid_iterator = torch.utils.data.DataLoader(test_partition,
                                                 batch_size=int(bsz),
                                         shuffle=True)
    



    test_iterator = data.DataLoader(test_data)

    return train_iterator, valid_iterator, bsz




class DataPartitioner_mnist(object):

    def __init__(self, data, sizes, cl_worker, new_het, homogen, classorder, indexes_toadd, trainset, data_ratio, seed=1234):
        self.data = data
        self.partitions = []
        rng = Random()
        rng.seed(seed)
        data_len = len(data)
        indexes = []
        no_of_labels = []
        label_chunk = []
        for i in range(10):
            no_of_labels.append(torch.sum(data.targets==i))
            label_chunk.append(torch.sum(data.targets==i)//((len(sizes)*cl_worker/10)/data_ratio)) #was //20

        cum_no_of_labels = np.cumsum(np.asarray(no_of_labels))
        
        
        if homogen == True:
            indexes = [x for x in range(0, data_len)]
            rng.shuffle(indexes)
        else:
            if type(data.targets) == list:
                datatarget_tensor = torch.Tensor(data.targets)
            else:
                datatarget_tensor = torch.Tensor(data.targets.float())
            sorted, indices = torch.sort(datatarget_tensor)
            indexes = indexes + indices.tolist()

        if new_het:
            indexes_new_temp = []
            indexes_new_temp += indexes

        frac = 1/len(classorder)
        
        partial_ratio = 1
        size = dist.get_world_size()
        for ind in range(len(sizes)):
            indxs_tmp = []
            targets_temp = []
            for j in range(cl_worker):                
                base_class = classorder[indexes_toadd[ind,j]]//(cl_worker*size/10)
                part_len = int(label_chunk[int(base_class)]*partial_ratio)
                rmn = int(classorder[indexes_toadd[ind,j]]%(cl_worker*size/10))
                if base_class != 0:
                    indxs_tmp = indxs_tmp + indexes[cum_no_of_labels[int(base_class-1)]+rmn*part_len:cum_no_of_labels[int(base_class-1)]+(rmn+1)*part_len]
                else:
                    indxs_tmp = indxs_tmp + indexes[rmn*part_len:(rmn+1)*part_len]

            if not homogen:
                rng.shuffle(indxs_tmp)
            self.partitions.append(indxs_tmp)
        self.indexes_toadd = indexes_toadd
    def use(self, partition):
        return Partition(self.data, self.partitions[partition])
