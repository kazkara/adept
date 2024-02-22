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


class DataPartitioner(object):

    def __init__(self, data, sizes, cl_worker, new_het, homogen, classorder, indexes_toadd, trainset, data_ratio = 1.0,  seed=1234):
        self.data = data
        self.partitions = []
        rng = Random()
        rng.seed(seed)
        data_len = len(data)
        indexes = []

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
        

        size = dist.get_world_size()
        classorder_dyn = []
        classorder_dyn= classorder[:]
        classorder_temp = []
        coef = len(classorder)/(max(data.targets)+1)
        if trainset:
            indexes_toadd = -1*np.ones((size, cl_worker),dtype=np.int32)
        for ind in range(len(sizes)):
            part_len = int(frac * data_len) #IMPORTANT
            if trainset:
                
                for j in range(cl_worker):

                    classorder_temp = classorder_dyn[:]
                    for k in range(j+1):
                        if indexes_toadd[ind,k]>= 0:
                            temp_ind = indexes_toadd[ind,k]
                            rmn = temp_ind%coef
                            for m in range((temp_ind-(rmn)).astype(int),(temp_ind+coef-rmn).astype(int)):
                                if m in classorder_temp:
                                    classorder_temp.remove(m)
                                
                        else:
                            if len(classorder_temp)>1:
                                choose = random.randint(0, len(classorder_temp)-1)
                            else:
                                choose = 0

                            if not classorder_temp:
                                choose = random.randint(0, len(classorder)-1)
                                indexes_toadd[ind,k] = classorder[choose]
                                if classorder[choose] in classorder_dyn:
                                    classorder_dyn.remove(classorder_temp[choose]) 
                            else:
                                indexes_toadd[ind,k] = classorder_temp[choose]
                                if classorder_temp[choose] in classorder_dyn:
                                    classorder_dyn.remove(classorder_temp[choose])
            
            indxs_tmp = []
            targets_temp = []
            for j in range(cl_worker):
                if not new_het:
                    indxs_tmp = indxs_tmp + indexes[classorder[indexes_toadd[ind,j]]*part_len:(classorder[indexes_toadd[ind,j]]+1)*part_len]
                else:
                    indxs_tmp = indxs_tmp + indexes[classorder[indexes_toadd[ind,j]]*part_len:(classorder[indexes_toadd[ind,j]]*part_len+int(part_len*(cl_worker)*0.225))]
            
            if new_het:
                indexes_new_temp = [ m for m in indexes_new_temp if m not in indxs_tmp ]
                #indexes_tmp_tmp += indexes_new_temp
                perm_tmp = np.random.permutation(indexes_new_temp)
                indxs_tmp += perm_tmp[0:int(data_len/size-len(indxs_tmp))].tolist()


            rng.shuffle(indxs_tmp)
            newlen = int(len(indxs_tmp)*data_ratio)
            self.partitions.append(indxs_tmp[0:newlen])
        self.indexes_toadd = indexes_toadd

    def use(self, partition):
        return Partition(self.data, self.partitions[partition])

def sparse2coarse(targets):
    """Convert Pytorch CIFAR100 sparse targets to coarse targets.
    Usage:
        trainset = torchvision.datasets.CIFAR100(path)
        trainset.targets = sparse2coarse(trainset.targets)
    """
    coarse_labels = np.array([ 4,  1, 14,  8,  0,  6,  7,  7, 18,  3,  
                            3, 14,  9, 18,  7, 11,  3,  9,  7, 11,
                            6, 11,  5, 10,  7,  6, 13, 15,  3, 15,  
                            0, 11,  1, 10, 12, 14, 16,  9, 11,  5, 
                            5, 19,  8,  8, 15, 13, 14, 17, 18, 10, 
                            16, 4, 17,  4,  2,  0, 17,  4, 18, 17, 
                            10, 3,  2, 12, 12, 16, 12,  1,  9, 19,  
                            2, 10,  0,  1, 16, 12,  9, 13, 15, 13, 
                            16, 19,  2,  4,  6, 19,  5,  5,  8, 19, 
                            18,  1,  2, 15,  6,  0, 17,  8, 14, 13])
    return coarse_labels[targets]

    """ Partitioning CIFAR-10 """
def partition_dataset(homogen,classorder, class_dist, cl_worker, cifar100 = False, data_ratio = 1.0, tau=20, seed=1234, new_het=False):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406]
                                ,std=[0.229, 0.224, 0.225])
    
    # normalize = transforms.Normalize(mean=[0, 0, 0]
    #                             ,std=[0.229, 0.224, 0.225])
    # torch.manual_seed(seed)

    if not cifar100:
        train_data = datasets.CIFAR10(root='./data', train=True, transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            # transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            normalize,
        ]), download=True)

        test_data = datasets.CIFAR10(root='./data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ]))
    else:
        train_data = datasets.CIFAR100(root='./data', train=True, transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            # transforms.RandomCrop(32, 4),
            transforms.ToTensor()
                    ]), download=True)

        test_data = datasets.CIFAR100(root='./data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
        ]))
    
    #shuffle the data
    train_data = torch.utils.data.Subset(train_data, torch.randperm(len(train_data))).dataset
    test_data = torch.utils.data.Subset(test_data, torch.randperm(len(test_data))).dataset

    size = dist.get_world_size()
    if size==20:
        bsz =  50 
    else:
        bsz = 25 

    partition_sizes = [1.0 / size for _ in range(size)]
    
    
    indexes_toadd = []

    ## to use the same class distribution for each run
    if cl_worker == 1:
        indexes_toadd = np.arange(int(size)).reshape(int(size),1)
    
    if cifar100:
        temp_train_targets = np.asarray(train_data.targets)
        temp_test_targets = np.asarray(test_data.targets)
        train_data.targets = sparse2coarse(train_data.targets).tolist()
        test_data.targets = sparse2coarse(test_data.targets).tolist()
    
    train_partition = DataPartitioner(train_data, partition_sizes, cl_worker, new_het, homogen, classorder, indexes_toadd , False, data_ratio, seed)
    indexes_toadd = train_partition.indexes_toadd
    


    train_partition = train_partition.use(dist.get_rank())

    

    test_partition = DataPartitioner(test_data, partition_sizes, cl_worker, new_het, homogen, classorder, indexes_toadd,  False, data_ratio, seed)
    
    test_partition = test_partition.use(dist.get_rank())

    if cifar100:
        # Go back to fine labels
        train_partition.data.targets = temp_train_targets
        test_partition.data.targets = temp_test_targets

    n_train_examples = int(len(train_partition))
    n_test_examples = int(len(test_partition))


    bsz =  np.floor(n_train_examples/tau)

    train_iterator = torch.utils.data.DataLoader(train_partition,
                                          batch_size=int(bsz),
                                          shuffle=True)
    
    valid_iterator = torch.utils.data.DataLoader(test_partition,
                                         batch_size=100,
                                         shuffle=True)



    return train_iterator, valid_iterator, bsz


def partition_dataset_mnist(homogen,classorder, class_dist, cl_worker, data_ratio= 1, fmnist = False, tau=20, seed=1234 ,new_het=False):
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
    
    

    ## to use the same class distribution for each run
    if cl_worker == 1:
        indexes_toadd = np.arange(int(size)).reshape(int(size),1)
                

    train_partition = DataPartitioner_mnist(train_data, partition_sizes, cl_worker, new_het, homogen, classorder, indexes_toadd , False , data_ratio, seed)

    indexes_toadd = train_partition.indexes_toadd
    train_partition = train_partition.use(dist.get_rank())

    test_partition = DataPartitioner_mnist(test_data, partition_sizes, cl_worker, new_het, homogen, classorder, indexes_toadd,  False, 1, seed)
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
             #IMPORTANT
            indxs_tmp = []
            targets_temp = []
            for j in range(cl_worker):                
                base_class = classorder[indexes_toadd[ind,j]]//(cl_worker*size/10)
                #experiment
                part_len = int(label_chunk[int(base_class)]*partial_ratio)
                rmn = int(classorder[indexes_toadd[ind,j]]%(cl_worker*size/10))
                if base_class != 0:
                    indxs_tmp = indxs_tmp + indexes[cum_no_of_labels[int(base_class-1)]+rmn*part_len:cum_no_of_labels[int(base_class-1)]+(rmn+1)*part_len]
                else:
                    indxs_tmp = indxs_tmp + indexes[rmn*part_len:(rmn+1)*part_len]
                #targets_temp += classorder[indexes_toadd[ind,j]]//10

            if not homogen:
                rng.shuffle(indxs_tmp)
            self.partitions.append(indxs_tmp)
        self.indexes_toadd = indexes_toadd
    def use(self, partition):
        return Partition(self.data, self.partitions[partition])


