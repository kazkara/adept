
import os
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data

from random import Random
from torch.multiprocessing import Process
from torch.autograd import Variable
import torch.nn.utils.clip_grad as clip_grad
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torchvision.utils import make_grid

from torchmetrics.functional import structural_similarity_index_measure
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.kid import KernelInceptionDistance
from torchmetrics.image.inception import InceptionScore

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np

from torchvision.utils import save_image
import copy
import random
import time
import math

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
torch.manual_seed(0)
class transformed_data(Dataset):
  def __init__(self, img, device):
    self.img = img  #img path
    self.len = len(os.listdir(self.img))
    self.device = device

  def __getitem__(self, index):
    ls_img = sorted(os.listdir(self.img))

    img_file_path = os.path.join(self.img, ls_img[index])
    img_tensor = torch.load(img_file_path, map_location= self.device)


    return img_tensor

  def __len__(self):
    return self.len         

def evaluate_fid(og_iterator, other_iterator,device):
    
    epoch_fid= 0
    epoch_fid_std = 0
    t = 0
    
    with torch.no_grad():
        
        for i, (im1, im2) in enumerate(zip(og_iterator,other_iterator)):
            
            im1 = im1.to(device)
            im2 = im2.to(device)
            print(im1.shape)
            if not cifar100:
              im1 = im1.repeat(1,3,1,1)
              im2 = im2[0:200]
              im2 = torch.reshape(im2,(200,1,28,28))
              im2 = im2.repeat(1,3,1,1)
            else:
              im2 = im2[0:200]
              im2 = torch.reshape(im2,(200,3,32,32))

            im1 = (im1 * 255).byte()
            im2 = (im2 * 255).byte()
            if fd:
              fid = FrechetInceptionDistance(feature = 2048)
            
              fid.to(device)
              fid.update(im1, real=True)
              fid.update(im2, real=False)

              epoch_fid += fid.compute().item()

            else:
              fid = KernelInceptionDistance(feature = 2048, subsets = 100, subset_size=100)
            
              fid.to(device)
              fid.update(im1, real=True)
              fid.update(im2, real=False)
              epoch_fid += fid.compute()[0].item()
              epoch_fid_std += fid.compute()[1].item()

            
              t += 1

            break
    return epoch_fid 

size = 50
device = "cuda:0"
cifar100 = False
fdlist = [False, True]
directory = "" #enter directory
for fd in fdlist:

  val_fid = []
  for i in range(size):
      dataset_original = transformed_data(directory + "images/{}/original".format(i), device)
      dataset_pers = transformed_data(directory + "images/{}/local".format(i), device)
      original_iterator = DataLoader(dataset_original, batch_size=200, shuffle=False)
      pers_iterator = DataLoader(dataset_pers, batch_size=200, shuffle=False)
      val_fid.append(evaluate_fid(original_iterator, pers_iterator, device))
      print("rank" + str(i) + ": ", str(val_fid[-1]))
      
  val_fid_avg = np.mean(val_fid)
  val_fid.append(val_fid_avg)
  print(val_fid_avg)
  if fd:
    save_place_fid_txt =  directory + '_fid_list5' + '.txt'
  else:
    save_place_fid_txt =  directory + '_kid_list5' + '.txt' 
  with open(save_place_fid_txt, 'w') as filehandle:
      json.dump(val_fid, filehandle)