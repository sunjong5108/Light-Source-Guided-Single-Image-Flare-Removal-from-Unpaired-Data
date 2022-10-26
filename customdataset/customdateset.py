import os
import random
from PIL import Image
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader
import numpy as np
                   
class TrainLoadDataset(Dataset):
  def __init__(self, inp_dir, tar_dir):
    super(TrainLoadDataset, self).__init__()

    self.inp_dir = inp_dir
    self.tar_dir = tar_dir
    self.length = len(self.inp_dir)

  def __len__(self):
    return self.length

  def __getitem__(self, index):
    index_ = index % self.length

    inp_path = self.inp_dir[index_]
    tar_path = self.tar_dir[index_]

    inp_img = Image.open(inp_path)
    tar_img = Image.open(tar_path)
    
    inp_img = TF.resize(inp_img, size=(512, 512))
    tar_img = TF.resize(tar_img, size=(512, 512))

    inp_img = TF.to_tensor(inp_img)
    tar_img = TF.to_tensor(tar_img)

    aug    = random.randint(0, 1)

    # Data Augmentations
    if aug==1:
        inp_img = inp_img.flip(2)
        tar_img = tar_img.flip(2)

    filename = os.path.splitext(os.path.split(tar_path)[-1])[0] 

    return inp_img, tar_img, filename

class ValTestLoadDataset(Dataset):
  def __init__(self, inp_dir, tar_dir):
    super(ValTestLoadDataset, self).__init__()

    self.inp_dir = inp_dir
    self.tar_dir = tar_dir
    self.length = len(self.inp_dir)

  def __len__(self):
    return self.length

  def __getitem__(self, index):
    index_ = index % self.length

    inp_path = self.inp_dir[index_]
    tar_path = self.tar_dir[index_]

    inp_img = Image.open(inp_path)
    tar_img = Image.open(tar_path)
    
    inp_img = TF.resize(inp_img, size=(512, 512))
    tar_img = TF.resize(tar_img, size=(512, 512))

    inp_img = TF.to_tensor(inp_img)
    tar_img = TF.to_tensor(tar_img)

    filename = os.path.splitext(os.path.split(tar_path)[-1])[0] 

    return inp_img, tar_img, filename

