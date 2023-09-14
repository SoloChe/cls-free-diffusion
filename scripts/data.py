import torch
import random
from random import randint
from pathlib import Path
import numpy as np
import nibabel as nib
from tqdm import tqdm
import os
import logging
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms, datasets
import torch.nn.functional as F
from torchvision.utils import make_grid, save_image
from torchvision import transforms, datasets

#%% cifar10
class cifar10(Dataset):
    def __init__(self, data_dir, split, transform=None, ret_lab=False):
        self.data_dir = data_dir
        self.transform = transform
        self.ret_lab = ret_lab
        train = True if split == 'train' else False
        self.data = datasets.CIFAR10(root=data_dir, train=train, download=True)
        self.data_size = self.data.__len__()
        
    def __len__(self):
        return self.data_size
    
    def __getitem__(self, idx):
        x, y = self.data[idx]
        if self.transform is not None:
            x = self.transform(x)
        if not self.ret_lab:
            return x, {}
        else:
            return x, {'y':y}
    
def get_cifar10_iter(data_dir, batch_size, split, ret_lab=False, logger=None, training=True):
    transform=transforms.Compose([
                                  transforms.RandomHorizontalFlip(),
                                  transforms.ToTensor(),
                                  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                  ])
    data = cifar10(data_dir, split, ret_lab=ret_lab, transform=transform)
    loader = DataLoader(data, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)
    if logger is not None:
        logger.log(f"data_size: {data.__len__()}")

    if training:
        while True:
            yield from loader
    else:
        yield from loader
    return 

def get_data_iter(name, data_dir, batch_size, 
                  split='train', ret_lab=False, 
                  training=True, logger=None,
                  kwargs=None):
            

    if name.lower() == 'cifar10':
        return get_cifar10_iter(data_dir, batch_size, 
                                split=split, ret_lab=ret_lab,
                                logger=logger, training=training)
                               
    else:
        raise NotImplementedError
        



