import torchvision.transforms as standard_transforms
from torch.utils.data import DataLoader
import transform as own_transforms
import torch

import numpy as np
import os
import random
from scipy import io as sio
import sys
import torch
from torch.utils import data
from PIL import Image, ImageOps
import pandas as pd

from conf import cfg, cfg_SHHA, cfg_SHHB

class SHHA(data.Dataset):
    def __init__(self, data_path, mode, main_transform=None, img_transform=None, gt_transform=None):
        self.img_path = data_path + '/img'
        self.gt_path = data_path + '/den'
        self.data_files = [filename for filename in os.listdir(self.img_path) \
                           if os.path.isfile(os.path.join(self.img_path,filename))]
        self.num_samples = len(self.data_files) 
        self.main_transform=main_transform  
        self.img_transform = img_transform
        self.gt_transform = gt_transform     
    
    def __getitem__(self, index):
        fname = self.data_files[index]
        img, den = self.read_image_and_gt(fname)      
        if self.main_transform is not None:
            img, den = self.main_transform(img,den) 
        if self.img_transform is not None:
            img = self.img_transform(img)         
        if self.gt_transform is not None:
            den = self.gt_transform(den)               
        return img, den

    def __len__(self):
        return self.num_samples

    def read_image_and_gt(self,fname):
        img = Image.open(os.path.join(self.img_path,fname))
        if img.mode == 'L':
            img = img.convert('RGB')

        den = pd.read_csv(os.path.join(self.gt_path,os.path.splitext(fname)[0] + '.csv'), sep=',',header=None).values
        
        den = den.astype(np.float32, copy=False)    
        den = Image.fromarray(den)  
        return img, den    

    def get_num_samples(self):
        return self.num_samples   

def loading_data_SHHA():
    mean_std = cfg_SHHA.MEAN_STD
    log_para = cfg_SHHA.LOG_PARA
    factor = cfg_SHHA.LABEL_FACTOR
    train_main_transform = own_transforms.Compose([
    	own_transforms.RandomHorizontallyFlip()
    ])
    img_transform = standard_transforms.Compose([
        standard_transforms.ToTensor(),
        standard_transforms.Normalize(*mean_std)
    ])
    gt_transform = standard_transforms.Compose([
        own_transforms.GTScaleDown(factor),
        own_transforms.LabelNormalize(log_para)
    ])
    restore_transform = standard_transforms.Compose([
        own_transforms.DeNormalize(*mean_std),
        standard_transforms.ToPILImage()
    ])

    train_set = SHHA(cfg_SHHA.DATA_PATH+'/train', 'train',main_transform=train_main_transform, img_transform=img_transform, gt_transform=gt_transform)
    train_loader = DataLoader(train_set, batch_size=1, num_workers=8, shuffle=True, drop_last=True)
   
    val_set = SHHA(cfg_SHHA.DATA_PATH+'/test', 'test', main_transform=None, img_transform=img_transform, gt_transform=gt_transform)
    val_loader = DataLoader(val_set, batch_size=cfg_SHHA.VAL_BATCH_SIZE, num_workers=8, shuffle=True, drop_last=False)

    return train_loader, val_loader, restore_transform

class SHHB(data.Dataset):
    def __init__(self, data_path, mode, main_transform=None, img_transform=None, gt_transform=None):
        self.img_path = data_path + '/img'
        self.gt_path = data_path + '/den'
        self.data_files = [filename for filename in os.listdir(self.img_path) \
                           if os.path.isfile(os.path.join(self.img_path,filename))]
        self.num_samples = len(self.data_files) 
        self.main_transform=main_transform  
        self.img_transform = img_transform
        self.gt_transform = gt_transform     
    
    def __getitem__(self, index):
        fname = self.data_files[index]
        img, den = self.read_image_and_gt(fname)      
        if self.main_transform is not None:
            img, den = self.main_transform(img,den) 
        if self.img_transform is not None:
            img = self.img_transform(img)         
        if self.gt_transform is not None:
            den = self.gt_transform(den)               
        return img, den

    def __len__(self):
        return self.num_samples

    def read_image_and_gt(self,fname):
        img = Image.open(os.path.join(self.img_path,fname))
        if img.mode == 'L':
            img = img.convert('RGB')
            
        den = pd.read_csv(os.path.join(self.gt_path,os.path.splitext(fname)[0] + '.csv'), sep=',',header=None).values
        
        den = den.astype(np.float32, copy=False)    
        den = Image.fromarray(den)  
        return img, den    

    def get_num_samples(self):
        return self.num_samples
    
def loading_data_SHHB():
    mean_std = cfg_SHHB.MEAN_STD
    log_para = cfg_SHHB.LOG_PARA
    train_main_transform = own_transforms.Compose([
    	#own_transforms.RandomCrop(cfg_data.TRAIN_SIZE),
    	own_transforms.RandomHorizontallyFlip()
    ])
    val_main_transform = own_transforms.Compose([
        own_transforms.RandomCrop(cfg_SHHB.TRAIN_SIZE)
    ])
    val_main_transform = None
    img_transform = standard_transforms.Compose([
        standard_transforms.ToTensor(),
        standard_transforms.Normalize(*mean_std)
    ])
    gt_transform = standard_transforms.Compose([
        own_transforms.LabelNormalize(log_para)
    ])
    restore_transform = standard_transforms.Compose([
        own_transforms.DeNormalize(*mean_std),
        standard_transforms.ToPILImage()
    ])

    train_set = SHHB(cfg_SHHB.DATA_PATH+'/train', 'train',main_transform=train_main_transform, img_transform=img_transform, gt_transform=gt_transform)
    train_loader = DataLoader(train_set, batch_size=cfg_SHHB.TRAIN_BATCH_SIZE, num_workers=8, shuffle=True, drop_last=True)
    

    val_set = SHHB(cfg_SHHB.DATA_PATH+'/test', 'test', main_transform=val_main_transform, img_transform=img_transform, gt_transform=gt_transform)
    val_loader = DataLoader(val_set, batch_size=cfg_SHHB.VAL_BATCH_SIZE, num_workers=8, shuffle=True, drop_last=False)

    return train_loader, val_loader, restore_transform

def get_min_size(batch):

    min_ht = cfg_SHHA.TRAIN_SIZE[0]
    min_wd = cfg_SHHA.TRAIN_SIZE[1]

    for i_sample in batch:
        
        _,ht,wd = i_sample.shape
        if ht<min_ht:
            min_ht = ht
        if wd<min_wd:
            min_wd = wd
    return min_ht,min_wd

def random_crop(img,den,dst_size):

    _,ts_hd,ts_wd = img.shape

    x1 = random.randint(0, ts_wd - dst_size[1])//cfg_SHHA.LABEL_FACTOR*cfg_SHHA.LABEL_FACTOR
    y1 = random.randint(0, ts_hd - dst_size[0])//cfg_SHHA.LABEL_FACTOR*cfg_SHHA.LABEL_FACTOR
    x2 = x1 + dst_size[1]
    y2 = y1 + dst_size[0]

    label_x1 = x1//cfg_SHHA.LABEL_FACTOR
    label_y1 = y1//cfg_SHHA.LABEL_FACTOR
    label_x2 = x2//cfg_SHHA.LABEL_FACTOR
    label_y2 = y2//cfg_SHHA.LABEL_FACTOR

    return img[:,y1:y2,x1:x2], den[label_y1:label_y2,label_x1:label_x2]


