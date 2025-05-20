import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

import random 

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import torchvision.transforms as tfs
import albumentations as A
from  albumentations.pytorch import ToTensorV2
import cv2
from glob import glob
import numpy as np
import pandas as pd
import os

def pad(h, num_frames):
    # reshape as batch of vids and pad
    start_ix = 0
    h_ls = []
    for n in num_frames:
        h_ls.append(h[start_ix:(start_ix+n)])
        start_ix += n
    h = pad_sequence(h_ls)

    return h

def num_parameters(module):
    return sum(p.numel() for p in module.parameters() if p.requires_grad)

class EarlyStopper:
    def __init__(self, patience, init_best_value = 1e10):
        self.patience = patience
        self._count_no_improvement = 0
        self.best_val = init_best_value
        
    def stop(self, val):
        if val >= self.best_val:
            self._count_no_improvement += 1
        else:
            self.best_val = val
            self._count_no_improvement = 0
            
        if self._count_no_improvement > self.patience:
            return True
        else:
            return False
        
    def __repr__(self):
        return f"EarlyStopper(patience={self.patience}). Current best: {self.best_val:0.5f}. Steps without improvement: {self._count_no_improvement}"

def apply_transform(mode):
    
    if mode == 'train':
        transform = A.Compose([
                    A.OneOf([
                        A.RandomBrightness(p=1),
                        A.RandomContrast(p=1),
                        # A.RandomBrightnessContrast(p=0.2),
                        A.RandomGamma(p=1)
                        ], p=0.3),
                    A.OneOf([
                        A.MotionBlur(p=1),
                        A.MedianBlur(p=1),        
                        A.GaussNoise(p=1),
                        A.CLAHE(clip_limit=8.0, tile_grid_size=(8, 8),always_apply=False, p=1)
                        ], p=0.3),
                    A.HorizontalFlip(p=0.5),
                    # A.VerticalFlip(p=0.5),
                    A.Rotate(limit=10, p=0.5, border_mode=cv2.BORDER_REPLICATE),
                    A.Normalize(mean=(0.485, 0.456, 0.406),
                        std=(0.229, 0.224, 0.225),
                        max_pixel_value=255.0,
                        always_apply=False,
                        p=1.0,),
                    ToTensorV2()
                    ])
    else:
          transform = A.Compose([
                    # A.Resize(128, 128, interpolation = cv2.INTER_LINEAR),
                    A.Normalize(mean=(0.485, 0.456, 0.406),
                        std=(0.229, 0.224, 0.225),
                        max_pixel_value=255.0,
                        always_apply=False,
                        p=1.0,),
                    ToTensorV2()
                    ])
          
    return transform      
