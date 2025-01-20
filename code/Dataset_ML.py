
from torch.utils.data import Dataset, DataLoader
from glob import glob
import numpy as np
import pandas as pd
import os 

import cv2
import torch 
from utils_ML import *
import vidaug.augmentors as va

class video_dataset(Dataset):
    """ Video Dataset.
    
    """
    # def __init__(self, class0_csv_path, class1_csv_path, class2_csv_path, class3_csv_path, transforms, padding_type, is_train=True): # case 1
    def __init__(self, csv_path, transforms, img_size, is_train=True): 

        # class 0 / class 1, class 2 / class 3
        self.csv_path = csv_path
        
        self.video_df = pd.read_csv(self.csv_path)

        self.transforms = transforms
        self.is_train = is_train

        self.video_path_list = [str(i) for i in self.video_df[f'{img_size}_clip_path']] 
        
        '''
        # 5 artifacts class
        self.PRED_LABEL = [
            'A-line_lbl',
            'B-line_lbl',
            'Confluent B-line_lbl',
            'Consolidation_lbl',
            'Pleural effusion_lbl'
            ]
        '''
        
        # 4 artifacts class
        self.PRED_LABEL = [
            'A-line_lbl',
            'total-B-line_lbl',
            'Consolidation_lbl',
            'Pleural effusion_lbl'
            ]

    def __len__(self):
        return len(self.video_df)
    
    def __getitem__(self, idx):
         
        sampled_clip = load_video(self.video_path_list[idx])

        if self.is_train:
            # apply augmentation
            sometimes = lambda aug: va.Sometimes(0.5, aug)

            sigma = 0.7
            seq = va.Sequential([ # randomly rotates the video with a degree randomly choosen from [-10, 10]  
                sometimes(va.HorizontalFlip()),
                sometimes(va.RandomRotate(degrees=10)),
#                 sometimes(va.InverseOrder()),
#                 sometimes(va.GaussianBlur(sigma))
            ])
            sampled_clip = np.array(seq(sampled_clip))
        
        augmented_images = []
        for frame in sampled_clip:
            augmented_image = torch.from_numpy(self.transforms(image=frame)['image']).permute(2, 0, 1)
            augmented_images.append(augmented_image)
            
        torch_auged_clip = torch.concat([f[None] for f in augmented_images])

        label = torch.FloatTensor(np.zeros(len(self.PRED_LABEL), dtype=float))
        
        for i in range(0, len(self.PRED_LABEL)):
            if (self.video_df[self.PRED_LABEL[i].strip()].iloc[idx].astype('float') > 0):
                label[i] = self.video_df[self.PRED_LABEL[i].strip()].iloc[idx].astype('float')
        
        path = self.video_path_list[idx]
                    
        return torch_auged_clip, label, path
