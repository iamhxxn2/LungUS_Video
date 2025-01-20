import random 

import torch
import torchvision.transforms as tfs
import albumentations as A
from  albumentations.pytorch import ToTensorV2
import cv2
from glob import glob
import numpy as np
import pandas as pd
import os
import imageio

from torch import nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

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
    
# raw_video에서 Frame 추출하여 clip 형성
def sampling_frame(np_video, sampling_type, select_frame_num, sampling_period):
    
    total_frame_num = np_video.shape[0]

    # 만약 input 비디오의 frame수가 select_frame(min_frame)와 동일한 경우
    # 이 부분 selcet_frame_num이 16이면 에러남 왜냐하면 마지막 이미지가 그냥 검은 이미지라 15장이 되기 때문에 
    if total_frame_num == select_frame_num:
        return np_video
        # selected_clip = np_video

    # 만약 비디오의 frame수가 select_frame(min_frame)보다 클 경우 
    # -> random sampling / regular sampling 방법에 따라 진행     
    elif total_frame_num > select_frame_num:
        
        if sampling_type == 'random':

            selected_frame_index = random.sample(range(total_frame_num), select_frame_num)[:select_frame_num]

        # uniformly sampled along the whole temporal dimension
        elif sampling_type == 'regular':
            
            # 특정 Period로 추출했을 때 select_frame_num보다 작을 경우 Random하게 frame select
            if ((total_frame_num / sampling_period) < select_frame_num):
                selected_frame_index = random.sample(range(total_frame_num), select_frame_num)
            else:
                selected_frame_index = list(range(0, total_frame_num, sampling_period))[:select_frame_num]
        return np_video[selected_frame_index]
        # selected_clip = np_video[selected_frame_index]
    # return selected_clip

def apply_transforms(mode):
    
    if mode == 'train':
        transforms = A.Compose([
            # A.HorizontalFlip(p=0.5),
            A.Normalize()
            ], p=1) 
    else:
        transforms = A.Compose([
            A.Normalize()
            ], p=1)
    
    return transforms
'''
def load_video(video_path):
    
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    video_frame_num = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    raw_video = []
    
    # transform all frames with numpy array
    for i in range(video_frame_num):
        ret, frame = cap.read()
        if ret:
            raw_video.append(frame)
        else:
            break
    np_raw_video = np.array(raw_video)
    
    # sampling frame
    # sampled_clip = sampling_frame(np_raw_video, sampling_type=sampling_type, select_frame_num=select_frame_num, sampling_period=sampling_period)

    return np_raw_video
'''

def load_video(video_path):
    
    reader = imageio.get_reader(video_path)
    
    # 모든 프레임을 리스트로 저장
    frames = []
    for frame in reader:
        frames.append(frame)

    np_raw_video = np.array(frames)

    return np_raw_video

def gen_new_dir(new_dir):
    """make new directory
    Parameters:
        new_dir (str) -- new directory
    Return:
        None
    """
    try:
        if not os.path.exists(new_dir):
            os.makedirs(new_dir)
            print(f"New directory!: {new_dir}")
    except OSError:
        print("Error: Failed to create the directory.")

def checkpoint(seed_num, batch_size, version, frame_size, data_type, 
               train_layer, testset_rate, checkpoint_std, 
               model, model_output_class, save_path, model_name, 
               model_version, encoder_name, encoder_batch_size, 
               pooling_method, num_heads, fold_number, lr,
               best_epoch, best_valid_loss, best_valid_acc, 
               best_valid_auc, best_thresholds):
    """
    Saves checkpoint of torchvision model during training.
    Args:
        model: torchvision model to be saved
        best_loss: best val loss achieved so far in training
        epoch: current epoch of training
        LR: current learning rate in training
    Returns:
        None
    """

    print('saving')
    state = {
        'model': model.state_dict(),
        'best_epoch': best_epoch,
        'best_valid_loss': best_valid_loss,
        'best_valid_acc': best_valid_acc,
        'best_valid_auc': best_valid_auc,
        'best_valid_thres':best_thresholds
    }
    
    fold_path = os.path.join(save_path, f'seed{seed_num}_test{testset_rate}_std_{checkpoint_std}_{data_type}_{version}_{train_layer}_{model_output_class}_artifacts_duplicate_batch{batch_size}_{frame_size}_30frame_{model_name}_{model_version}_{encoder_name}_{encoder_batch_size}_{pooling_method}_{num_heads}head_fold{fold_number}_lr{lr}_checkpoint')
    os.makedirs(os.path.dirname(fold_path), exist_ok=True)
    torch.save(state, fold_path)

def checkpoint2(seed_num, batch_size, version, frame_size, data_type, 
               train_layer, testset_rate, checkpoint_std, 
               model, model_output_class, save_path, model_name, 
               model_version, encoder_name, encoder_batch_size, 
               pooling_method, num_heads, kernel_width, fold_number, lr,
               best_epoch, best_valid_loss, best_valid_acc, 
               best_valid_auc, best_thresholds):
    """
    Saves checkpoint of torchvision model during training.
    Args:
        model: torchvision model to be saved
        best_loss: best val loss achieved so far in training
        epoch: current epoch of training
        LR: current learning rate in training
    Returns:
        None
    """

    print('saving')
    state = {
        'model': model.state_dict(),
        'best_epoch': best_epoch,
        'best_valid_loss': best_valid_loss,
        'best_valid_acc': best_valid_acc,
        'best_valid_auc': best_valid_auc,
        'best_valid_thres':best_thresholds
    }
    
    fold_path = os.path.join(save_path, f'seed{seed_num}_test{testset_rate}_std_{checkpoint_std}_{data_type}_{version}_{train_layer}_{model_output_class}_artifacts_duplicate_batch{batch_size}_{frame_size}_30frame_{model_name}_{model_version}_{encoder_name}_{encoder_batch_size}_{pooling_method}_{num_heads}head_{kernel_width}ksize_fold{fold_number}_lr{lr}_checkpoint')
    os.makedirs(os.path.dirname(fold_path), exist_ok=True)
    torch.save(state, fold_path)

