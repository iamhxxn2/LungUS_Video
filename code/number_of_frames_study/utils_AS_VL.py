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

def checkpoint(batch_size, version, frame_num, frame_size, data_type, 
               train_layer, testset_rate, checkpoint_std, 
               model, model_output_class, save_path, model_name, 
               encoder_name, encoder_batch_size, pooling_method, 
               fold_number, best_epoch, best_valid_loss, 
               best_valid_acc, best_valid_auc, best_thresholds, lr):
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
    
    fold_path = os.path.join(save_path, f'test{testset_rate}_std_{checkpoint_std}_{data_type}_{version}_{train_layer}_{model_output_class}_artifacts_duplicate_batch{batch_size}_{frame_size}_{frame_num}frame_{model_name}_{encoder_name}_{encoder_batch_size}_{pooling_method}_fold{fold_number}_lr{lr}_checkpoint')
    os.makedirs(os.path.dirname(fold_path), exist_ok=True)
    torch.save(state, fold_path)
    
def checkpoint2(seed_num, batch_size, version, frame_num, frame_size, 
                data_type, train_layer, testset_rate, checkpoint_std, 
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
    
    fold_path = os.path.join(save_path, f'seed{seed_num}_test{testset_rate}_std_{checkpoint_std}_{data_type}_{version}_{train_layer}_{model_output_class}_artifacts_duplicate_batch{batch_size}_{frame_size}_{frame_num}frame_{model_name}_{model_version}_{encoder_name}_{encoder_batch_size}_{pooling_method}_{num_heads}head_{kernel_width}ksize_fold{fold_number}_lr{lr}_checkpoint')
    os.makedirs(os.path.dirname(fold_path), exist_ok=True)
    torch.save(state, fold_path)