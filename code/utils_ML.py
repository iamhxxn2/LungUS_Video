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

def Saved_items(save_path, model_name, clip_frame_num, sampling_period, fold_number, epoch_losses_train, epoch_losses_val, epoch_accs_train, epoch_accs_val, epoch_auc_train, epoch_auc_val, time_elapsed, batch_size):
    """
    Saves checkpoint of torchvision model during training.
    Args:

        epoch_losses_train: training losses over epochs
        epoch_losses_val: validation losses over epochs

    """
    print('saving')
    state2 = {
        'epoch_losses_train': epoch_losses_train,
        'epoch_losses_val': epoch_losses_val,
        'epoch_accs_train': epoch_accs_train,
        'epoch_accs_val': epoch_accs_val,
        'epoch_auc_train': epoch_auc_train,
        'epoch_auc_valid': epoch_auc_val,
        'time_elapsed': time_elapsed,
        "batch_size": batch_size
    }
    
    fold_path = os.path.join(save_path, f'USVN_fold{fold_number}_{clip_frame_num}_{sampling_period}_s1')
    torch.save(state2, fold_path)

def checkpoint(seed_num, batch_size, version, frame_size, 
               train_layer, testset_rate, checkpoint_std, 
               model, model_output_class, save_path, model_name, 
               encoder_name, pooling_method, fold_number, best_epoch, 
               best_valid_loss, best_valid_acc, best_valid_auc, best_thresholds, lr):
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
    
    fold_path = os.path.join(save_path, f'seed{seed_num}_test{testset_rate}_std_{checkpoint_std}_{version}_{train_layer}_{model_output_class}_artifacts_duplicate_batch{batch_size}_{frame_size}_30frame_{model_name}_{encoder_name}_{pooling_method}_fold{fold_number}_lr{lr}_checkpoint')
    os.makedirs(os.path.dirname(fold_path), exist_ok=True)
    torch.save(state, fold_path)

def checkpoint_asvl(seed_num, batch_size, version, frame_num, frame_size, 
                train_layer, testset_rate, checkpoint_std, 
               model, model_output_class, save_path, model_name, encoder_name, 
               pooling_method, num_heads, kernel_width, fold_number, lr,
               best_epoch, best_valid_loss, best_valid_acc, 
               best_valid_auc, best_thresholds):

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
    
def checkpoint2(seed_num, batch_size, version, frame_size, 
               train_layer, testset_rate, checkpoint_std, 
               model, model_output_class, save_path, model_name, encoder_name,
               pooling_method, num_heads, kernel_width, fold_number, lr,
               best_epoch, best_valid_loss, best_valid_acc, 
               best_valid_auc, best_thresholds):

    print('saving')
    state = {
        'model': model.state_dict(),
        'best_epoch': best_epoch,
        'best_valid_loss': best_valid_loss,
        'best_valid_acc': best_valid_acc,
        'best_valid_auc': best_valid_auc,
        'best_valid_thres':best_thresholds
    }
    
    fold_path = os.path.join(save_path, f'seed{seed_num}_test{testset_rate}_std_{checkpoint_std}_{version}_{train_layer}_{model_output_class}_artifacts_duplicate_batch{batch_size}_{frame_size}_30frame_{model_name}_{encoder_name}_{pooling_method}_{num_heads}head_{kernel_width}ksize_fold{fold_number}_lr{lr}_checkpoint')
    os.makedirs(os.path.dirname(fold_path), exist_ok=True)
    torch.save(state, fold_path)

def checkpoint2_asvl(seed_num, batch_size, version, frame_num, frame_size, 
                train_layer, testset_rate, checkpoint_std, 
               model, model_output_class, save_path, model_name, encoder_name,
               pooling_method, num_heads, kernel_width, fold_number, lr,
               best_epoch, best_valid_loss, best_valid_acc, 
               best_valid_auc, best_thresholds):

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

