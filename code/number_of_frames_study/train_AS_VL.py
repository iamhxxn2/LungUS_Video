import argparse
import os 
import csv
import wandb
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import pickle

import torch
import torch.nn as nn
import torch.optim as optim
# from timm import optim
import torch.nn.functional as F
from torch.optim import lr_scheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau

import torchvision
from torchvision import models
import timm
from torch.utils.data import DataLoader, WeightedRandomSampler, default_collate
import random
from tqdm import tqdm
import time
from collections import OrderedDict

from Dataset_AS_VL import *
from utils_AS_VL import *

from LUV_Net import *
from LUV_Net_ import *

import vidaug.augmentors as va

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, ConfusionMatrixDisplay
from sklearn.metrics import roc_auc_score
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import roc_curve

import math

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--frame_num', type=int, default=30,help='clip_frame_num')
    
    # model 
    parser.add_argument('--model_name', type=str, default='LUVM',help='model')
    parser.add_argument('--model_version', type=str, default='v3',help='model')
    parser.add_argument('--encoder_name', type=str, default='imgnet_init_densenet161',help='encoder_name')
    parser.add_argument('--encoder_batch_size', type=int, default=32 ,help='encoder_name')

    # select fold & seed num
    parser.add_argument('--seed_num', type=int, default=1234,help='seed_num')
    parser.add_argument('--fold_num', type=str, default=3,help='fold_num')

    # set task
    parser.add_argument('--task', type=str, default='multi_label',help='task')
    
    # set padding type
    parser.add_argument('--version', type=str, default='version_1',help='label_version')
    parser.add_argument('--frame_size', type=int, default=256,help='frame_size')
    parser.add_argument('--fps', type=int, default=10,help='fps')
    
    parser.add_argument('--pooling_method', type=str, default='attn_multilabel_conv',help='pooling_method')
    parser.add_argument('--num_heads', type=int, default=32,help='attn_head_num')
    parser.add_argument('--kernel_width', type=int, default=5,help='kernel_width')
    parser.add_argument('--train_layer', type=str, default='all',help='train_layer')

    # save_path to save checkpoint
    parser.add_argument('--save_path', type=str, default='/home/work/LUS/Results/video_model2/',help='save_path')
    
    parser.add_argument('--base_path', type=str, default='/home/work/LUS/Dataset/csv_files/clip_multilabel_classification/',help='base_path')
    
    parser.add_argument('--model_output_class', type=int, default=5,help='model_output_class')
    parser.add_argument('--model_test_rate', type=str, default='0.2',help='model_test_rate')
    parser.add_argument('--data_type', type=str, default='before_all_data',help='data_type')
    
    # set batch_size
    parser.add_argument('--batch_size', type=int, default=2, help='batch_size')
    parser.add_argument('--accumulation_steps', type=int, default=1, help='Gradient accumulation steps')
    parser.add_argument('--lr', type=float, default=1e-6, help='learning_rate')
    
    parser.add_argument('--total_epochs', type=int, default=200, help='the number of training epochs.')
    parser.add_argument('--stop_patience_num', type=int, default=30, help='the number of training epochs.')
    
    # trainable parameter

    args = parser.parse_args()
    return args

def set_all_seeds(SEED):
    # REPRODUCIBILITY
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(SEED)
    random.seed(SEED)

def collate_video(batch_list):
    """
    A custom collate function to be passed to the callate_fn argument when creating a pytorch dataloader.
    This is necessary because videos have different lengths. We handle by combining all videos along the time 
    dimension and returning the number of frames in each video.
    """
    vids = torch.concat([b[0] for b in batch_list])
    # num_frames = [b.shape[0] for b in batch_list]
    labels = [b[1] for b in batch_list]
    # record = {
    #     'video': vids,
    #     'num_frames': num_frames
    # }

    # use pytorch's default collate function for remaining items
    # for b in batch_list:
    #     b.pop('video')
    # record.update(default_collate(batch_list))

    return vids, labels

def train():

    # parse arguments
    args = parse_args()
    
    #wand config
    config = {
                "model" : args.model_name,
                "frame_size" : args.frame_size,
                "epochs": args.total_epochs,
                "patience_num": args.stop_patience_num,
                'fold_num':args.fold_num,
                "sheduler":'None',
                "batch_size": args.batch_size*args.accumulation_steps,
                "task": args.task,
                "seed_num":args.seed_num
                }
    wandb.init(project='LUS_video_model2', config=config)
    wandb.run.name = f'seed_{args.seed_num}_test{args.model_test_rate}_{args.data_type}_{args.version}_clip{args.frame_num}_{args.model_name}_{args.model_version}_lr{args.lr}_{args.encoder_name}_{args.encoder_batch_size}_{args.task}_{args.model_output_class}_artifacts_{args.frame_size}_duplicate_{args.batch_size*args.accumulation_steps}batch_{args.pooling_method}_{args.num_heads}head_{args.kernel_width}ksize_fold{args.fold_num}_{args.train_layer}_train'
    wandb.run.save()

    # set seed
    set_all_seeds(args.seed_num)

    # train csv_path
    train_csv_path = os.path.join(args.base_path, f'{args.data_type}/{args.version}/clip_length_ablation_study/{args.frame_num}_clip/fold_{args.fold_num}/train.csv')

    # valid csv_path
    valid_csv_path = os.path.join(args.base_path, f'{args.data_type}/{args.version}/clip_length_ablation_study/{args.frame_num}_clip/fold_{args.fold_num}/valid.csv')

    # make dataset
    train_dataset = video_dataset(train_csv_path, transforms = apply_transforms(mode='train'), img_size = args.frame_size, frame_num = args.frame_num, is_train = True)
    valid_dataset = video_dataset(valid_csv_path, transforms = apply_transforms(mode=None), img_size = args.frame_size, frame_num = args.frame_num, is_train = False) 

    if args.model_name == 'LUV_Net':
        # dataloader
        train_dataloader =  torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle = True, collate_fn=collate_video, drop_last=True)
        valid_dataloader =  torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch_size, shuffle = False, collate_fn=collate_video, drop_last=True)

    #len_dataloader
    len_train_dataset = len(train_dataloader.dataset)
    len_valid_dataset = len(valid_dataloader.dataset)

    len_train_dl = len(train_dataloader)
    len_valid_dl = len(valid_dataloader)

    print('train_dataset : ', len_train_dataset)
    print('valid_dataset : ', len_valid_dataset)
    
    print('train_dl : ', len_train_dl)
    print('valid_dl : ', len_valid_dl)

    # 1. Multi GPU
    ##### visible device 지정해주는 부분
#     os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#     os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"'{device}' is available.")
#     print('Current cuda device:', torch.cuda.current_device())
#     print('Count of using GPUs:', torch.cuda.device_count())

    # 2. Single GPU
    gpu_index = 0
    device = torch.device(f"cuda:{gpu_index}" if torch.cuda.is_available() else "cpu")
    print(f"'{device}' is available.")
    
    # image encoder
    if args.model_name == 'LUV_Net' or args.model_name == 'LUV_Net_':
        if args.encoder_name == 'densenet161':
            encoder_lr = 1e-6
            encoder_seed = 234
            
            weight_path = f'/home/work/LUS/Results/frame_base/multi_label/seed_{encoder_seed}_test{args.model_test_rate}_std_loss_{args.data_type}_{args.version}_5_artifacts_densenet161_lr{encoder_lr}_batch{args.encoder_batch_size}_{args.frame_size}_fold{args.fold_num}_noaug_checkpoint'
            encoder = timm.create_model('densenet161', pretrained=False, num_classes=0)
            check_point = torch.load(weight_path, map_location='cpu')

            new_state_dict = {k: v for k, v in check_point['model'].items() if 'classifier' not in k}

            encoder.load_state_dict(new_state_dict, strict=False)

        # Use imgnet pretrained densenet 161
        elif args.encoder_name == 'imgnet_init_densenet161':
            encoder = timm.create_model('densenet161', pretrained=True, num_classes=0)
        
        elif args.encoder_name == 'densenet161_scratch':
            encoder = timm.create_model('densenet161', pretrained=False, num_classes=0)
            
        elif args.encoder_name == 'imgnet_init_resnet50':
            encoder = timm.create_model('resnet50', pretrained=True, num_classes=0)
        
        elif args.encoder_name == 'resnet50_scratch':
            encoder = timm.create_model('resnet50', pretrained=False, num_classes=0)

#     num_heads = 32

    # Multi GPU
    # multi-gpu 사용시 조절 필요
#     num_gpu = 2
#     num_frames = [args.frame_num]*(args.batch_size // num_gpu)
#     if args.model_version == 'v1':
#         model = MedVidNet_multi_attn(encoder, args.num_heads, pooling_method = args.pooling_method)
#     elif args.model_version == 'v2':
#         model = MedVidNet_multi_attn_conv(encoder, args.num_heads, pooling_method = args.pooling_method)
#     elif args.model_version == 'v4':
#         total_frame_num = args.frame_num
#         model = MedVidNet_conv(encoder, pooling_method = args.pooling_method, total_frame_num = total_frame_num)

#     model = torch.nn.DataParallel(model, device_ids=[0, 1]).to(device=device)

#     if args.train_layer == 'pooling':
#         for param in model.module.encoder.parameters():
#             param.requires_grad = False
#     elif args.train_layer == 'all':
#         for param in model.module.encoder.parameters():
#             param.requires_grad = True

    # Single GPU
    if args.model_name == "LUV_Net":
        num_frames = [args.frame_num]*args.batch_size
        model = LUV_Net(encoder, args.num_heads, pooling_method = args.pooling_method, kernel_width = args.kernel_width)

    elif args.model_name == "LUV_Net_":
        num_frames = [args.frame_num]*args.batch_size
        model = LUV_Net_(encoder, args.num_heads, pooling_method = args.pooling_method, kernel_width = args.kernel_width)
    
    if args.train_layer == 'pooling':
        for param in model.encoder.parameters():
            param.requires_grad = False
    elif args.train_layer == 'all':
        for param in model.encoder.parameters():
            param.requires_grad = True

    model = model.to(device) 

    criterion = nn.BCEWithLogitsLoss()

    sigmoid = nn.Sigmoid()

#     optimizer = optim.AdamP(model.parameters(), lr=LR, weight_decay = 1.e-3)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
#     scheduler = ReduceLROnPlateau(optimizer, factor = 0.1, patience = 3, verbose = True)
    
    # log train / valid loss & acc
    wandb.watch(model, criterion, log="all")

    epoch_losses_train = []
    epoch_losses_val = []
    
    epoch_accs_train = []
    epoch_accs_val = []
    
    epoch_auc_train = []
    epoch_auc_val = []
    
    since = time.time()
    
    #compute best_validation_loss for early stopping
    best_loss = float("inf")
    best_auc = 0
    best_epoch = 0

    best_train_loss = 0.0
    best_train_acc = 0.0
    best_train_auc = 0.0

    best_valid_loss = 0.0
    best_valid_acc = 0.0
    best_valid_auc = 0.0
    
    # num_classes = 3

    for epoch in tqdm(range(0, args.total_epochs + 1)):
        print('-'*50)
        print('Epoch {}/{}'.format(epoch, args.total_epochs))
        
        train_epoch_acc = 0
        train_running_loss = 0.0
        train_epoch_auc = 0.0
        
        train_preds = []
        train_trues = []

        for batch_ndx, data in enumerate(train_dataloader):

            train_imgs, train_labels = data
            train_imgs = train_imgs.float().to(device)
            train_labels = torch.stack([label.to(device) for label in train_labels])
#             print(train_imgs.shape)
#             print(train_labels.shape)
            
            model.train()
        
            # model output
            y_preds, attentions = model(train_imgs, num_frames)

#             print(y_preds.shape)
            # train_labels = train_labels.to(torch.float32)
            train_loss = criterion(y_preds, train_labels)
            
            y_preds = sigmoid(y_preds.squeeze())
            
            y_preds_np = y_preds.data.cpu().numpy()
            y_preds_np = np.where(y_preds_np >= 0.5, 1, 0)
            
            np_train_labels = train_labels.cpu().numpy()
            
            # train_batch_accuracy
            train_batch_acc = accuracy_score(np_train_labels, y_preds_np)
            
            train_preds.append(y_preds.cpu().detach().numpy())
            train_trues.append(train_labels.cpu().detach().numpy())
            
            train_running_loss += train_loss
            
            train_loss = train_loss / args.accumulation_steps
            train_loss.backward()
            
            if (batch_ndx + 1) % args.accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                
            train_epoch_acc += train_batch_acc
            train_running_loss += train_loss
        
        # calculate train auc
        train_true = np.concatenate(train_trues)
        train_pred = np.concatenate(train_preds)
        train_epoch_auc = roc_auc_score(train_true, train_pred, multi_class='ovr')
        # train_epoch_auc = roc_auc_score(train_true, train_pred)
        
        epoch_loss_train = train_running_loss / len_train_dl
        epoch_acc_train = train_epoch_acc / len_train_dl
        
        epoch_losses_train.append(epoch_loss_train.item())
        epoch_accs_train.append(epoch_acc_train.item())
        epoch_auc_train.append(train_epoch_auc.item())

        # wandb log
        wandb.log({"train_loss": epoch_loss_train, "train_acc": epoch_acc_train, "train_auc": train_epoch_auc}, step = epoch)

        print("train_loss:", epoch_loss_train, ", train_acc:", epoch_acc_train, ", train_auc:", train_epoch_auc)
            
        #######################
        # start validation 
        val_epoch_acc = 0
        val_running_loss = 0.0
        val_epoch_auc= 0.0
        best_thresholds = None
        
        print('start validation')
        model.eval()
        with torch.no_grad():

            val_preds = []
            val_trues = []    
            
            for batch_ndx, data in enumerate(valid_dataloader):
                
                val_imgs, val_labels = data
                val_imgs = val_imgs.float().to(device)
                val_labels = torch.stack([label.to(device) for label in val_labels])
                
                y_preds, attentions = model(val_imgs, num_frames)
                
                val_loss = criterion(y_preds, val_labels)
                
                y_preds = sigmoid(y_preds.squeeze())

                y_preds_np = y_preds.data.cpu().numpy()
                y_preds_np = np.where(y_preds_np >= 0.5, 1, 0)

                np_val_labels = val_labels.cpu().numpy()

                # train_batch_accuracy
                val_batch_acc = accuracy_score(np_val_labels, y_preds_np)

                val_preds.append(y_preds.cpu().detach().numpy())
                val_trues.append(val_labels.cpu().detach().numpy())

                val_epoch_acc += val_batch_acc
                val_running_loss += val_loss

            # calculate train auc
            val_true = np.concatenate(val_trues)
            val_pred = np.concatenate(val_preds)
            val_epoch_auc = roc_auc_score(val_true, val_pred, multi_class='ovr')
            
            epoch_loss_val = val_running_loss / len_valid_dl
            epoch_acc_valid = val_epoch_acc / len_valid_dl
            
            epoch_losses_val.append(epoch_loss_val.item())
            epoch_accs_val.append(epoch_acc_valid.item())
            epoch_auc_val.append(val_epoch_auc.item())

#             scheduler.step(epoch_loss_val)

            # wandb log
            wandb.log({"valid_loss": epoch_loss_val, "valid_acc": epoch_acc_valid, "valid_auc": val_epoch_auc}, step = epoch)
            
            
            print("valid_loss:", epoch_loss_val, ", valid_acc:", epoch_acc_valid, ", valid_auc:", val_epoch_auc)
            
            # checkpoint model if has best val loss yet
            if epoch_loss_val < best_loss:

                checkpoint_std_loss = "loss"
                best_loss = epoch_loss_val
#                 best_thresholds = current_thresholds

                best_epoch = epoch
                best_train_loss = epoch_loss_train
                best_train_acc = epoch_acc_train
                best_train_auc = train_epoch_auc

                best_valid_loss = epoch_loss_val
                best_valid_acc = epoch_acc_valid
                best_valid_auc = val_epoch_auc

                # best thresholds 계산
                current_thresholds = []

                for i in range(4):
                    fpr, tpr, thres = roc_curve(val_true[:, i], val_pred[:, i])
                    J = tpr - fpr
                    ix = np.argmax(J)

                    # 반환된 임계값 배열에서 샘플로 데이터를 추출.
                    # threshold[0]은 max(예측확률)+1로 임의 설정됨. 이를 제외하기 위해 np.arange는 1부터 시작
                    thr_index = thres[1:]

                    best_thr = thr_index[ix]
                    current_thresholds.append(best_thr)

                print(f"Epoch {epoch+1}, best threshold for each label: {current_thresholds}")

                best_thresholds = current_thresholds

#                 checkpoint(args.batch_size*args.accumulation_steps, args.version, args.frame_size, 
#                    args.data_type, args.train_layer, args.model_test_rate, checkpoint_std_loss, 
#                    model, args.model_output_class, args.save_path, args.model_name, 
#                    args.model_version, args.encoder_name, args.encoder_batch_size, 
#                    args.pooling_method, args.num_heads, args.fold_num, args.lr,
#                    best_epoch, best_valid_loss, best_valid_acc, 
#                    best_valid_auc, best_thresholds)

                checkpoint2(args.seed_num, args.batch_size*args.accumulation_steps, args.version, args.frame_num, args.frame_size, 
                           args.data_type, args.train_layer, args.model_test_rate, checkpoint_std_loss, 
                           model, args.model_output_class, args.save_path, args.model_name, 
                           args.model_version, args.encoder_name, args.encoder_batch_size, 
                           args.pooling_method, args.num_heads, args.kernel_width, args.fold_num, args.lr,
                           best_epoch, best_valid_loss, best_valid_acc, 
                           best_valid_auc, best_thresholds)
            '''
            if best_auc < val_epoch_auc:

                checkpoint_std_auc = "auc"

                best_auc = val_epoch_auc
#                 best_thresholds = current_thresholds

                best_epoch = epoch
                best_train_loss = epoch_loss_train
                best_train_acc = epoch_acc_train
                best_train_auc = train_epoch_auc

                best_valid_loss = epoch_loss_val
                best_valid_acc = epoch_acc_valid
                best_valid_auc = val_epoch_auc

                # best thresholds 계산
                current_thresholds = []

                for i in range(args.model_output_class):
                    fpr, tpr, thres = roc_curve(val_true[:, i], val_pred[:, i])
                    J = tpr - fpr
                    ix = np.argmax(J)

                    # 반환된 임계값 배열에서 샘플로 데이터를 추출.
                    # threshold[0]은 max(예측확률)+1로 임의 설정됨. 이를 제외하기 위해 np.arange는 1부터 시작
                    thr_index = thres[1:]

                    best_thr = thr_index[ix]
                    current_thresholds.append(best_thr)

                print(f"Epoch {epoch+1}, best threshold for each label: {current_thresholds}")

                best_thresholds = current_thresholds
                
                checkpoint(args.batch_size, args.version, args.frame_size, args.data_type, args.train_layer, args.model_test_rate, checkpoint_std_auc, model, args.model_output_class, args.save_path, args.model_name, args.encoder_name, args.encoder_batch_size, args.pooling_method, args.fold_num, best_epoch, best_valid_loss, best_valid_acc, best_valid_auc, best_thresholds)
            '''
            
            # break if no val loss improvement in stop_patience_num" epochs
            if ((epoch - best_epoch) >= args.stop_patience_num):
                print(f"no improvement in {args.stop_patience_num} epochs, break")
                break   
                
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))   

if __name__ == '__main__':
    train()

