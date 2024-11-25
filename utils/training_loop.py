import os
import cv2
import math
import time
import torch
import numpy as np

from torch.utils.data import DataLoader

from tqdm import tqdm
import pickle 
from matplotlib import pyplot as plt

from utils.dataset_frames import AccessMathDataset

from .config import *

def get_learning_rate(step, step_per_epoch):
    # return 21e-5
    return 6e-5
    
def flow2rgb(flow_map_np):
    h, w, _ = flow_map_np.shape
    rgb_map = np.ones((h, w, 3)).astype(np.float32)
    normalized_flow_map = flow_map_np / (np.abs(flow_map_np).max())

    rgb_map[:, :, 0] += normalized_flow_map[:, :, 0]
    rgb_map[:, :, 1] -= 0.5 * \
        (normalized_flow_map[:, :, 0] + normalized_flow_map[:, :, 1])
    rgb_map[:, :, 2] += normalized_flow_map[:, :, 1]
    return rgb_map.clip(0, 1)


def train(model):
    train_losses, val_losses = [], []
    min_val_loss, es, patience, max_psnr = 999,0,7, 0
    step = 0
    nr_eval = 0
    dataset = AccessMathDataset('train')
    train_data = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory, shuffle=True)#, drop_last=True)
#     train_data = DataLoader(dataset, batch_size=batch_size)#, num_workers=8, pin_memory=True, drop_last=True)
    step_per_epoch = train_data.__len__()
    dataset_val = AccessMathDataset('validation')
    val_data = DataLoader(dataset_val, batch_size=batch_size, pin_memory=pin_memory, num_workers=num_workers)
#     val_data = DataLoader(dataset_val, batch_size=batch_size)#, pin_memory=True, num_workers=8)
    print('training...')
    time_stamp = time.time()
    for epoch in range(max_epochs):
        pbar = tqdm(enumerate(train_data), total=len(train_data), desc=f"Epoch{epoch}, Step", position=0)
        train_loss_all = []
        for i, data in pbar:
            data_time_interval = time.time() - time_stamp
            time_stamp = time.time()
            data_gpu, timestep = data
            data_gpu = data_gpu.to(device) / 255.
            timestep = timestep.to(device)
            
            data_gpu = data_gpu.reshape(-1, *data_gpu.shape[2:]) # added
            
            if is_context == False:
                imgs = data_gpu[:, :6]
                gt = data_gpu[:, 6:9]
            else:
                imgs = data_gpu[:, :12]
                gt = data_gpu[:, 12:]
                
            
            learning_rate = get_learning_rate(step, step_per_epoch) * world_size / 4
            #TODO: pass timestep if you are training RIFEm
            pred, info = model.update(imgs, gt, learning_rate, training=True)
            train_time_interval = time.time() - time_stamp
            time_stamp = time.time()
            # if step % 200 == 1:
            #     print(f"Step=>{step} | learning_rate={learning_rate} | loss/l1={info['loss_l1']} | loss/tea={info['loss_tea']} | loss/distill={info['loss_distill']}")
            if (step>200) and (step % 200 == 0):
                gt = (gt.permute(0, 2, 3, 1).detach(
                ).cpu().numpy() * 255).astype('uint8')
                mask = (torch.cat((info['mask'], info['mask_tea']), 3).permute(
                    0, 2, 3, 1).detach().cpu().numpy() * 255).astype('uint8')
                pred = (pred.permute(0, 2, 3, 1).detach(
                ).cpu().numpy() * 255).astype('uint8')
                merged_img = (info['merged_tea'].permute(
                    0, 2, 3, 1).detach().cpu().numpy() * 255).astype('uint8')
                flow0 = info['flow'].permute(0, 2, 3, 1).detach().cpu().numpy()
                flow1 = info['flow_tea'].permute(
                    0, 2, 3, 1).detach().cpu().numpy()
                for i in range(2):
                    imgs = np.concatenate((merged_img[i], pred[i], gt[i]), 1)[
                        :, :, ::-1]
                    cv2.imwrite(os.path.join(train_log_path, f"img{i}_merged_{step}.jpeg"), imgs)
                    cv2.imwrite(os.path.join(train_log_path, f"img{i}_flow_{step}.jpeg"), np.concatenate(
                        (flow2rgb(flow0[i]), flow2rgb(flow1[i])), 1))
                    cv2.imwrite(os.path.join(
                        train_log_path, f"img{i}_mask_{step}.jpeg"), mask[i])
           
            # pbar.set_description(f"Epoch {i}")
            pbar.set_postfix({'lr':learning_rate,'l_l1': info['loss_l1'].detach().item(), 'l_tea':info['loss_tea'].detach().item(), 'l_dist':info['loss_distill'].detach().item()})
            
            # print('epoch:{} {}/{} time:{:.2f}+{:.2f} loss_l1:{:.4e}'.format(epoch, i,step_per_epoch, data_time_interval, train_time_interval, info['loss_l1']))
            step += 1
            train_loss_all.append(info['loss_all'])
        
        train_losses.append(np.array(train_loss_all).mean())
        
        nr_eval += 1
        c_psnr, val_loss = evaluate(model, val_data, step)
        val_losses.append(val_loss)
        if val_loss < min_val_loss:
            min_val_loss = val_loss
            model.save_model(train_log_path)
            es = 0
        elif val_loss > min_val_loss:
            es += 1
            if es == patience:
                print("Early stopping at epoch", epoch)
                break
        if max_psnr < c_psnr:
            max_psnr = c_psnr
            model.save_model(val_log_path)

            
        if epoch>=50 and epoch%50==0:
            model_save_path = f"model_after_epoch{epoch}"
            os.makedirs(model_save_path,exist_ok=True)
            model.save_model(model_save_path)

    # plots
    train_losses, val_losses = np.array(train_losses), np.array(val_losses)
    plt.plot(train_losses)
    plt.plot(val_losses)
    plt.legend(['Train', 'Val'])
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.show()
    
    # saving losses
    with open(f'{train_log_path}/losses.pkl', 'wb') as f:
        pickle.dump({'train_losses':train_losses, 'val_losses':val_losses}, f)

def evaluate(model, val_data, nr_eval, is_tqdm=False):
    loss_l1_list = []
    loss_distill_list = []
    loss_tea_list = []
    val_loss_all = []
    psnr_list = []
    psnr_list_teacher = []
    time_stamp = time.time()
    if is_tqdm:
        pbar = tqdm(enumerate(val_data), total = len(val_data))
    else:
        pbar = enumerate(val_data)
    for i, data in pbar:
#   for i, data in enumerate(val_data):

        data_gpu, timestep = data
        data_gpu = data_gpu.to(device) / 255.
            
        data_gpu = data_gpu.reshape(-1, *data_gpu.shape[2:]) # added
        
        if is_context == False:
            imgs = data_gpu[:, :6]
            gt = data_gpu[:, 6:9]
        else:
            imgs = data_gpu[:, :12]
            gt = data_gpu[:, 12:]
        
        with torch.no_grad():
            pred, info = model.update(imgs, gt, training=False)
            merged_img = info['merged_tea']
        loss_l1_list.append(info['loss_l1'].cpu().numpy())
        loss_tea_list.append(info['loss_tea'].cpu().numpy())
        loss_distill_list.append(info['loss_distill'].cpu().numpy())
        val_loss_all.append(info['loss_all'])
        for j in range(gt.shape[0]):
            psnr = -10 * \
                math.log10(torch.mean(
                    (gt[j] - pred[j]) * (gt[j] - pred[j])).cpu().data)
            psnr_list.append(psnr)
            psnr = -10 * \
                math.log10(torch.mean(
                    (merged_img[j] - gt[j]) * (merged_img[j] - gt[j])).cpu().data)
            psnr_list_teacher.append(psnr)
    eval_time_interval = time.time() - time_stamp

    print(f"nr_eval => {nr_eval} | psnr={np.array(psnr_list).mean()} | psnr_teacher={np.array(psnr_list_teacher).mean()}")
    return np.array(psnr_list).mean(), np.array(val_loss_all).mean()