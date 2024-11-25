import os
import torch
import numpy as np
from torch.utils.data import Dataset
import csv
from decord import VideoReader
from decord import cpu, gpu

from .config import is_context, num_instances, num_frames

class AccessMathDataset(Dataset):
    def __init__(self, split_name):
        self.split_name = split_name
        self.h = 448
        self.w = 448
        self.data_root = './dataset/VI_dataset_mix_224_10s'
        
        self.csv_path = os.path.join(self.data_root, 'splits.csv')
        self.trainlist, self.testlist = [], []
        with open(self.csv_path, 'r')as f:
            reader = csv.reader(f)
            reader.__next__()
            for name, split in reader:
                if split == 'train':
                    self.trainlist += [name]
                else:
                    self.testlist += [name]
        self.load_data()
        
        self.all_imgs_idx = self.get_img_idx(300, num_frames)


    def get_img_idx(self, num_frames, n):
        extreme_frames = range(0, num_frames, n)  # 300, 7
        extreme_frames = np.array(
            [[extreme_frames[i], extreme_frames[i+1]-1] for i in range(len(extreme_frames)-1)])
        gt = extreme_frames.sum(1)//2
        
        if is_context == False:
            all_imgs = np.zeros((len(extreme_frames), 3), dtype=int)
            all_imgs[:, 0] = extreme_frames[:, 0]
            all_imgs[:, 1] = extreme_frames[:, 1]
            all_imgs[:, 2] = gt
        else:
            all_imgs = np.zeros((len(extreme_frames), 5), dtype=int)
            
            all_imgs[:, 0] = extreme_frames[:, 0]
            all_imgs[:, 1] = extreme_frames[:, 0]+1
            all_imgs[:, 2] = extreme_frames[:, 1]-1
            all_imgs[:, 3] = extreme_frames[:, 1]
            all_imgs[:, 4] = gt
        
        # only 9 out of 42 --- to reduce the time
        random_imgs = np.linspace(0,len(extreme_frames)-1, num_instances, dtype=int)
        all_imgs = all_imgs[random_imgs]
        
        all_imgs = all_imgs.flatten()  # 42*3
        return all_imgs

    def __len__(self):
        return len(self.meta_data)

    def load_data(self):
        cnt = int(len(self.trainlist) * 0.90)
        if self.split_name == 'train':
            self.meta_data = self.trainlist[:cnt]
        elif self.split_name == 'test':
            self.meta_data = self.testlist
        else:
            self.meta_data = self.trainlist[cnt:]
    

    def getimg(self, index):
        vid_path = os.path.join(self.data_root, self.meta_data[index])
        vr = VideoReader(vid_path, ctx=cpu(0))
        num_frames = len(vr)

        n = 7
        timestep = 0.5

        all_imgs = self.all_imgs_idx
        all_imgs = vr.get_batch(all_imgs).asnumpy()

        return all_imgs, timestep

    def __getitem__(self, index):
        all_imgs, timestep = self.getimg(index)
        
        if is_context == False:
            all_imgs = torch.tensor(all_imgs, dtype=torch.uint8).permute(0,3,1,2).reshape(-1, 9, *all_imgs.shape[1:-1])
        else:
            all_imgs = torch.tensor(all_imgs, dtype=torch.uint8).permute(0,3,1,2).reshape(-1, 15, *all_imgs.shape[1:-1])
            
        return all_imgs, torch.tensor(timestep, dtype=torch.float16)