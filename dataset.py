import os
import torch
import numpy as np
import pandas as pd
from datetime import datetime
from torchvision import transforms
from torch.utils.data import Dataset

class NimrodDataset(Dataset):
    def __init__(self, file, target_year, record_file, 
            nonzeros_th=150000, in_step=4, out_step=18, 
            cropsize=256, dbz=True, return_time=False, data_type="train"):

        days = 366 if target_year % 4 == 0 else 365
        self.map = np.memmap(file, shape=(288*days, 512, 512), dtype=np.int16)
        self.record_df = pd.read_csv(record_file)
        
        self.in_step = in_step
        self.out_step = out_step
        self.cropsize = cropsize
        self.dbz = dbz
        self.return_time = return_time
        self.data_type = data_type
        self.rain_th = 0.1

        self.target_index_list = self.record_df.index[self.record_df.nonzeros>=nonzeros_th]

        if self.data_type == "train":
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomCrop(cropsize),
                transforms.ToTensor()
                ])
        else:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.CenterCrop(cropsize),
                transforms.ToTensor()
                ])

    def __len__(self):
        return len(self.target_index_list)
    
    def __getitem__(self, index):
        idx = self.target_index_list[index]
        
        if idx - self.in_step < 0:
            ranger = range(0, self.in_step+self.out_step)
        elif idx + self.out_step >= len(self.map):
            ranger = range(len(self.map)-(self.in_step+self.out_step), len(self.map))
        else:
            ranger = range(idx-self.in_step, idx+self.out_step)
        
        img = self.map[ranger]

        if self.dbz:
            img[img>self.rain_th] = np.log10(((img[img>self.rain_th] **  (8 / 5)) * 200)) * 10
            img[img<=self.rain_th] = -15
        
        img = torch.cat([self.transform(img[i]) for i in range(self.in_step+self.out_step)], dim=0) / 32

        if self.return_time:
            time = self.record_df.iloc[ranger, 0].to_list()
            return img, time
        
        return img

class MultiNimrodDataset(Dataset):
    def __init__(self, root, target_year_list,
            nonzeros_th=200000, in_step=4, out_step=18,
            cropsize=256, dbz=True, return_time=False, data_type="train"):
        self.root = root
        self.target_year_list = target_year_list
        self.nonzeros_th = nonzeros_th
        self.in_step = in_step
        self.out_step = out_step
        self.cropsize = cropsize
        self.dbz = dbz
        self.return_time = return_time
        self.data_type = data_type
        
        self.rain_th = 0.1 # not in argument

        self.map_list = []
        self.record_df = []
        self.end_indices = []
        
        end_index = 0
        for year in target_year_list:
            days = 366 if year % 4 == 0 else 365

            temp_map = np.memmap(os.path.join(root, f"Nimrod_{year}.dat"), shape=(288*days, 512, 512), dtype=np.int16) 
            temp_df = pd.read_csv(os.path.join(root, "log", f"rain_record_{year}.csv"))

            self.map_list.append(temp_map)
            self.record_df.append(temp_df)
            end_index += temp_df.index[-1]
            self.end_indices.append(end_index)

        self.record_df = pd.concat(self.record_df, axis=0)
        self.record_df = self.record_df.reset_index()
        
        self.target_index_list = self.record_df.index[self.record_df.nonzeros>=nonzeros_th]
        
        if self.data_type == "train":
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomCrop(cropsize),
                transforms.ToTensor()
                ])
        else:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.CenterCrop(cropsize),
                transforms.ToTensor()
                ])

    def __len__(self):
        return len(self.target_index_list)

    def __getitem__(self, index):
        idx = self.target_index_list[index]
        map_idx = [1 if end_index <= idx else 0 for end_index in self.end_indices]
        map_idx = sum(map_idx) - 1
        memmap = self.map_list[map_idx]

        if map_idx > 0:
            idx = idx - self.end_indices[map_idx-1]
        
        if idx - self.in_step < 0:
            ranger = range(0, self.in_step+self.out_step)
        elif idx + self.out_step >= len(memmap):
            ranger = range(len(memmap)-(self.in_step+self.out_step), len(memmap))
        else:
            ranger = range(idx-self.in_step, idx+self.out_step)

        img = memmap[ranger]

        if self.dbz:
            img[img>self.rain_th] = np.log10(((img[img>self.rain_th] **  (8 / 5)) * 200)) * 10
            img[img<=self.rain_th] = -15

        img = torch.cat([self.transform(img[i]) for i in range(self.in_step+self.out_step)], dim=0) / 32
        #img = img.to(torch.float16)

        if self.return_time:
            if map_idx > 0:
                ranger = range(ranger[0]-self.end_indices[map_idx-1], ranger[-1]-self.end_indices[map_idx-1])
            print(ranger)
            time = self.record_df.iloc[ranger, 1].to_list()
            return img, time

        return img


if __name__ == "__main__":
    path = "/home/yihan/yh/research/Nimrod/"
    years = [2016, 2017, 2018]
    dataset = MultiNimrodDataset(path, years, return_time=True)
    print(len(dataset))
    print(dataset[0][0].dtype)
