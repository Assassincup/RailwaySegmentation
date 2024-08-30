from torch.utils.data import Dataset
import torch
from torch.utils.data import DataLoader
import numpy as np
import configparser
from collections import OrderedDict
import os
import torch.nn as nn
import torch.distributed as dist
import numpy as np
from tqdm import *
import argparse
import json
import time
import random

from ocnn.octree import Points, Octree
from ocnn.dataset import CollateBatch
from thsolver import Dataset
from typing import List


def centralization(points):
    points[:, 0] -= np.mean(points[:, 0])
    return points

    
class RepcDataset(Dataset):
    def __init__(self, mode,centralization_flag):
        self.data_list_path = f'/home/lzx/code/RAILSEGMENT/data/preprocess/repc/data_info/repc_{mode}_info.json'
        self.data_path = GetDataPath(self.data_list_path)
        # print("self.data_path",len(self.data_path))
        self.all_sample = []
        self.centralization_flag = centralization_flag
        for repc_path in self.data_path:
            repc_piece = np.fromfile(repc_path,dtype=np.float32).reshape(-1,5)
            #repc_piece = torch.tensor(repc_piece)
            self.all_sample.append(repc_piece)
        # print(len(self.all_sample))


    def __getitem__(self, item):
        scale_factor = 25.0 #depth=12
        self.all_sample[item] = self.all_sample[item]
        data = self.all_sample[item][:, :3].astype(np.float32)
        if(self.centralization_flag == True):
            data = centralization(data)
        label = self.all_sample[item][:, 3].astype(np.float32)
        
        xyz = data
        center = (xyz.min(axis=0) + xyz.max(axis=0)) / 2.0
        xyz = (xyz - center) / scale_factor  # xyz in [-1, 1]
        
        points = Points(points=torch.from_numpy(xyz), labels=torch.from_numpy(label))
        inbox_mask = points.clip(min=-1, max=1)
        
        
        return_data = {
            "points":points,
            "inbox_mask":inbox_mask,
            "labels":label
        }
        return return_data

    def __len__(self):
        return len(self.data_path)




def GetDataPath(info_path):
    with open(info_path,"r",encoding="utf-8") as f:
        data = json.load(f)
    data_list = data["data_list"]
    return data_list


class CollateBatch:

  def __init__(self, cutmix: int = 0):
    super().__init__()
    self.cutmix = cutmix

  def __call__(self, batch: list):
    assert type(batch) == list
    # a list of dicts -> a dict of lists
    outputs = {key: [b[key] for b in batch] for key in batch[0].keys()}
    return outputs

def build_dataloader(mode):
    if mode =="train":
        dataset = RepcDataset(mode='train',centralization_flag=True)
        sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        #sampler = None
        loader = DataLoader(dataset=dataset, batch_size=4,sampler=sampler,pin_memory=True,num_workers=4,collate_fn=CollateBatch(0))

    else:
        dataset = RepcDataset(mode='test',centralization_flag=True)
        #sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        sampler = None
        loader = DataLoader(dataset=dataset, batch_size=1, shuffle=False ,sampler=sampler,pin_memory=True,num_workers=4,collate_fn=CollateBatch(0))
    return dataset, loader, sampler



if __name__ == "__main__":
    class_name = ['void', 'tree', 'line', 'pole', 'building', 'rail', 'pathway', 'bar', 'hillside', 'tunnel',
                  'platform']
    traindata = RepcDataset(mode='test',centralization_flag=True)
    print(len(traindata))
    print(traindata[0].keys())
    loader = DataLoader(dataset=traindata, batch_size=1,sampler=None,pin_memory=True,num_workers=4,collate_fn=CollateBatch(0))
    for i in loader:
        print(len(i))
        print(type(i))
        print(i.keys())
        print(i)
        break

    


