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


def centralization(points):
    points[:, 0] -= np.mean(points[:, 0])
    return points

    
class RepcDataset(Dataset):
    def __init__(self, mode,centralization_flag):
        self.voxel_size = 0.05
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
        #点云获取顺序
        order = np.argsort(self.all_sample[item][:,4],axis=0)
        self.all_sample[item] = self.all_sample[item][order]
        
        #random_order option
        #np.random.shuffle(self.all_sample[item])

        # point_cloud = self.all_sample[item]
        # point_cloud = centralization(point_cloud) #防止x值太大导致溢出
        # voxel_coordinate_all = np.round(point_cloud[:, :3] / self.voxel_size).astype(np.float32)
        # voxel_coordinate_all -= voxel_coordinate_all.min(0, keepdims=1)


       
        data = self.all_sample[item][:, :3].astype(np.float32)
        # print("dataset_data.shape",data.shape)
        # print("dataset_data",data)
        if(self.centralization_flag == True):
            data = centralization(data)
        # print("centralization_data.shape",data.shape)
        # print("centralization_data",data)
        voxel_coordinate_all = np.round(data[:, :3] / self.voxel_size).astype(np.float32)
        # print("voxel_coordinate_all.shape",voxel_coordinate_all.shape)
        # print("voxel_coordinate_all",voxel_coordinate_all)
        voxel_coordinate_all -= voxel_coordinate_all.min(0, keepdims=1)
        # print("voxel_coordinate_all.shape",voxel_coordinate_all.shape)
        # print("voxel_coordinate_all",voxel_coordinate_all)
        label = self.all_sample[item][:, 3].astype(np.float32)
        grid_size = np.array([0.5],dtype=np.float32)
        offset = np.array([32768],dtype=np.int64)
        
        return_data = {
            "feat":data,
            "labels":label,
            "coord":data,
            "grid_size":grid_size,
            "offset":offset
        }
        return return_data

    def __len__(self):
        return len(self.data_path)




def GetDataPath(info_path):
    with open(info_path,"r",encoding="utf-8") as f:
        data = json.load(f)
    data_list = data["data_list"]
    return data_list


def build_dataloader(mode):
    if mode =="train":
        dataset = RepcDataset(mode='train',centralization_flag=True)
        sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        #sampler = None
        loader = DataLoader(dataset=dataset, batch_size=1,sampler=sampler,pin_memory=True,num_workers=4)

    else:
        dataset = RepcDataset(mode='test',centralization_flag=True)
        #sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        sampler = None
        loader = DataLoader(dataset=dataset, batch_size=1, shuffle=False ,sampler=sampler,pin_memory=True,num_workers=4)
    return dataset, loader, sampler



if __name__ == "__main__":
    data = RepcDataset(mode='test',centralization_flag=True)
    data1 = data[0]

'''
dataset_data.shape (5, 3)
dataset_data [[ 6.54860e+02 -2.17400e+00 -3.42000e-01]
 [ 6.55067e+02  5.37000e-01  6.29800e+00]
 [ 6.60385e+02  9.03000e-01 -2.10000e-01]
 [ 6.59365e+02 -2.40700e+00  4.41600e+00]
 [ 6.73413e+02 -6.14000e-01 -2.43000e-01]]
centralization_data.shape (5, 3)
centralization_data [[-5.7580566  -2.174      -0.342     ]
 [-5.5510254   0.537       6.298     ]
 [-0.23303223  0.903      -0.21      ]
 [-1.2530518  -2.407       4.416     ]
 [12.794983   -0.614      -0.243     ]]
voxel_coordinate_all.shape (5, 3)
voxel_coordinate_all [[-115.  -43.   -7.]
 [-111.   11.  126.]
 [  -5.   18.   -4.]
 [ -25.  -48.   88.]
 [ 256.  -12.   -5.]]
voxel_coordinate_all.shape (5, 3)
voxel_coordinate_all [[  0.   5.   0.]
 [  4.  59. 133.]
 [110.  66.   3.]
 [ 90.   0.  95.]
 [371.  36.   2.]]

'''

