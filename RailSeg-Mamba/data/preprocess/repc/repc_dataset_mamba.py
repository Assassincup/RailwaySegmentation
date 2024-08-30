'''
This file is modified from https://github.com/mit-han-lab/spvnas
'''

import os
import numpy as np
import torch
from torch.utils import data

from .repc_dataset import RepcDataset
from torchsparse import SparseTensor
from torchsparse.utils.collate import sparse_collate_fn
from torchsparse.utils.quantize import sparse_quantize
from itertools import accumulate
from .seg_utils import aug_points
from collections import defaultdict
import torch
from torch.utils.data import DataLoader



class RepcDatasetSparse(data.Dataset):
    def __init__(self,
                 mode: str = "train"):
        super().__init__()
        self.mode = mode
        self.class_names = ['void', 'tree', 'line', 'pole', 'building', 'rail', 'pathway', 'bar', 'hillside', 'tunnel',
                  'platform', 'mean']
        self.point_cloud_dataset = RepcDataset(mode=mode,centralization_flag=False)
        self.voxel_size = 0.02

        # self.if_flip = True
        # self.if_scale = True
        # self.scale_axis = "xyz"
        # self.scale_range = [0.9, 1.1]
        # self.if_jitter = True
        # self.if_rotate = True

    def __len__(self):
        return len(self.point_cloud_dataset)

    def __getitem__(self, index):
        data = self.get_single_sample(index)
        return data

    # 生成一个样本数据的函数
    def get_single_sample(self, index):
        pc_data = self.point_cloud_dataset[index]
        point_cloud = pc_data["points"]
        point_label = pc_data["labels"].reshape(-1)

        # if self.mode == "train":
        #     point_cloud[:, :3] = aug_points(
        #         xyz=point_cloud[:, :3],
        #         if_flip=self.if_flip,
        #         if_scale=self.if_scale,
        #         scale_axis=self.scale_axis,
        #         scale_range=self.scale_range,
        #         if_jitter=self.if_jitter,
        #         if_rotate=self.if_rotate,
        #     )

        voxel_coordinate_all = np.round(point_cloud[:, :3] / self.voxel_size).astype(np.int32)
        voxel_coordinate_all -= voxel_coordinate_all.min(0, keepdims=1)
        #_, index, inverse_map = sparse_quantize(voxel_coordinate_all, return_index=True, return_inverse=True)

        #voxel_coordinate_sample = voxel_coordinate_all[index]
        #label_sample = point_label[index]

        sample_data = {
            'points': voxel_coordinate_all, #将空间中的float类型坐标的点转化成int类型的坐标，分辨率为voxel_size
            #'labels': label_sample,
            'labels_all': point_label,
            #'inverse_map': inverse_map,

        }

        return sample_data

    # collate_fn是以函数作为参数进行传递, 那么其一定有默认参数. 这个默认参数就是getitem函数返回的数据项的batch形成的列表.
    #     将sample整理成batch的函数


def build_dataloader(mode):
    if mode =="train":
        dataset = RepcDataset(mode='train',centralization_flag=False)
        sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        loader = DataLoader(dataset=dataset, batch_size=1,sampler=sampler,pin_memory=True,num_workers=4)

    else:
        dataset = RepcDataset(mode='test',centralization_flag=False)
        sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        loader = DataLoader(dataset=dataset, batch_size=1, shuffle=False ,sampler=sampler,pin_memory=True,num_workers=4)
    return dataset, loader, sampler




if __name__ == "__main__":
    a = RepcDatasetSparse()
    b = a[0]
    print(b)