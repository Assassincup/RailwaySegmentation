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
        self.point_cloud_dataset = RepcDataset(mode=mode,centralization_flag=True)
        self.voxel_size = 0.05

        self.if_flip = True
        self.if_scale = True
        self.scale_axis = "xyz"
        self.scale_range = [0.9, 1.1]
        self.if_jitter = True
        self.if_rotate = True

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
        point_cloud_all = point_cloud
        _, index, inverse_map = sparse_quantize(voxel_coordinate_all, return_index=True, return_inverse=True)

        point_cloud_sample = point_cloud_all[index]
        voxel_coordinate_sample = voxel_coordinate_all[index]
        label_sample = point_label[index]

        point_sparse_sample = SparseTensor(point_cloud_sample, voxel_coordinate_sample)
        label_sparse_sample = SparseTensor(label_sample, voxel_coordinate_sample)
        label_sparse_all = SparseTensor(point_label, voxel_coordinate_all)
        inverse_map = SparseTensor(inverse_map, voxel_coordinate_all)

        sample_data = {
            'points': point_sparse_sample,
            'labels': label_sparse_sample,
            'labels_all': label_sparse_all,
            'inverse_map': inverse_map,

        }

        return sample_data

    # collate_fn是以函数作为参数进行传递, 那么其一定有默认参数. 这个默认参数就是getitem函数返回的数据项的batch形成的列表.
    #     将sample整理成batch的函数
    @staticmethod
    def collate_batch(inputs):
        batch_data = sparse_collate_fn(inputs)
        # print("sparse_collate_fn")
        return batch_data

    @staticmethod
    def collate_batch_test(inputs):
        batch_data = sparse_collate_fn(inputs)
        return batch_data




def build_dataloader(
        mode:str = "train",
        dist:bool = False

):
    dataset = RepcDatasetSparse(mode)
    if dist:
        if mode == "train":
            # sampler的作用就是根据rank将dataset拆开分到各个分进程中，内部会分辨rank和world_size
            #使用sampler的时候如果要shuffle的话，要注意在train函数中使用set_epoch将当前epoch传入，以此和seed一起生成随机数
            sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        else:
            sampler = torch.utils.data.distributed.DistributedSampler(dataset,shuffle=False)
    else:
        sampler = None


    if mode == "train":
        dataloader = DataLoader(
            dataset,
            batch_size=16,
            pin_memory=True,#在加载数据到GPU之前，先加载到内存的专门区域，加快数据放到GPU的速度
            num_workers=4,
            shuffle=(sampler is None) and (mode == "train"),
            collate_fn=dataset.collate_batch,
            drop_last=False,
            sampler = sampler,
            persistent_workers=True #加载完一个epoch数据不会关闭worker进程而是继续加载下一个epoch的数据
        )
    else:
        dataloader = DataLoader(
            dataset,
            batch_size=1,
            pin_memory=True,
            num_workers=4,
            shuffle=(sampler is None) and (mode != "train"),
            collate_fn=dataset.collate_batch_test,
            drop_last=False,
            sampler=sampler,
            persistent_workers=True
        )

    return dataset , dataloader , sampler




if __name__ == "__main__":
    a = RepcDatasetSparse()
    b = a[0]
    print(b)