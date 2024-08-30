from .utils_randla.data_process import DataProcessing as DP
from .utils_randla.config import ConfigSemanticKITTI as cfg
from os.path import join
import numpy as np
import pickle
import torch.utils.data as torch_data
import torch
import json
from torch.utils.data import DataLoader

def centralization(points):
    points[:, 0] -= np.mean(points[:, 0])
    return points
def GetDataPath(info_path):
    with open(info_path,"r",encoding="utf-8") as f:
        data = json.load(f)
    data_list = data["data_list"]
    return data_list

class RepcDataset(torch_data.Dataset):
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
        data = self.all_sample[item][:, :3].astype(np.float32)
        if(self.centralization_flag == True):
            data = centralization(data)
        label = self.all_sample[item][:, 3].astype(np.float32)
        selected_idx = np.arange(0,32768,1,np.float32)
        cloud_ind = np.array([item],dtype=np.int32)
        
        return data,label,selected_idx,cloud_ind

    def __len__(self):
        return len(self.data_path)


    def tf_map(self, batch_pc, batch_label, batch_pc_idx, batch_cloud_idx):
        features = batch_pc
        input_points = []
        input_neighbors = []
        input_pools = []
        input_up_samples = []

        for i in range(cfg.num_layers):
            neighbour_idx = DP.knn_search(batch_pc, batch_pc, cfg.k_n)
            sub_points = batch_pc[:, :batch_pc.shape[1] // cfg.sub_sampling_ratio[i], :]
            pool_i = neighbour_idx[:, :batch_pc.shape[1] // cfg.sub_sampling_ratio[i], :]
            up_i = DP.knn_search(sub_points, batch_pc, 1)
            input_points.append(batch_pc)
            input_neighbors.append(neighbour_idx)
            input_pools.append(pool_i)
            input_up_samples.append(up_i)
            batch_pc = sub_points

        input_list = input_points + input_neighbors + input_pools + input_up_samples
        input_list += [features, batch_label, batch_pc_idx, batch_cloud_idx]

        return input_list

    def collate_fn(self, batch):

        selected_pc, selected_labels, selected_idx, cloud_ind = [], [], [], []
        for i in range(len(batch)):
            selected_pc.append(batch[i][0])
            selected_labels.append(batch[i][1])
            selected_idx.append(batch[i][2])
            cloud_ind.append(batch[i][3])

        selected_pc = np.stack(selected_pc)
        selected_labels = np.stack(selected_labels)
        selected_idx = np.stack(selected_idx)
        cloud_ind = np.stack(cloud_ind)

        flat_inputs = self.tf_map(selected_pc, selected_labels, selected_idx, cloud_ind)

        num_layers = cfg.num_layers
        inputs = {}
        inputs['xyz'] = []
        for tmp in flat_inputs[:num_layers]:
            inputs['xyz'].append(torch.from_numpy(tmp).float())
        inputs['neigh_idx'] = []
        for tmp in flat_inputs[num_layers: 2 * num_layers]:
            inputs['neigh_idx'].append(torch.from_numpy(tmp).long())
        inputs['sub_idx'] = []
        for tmp in flat_inputs[2 * num_layers:3 * num_layers]:
            inputs['sub_idx'].append(torch.from_numpy(tmp).long())
        inputs['interp_idx'] = []
        for tmp in flat_inputs[3 * num_layers:4 * num_layers]:
            inputs['interp_idx'].append(torch.from_numpy(tmp).long())
        inputs['features'] = torch.from_numpy(flat_inputs[4 * num_layers]).transpose(1, 2).float()
        inputs['labels'] = torch.from_numpy(flat_inputs[4 * num_layers + 1]).long()

        return inputs

def build_dataloader(mode):
    if mode =="train":
        dataset = RepcDataset(mode='train',centralization_flag=True)
        sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        #sampler = None
        loader = DataLoader(dataset=dataset, batch_size=6,sampler=sampler,pin_memory=True,num_workers=4,collate_fn=dataset.collate_fn)

    else:
        dataset = RepcDataset(mode='test',centralization_flag=True)
        #sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        sampler = None
        loader = DataLoader(dataset=dataset, batch_size=1, shuffle=False ,sampler=sampler,pin_memory=True,num_workers=4,collate_fn=dataset.collate_fn)
    return dataset, loader, sampler


if __name__ == "__main__":
    dataset,dataloader,sample = build_dataloader("train")
    print(len(dataloader)) 
    for item in dataloader:
        print(item.keys())
        break