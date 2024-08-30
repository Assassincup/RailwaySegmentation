import os
import json


train_num = 23119
#92477,46305
def get_repc_info(split):
    root_dir = "/home/lzx/DATASET/repc2bin_8192_4_2side_index"
    print("项目根目录:",root_dir)
    data_infos = dict()
    data_infos['metainfo'] = dict(DATASET='repc')
    data_mode = split
    data_list = []
    if data_mode == "train":
        for j in range(0,train_num):
            if j%10 != 0:
                data = f"{os.path.join(root_dir,str(j).zfill(7))}.bin"
                data_list.append(data)
                data_infos.update(dict(data_list=data_list))
    if data_mode == "test":
        for j in range(0,train_num):
            if j%10 ==0:
                data = f"{os.path.join(root_dir,str(j).zfill(7))}.bin"
                data_list.append(data)
                data_infos.update(dict(data_list=data_list))
    return data_infos


def write_data(data_path,split,dataset='repc'):
    data_info = get_repc_info(split)
    if (not os.path.exists(data_path)):
        print("path is not exist")
        os.makedirs(data_path)
    write_path = os.path.join(data_path, f"{dataset}_{split}_info.json")
    with open(write_path, 'w', encoding='utf-8') as f:
        json.dump(data_info, f, indent=2)
        print("write done")

def create_repc_info_file(save_path):
    write_data(save_path,split='train')
    print(f"repc info train is saved to {save_path} ")
    write_data(save_path,split='test')
    print(f"repc info test is saved to {save_path} ")

if __name__ == "__main__":
    create_repc_info_file("data_info")