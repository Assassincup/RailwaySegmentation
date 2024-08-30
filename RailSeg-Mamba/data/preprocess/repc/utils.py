import torch
from repc_reader import readXYZCls
import os


def get_folder_repc(file_root, num_sample, mode,side,return_index=False):
    all_file_sample = []
    repc_index = 0
    for file_name in os.listdir(file_root):
        if mode == 'test' and (repc_index+1) % 3 != 0:
            repc_index += 1
            continue
        print(mode,'文件：',file_name)
        file_path = os.path.join(file_root, file_name)
        data = readXYZCls(file_path)
        if side == "2side":
            file_sample = get_all_sample_2side(data, num_sample,return_index=return_index)
        else:
            file_sample = get_all_sample(data, num_sample)
        if len(file_sample) > 0:
            all_file_sample.extend(file_sample)
        repc_index += 1
    return all_file_sample


def get_all_sample_2side(points,num_sample,return_index=False):
    points = torch.tensor(points)
    if return_index:
        xyzl = points[:, (1, 3, 2, 4, 0)]
    else:
        xyzl = points[:, (1, 3, 2, 4)]
    handle_data_label = (xyzl[:, 2] > -2) & (-20 < xyzl[:, 1]) & (xyzl[:, 1] < 20)
    handle_data_index = torch.nonzero(handle_data_label).squeeze()
    handle_data = xyzl[handle_data_index]

    section_data = get_section_data(handle_data, num_sample, 20)
    all_sample = seciton_to_sample(section_data, num_sample)
    return all_sample

def get_all_sample(points,num_sample):
    points = torch.tensor(points)
    xyzl = points[:, (1, 3, 2, 4)]
    handle_data_label = (xyzl[:, 2] > -2) & (-20 < xyzl[:, 1]) & (xyzl[:, 1] < 20)
    handle_data_index = torch.nonzero(handle_data_label).squeeze()
    handle_data = xyzl[handle_data_index]

    left_label = handle_data[:, 1] < 0
    left_index = torch.nonzero(left_label).squeeze()
    left_data = handle_data[left_index]

    right_label = handle_data[:, 1] >= 0
    right_index = torch.nonzero(right_label).squeeze()
    right_data = handle_data[right_index]

    left_section_data = get_section_data(left_data, num_sample, 20)
    right_section_data = get_section_data(right_data, num_sample, 20)

    left_section_data += right_section_data
    all_sample = seciton_to_sample(left_section_data, num_sample)
    return all_sample


def get_section_data(data, sample_num, split_length):
    all_section_data = []
    data_num = data.size(0)
    start_x = 0
    end_x = data[-1, 0].item()
    start_index = 0
    i = 0
    while i < data_num:
        if i == 0:
            start_x = data[0, 0].item()
        else:
            current_x = data[i, 0].item()
            if abs(current_x - start_x) >= split_length:
                section_data = data[start_index:i]
                all_section_data.append(section_data)
                start_x = current_x
                start_index = i
        i += sample_num
    return all_section_data


def seciton_to_sample(all_section_data, sample_num):
    section_num = len(all_section_data)
    all_sample = []
    for i in range(section_num):
        section_data = all_section_data[i]
        section_data_num = section_data.size(0)
        randperm = torch.randperm(section_data_num)
        rand_section_data = section_data[randperm]

        if section_data_num >= sample_num:
            k = section_data_num // sample_num
            for j in range(k):
                sample_data = rand_section_data[j * sample_num:(j + 1) * sample_num]
                all_sample.append(sample_data)
    return all_sample
