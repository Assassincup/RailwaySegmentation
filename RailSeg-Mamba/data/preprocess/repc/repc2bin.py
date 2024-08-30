from utils import get_folder_repc
import numpy as np
import os



def write_bin(data_path,num_sample,mode,save_path):
    points_data = get_folder_repc(data_path,num_sample,mode,"2side",return_index=True)
    for index,point_piece in enumerate(points_data):
        save_data = np.array(point_piece,dtype=np.float32)
        save_name = f"{os.path.join(save_path,str(index).zfill(7))}.bin"
        save_data.tofile(save_name)


def read_bin(data_path):
    points_list = []
    for root,dirs,files in os.walk(data_path):
        for file in files:
            assert file.endswith('.bin')
            bin_name = os.path.join(root,file)
            data = np.fromfile(bin_name,dtype=np.float32).reshape(-1,4)
            points_list.append(data)
            break
    print(len(points_list))
    print(points_list[0].shape)
    print(points_list[0][0][3])

def change_name(data_path):
    for root,dirs,files in os.walk(data_path):
        for file in files:
            if file.endswith('.bin'):
                
                new_name = file.split('\\')[-1]
                os.rename(file,new_name)
                


if __name__ == "__main__":
    # data_path = "/home/lzx/DATASET/repc2bin_345label"
    # read_bin(data_path)

    data_path = '/home/lzx/DATASET/repc/forpaper/72_76_3'
    save_path = '/home/lzx/DATASET/forpapermax1'
    #data_path = 'D:\\DataSet\\repc_test'
    num_sample = 8192*64
    mode = 'train'
    write_bin(data_path,num_sample,mode,save_path)
    print("ok")

    # change_name('/home/lzx/code/ModelTrain-DDP')




