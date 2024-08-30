import torch



def train_spvcnn(model, batch):
    result = model(batch)
    return result


def val_spvcnn(model, batch):
    data = batch["points"]
    result = model(batch)
    labels_all = batch["labels_all"]
    inverse_map = batch["inverse_map"]
    pre_labels_all = []
    true_labels_all = []
    # print("遍历次数",inverse_map.C[:,-1].max())
    for idx in range(inverse_map.C[:, -1].max() + 1):  # 这个地方是inverse_map而不是data，因为inverse有所有点的坐标，可以完整遍历
        x_plane_idx = (data.C[:, -1] == idx).cpu().numpy()  # 数据放到cpu中方便numpy处理
        # print("采样点的个数",len(x_plane_idx))
        point_plane_reverse_idx = inverse_map.F[inverse_map.C[:, -1] == idx].cpu().numpy()
        x_plane_label_idx = (labels_all.C[:, -1] == idx).cpu().numpy()
        pre_labels = result[x_plane_idx][point_plane_reverse_idx].argmax(1)
        # print('len1',len(labels_all.F))
        # print("len2",len(x_plane_label_idx))
        # print("len(labels_all)",len(labels_all.F))
        # print("labels_all.F",labels_all.F[[0,1,2,3,4,5]])
        # print("x_plane_label_idx",x_plane_label_idx[[0,1,2,3,4,5]])

        true_labels = labels_all.F[x_plane_label_idx]
        # print("len(true_labels)",len(true_labels))
        # print("true_labels",true_labels[0])
        pre_labels_all.append(pre_labels)
        true_labels_all.append(true_labels)
        # print("pre_labels_all",pre_labels_all)
        # print("true_labels_all",true_labels_all)
        # print("完成一轮")
    pre_labels_all = torch.cat(pre_labels_all, 0).cpu()
    true_labels_all = torch.cat(true_labels_all, 0).cpu()
    return pre_labels_all, true_labels_all, result
