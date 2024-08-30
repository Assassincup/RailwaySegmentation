from .pct import down_sample
import torch

expantion =4


def train_pct(model, batch):
    data = batch["points"][:, :,:3]
    fps_idx_1, fps_idx_2, group_idx_1, group_idx_2 = down_sample(data, 512*expantion, 256*expantion, 32*expantion)
    result = model(data, fps_idx_1, fps_idx_2, group_idx_1, group_idx_2)
    return result


def val_pct(model, batch):
    data = batch["points"][:, :,:3]
    fps_idx_1, fps_idx_2, group_idx_1, group_idx_2 = down_sample(data, 512*expantion, 256*expantion, 32*expantion)
    result0 = model(data, fps_idx_1, fps_idx_2, group_idx_1, group_idx_2)
    # sprint(result.shape)
    result = result0.max(dim=1)[1].cpu().numpy().reshape(-1)
    label = batch["labels"].type(torch.long).cpu().numpy().reshape(-1)
    return result, label, result0


