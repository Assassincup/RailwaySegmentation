import torch




def train_pointnet(model, batch):
    data = batch["points"][:, :,:3]
    result = model(data)
    return result


def val_pointnet(model, batch):
    data = batch["points"][:, :,:3]
    result_0 = model(data)
    # sprint(result.shape)
    result = result_0.max(dim=1)[1].cpu().numpy().reshape(-1)
    label = batch["labels"].type(torch.long).cpu().numpy().reshape(-1)
    return result, label,result_0

