import torch



def train_point_transformerv3(model, batch):
    batch["feat"] = batch["feat"].squeeze(0)
    batch["coord"] = batch["coord"].squeeze(0)
    batch["grid_size"] = batch["grid_size"].squeeze(0)
    batch["offset"] = batch["offset"].squeeze(0).to(torch.int64)
    #print("batch[offset]",batch["offset"])


    result = model(batch)
    return result


def val_point_transformerv3(model, batch):
    batch["feat"] = batch["feat"].squeeze(0)
    batch["coord"] = batch["coord"].squeeze(0)
    batch["grid_size"] = batch["grid_size"].squeeze(0)
    batch["offset"] = batch["offset"].squeeze(0).to(torch.int64)
    result0 = model(batch)
    #print(result.shape)
    result = result0.max(dim=1)[1].cpu().numpy().reshape(-1)
    label = batch["labels"].type(torch.long).cpu().numpy().reshape(-1)
    return result, label,result0


