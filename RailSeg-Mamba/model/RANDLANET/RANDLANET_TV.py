import torch



def train_randla(model, batch):
    #data = batch["points"][:, :,:3]
    result = model(batch)
    return result["logits"],result["labels"]


def val_randla(model, batch):
    #data = batch["points"][:, :,:3]
    result_0 = model(batch)
    result_1 = result_0["logits"]
    # sprint(result.shape)
    result = result_1.max(dim=1)[1].cpu().numpy().reshape(-1)
    label = result_0["labels"].type(torch.long).cpu().numpy().reshape(-1)
    return result, label, result_1

