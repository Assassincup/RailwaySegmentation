import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torch.nn.functional as F
from .pointnet_utils import PointNetEncoder, feature_transform_reguliarzer


class POINTNET(nn.Module):
    def __init__(self, num_class):
        super(POINTNET, self).__init__()
        self.k = num_class
        self.feat = PointNetEncoder(global_feat=False, feature_transform=True, channel=3)
        self.conv1 = torch.nn.Conv1d(1088, 512, 1)
        self.conv2 = torch.nn.Conv1d(512, 256, 1)
        self.conv3 = torch.nn.Conv1d(256, 128, 1)
        self.conv4 = torch.nn.Conv1d(128, self.k, 1)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)

    def forward(self, x):
        x = x.permute(0,2,1)
        batchsize = x.size()[0]
        n_pts = x.size()[2]
        x, trans, trans_feat = self.feat(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.conv4(x)
        x = x.transpose(2,1).contiguous()
        x = F.log_softmax(x.view(-1,self.k), dim=-1)
        x = x.view(batchsize, n_pts, self.k)
        x = x.permute(0,2,1)
        return x

class get_loss(torch.nn.Module):
    def __init__(self, mat_diff_loss_scale=0.001):
        super(get_loss, self).__init__()
        self.mat_diff_loss_scale = mat_diff_loss_scale

    def forward(self, pred, target, trans_feat, weight):
        loss = F.nll_loss(pred, target, weight = weight)
        mat_diff_loss = feature_transform_reguliarzer(trans_feat)
        total_loss = loss + mat_diff_loss * self.mat_diff_loss_scale
        return total_loss


if __name__ == "__main__":
    model = POINTNET(11)
    # 计算网络参数
    total = sum([param.nelement() for param in model.parameters()])
    # 精确地计算：1MB=1024KB=1048576字节
    print('Number of parameter: % .4fM' % (total / 1e6))

"""
Number of parameter:  3.5285M
"""