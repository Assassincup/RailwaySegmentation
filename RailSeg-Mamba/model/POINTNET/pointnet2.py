import torch.nn as nn
import torch.nn.functional as F
from .pointnet2_utils import PointNetSetAbstraction,PointNetFeaturePropagation


class POINTNET2(nn.Module):
    def __init__(self, num_classes):
        super(POINTNET2, self).__init__()
        self.extension = 1
        self.redius_enxtension = 1
        self.sa1 = PointNetSetAbstraction(1024*self.extension, 0.1*self.redius_enxtension, 32*self.extension, 3 + 3, [32, 32, 64], False) #1024 0.1
        self.sa2 = PointNetSetAbstraction(256*self.extension, 0.2*self.redius_enxtension, 32*self.extension, 64 + 3, [64, 64, 128], False) #0.2
        self.sa3 = PointNetSetAbstraction(64*self.extension, 0.4*self.redius_enxtension, 32*self.extension, 128 + 3, [128, 128, 256], False) #0.4
        self.sa4 = PointNetSetAbstraction(16*self.extension, 0.8*self.redius_enxtension, 32*self.extension, 256 + 3, [256, 256, 512], False) #0.8
        self.fp4 = PointNetFeaturePropagation(768, [256, 256])
        self.fp3 = PointNetFeaturePropagation(384, [256, 256])
        self.fp2 = PointNetFeaturePropagation(320, [256, 128])
        self.fp1 = PointNetFeaturePropagation(128, [128, 128, 128])
        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(128, num_classes, 1)

    def forward(self, xyz):
        xyz = xyz.permute(0,2,1)

        l0_points = xyz
        l0_xyz = xyz[:,:3,:]

        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)

        l3_points = self.fp4(l3_xyz, l4_xyz, l3_points, l4_points)
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(l0_xyz, l1_xyz, None, l1_points)

        x = self.drop1(F.relu(self.bn1(self.conv1(l0_points))))
        x = self.conv2(x)
        x = F.log_softmax(x, dim=1)
        
        return x


class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()
    def forward(self, pred, target, trans_feat, weight):
        total_loss = F.nll_loss(pred, target, weight=weight)

        return total_loss

if __name__ == "__main__":
    model = POINTNET2(11)
    # 计算网络参数
    total = sum([param.nelement() for param in model.parameters()])
    # 精确地计算：1MB=1024KB=1048576字节
    print('Number of parameter: % .4fM' % (total / 1e6))

    """
    Number of parameter:  0.9678M
    """