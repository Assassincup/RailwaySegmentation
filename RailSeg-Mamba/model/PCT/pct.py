from torch import nn
import torch
import torch.nn.functional as F
import numpy as np

class PCSeg(nn.Module):
    def __init__(self,part_num=11):
        super().__init__()
        self.part_num = part_num
        self.conv1 = nn.Conv1d(3, 128, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(128, 128, kernel_size=1, bias=False)

        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(128)

        self.gather_local_0 = Local_op(in_channels=256, out_channels=256)
        self.gather_local_1 = Local_op(in_channels=512, out_channels=512)

        self.pt_last = Point_Transformer_Last()
        self.conv_fuse = nn.Sequential(nn.Conv1d(2560, 1024, kernel_size=1, bias=False),
                                       nn.BatchNorm1d(1024),
                                       nn.LeakyReLU(negative_slope=0.2))
        self.label_conv = nn.Sequential(nn.Conv1d(16, 64, kernel_size=1, bias=False),
                                        nn.BatchNorm1d(64),
                                        nn.LeakyReLU(negative_slope=0.2))
        self.convs1 = nn.Conv1d(1024 * 2 + 128, 1024, 1)
        self.dp1 = nn.Dropout(0.5)
        self.convs2 = nn.Conv1d(1024, 512, 1)
        self.convs3 = nn.Conv1d(512, 256, 1)
        self.convs4 = nn.Conv1d(256, part_num, 1)
        self.bns1 = nn.BatchNorm1d(1024)
        self.bns2 = nn.BatchNorm1d(512)
        self.bns3 = nn.BatchNorm1d(256)

        self.relu = nn.ReLU()

        self.sigmoid = nn.Sigmoid()

    def forward(self,x,fps_idx_1, fps_idx_2, group_idx_1, group_idx_2):
        #image = batch_dict["image"]
        xyz = x
        x = x.permute(0, 2, 1)
        batch_size, _, N = x.size()
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        points_feature = x
        x = x.permute(0, 2, 1)
        new_xyz, new_feature = sample_and_group(xyz=xyz, points=x, fps_idx=fps_idx_1,
                                                group_idx=group_idx_1)  # B 512 32 256   B 512 3
        feature_0 = self.gather_local_0(new_feature)
        feature = feature_0.permute(0, 2, 1)
        new_xyz, new_feature = sample_and_group(xyz=new_xyz, points=feature, fps_idx=fps_idx_2,
                                                group_idx=group_idx_2)  # B 256 32 512  B 256 3
        feature_1 = self.gather_local_1(new_feature)
        x = self.pt_last(feature_1)
        x = torch.cat((x, feature_1), dim=1)
        x = self.conv_fuse(x)

        x_max = F.adaptive_max_pool1d(x, 1)
        x_max = x_max.view(batch_size, -1)
        x_max = x_max.unsqueeze(-1)
        x_max_feature = x_max.repeat(1, 1, N)

        x_avg = F.adaptive_avg_pool1d(x, 1)
        x_avg = x_avg.view(batch_size, -1)
        x_avg = x_avg.unsqueeze(-1)
        x_avg_feature = x_avg.repeat(1, 1, N)
        global_feature = torch.cat((x_max_feature, x_avg_feature), 1)
        x = torch.cat((global_feature, points_feature), 1)
        x = self.relu(self.bns1(self.convs1(x)))
        x = self.dp1(x)
        x = self.relu(self.bns2(self.convs2(x)))
        x = self.dp1(x)
        x = self.relu(self.bns3(self.convs3(x)))
        x = self.convs4(x)
        return x

'''
result_all.shape (75759616,)
label_all.shape (75759616,)
part_num 11
len:ious 12
void 13.3
tree 82.5
line 87.4
pole 69.6
building 78.8
rail 70.6
pathway 94.9
bar 86.0
hillside 89.3
tunnel 98.4
platform 69.7
mean 82.7
'''


def sample_and_group(xyz, points, fps_idx, group_idx):
    new_xyz = ids_to_points_2dim(xyz, fps_idx)
    grouped_xyz = ids_to_points_3dim(xyz, group_idx)
    B, S, N = group_idx.shape
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, -1)
    new_points = ids_to_points_2dim(points, fps_idx)
    grouped_points = ids_to_points_3dim(points, group_idx)
    grouped_points_norm = grouped_points - new_points.view(B, S, 1, -1)
    new_points = torch.cat([grouped_points_norm, new_points.view(B, S, 1, -1).repeat(1, 1, N, 1)], dim=-1)
    return new_xyz,new_points



class Local_op(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Local_op, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        b, n, s, d = x.size()  # torch.Size([32, 512, 32, 6])
        x = x.permute(0, 1, 3, 2)
        x = x.reshape(-1, d, s)
        batch_size, _, N = x.size()
        x = F.relu(self.bn1(self.conv1(x))) # B, D, N
        x = F.relu(self.bn2(self.conv2(x))) # B, D, N
        x = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x = x.reshape(b, n, -1).permute(0, 2, 1)
        return x


class Point_Transformer_Last(nn.Module):
    def __init__(self, channels=512):
        super(Point_Transformer_Last, self).__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=1, bias=False)

        self.bn1 = nn.BatchNorm1d(channels)
        self.bn2 = nn.BatchNorm1d(channels)

        self.sa1 = SA_Layer(channels)
        self.sa2 = SA_Layer(channels)
        self.sa3 = SA_Layer(channels)
        self.sa4 = SA_Layer(channels)

    def forward(self, x):
        #
        # b, 3, npoint, nsample
        # conv2d 3 -> 128 channels 1, 1
        # b * npoint, c, nsample
        # permute reshape
        batch_size, _, N = x.size() # x:1,256,a

        # B, D, N
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x1 = self.sa1(x)
        x2 = self.sa2(x1)
        x3 = self.sa3(x2)
        x4 = self.sa4(x3)
        x = torch.cat((x1, x2, x3, x4), dim=1)

        return x


class SA_Layer(nn.Module):
    def __init__(self, channels):
        super(SA_Layer, self).__init__()
        self.q_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.k_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.q_conv.weight = self.k_conv.weight
        self.q_conv.bias = self.k_conv.bias

        self.v_conv = nn.Conv1d(channels, channels, 1)
        self.trans_conv = nn.Conv1d(channels, channels, 1)
        self.after_norm = nn.BatchNorm1d(channels)
        self.act = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x): # x:1,256,a
        # b, n, c
        x_q = self.q_conv(x).permute(0, 2, 1)  # x_q:1,a,64
        # b, c, n
        x_k = self.k_conv(x) # x:1,64,a
        x_v = self.v_conv(x) # x:1,64,a
        # b, n, n
        energy = torch.bmm(x_q, x_k) #energy 1,a,a

        attention = self.softmax(energy)
        attention = attention / (1e-9 + attention.sum(dim=1, keepdim=True))
        # b, c, n
        x_r = torch.bmm(x_v, attention) # x_r: 1,64,a
        # x_r = self.act(self.after_norm(self.trans_conv(x - x_r)))
        x_r = self.act(self.trans_conv(x - x_r))
        x = x + x_r
        return x



def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        # a = xyz[batch_indices,farthest]
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids


def ids_to_points_2dim(points, idx):
    device = points.device
    B, N = idx.shape
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view((B, 1)).repeat((1, N))
    new_points = points[batch_indices, idx, :]
    return new_points

def ids_to_points_3dim(points, idx):
    device = points.device
    B, N, C= idx.shape
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view((B, 1, 1)).repeat((1, N, C))
    new_points = points[batch_indices, idx, :]
    return new_points




def knn_point(nsample, xyz, new_xyz):
    """
    Input:
        nsample: max sample number in local region
        xyz: all points, [B, N, C]
        new_xyz: query points, [B, S, C]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    sqrdists = square_distance(new_xyz, xyz)
    _, group_idx = torch.topk(sqrdists, nsample, dim = -1, largest=False, sorted=False)
    return group_idx

def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.
    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def down_sample(xyz, npoint_1, npoint_2, nsample):
    B, N, C = xyz.shape
    xyz = xyz.contiguous()
    fps_idx_1 = farthest_point_sample(xyz, npoint_1)
    sample_xyz_1 = ids_to_points_2dim(xyz, fps_idx_1)
    group_idx_1 = knn_point(nsample, xyz, sample_xyz_1)
    fps_idx_2 = farthest_point_sample(sample_xyz_1, npoint_2)
    sample_xyz_2 = ids_to_points_2dim(sample_xyz_1, fps_idx_2)
    group_idx_2 = knn_point(nsample, sample_xyz_1, sample_xyz_2)
    return fps_idx_1,fps_idx_2,group_idx_1, group_idx_2




def get_numpy_data(batch_size=1):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dummy_input = 2*torch.ones((batch_size, 8192, 3)).to(device)

    fps_idx_1, fps_idx_2, group_idx_1, group_idx_2 = down_sample(dummy_input, 512, 256, 32)
    return dummy_input,fps_idx_1, fps_idx_2, group_idx_1, group_idx_2

def centralization(points):
    points[:, 0] -= np.mean(points[:, 0])
    return points


if __name__ == "__main__":
    model = PCSeg(11)
    # 计算网络参数
    total = sum([param.nelement() for param in model.parameters()])
    # 精确地计算：1MB=1024KB=1048576字节
    print('Number of parameter: % .4fM' % (total / 1e6))

'''
Number of parameter:  9.0860M
'''