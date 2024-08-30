import torch
import torchsparse.nn.functional as F
from torchsparse import PointTensor, SparseTensor
from torchsparse.nn.utils import get_kernel_offsets

__all__ = ['initial_voxelize', 'point_to_voxel', 'voxel_to_point']


# z: PointTensor
# return: SparseTensor
def initial_voxelize(z, init_res, after_res):
    new_float_coord = torch.cat(
        [(z.C[:, :3] * init_res) / after_res, z.C[:, -1].view(-1, 1)], 1,
    )

    pc_hash = F.sphash(torch.floor(new_float_coord).int())#使用坐标计算哈希值
    sparse_hash = torch.unique(pc_hash)#找出所有不重复哈希值构建哈希表，相当于找到所有非空voxel
    idx_query = F.sphashquery(pc_hash, sparse_hash)#使用坐标哈希值获取哈希表中的索引
    counts = F.spcount(idx_query.int(), len(sparse_hash))#获取哈希值个数？

    inserted_coords = F.spvoxelize(
        torch.floor(new_float_coord),
        idx_query,
        counts,
    )#使用spvoxelize对索引相同的坐标取平均得到所有唯一voxel（感觉是无意义操作，因为已经有唯一哈希表序列（在torch.unique中获得））（想法错误）
    #在sparse_hash中获得的是唯一哈希值，不是具体坐标，使用spvoxelize来从相同idx_query索引的坐标中得到唯一坐标。
    inserted_coords = torch.round(inserted_coords).int()
    inserted_feat = F.spvoxelize(z.F, idx_query, counts)#使用spvoxelize从相同idx_query索引的特征中得到平均值特征

    new_tensor = SparseTensor(inserted_feat, inserted_coords, 1) #用特征和坐标组成sparse_tensor，stride为1
    #cmaps是用来记录stride和coords对应关系的，如果已经知道上个stride对应的coordinate，三维卷积时就可以直接得到输出对应的coordinate
    #kmaps是用来记录一些卷积操作时需要用到的信息
    new_tensor.cmaps.setdefault(new_tensor.stride, new_tensor.coords)

    z.additional_features['idx_query'][1] = idx_query
    z.additional_features['counts'][1] = counts
    z.C = new_float_coord

    return new_tensor


# x: SparseTensor, z: PointTensor
# return: SparseTensor
def point_to_voxel(x, z):
    if z.additional_features is None or z.additional_features.get(
            'idx_query') is None or z.additional_features['idx_query'].get(
                x.s) is None:
        pc_hash = F.sphash(
            torch.cat([
                torch.floor(z.C[:, :3] / x.s[0]).int() * x.s[0],
                z.C[:, -1].int().view(-1, 1)
            ], 1))
        sparse_hash = F.sphash(x.C)
        idx_query = F.sphashquery(pc_hash, sparse_hash)
        counts = F.spcount(idx_query.int(), x.C.shape[0])
        z.additional_features['idx_query'][x.s] = idx_query
        z.additional_features['counts'][x.s] = counts
    else:
        idx_query = z.additional_features['idx_query'][x.s]
        counts = z.additional_features['counts'][x.s]

    inserted_feat = F.spvoxelize(z.F, idx_query, counts)
    new_tensor = SparseTensor(inserted_feat, x.C, x.s)
    new_tensor.cmaps = x.cmaps
    new_tensor.kmaps = x.kmaps

    return new_tensor


# x: SparseTensor, z: PointTensor
# return: PointTensor
def voxel_to_point(x, z, nearest=False):
    if z.idx_query is None or z.weights is None or z.idx_query.get(
            x.s) is None or z.weights.get(x.s) is None: #这里判断点的坐标对应的voxel信息是否已经记录，如果已经有的话可以直接使用，相当于缓存
        off = get_kernel_offsets(2, x.s, 1, device=z.F.device)
        old_hash = F.sphash(
            torch.cat([
                torch.floor(z.C[:, :3] / x.s[0]).int() * x.s[0],
                z.C[:, -1].int().view(-1, 1)
            ], 1), off)
        #这个计算off和off对应old_hash的操作是需要的，因为不同的kernel卷积后coordinate会变化，这里应该是将pointtensor映射到与sparsetensor
        #相同的coordinate坐标系中，然后进行索引的选取。
        pc_hash = F.sphash(x.C.to(z.F.device))
        #old_hash是点特征坐标对应的哈希值，pc_hash是voxel特征坐标对应的哈希值
        idx_query = F.sphashquery(old_hash, pc_hash)
        weights = F.calc_ti_weights(z.C, idx_query,
                                    scale=x.s[0]).transpose(0, 1).contiguous()
        #文章中提到使用了三次线性插值来计算每个点的特征权重，具体怎么算的没看到
        idx_query = idx_query.transpose(0, 1).contiguous()
        if nearest:
            weights[:, 1:] = 0.
            idx_query[:, 1:] = -1
        new_feat = F.spdevoxelize(x.F, idx_query, weights)
        new_tensor = PointTensor(new_feat,
                                 z.C,
                                 idx_query=z.idx_query,
                                 weights=z.weights)
        new_tensor.additional_features = z.additional_features
        new_tensor.idx_query[x.s] = idx_query
        new_tensor.weights[x.s] = weights
        z.idx_query[x.s] = idx_query #这里更新记录本次操作后点的序列和权重
        z.weights[x.s] = weights

    else:
        new_feat = F.spdevoxelize(x.F, z.idx_query.get(x.s), z.weights.get(x.s))
        new_tensor = PointTensor(new_feat,
                                 z.C,
                                 idx_query=z.idx_query,
                                 weights=z.weights)
        new_tensor.additional_features = z.additional_features

    return new_tensor
