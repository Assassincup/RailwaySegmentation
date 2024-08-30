import torch
import ocnn


def train_octformer(model, batch):
    batch = process_batch(batch)
    octree, points = batch['octree'], batch['points']
    data = get_input_feature(octree=octree)
    query_pts = torch.cat([points.points, points.batch_id], dim=1)
    result = model(data, octree, octree.depth, query_pts)
    return result,points.labels


def val_octformer(model, batch):
    batch = process_batch(batch)
    octree, points = batch['octree'], batch['points']
    data = get_input_feature(octree=octree)
    query_pts = torch.cat([points.points, points.batch_id], dim=1)
    result_0 = model(data, octree, octree.depth, query_pts)
    
    # sprint(result.shape)
    result = result_0.max(dim=1)[1].cpu().numpy().reshape(-1)
    label = points.labels.type(torch.long).cpu().numpy().reshape(-1)
    return result, label, result_0, points.labels

def process_batch(batch):
    def points2octree(points):
        octree = ocnn.octree.Octree(depth=10, full_depth=2) #25米 2的10-1=9次方 512 25/512约等于0.05米的voxelsize
        octree.build_octree(points)
        return octree

    if 'octree' in batch:
        batch['octree'] = batch['octree'].cuda(non_blocking=True)
        batch['points'] = batch['points'].cuda(non_blocking=True)
    else:
        points = [pts.cuda(non_blocking=True) for pts in batch['points']]
        octrees = [points2octree(pts) for pts in points]
        octree = ocnn.octree.merge_octrees(octrees)
        octree.construct_all_neigh()
        batch['points'] = ocnn.octree.merge_points(points)
        batch['octree'] = octree
    return batch

def get_input_feature(octree):
    octree_feature = ocnn.modules.InputFeature(feature="P", nempty=True)
    data = octree_feature(octree)
    return data


'''
14depth
data.shape torch.Size([236920, 3])
self.in_shape (tensor(31417), 3)

data.shape torch.Size([259264, 3])
self.in_shape (tensor(32742), 3)

data.shape torch.Size([259264, 3])
self.in_shape (tensor(32742), 3)

12
data.shape torch.Size([236920, 3])
self.in_shape (tensor(31417), 3)
'''
