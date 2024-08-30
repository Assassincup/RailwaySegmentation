import torch
import torch.nn as nn
import torch.nn.functional as F
import model.RANDLANET.pytorch_utils as pt_utils


class Network(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.fc0 = pt_utils.Conv1d(3, 8, kernel_size=1, bn=True)

        self.dilated_res_blocks = nn.ModuleList()
        d_in = 8
        for i in range(self.config.num_layers):
            d_out = self.config.d_out[i]
            self.dilated_res_blocks.append(Dilated_res_block(d_in, d_out))
            d_in = 2 * d_out

        d_out = d_in
        self.decoder_0 = pt_utils.Conv2d(d_in, d_out, kernel_size=(1, 1), bn=True)

        self.decoder_blocks = nn.ModuleList()
        for j in range(self.config.num_layers):
            if j < 4:
                d_in = d_out + 2 * self.config.d_out[-j-2]
                d_out = 2 * self.config.d_out[-j-2]
            else:
                d_in = 4 * self.config.d_out[-5]
                d_out = 2 * self.config.d_out[-5]
            self.decoder_blocks.append(pt_utils.Conv2d(d_in, d_out, kernel_size=(1, 1), bn=True))

        self.fc1 = pt_utils.Conv2d(d_out, 64, kernel_size=(1, 1), bn=True)
        self.fc2 = pt_utils.Conv2d(64, 32, kernel_size=(1, 1), bn=True)
        self.dropout = nn.Dropout(0.6)
        self.dropout1 = nn.Dropout(0.3)
        self.fc3 = pt_utils.Conv2d(32, self.config.num_classes, kernel_size=(1, 1), bn=False, activation=None)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, end_points):
        #print(end_points.keys())
        features = end_points['features']  # Batch*channel*npoints
        #print("features0.shape",features.shape)
        features = self.fc0(features)
        #print("features1.shape",features.shape)

        features = features.unsqueeze(dim=3)  # Batch*channel*npoints*1
        #print("features2.shape",features.shape)

        # ###########################Encoder############################
        f_encoder_list = []
        for i in range(self.config.num_layers):
            f_encoder_i = self.dilated_res_blocks[i](features, end_points['xyz'][i], end_points['neigh_idx'][i])

            f_sampled_i = self.random_sample(f_encoder_i, end_points['sub_idx'][i])
            features = f_sampled_i
            if i == 0:
                f_encoder_list.append(f_encoder_i)
            f_encoder_list.append(f_sampled_i)
        # for i in range(len(f_encoder_list)):
        #     print(f"f_encoder_list{i}",f_encoder_list[i].shape)
            '''
f_encoder_list0 torch.Size([8, 32, 32768, 1])
f_encoder_list1 torch.Size([8, 32, 8192, 1])
f_encoder_list2 torch.Size([8, 128, 2048, 1])
f_encoder_list3 torch.Size([8, 256, 512, 1])
f_encoder_list4 torch.Size([8, 512, 128, 1])
features3.shape torch.Size([8, 512, 128, 1])
f_encoder_list-2.shape torch.Size([8, 256, 512, 1])
f_interp_i.shape torch.Size([8, 512, 512, 1])
f_encoder_list-3.shape torch.Size([8, 128, 2048, 1])
f_interp_i.shape torch.Size([8, 256, 2048, 1])
f_encoder_list-4.shape torch.Size([8, 32, 8192, 1])
f_interp_i.shape torch.Size([8, 128, 8192, 1])
f_encoder_list-5.shape torch.Size([8, 32, 32768, 1])
f_interp_i.shape torch.Size([8, 32, 32768, 1])
f_decoder_list0 torch.Size([8, 256, 512, 1])
f_decoder_list1 torch.Size([8, 128, 2048, 1])
f_decoder_list2 torch.Size([8, 32, 8192, 1])
f_decoder_list3 torch.Size([8, 32, 32768, 1])


f_encoder_list0 torch.Size([8, 32, 32768, 1])
f_encoder_list1 torch.Size([8, 32, 8192, 1])
f_encoder_list2 torch.Size([8, 128, 2048, 1])
f_encoder_list3 torch.Size([8, 256, 512, 1])
f_encoder_list4 torch.Size([8, 512, 128, 1])
f_encoder_list5 torch.Size([8, 1024, 32, 1])
features3.shape torch.Size([8, 1024, 32, 1])
f_encoder_list-2.shape torch.Size([8, 512, 128, 1])
f_interp_i.shape torch.Size([8, 1024, 128, 1])
f_encoder_list-3.shape torch.Size([8, 256, 512, 1])
f_interp_i.shape torch.Size([8, 512, 512, 1])
f_encoder_list-4.shape torch.Size([8, 128, 2048, 1])
f_interp_i.shape torch.Size([8, 256, 2048, 1])
f_encoder_list-5.shape torch.Size([8, 32, 8192, 1])
f_interp_i.shape torch.Size([8, 128, 8192, 1])
f_encoder_list-6.shape torch.Size([8, 32, 32768, 1])
f_interp_i.shape torch.Size([8, 32, 32768, 1])


f_encoder_list0 torch.Size([4, 32, 32768, 1])
f_encoder_list1 torch.Size([4, 32, 8192, 1])
f_encoder_list2 torch.Size([4, 128, 2048, 1])
f_encoder_list3 torch.Size([4, 256, 512, 1])
f_encoder_list4 torch.Size([4, 512, 128, 1])
f_encoder_list5 torch.Size([4, 1024, 32, 1])
f_encoder_list6 torch.Size([4, 1024, 8, 1])
features3.shape torch.Size([4, 1024, 8, 1])
f_encoder_list-2.shape torch.Size([4, 1024, 32, 1])
f_interp_i.shape torch.Size([4, 1024, 32, 1])
f_encoder_list-3.shape torch.Size([4, 512, 128, 1])
f_interp_i.shape torch.Size([4, 1024, 128, 1])
f_encoder_list-4.shape torch.Size([4, 256, 512, 1])
f_interp_i.shape torch.Size([4, 512, 512, 1])
f_encoder_list-5.shape torch.Size([4, 128, 2048, 1])
f_interp_i.shape torch.Size([4, 256, 2048, 1])
f_encoder_list-6.shape torch.Size([4, 32, 8192, 1])
f_interp_i.shape torch.Size([4, 128, 8192, 1])
f_encoder_list-7.shape torch.Size([4, 32, 32768, 1])
f_interp_i.shape torch.Size([4, 32, 32768, 1])

            '''
        # ###########################Encoder############################

        features = self.decoder_0(f_encoder_list[-1])

        # print("features3.shape",features.shape)
        

        # ###########################Decoder############################
        f_decoder_list = []
        for j in range(self.config.num_layers):
            f_interp_i = self.nearest_interpolation(features, end_points['interp_idx'][-j - 1])
            # print(f"f_encoder_list{-j-2}.shape",f_encoder_list[-j - 2].shape)
            # print("f_interp_i.shape",f_interp_i.shape)
            f_decoder_i = self.decoder_blocks[j](torch.cat([f_encoder_list[-j - 2], f_interp_i], dim=1))
            # decoder[id] d_in            d_out  f_encoder_list[-id-2]   f_interp_i
            #         0   512+2*128=736   128*2  256                     512
            #         1   256+2*64 =384   64*2   128                     256
            #         2   128+2*16 = 160  16*2   32                      128
            #         3   16*4=64         16*2   32                      32
            
            
            #         0   1024+256*2=1536 256*2  512                     1024
            #         1   512+128*2=768   128*2  256                     512
            #         2   256+64*2 =384   64*2   128                     256
            #         3   128+16*2 =160   16*2   32                      128
            #         4   16*4            16*2   32                      32
            
            #         0   1024+512*2=2048 512*2  1024                    1024
            #         1   1024+256*2=1536 256*2  512                     1024
            #         2   512+128*2 = 768 128*2  256                     512
            #         3   256+64*2  = 384 64*2   128                     256
            #         4   128+16*2  =160  16*2   32                      128
            #         5   16*4      =64   16*2   32                      32
            features = f_decoder_i
            f_decoder_list.append(f_decoder_i)
        # for i in range(len(f_decoder_list)):
        #     print(f"f_decoder_list{i}",f_decoder_list[i].shape)
        # ###########################Decoder############################

        features = self.fc1(features)
        #print("features4.shape",features.shape)
        features = self.fc2(features)
        #print("features5.shape",features.shape)
        features = self.dropout(features)
        #print("features6.shape",features.shape)
        features = self.fc3(features)
        #print("features7.shape",features.shape)
        f_out = features.squeeze(3)
        f_out = self.softmax(f_out)
        #print("f_out.shape",f_out.shape)

        end_points['logits'] = f_out
        return end_points

    @staticmethod
    def random_sample(feature, pool_idx):
        """
        :param feature: [B, N, d] input features matrix
        :param pool_idx: [B, N', max_num] N' < N, N' is the selected position after pooling
        :return: pool_features = [B, N', d] pooled features matrix
        """
        feature = feature.squeeze(dim=3)  # batch*channel*npoints
        num_neigh = pool_idx.shape[-1]
        d = feature.shape[1]
        batch_size = pool_idx.shape[0]
        pool_idx = pool_idx.reshape(batch_size, -1)  # batch*(npoints,nsamples)
        pool_features = torch.gather(feature, 2, pool_idx.unsqueeze(1).repeat(1, feature.shape[1], 1))
        pool_features = pool_features.reshape(batch_size, d, -1, num_neigh)
        pool_features = pool_features.max(dim=3, keepdim=True)[0]  # batch*channel*npoints*1
        return pool_features

    @staticmethod
    def nearest_interpolation(feature, interp_idx):
        """
        :param feature: [B, N, d] input features matrix
        :param interp_idx: [B, up_num_points, 1] nearest neighbour index
        :return: [B, up_num_points, d] interpolated features matrix
        """
        feature = feature.squeeze(dim=3)  # batch*channel*npoints
        batch_size = interp_idx.shape[0]
        up_num_points = interp_idx.shape[1]
        interp_idx = interp_idx.reshape(batch_size, up_num_points)
        interpolated_features = torch.gather(feature, 2, interp_idx.unsqueeze(1).repeat(1, feature.shape[1], 1))
        interpolated_features = interpolated_features.unsqueeze(3)  # batch*channel*npoints*1
        return interpolated_features


class Dilated_res_block(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()

        self.mlp1 = pt_utils.Conv2d(d_in, d_out//2, kernel_size=(1, 1), bn=True)
        self.lfa = Building_block(d_out)
        self.mlp2 = pt_utils.Conv2d(d_out, d_out*2, kernel_size=(1, 1), bn=True, activation=None)
        self.shortcut = pt_utils.Conv2d(d_in, d_out*2, kernel_size=(1, 1), bn=True, activation=None)

    def forward(self, feature, xyz, neigh_idx):
        f_pc = self.mlp1(feature)  # Batch*channel*npoints*1
        f_pc = self.lfa(xyz, f_pc, neigh_idx)  # Batch*d_out*npoints*1
        f_pc = self.mlp2(f_pc)
        shortcut = self.shortcut(feature)
        return F.leaky_relu(f_pc+shortcut, negative_slope=0.2)


class Building_block(nn.Module):
    def __init__(self, d_out):  # d_in = d_out//2
        super().__init__()
        self.mlp1 = pt_utils.Conv2d(10, d_out//2, kernel_size=(1, 1), bn=True)
        self.att_pooling_1 = Att_pooling(d_out, d_out//2)

        self.mlp2 = pt_utils.Conv2d(d_out//2, d_out//2, kernel_size=(1, 1), bn=True)
        self.att_pooling_2 = Att_pooling(d_out, d_out)

    def forward(self, xyz, feature, neigh_idx):  # feature: Batch*channel*npoints*1
        f_xyz = self.relative_pos_encoding(xyz, neigh_idx)  # batch*npoint*nsamples*10
        f_xyz = f_xyz.permute((0, 3, 1, 2))  # batch*10*npoint*nsamples
        f_xyz = self.mlp1(f_xyz)
        # batch*npoint*nsamples*channel
        f_neighbours = self.gather_neighbour(feature.squeeze(-1).permute((0, 2, 1)), neigh_idx)
        f_neighbours = f_neighbours.permute((0, 3, 1, 2))  # batch*channel*npoint*nsamples
        f_concat = torch.cat([f_neighbours, f_xyz], dim=1)
        f_pc_agg = self.att_pooling_1(f_concat)  # Batch*channel*npoints*1

        f_xyz = self.mlp2(f_xyz)
        # batch*npoint*nsamples*channel
        f_neighbours = self.gather_neighbour(f_pc_agg.squeeze(-1).permute((0, 2, 1)), neigh_idx)
        f_neighbours = f_neighbours.permute((0, 3, 1, 2))  # batch*channel*npoint*nsamples
        f_concat = torch.cat([f_neighbours, f_xyz], dim=1)
        f_pc_agg = self.att_pooling_2(f_concat)
        return f_pc_agg

    def relative_pos_encoding(self, xyz, neigh_idx):
        neighbor_xyz = self.gather_neighbour(xyz, neigh_idx)  # batch*npoint*nsamples*3

        xyz_tile = xyz.unsqueeze(2).repeat(1, 1, neigh_idx.shape[-1], 1)  # batch*npoint*nsamples*3
        relative_xyz = xyz_tile - neighbor_xyz  # batch*npoint*nsamples*3
        # batch*npoint*nsamples*1
        relative_dis = torch.sqrt(torch.sum(torch.pow(relative_xyz, 2), dim=-1, keepdim=True))
        # batch*npoint*nsamples*10
        relative_feature = torch.cat([relative_dis, relative_xyz, xyz_tile, neighbor_xyz], dim=-1)
        return relative_feature

    @staticmethod
    def gather_neighbour(pc, neighbor_idx):  # pc: batch*npoint*channel
        # gather the coordinates or features of neighboring points
        batch_size = pc.shape[0]
        num_points = pc.shape[1]
        d = pc.shape[2]
        index_input = neighbor_idx.reshape(batch_size, -1)
        features = torch.gather(pc, 1, index_input.unsqueeze(-1).repeat(1, 1, pc.shape[2]))
        features = features.reshape(batch_size, num_points, neighbor_idx.shape[-1], d)  # batch*npoint*nsamples*channel
        return features


class Att_pooling(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.fc = nn.Conv2d(d_in, d_in, (1, 1), bias=False)
        self.mlp = pt_utils.Conv2d(d_in, d_out, kernel_size=(1, 1), bn=True)

    def forward(self, feature_set):

        att_activation = self.fc(feature_set)
        att_scores = F.softmax(att_activation, dim=3)
        f_agg = feature_set * att_scores
        f_agg = torch.sum(f_agg, dim=3, keepdim=True)
        f_agg = self.mlp(f_agg)
        return f_agg

"""
Number of parameter:  4.9841M
"""