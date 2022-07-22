import torch
from torch import nn

import numpy as np



class norm_data(nn.Module):
    r"""
    N: samples num
    C: 3, which means x, y, z of each joint
    V: joint num
    T: frame num
    """
    def __init__(self, norm_channels=3, num_node=17):
        super(norm_data, self).__init__()

        self.bn = nn.BatchNorm1d(norm_channels * num_node)

    def forward(self, x):
        N, C, V, T = x.size()
        x = x.view(N, -1, T)
        x = self.bn(x)
        x = x.view(N, -1, V, T).contiguous()
        return x

class embed(nn.Module):
    def __init__(self, in_channels=3, out_channels=128, norm=True, bias=True):
        super(embed, self).__init__()

        if norm:
            self.embeding = nn.Sequential(
                norm_data(in_channels),

                nn.Conv2d(in_channels, 64, kernel_size=1),
                nn.ReLU(),
                nn.Conv2d(64, out_channels, kernel_size=1),
                nn.ReLU(),
            )
        else:
            self.embeding = nn.Sequential(
                nn.Conv2d(in_channels, 64, kernel_size=1),
                nn.ReLU(),
                nn.Conv2d(64, out_channels, kernel_size=1),
                nn.ReLU(),
            )

    def forward(self, x):
        return self.embeding(x)

class block_idx_info(nn.Module):
    def __init__(self, args, out_jpt_channels=64, out_frm_channels=64*4):
        super().__init__()
        train = args.train if args != None else 1
        bs = args.batch_size if args != None else 64
        bs_test = args.batch_size_test if args != None else 32 # 20211216
        self.seg = args.seg if args != None else 20
        self.device = args.device if args != None else 0
        V = 17 # 25

        # create tensor
        sample_num = bs if train else bs_test * 20 # 32*5
        self.idx_jpt = self.one_hot(sample_num, self.seg, V) # (bs, self.seg, V, V)
        self.idx_frm = self.one_hot(sample_num, V, self.seg) # (bs, V, self.seg, self,seg)
        self.idx_jpt = self.idx_jpt.permute(0, 3, 2, 1).contiguous() # (-1, V, V, self.seg)
        self.idx_frm = self.idx_frm.permute(0, 3, 1, 2).contiguous() # (-1, self.seg, V, self.seg)

        # embedding
        self.embed_idx_jpt = embed(V, out_jpt_channels, norm=False)
        self.embed_idx_frm = embed(self.seg, out_frm_channels, norm=False)

    def forward(self):
        idx_jpt = self.embed_idx_jpt(self.idx_jpt.cuda(self.device))
        idx_frm = self.embed_idx_frm(self.idx_frm.cuda(self.device))

        return idx_jpt, idx_frm

    def one_hot(self, bs, dim1, dim2):
        y = torch.arange(dim2).unsqueeze(-1) # shape: (25, 1) if dim2 is 25
        y_onehot = torch.FloatTensor(dim2, dim2).zero_() # shape (25, 25) with element 0

        y_onehot.scatter_(1, y, 1)

        y_onehot = y_onehot.unsqueeze(0).unsqueeze(0) # shape (1, 1, 25, 25)
        y_onehot = y_onehot.repeat(bs, dim1, 1, 1) # shape (bs, dim1, 25, 25)

        return y_onehot # (bs, dim1, dim2, dim2)



class block_joint_input(nn.Module):
    '''
    mode:
        0 for only joint
        1 for only t_joint
        2 for joint + t_joint
    '''
    def __init__(self, in_channels=3, out_channels=64, mode=2):
        super().__init__()
        self.mode = mode

        self.in_channels = in_channels

        # embeding
        self.embed_joint = embed(in_channels, out_channels, norm=True)
        self.embed_t_jpt = embed(in_channels, out_channels, norm=True)
    
    def forward(self, input):
        N, T, V_C = input.shape
        V = 25 # revised on 20220712
        C = V_C // V

        ## joint info in each frame
        input = input.view(N, T, V, C) # shape: (N, self.seg, 25, 3) # 3D position
        if C != self.in_channels:
            input = input[:, :, :, :self.in_channels] # 2D position # (N, seg, 25, 2)
            C = self.in_channels

        input = input.permute(0, 3, 2, 1).contiguous() # shape: (N, C, V, T)
        #jnt = self.embed_joint(input) # position, i.t., x, y, z

        # by gzb: 20220712
        #delete_17 = [0, 2, 7, 11, 14, 18, 22, 24]
        idx_17 = [1, 3, 4, 5, 6, 8, 9, 10, 12, 13, 15, 16, 17, 19, 20, 21, 23]
        _device = input.get_device()
        #input_new = np.delete(input.numpy(), delete_17, axis=2)
        #input = torch.from_numpy(input_new)
        input = torch.index_select(input, 2, torch.tensor(idx_17).cuda(_device)).cuda(_device)
        V = 17

        ## motion joint info between frames
        t_jpt = input[:, :, :, 1:] - input[:, :, :, 0:-1] # (N, C, V, T-1)
        t_jpt = torch.cat([t_jpt.new(N, C, V, 1).zero_(), t_jpt], dim=-1) # (N, C, V, T)
      
        if self.mode == 0:
            output = self.embed_joint(input) # position, i.t., x, y, z
        elif self.mode == 1:
            output = self.embed_t_jpt(t_jpt)
        else:
            output = self.embed_joint(input) + self.embed_t_jpt(t_jpt)

        return output


class block_bone_input(nn.Module):
    '''
    mode:
        0 for only bone
        1 for only t_bone
        2 for bone + t_bone
    '''
    def __init__(self, in_channels=3, out_channels=64, mode=2):
        super().__init__()
        self.mode = mode

        self.ntu_pairs = (
            (1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5),
            (7, 6), (8, 7), (9, 21), (10, 9), (11, 10), (12, 11),
            (13, 1), (14, 13), (15, 14), (16, 15), (17, 1), (18, 17),
            (19, 18), (20, 19), (22, 23), (21, 21), (23, 8), (24, 25),(25, 12)
        )

        self.embed_bone = embed(in_channels, out_channels, norm=True)
        self.embed_t_bone = embed(in_channels, out_channels, norm=True)

    def forward(self, input):
        N, T, V_C = input.shape
        V = 25
        C = V_C // V

        input = input.view(N, T, V, C) # shape: (N, self.seg, 25, 3) # 3D position
        input = input.permute(0, 3, 2, 1).contiguous() # shape: (N, C, V, T)
        bone = torch.zeros_like(input) # (N, C, V, T)
        for v1, v2 in self.ntu_pairs:
            bone[:, :, v1-1, :] = input[:, :, v1-1, :] - input[:, :, v2-1, :] # (N, C, V, T)

        t_bone = torch.zeros_like(bone)
        t_bone[:, :, :, :-1] = bone[:, :, :, 1:] - bone[:, :, :, :-1] # (N, C, V, T)

        if self.mode == 0:
            output = self.embed_bone(bone)
        elif self.mode == 1:
            output = self.embed_t_bone(t_bone) 
        else:
            output = self.embed_bone(bone) + self.embed_t_bone(t_bone) # (N, out_C, V, T)

        return output


