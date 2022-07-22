import copy as cp

import torch
import torch.nn as nn
from mmcv.runner import load_checkpoint

from ...utils import Graph, cache_checkpoint

EPS = 1e-4

import torch
import torch.nn as nn
from mmcv.cnn import constant_init, kaiming_init, normal_init
from mmcv.runner import load_checkpoint
from mmcv.utils import _BatchNorm

from ...utils import get_root_logger
from ..builder import BACKBONES

import numpy as np

from torch.autograd import Variable
from .graph import Graph as GraphA
from .tools_cigcn import SelfAtt, unit_tcn



def conv_init(conv):
    if conv.weight is not None:
        nn.init.kaiming_normal_(conv.weight, mode='fan_out')
    if conv.bias is not None:
        nn.init.constant_(conv.bias, 0)


def bn_init(bn, scale):
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)


# from tool_inputs.py
class norm_data(nn.Module):
    r"""
    N: samples num
    C: 3, which means x, y, z of each joint
    V: joint num
    T: frame num
    """
    def __init__(self, norm_channels=3, num_node=25):
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

class IdxBlock(nn.Module):
    '''
    Args:
        T: clip_lens
        V: num_joints
        out_channels
    shape:
        input:
        output:
    '''
    def __init__(self, T=20, V=25, out_channels=64, mode='jpt'):
        super().__init__()
        self.mode = mode

        # create tensor
        if mode ==  'jpt':
            # for jpt
            self.idx_info = self.one_hot(T, V) # (1, self.seg, V, V)
            self.idx_info = self.idx_info.permute(0, 3, 2, 1).contiguous() # (1, V, V, self.seg)
            self.embed_idx = embed(V, out_channels, norm=False)
        else:
            # for frm
            self.idx_info = self.one_hot(V, T) # (1, V, self.seg, self,seg)
            self.idx_info = self.idx_info.permute(0, 3, 1, 2).contiguous() # (1, self.seg, V, self.seg)
            self.embed_idx = embed(T, out_channels, norm=False)
        
    def forward(self, input):
        device = input.get_device()

        # get the batch_size of input
        bs = input.shape[0]
        if len(self.idx_info) != bs:
            self.idx_info = self.idx_info.repeat(bs, 1, 1, 1) # (bs, V, V, T)

        idx_info = self.embed_idx(self.idx_info.cuda(device)) # (bs, T, V, T)

        return idx_info

    def one_hot(self, dim1, dim2):
        y = torch.arange(dim2).unsqueeze(-1) # shape: (25, 1) if dim2 is 25
        y_onehot = torch.FloatTensor(dim2, dim2).zero_() # shape (25, 25) with element 0

        y_onehot.scatter_(1, y, 1) # shape (25, 25) with dia elemeter is 1

        y_onehot = y_onehot.unsqueeze(0).unsqueeze(0) # shape (1, 1, 25, 25)
        y_onehot = y_onehot.repeat(1, dim1, 1, 1) # shape (bs, dim1, 25, 25)

        return y_onehot


class block_joint_input(nn.Module):
    '''
    shape:
        input: N C_in V T
        output: N C_out V T
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
        N, C, V, T = input.shape

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
    shape:
        input: N C_in V T
        output: N C_out V T
    mode:
        0 for only joint
        1 for only t_joint
        2 for joint + t_joint
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




class CBRBlock(nn.Module):
    '''
    shape:
        input: N C T V
        output: N C T V if stride == 1
                N C T/2 V if stride == 2
    '''
    def __init__(self,
                 in_channels, 
                 out_channels,
                 mode='TV',
                 kernel_size=1, 
                 stride=1,
                 dilation=1, 
                 relu=True):
        super().__init__()

        ks = kernel_size

        if mode == 'TV':
            _kernel_size = (ks, 1)
            _padding = ((ks + (ks - 1) * (dilation - 1) - 1) //2, 0)
            _stride = (stride, 1)
            _dilation = (dilation, 1)
        else:
            _kernel_size = (1, ks)
            _padding = (0, (ks + (ks - 1) * (dilation - 1) - 1) //2)
            _stride = (1, stride)
            _dilation = (1, dilation)

        self.conv = nn.Conv2d(in_channels, 
                              out_channels, 
                              kernel_size=_kernel_size,
                              padding=_padding,
                              stride=_stride,
                              dilation=_dilation)
        self.bn = nn.BatchNorm2d(out_channels)

        if relu == True:
            self.relu = nn.ReLU(inplace=True)
        else:
            self.relu = lambda x: x

    def forward(self, x):
        x = self.relu(self.bn(self.conv(x)))

        return x


class ResBlock(nn.Module):
    def __init__(self,
                 in_channels, 
                 out_channels,
                 mode='TV',
                 kernel_size=1, 
                 stride=1,
                 residual=True):
        super().__init__()

        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        
        else:
            self.residual = CBRBlock(in_channels, 
                                     out_channels,
                                     mode=mode,
                                     kernel_size=kernel_size,
                                     stride=stride,
                                     relu=False)

    def forward(self, x):
        x = self.residual(x)

        return x


class CNNBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 mode='TV',
                 kernel_size=3,
                 stride=1,
                 dilations=[1,2,3,4],
                 residual=True):
        super().__init__()

        assert out_channels % (len(dilations) + 2) == 0, '# out channels should be multiples of # branches'

        self.num_branches = len(dilations) + 2
        branch_channels = out_channels // self.num_branches

        if type(kernel_size) == list:
            assert len(kernel_size) == len(dilations)
        else:
            kernel_size = [kernel_size] * len(dilations)

        self.branches = nn.ModuleList([
            nn.Sequential(
                CBRBlock(
                    in_channels,
                    branch_channels, 
                    mode=mode, 
                ),
                CBRBlock(
                    branch_channels,
                    branch_channels,
                    mode=mode,
                    kernel_size=_ks,
                    stride=stride,
                    dilation=_dilation,
                    relu=False
                ),
                #SelfAtt(branch_channels, branch_channels, mode='spatial')
            ) for _ks, _dilation in zip(kernel_size, dilations)
        ])

        # Additional Max & 1x1 branch
        self.branches.append(
            nn.Sequential(
                CBRBlock(in_channels, branch_channels, mode=mode),
                nn.MaxPool2d(kernel_size=(3, 1), stride=(stride, 1), padding=(1, 0)),
                nn.BatchNorm2d(branch_channels),  # 为什么还要加bn
                #SelfAtt(branch_channels, branch_channels, mode='spatial')
            )
        )

        self.branches.append(CBRBlock(in_channels, branch_channels, mode=mode, stride=stride, relu=False)
            #nn.Sequential(
            #    CBRBlock(in_channels, branch_channels, mode=mode, stride=stride, relu=False),
                #SelfAtt(branch_channels, branch_channels, mode='spatial')
            #)
        )

        self.residual = ResBlock(
                            in_channels, 
                            out_channels, 
                            mode='TV', 
                            stride=stride, 
                            residual=residual)


    def forward(self, x):
        res = self.residual(x)
        branch_outs = []
        for tempconv in self.branches:
            out = tempconv(x)
            branch_outs.append(out)

        out = torch.cat(branch_outs, dim=1)
        out += res

        return out



class GCN(nn.Module):
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 rel_reduction=8):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        if in_channels == 3:
            rel_channels = 8
        else:
            rel_channels = in_channels // rel_reduction

        self.conv1 = nn.Conv2d(in_channels, rel_channels, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels, rel_channels, kernel_size=1)
        self.conv3 = nn.Conv2d(rel_channels, out_channels, kernel_size=1)

        self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1)

        self.tanh = nn.Tanh()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)

    def forward(self, x, A=None, alpha=1):
        inputs = self.residual(x) # N out_channels T V

        x1 = self.conv1(x).mean(-2).unsqueeze(-1) # N C_rel V 1
        x2 = self.conv2(x).mean(-2).unsqueeze(-2) # N C_rel 1 V
        x1 = self.tanh(x1 - x2) # N C_rel V V

        A = A.unsqueeze(0).unsqueeze(0) if A is not None else 0 # 1 1 V V 
        A = self.conv3(x1) * alpha + A # N C V V

        output = torch.einsum('ncuv,nctv->nctu', A, inputs)

        return output


class GCNBlock(nn.Module):
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 A,
                 adaptive=True, 
                 residual=True):
        super().__init__()

        self.adaptive = adaptive
        self.num_subset = A.shape[0]

        # conv for gcn
        self.convs = nn.ModuleList()
        for i in range(self.num_subset):
            self.convs.append(GCN(in_channels, out_channels))

        # residual
        self.residual = ResBlock(in_channels, out_channels, mode='TV', residual=residual)

        # perturbation
        if self.adaptive:
            self.PA = nn.Parameter(torch.from_numpy(A.astype(np.float32)))
        else:
            self.A = Variable(torch.from_numpy(A.astype(np.float32)), requires_grad=False)

        self.alpha = nn.Parameter(torch.zeros(1))
        self.bn = nn.BatchNorm2d(out_channels)
        self.soft = nn.Softmax(-2)
        self.relu = nn.ReLU(inplace=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)
        bn_init(self.bn, 1e-6)

    def forward(self, x):
        y = None
        if self.adaptive:
            A = self.PA
        else:
            A = self.A.cuda(x.get_device())

        for i in range(self.num_subset):
            z = self.convs[i](x, A[i], self.alpha)
            y = z + y if y is not None else z

        y = self.relu(self.bn(y) + self.residual(x))

        return y


class GCN_TCN(nn.Module):
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 A,
                 kernel_size=5,
                 stride=1,
                 dilations=[1,2],
                 residual=True, 
                 adaptive=True):
        super().__init__()

        self.gcn = GCNBlock(in_channels, out_channels, A, adaptive=adaptive)
        self.cnn = CNNBlock(
                        out_channels,
                        out_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        dilations=dilations,
                        residual=False)

        self.relu = nn.ReLU(inplace=True)

        self.residual = ResBlock(
                            in_channels, 
                            out_channels,
                            kernel_size=kernel_size,
                            stride=stride,
                            residual=residual)

    def forward(self, x):
        y = self.relu(self.cnn(self.gcn(x)) + self.residual(x))

        return y



@BACKBONES.register_module()
class GTCN(nn.Module):
    def __init__(self,  
                 V=25,
                 in_channels=3,
                 adaptive=True,
                 pretrained=None,):
        super(GTCN, self).__init__()

        self.graph = GraphA()

        A = self.graph.A # 3,25,25

        self.num_point = V
        self.data_bn = nn.BatchNorm1d(in_channels * V)

        base_channel = 64
        self.l1 = GCN_TCN(in_channels, base_channel, A, residual=False, adaptive=adaptive)
        self.l2 = GCN_TCN(base_channel, base_channel, A, adaptive=adaptive)
        self.l3 = GCN_TCN(base_channel, base_channel, A, adaptive=adaptive)
        self.l4 = GCN_TCN(base_channel, base_channel, A, adaptive=adaptive)
        self.l5 = GCN_TCN(base_channel, base_channel*2, A, stride=2, adaptive=adaptive)
        self.l6 = GCN_TCN(base_channel*2, base_channel*2, A, adaptive=adaptive)
        self.l7 = GCN_TCN(base_channel*2, base_channel*2, A, adaptive=adaptive)
        self.l8 = GCN_TCN(base_channel*2, base_channel*4, A, stride=2, adaptive=adaptive)
        self.l9 = GCN_TCN(base_channel*4, base_channel*4, A, adaptive=adaptive)
        self.l10 = GCN_TCN(base_channel*4, base_channel*4, A, adaptive=adaptive)

        self.pretrained = pretrained


    def init_weights(self):
        """Initiate the parameters either from existing checkpoint or from
        scratch."""
        if isinstance(self.pretrained, str):
            logger = get_root_logger()
            logger.info(f'load model from: {self.pretrained}')

            load_checkpoint(self, self.pretrained, strict=False, logger=logger)

        elif self.pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
                elif isinstance(m, nn.Linear):
                    normal_init(m)
                elif isinstance(m, _BatchNorm):
                    constant_init(m, 1)
        else:
            raise TypeError('pretrained must be a str or None')



    def forward(self, inputs):
        if len(inputs.shape) == 5:
            N, C, T, V, _ = inputs.shape
            inputs = inputs.permute(0, 4, 1, 3, 2).contiguous().view(-1, C, V, T) # N*M C V T

        x = inputs.view(-1, C*V, T)
        x = self.data_bn(x)
        x = x.view(-1, C, V, T).permute(0, 1, 3, 2).contiguous() # N*M, C, T, V
        
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.l5(x)
        x = self.l6(x)
        x = self.l7(x)
        x = self.l8(x)
        x = self.l9(x)
        x = self.l10(x) # N*M, C, T, V

        return x


@BACKBONES.register_module()
class GTCNIdxNoAtt(nn.Module):
    def __init__(self,  
                 V=25,
                 in_channels=3,
                 adaptive=True,
                 clip_len=20,
                 idx_mode='jpt',
                 pretrained=None,):
        super(GTCNIdxNoAtt, self).__init__()

        self.graph = GraphA()

        A = self.graph.A # 3,25,25

        self.num_point = V
        self.data_bn = nn.BatchNorm1d(in_channels * V)

        self.block_input_jnt = block_joint_input(in_channels=in_channels, out_channels=32)
        self.idx_block = IdxBlock(T=clip_len, V=V, out_channels=32, mode=idx_mode)

        base_channel = 64
        #self.l1 = GCN_TCN(in_channels, base_channel, A, residual=False, adaptive=adaptive)
        self.l1 = GCN_TCN(base_channel, base_channel, A, residual=False, adaptive=adaptive)
        self.l2 = GCN_TCN(base_channel, base_channel, A, adaptive=adaptive)
        self.l3 = GCN_TCN(base_channel, base_channel, A, adaptive=adaptive)
        self.l4 = GCN_TCN(base_channel, base_channel, A, adaptive=adaptive)
        self.l5 = GCN_TCN(base_channel, base_channel*2, A, stride=2, adaptive=adaptive)
        self.l6 = GCN_TCN(base_channel*2, base_channel*2, A, adaptive=adaptive)
        self.l7 = GCN_TCN(base_channel*2, base_channel*2, A, adaptive=adaptive)
        self.l8 = GCN_TCN(base_channel*2, base_channel*4, A, stride=2, adaptive=adaptive)
        self.l9 = GCN_TCN(base_channel*4, base_channel*4, A, adaptive=adaptive)
        self.l10 = GCN_TCN(base_channel*4, base_channel*4, A, adaptive=adaptive)

        self.pretrained = pretrained


    def init_weights(self):
        """Initiate the parameters either from existing checkpoint or from
        scratch."""
        if isinstance(self.pretrained, str):
            logger = get_root_logger()
            logger.info(f'load model from: {self.pretrained}')

            load_checkpoint(self, self.pretrained, strict=False, logger=logger)

        elif self.pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
                elif isinstance(m, nn.Linear):
                    normal_init(m)
                elif isinstance(m, _BatchNorm):
                    constant_init(m, 1)
        else:
            raise TypeError('pretrained must be a str or None')



    def forward(self, inputs):
        if len(inputs.shape) == 5:
            N, C, T, V, _ = inputs.shape
            inputs = inputs.permute(0, 4, 1, 3, 2).contiguous().view(-1, C, V, T) # N*M C V T

        x = inputs.view(-1, C*V, T)
        x = self.data_bn(x)
        #x = x.view(-1, C, V, T).permute(0, 1, 3, 2).contiguous() # N*M, C, T, V

        x = x.view(-1, C, V, T)

        idx_jpt = self.idx_block(x).permute(0, 1, 3, 2).contiguous() # (N, 32, T, V)
        x = self.block_input_jnt(x).permute(0, 1, 3, 2).contiguous() # (N, 32, T, V)
        x = torch.cat([x, idx_jpt], 1) # (N, 64, T, V)
        
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.l5(x)
        x = self.l6(x)
        x = self.l7(x)
        x = self.l8(x)
        x = self.l9(x)
        x = self.l10(x) # N*M, C, T, V

        return x


@BACKBONES.register_module()
class GTCNIdxNoAttInput(nn.Module):
    def __init__(self,  
                 V=25,
                 in_channels=3,
                 adaptive=True,
                 clip_len=20,
                 idx_mode='jpt',
                 pretrained=None,):
        super(GTCNIdxNoAttInput, self).__init__()

        self.graph = GraphA()

        A = self.graph.A # 3,25,25

        self.num_point = V
        self.data_bn = nn.BatchNorm1d(in_channels * V)

        self.block_input_jnt = block_joint_input(in_channels=in_channels, out_channels=32)
        self.idx_block = IdxBlock(T=clip_len, V=V, out_channels=32, mode=idx_mode)

        base_channel = 64
        #self.l1 = GCN_TCN(in_channels, base_channel, A, residual=False, adaptive=adaptive)
        self.l1 = GCN_TCN(base_channel, base_channel, A, residual=False, adaptive=adaptive)
        self.l2 = GCN_TCN(base_channel, base_channel, A, adaptive=adaptive)
        self.l3 = GCN_TCN(base_channel, base_channel, A, adaptive=adaptive)
        self.l4 = GCN_TCN(base_channel, base_channel, A, adaptive=adaptive)
        self.l5 = GCN_TCN(base_channel, base_channel*2, A, stride=2, adaptive=adaptive)
        self.l6 = GCN_TCN(base_channel*2, base_channel*2, A, adaptive=adaptive)
        self.l7 = GCN_TCN(base_channel*2, base_channel*2, A, adaptive=adaptive)
        self.l8 = GCN_TCN(base_channel*2, base_channel*4, A, stride=2, adaptive=adaptive)
        self.l9 = GCN_TCN(base_channel*4, base_channel*4, A, adaptive=adaptive)
        self.l10 = GCN_TCN(base_channel*4, base_channel*4, A, adaptive=adaptive)

        self.pretrained = pretrained


    def init_weights(self):
        """Initiate the parameters either from existing checkpoint or from
        scratch."""
        if isinstance(self.pretrained, str):
            logger = get_root_logger()
            logger.info(f'load model from: {self.pretrained}')

            load_checkpoint(self, self.pretrained, strict=False, logger=logger)

        elif self.pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
                elif isinstance(m, nn.Linear):
                    normal_init(m)
                elif isinstance(m, _BatchNorm):
                    constant_init(m, 1)
        else:
            raise TypeError('pretrained must be a str or None')



    def forward(self, inputs):
        if len(inputs.shape) == 5:
            N, C, T, V, _ = inputs.shape
            inputs = inputs.permute(0, 4, 1, 3, 2).contiguous().view(-1, C, V, T) # N*M C V T

        #x = inputs.view(-1, C*V, T)
        #x = self.data_bn(x)
        #x = x.view(-1, C, V, T).permute(0, 1, 3, 2).contiguous() # N*M, C, T, V

        #x = x.view(-1, C, V, T)
        
        #idx_jpt = self.idx_block(inputs).permute(0, 1, 3, 2).contiguous() # (N, 32, T, V)
        #x = self.block_input_jnt(inputs).permute(0, 1, 3, 2).contiguous() # (N, 32, T, V)
        #x = torch.cat([x, idx_jpt], 1) # (N, 64, T, V)

        idx_jpt = self.idx_block(inputs).permute(0, 1, 3, 2).contiguous() # (N, 32, T, V)
        x = self.block_input_jnt(inputs).permute(0, 1, 3, 2).contiguous() # (N, 32, T, V)
        x = torch.cat([x, idx_jpt], 1) # (N, 64, T, V)

        
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.l5(x)
        x = self.l6(x)
        x = self.l7(x)
        x = self.l8(x)
        x = self.l9(x)
        x = self.l10(x) # N*M, C, T, V

        return x


@BACKBONES.register_module()
class GTCNIdxNoAttDjnt(nn.Module):
    def __init__(self,  
                 V=25,
                 in_channels=3,
                 adaptive=True,
                 clip_len=20,
                 idx_mode='jpt',
                 pretrained=None,):
        super(GTCNIdxNoAttDjnt, self).__init__()

        self.graph = GraphA()

        A = self.graph.A # 3,25,25

        self.num_point = V
        self.data_bn = nn.BatchNorm1d(in_channels * V)
        self.data_bn2 = nn.BatchNorm1d(in_channels * V)


        #self.block_input_jnt = block_joint_input(in_channels=in_channels, out_channels=32)
        #self.idx_block = IdxBlock(T=clip_len, V=V, out_channels=32, mode=idx_mode)

        base_channel = 64
        #self.l1 = GCN_TCN(in_channels, base_channel, A, residual=False, adaptive=adaptive)

        self.l11 = GCN_TCN(in_channels, base_channel, A, residual=False, adaptive=adaptive)
        self.l12 = GCN_TCN(in_channels, base_channel, A, residual=False, adaptive=adaptive)

        self.l2 = GCN_TCN(base_channel, base_channel, A, adaptive=adaptive)
        self.l3 = GCN_TCN(base_channel, base_channel, A, adaptive=adaptive)
        self.l4 = GCN_TCN(base_channel, base_channel, A, adaptive=adaptive)
        self.l5 = GCN_TCN(base_channel, base_channel*2, A, stride=2, adaptive=adaptive)
        self.l6 = GCN_TCN(base_channel*2, base_channel*2, A, adaptive=adaptive)
        self.l7 = GCN_TCN(base_channel*2, base_channel*2, A, adaptive=adaptive)
        self.l8 = GCN_TCN(base_channel*2, base_channel*4, A, stride=2, adaptive=adaptive)
        self.l9 = GCN_TCN(base_channel*4, base_channel*4, A, adaptive=adaptive)
        self.l10 = GCN_TCN(base_channel*4, base_channel*4, A, adaptive=adaptive)

        self.pretrained = pretrained


    def init_weights(self):
        """Initiate the parameters either from existing checkpoint or from
        scratch."""
        if isinstance(self.pretrained, str):
            logger = get_root_logger()
            logger.info(f'load model from: {self.pretrained}')

            load_checkpoint(self, self.pretrained, strict=False, logger=logger)

        elif self.pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
                elif isinstance(m, nn.Linear):
                    normal_init(m)
                elif isinstance(m, _BatchNorm):
                    constant_init(m, 1)
        else:
            raise TypeError('pretrained must be a str or None')



    def forward(self, inputs):
        if len(inputs.shape) == 5:
            N, C, T, V, _ = inputs.shape
            inputs = inputs.permute(0, 4, 1, 3, 2).contiguous().view(-1, C, V, T) # N*M C V T

            N2 = inputs.shape[0] # may be N*M

            # t_jpt
            t_jpt = inputs[:, :, :, 1:] - inputs[:, :, :, 0:-1] # (N, C, V, T-1)
            t_jpt = torch.cat([t_jpt.new(N2, C, V, 1).zero_(), t_jpt], dim=-1) # (N, C, V, T)

        # for input
        x = inputs.view(-1, C*V, T)
        x = self.data_bn(x)
        x = x.view(-1, C, V, T).permute(0, 1, 3, 2).contiguous() # N*M, C, T, V

        # for t_jnt
        t_x = t_jpt.view(-1, C*V, T)
        t_x = self.data_bn2(t_x)
        t_x = t_x.view(-1, C, V, T).permute(0, 1, 3, 2).contiguous() # N*M, C, T, V

        x = self.l11(x) + self.l12(t_x)


        #x = x.view(-1, C, V, T)
        
        #idx_jpt = self.idx_block(inputs).permute(0, 1, 3, 2).contiguous() # (N, 32, T, V)
        #x = self.block_input_jnt(inputs).permute(0, 1, 3, 2).contiguous() # (N, 32, T, V)
        #x = torch.cat([x, idx_jpt], 1) # (N, 64, T, V)
        
        #x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.l5(x)
        x = self.l6(x)
        x = self.l7(x)
        x = self.l8(x)
        x = self.l9(x)
        x = self.l10(x) # N*M, C, T, V

        return x



class CNNBlock_branch2(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 mode='TV',
                 kernel_size=3,
                 stride=1,
                 dilations=[1,2],
                 residual=True):
        super().__init__()

        assert out_channels % (len(dilations)) == 0, '# out channels should be multiples of # branches'

        self.num_branches = len(dilations)
        branch_channels = out_channels // self.num_branches

        if type(kernel_size) == list:
            assert len(kernel_size) == len(dilations)
        else:
            kernel_size = [kernel_size] * len(dilations)

        self.branches = nn.ModuleList([
            nn.Sequential(
                CBRBlock(
                    in_channels,
                    branch_channels, 
                    mode=mode, 
                ),
                CBRBlock(
                    branch_channels,
                    branch_channels,
                    mode=mode,
                    kernel_size=_ks,
                    stride=stride,
                    dilation=_dilation,
                    relu=False
                ),
                SelfAtt(branch_channels, branch_channels, mode='spatial')
            ) for _ks, _dilation in zip(kernel_size, dilations)
        ])

        self.residual = ResBlock(
                            in_channels, 
                            out_channels, 
                            mode='TV', 
                            stride=stride, 
                            residual=residual)


    def forward(self, x):
        res = self.residual(x)
        branch_outs = []
        for tempconv in self.branches:
            out = tempconv(x)
            branch_outs.append(out)

        out = torch.cat(branch_outs, dim=1)
        out += res

        return out


class GCN_TCN_branch2(nn.Module):
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 A,
                 kernel_size=3,
                 stride=1,
                 dilations=[1,2],
                 residual=True, 
                 adaptive=True):
        super().__init__()

        self.gcn = GCNBlock(in_channels, out_channels, A, adaptive=adaptive)
        self.cnn = CNNBlock_branch2(
                        out_channels,
                        out_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        dilations=dilations,
                        residual=False)

        self.relu = nn.ReLU(inplace=True)

        self.residual = ResBlock(
                            in_channels, 
                            out_channels,
                            kernel_size=kernel_size,
                            stride=stride,
                            residual=residual)

    def forward(self, x):
        y = self.relu(self.cnn(self.gcn(x)) + self.residual(x))

        return y

# bad, deleted, epoch1 24.62 (top1)
@BACKBONES.register_module()
class GTCNBranch2(nn.Module):
    def __init__(self,  
                 V=25,
                 in_channels=3,
                 adaptive=True,
                 pretrained=None,):
        super(GTCNBranch2, self).__init__()

        self.graph = GraphA()

        A = self.graph.A # 3,25,25

        self.num_point = V
        self.data_bn = nn.BatchNorm1d(in_channels * V)

        base_channel = 64
        self.l1 = GCN_TCN_branch2(in_channels, base_channel, A, residual=False, adaptive=adaptive)
        self.l2 = GCN_TCN_branch2(base_channel, base_channel, A, adaptive=adaptive)
        #self.l3 = GCN_TCN(base_channel, base_channel, A, adaptive=adaptive)
        #self.l4 = GCN_TCN(base_channel, base_channel, A, adaptive=adaptive)
        self.l5 = GCN_TCN_branch2(base_channel, base_channel*2, A, stride=1, adaptive=adaptive)
        self.l6 = GCN_TCN_branch2(base_channel*2, base_channel*2, A, adaptive=adaptive)
        #self.l7 = GCN_TCN(base_channel*2, base_channel*2, A, adaptive=adaptive)
        self.l8 = GCN_TCN_branch2(base_channel*2, base_channel*4, A, stride=1, adaptive=adaptive)
        #self.l9 = GCN_TCN(base_channel*4, base_channel*4, A, adaptive=adaptive)
        self.l10 = GCN_TCN_branch2(base_channel*4, base_channel*4, A, adaptive=adaptive)

        self.pretrained = pretrained


    def init_weights(self):
        """Initiate the parameters either from existing checkpoint or from
        scratch."""
        if isinstance(self.pretrained, str):
            logger = get_root_logger()
            logger.info(f'load model from: {self.pretrained}')

            load_checkpoint(self, self.pretrained, strict=False, logger=logger)

        elif self.pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
                elif isinstance(m, nn.Linear):
                    normal_init(m)
                elif isinstance(m, _BatchNorm):
                    constant_init(m, 1)
        else:
            raise TypeError('pretrained must be a str or None')



    def forward(self, inputs):
        if len(inputs.shape) == 5:
            N, C, T, V, _ = inputs.shape
            inputs = inputs.permute(0, 4, 1, 3, 2).contiguous().view(-1, C, V, T) # N*M C V T

        x = inputs.view(-1, C*V, T)
        x = self.data_bn(x)
        x = x.view(-1, C, V, T).permute(0, 1, 3, 2).contiguous() # N*M, C, T, V
        
        x = self.l1(x)
        x = self.l2(x)
        #x = self.l3(x)
        #x = self.l4(x)
        x = self.l5(x)
        x = self.l6(x)
        #x = self.l7(x)
        x = self.l8(x)
        #x = self.l9(x)
        x = self.l10(x) # N*M, C, T, V

        return x




@BACKBONES.register_module()
class DeepGTCN(nn.Module):
    def __init__(self,  
                 V=25,
                 in_channels=3,
                 adaptive=True,
                 pretrained=None,):
        super(DeepGTCN, self).__init__()

        self.graph = GraphA()

        A = self.graph.A # 3,25,25

        self.num_point = V
        self.data_bn = nn.BatchNorm1d(in_channels * V)

        base_channel = 64
        self.l1 = GCN_TCN(in_channels, base_channel, A, residual=False, adaptive=adaptive)
        self.l2 = GCN_TCN(base_channel, base_channel, A, adaptive=adaptive)
        self.l3 = GCN_TCN(base_channel, base_channel, A, adaptive=adaptive)
        self.l4 = GCN_TCN(base_channel, base_channel, A, adaptive=adaptive)

        self.l5 = GCN_TCN(base_channel, base_channel*2, A, stride=2, adaptive=adaptive)
        self.l6 = GCN_TCN(base_channel*2, base_channel*2, A, adaptive=adaptive)
        self.l7 = GCN_TCN(base_channel*2, base_channel*2, A, adaptive=adaptive)

        self.l8 = GCN_TCN(base_channel*2, base_channel*4, A, stride=2, adaptive=adaptive)
        self.l9 = GCN_TCN(base_channel*4, base_channel*4, A, adaptive=adaptive)
        self.l10 = GCN_TCN(base_channel*4, base_channel*4, A, adaptive=adaptive)

        self.pretrained = pretrained


    def init_weights(self):
        """Initiate the parameters either from existing checkpoint or from
        scratch."""
        if isinstance(self.pretrained, str):
            logger = get_root_logger()
            logger.info(f'load model from: {self.pretrained}')

            load_checkpoint(self, self.pretrained, strict=False, logger=logger)

        elif self.pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
                elif isinstance(m, nn.Linear):
                    normal_init(m)
                elif isinstance(m, _BatchNorm):
                    constant_init(m, 1)
        else:
            raise TypeError('pretrained must be a str or None')



    def forward(self, inputs):
        if len(inputs.shape) == 5:
            N, C, T, V, _ = inputs.shape
            inputs = inputs.permute(0, 4, 1, 3, 2).contiguous().view(-1, C, V, T) # N*M C V T

        x = inputs.view(-1, C*V, T)
        x = self.data_bn(x)
        x = x.view(-1, C, V, T).permute(0, 1, 3, 2).contiguous() # N*M, C, T, V
        
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.l5(x)
        x = self.l6(x)
        x = self.l7(x)
        x = self.l8(x)
        x = self.l9(x)
        x = self.l10(x) # N*M, C, T, V

        #x = self.l11(x)

        return x



@BACKBONES.register_module()
class GTCNDjntIdx(nn.Module):
    def __init__(self,  
                 V=25,
                 in_channels=3,
                 adaptive=True,
                 clip_len=20,
                 idx_mode='jpt',
                 pretrained=None,):
        super(GTCNDjntIdx, self).__init__()

        self.graph = GraphA()

        A = self.graph.A # 3,25,25

        self.num_point = V
        self.data_bn = nn.BatchNorm1d(in_channels * V)
        self.data_bn2 = nn.BatchNorm1d(in_channels * V)


        #self.block_input_jnt = block_joint_input(in_channels=in_channels, out_channels=32)
        self.idx_block = IdxBlock(T=clip_len, V=V, out_channels=32, mode=idx_mode)

        base_channel = 64
        #self.l1 = GCN_TCN(in_channels, base_channel, A, residual=False, adaptive=adaptive)

        self.l11 = GCN_TCN(in_channels, base_channel //2, A, residual=False, adaptive=adaptive)
        self.l12 = GCN_TCN(in_channels, base_channel //2, A, residual=False, adaptive=adaptive)

        self.l2 = GCN_TCN(base_channel, base_channel, A, adaptive=adaptive)
        self.l3 = GCN_TCN(base_channel, base_channel, A, adaptive=adaptive)
        self.l4 = GCN_TCN(base_channel, base_channel, A, adaptive=adaptive)
        self.l5 = GCN_TCN(base_channel, base_channel*2, A, stride=2, adaptive=adaptive)
        self.l6 = GCN_TCN(base_channel*2, base_channel*2, A, adaptive=adaptive)
        self.l7 = GCN_TCN(base_channel*2, base_channel*2, A, adaptive=adaptive)
        self.l8 = GCN_TCN(base_channel*2, base_channel*4, A, stride=2, adaptive=adaptive)
        self.l9 = GCN_TCN(base_channel*4, base_channel*4, A, adaptive=adaptive)
        self.l10 = GCN_TCN(base_channel*4, base_channel*4, A, adaptive=adaptive)

        self.pretrained = pretrained


    def init_weights(self):
        """Initiate the parameters either from existing checkpoint or from
        scratch."""
        if isinstance(self.pretrained, str):
            logger = get_root_logger()
            logger.info(f'load model from: {self.pretrained}')

            load_checkpoint(self, self.pretrained, strict=False, logger=logger)

        elif self.pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
                elif isinstance(m, nn.Linear):
                    normal_init(m)
                elif isinstance(m, _BatchNorm):
                    constant_init(m, 1)
        else:
            raise TypeError('pretrained must be a str or None')



    def forward(self, inputs):
        if len(inputs.shape) == 5:
            N, C, T, V, _ = inputs.shape
            inputs = inputs.permute(0, 4, 1, 3, 2).contiguous().view(-1, C, V, T) # N*M C V T

            N2 = inputs.shape[0] # may be N*M

            # t_jpt
            t_jpt = inputs[:, :, :, 1:] - inputs[:, :, :, 0:-1] # (N, C, V, T-1)
            t_jpt = torch.cat([t_jpt.new(N2, C, V, 1).zero_(), t_jpt], dim=-1) # (N, C, V, T)

        idx_jpt = self.idx_block(inputs).permute(0, 1, 3, 2).contiguous() # (N, 32, T, V)

        # for input
        x = inputs.view(-1, C*V, T)
        x = self.data_bn(x)
        x = x.view(-1, C, V, T).permute(0, 1, 3, 2).contiguous() # N*M, C, T, V

        # for t_jnt
        t_x = t_jpt.view(-1, C*V, T)
        t_x = self.data_bn2(t_x)
        t_x = t_x.view(-1, C, V, T).permute(0, 1, 3, 2).contiguous() # N*M, C, T, V

        x = self.l11(x) + self.l12(t_x) # N*M C T V

        x = torch.cat([x, idx_jpt], 1) # (N, 64, T, V)
        
        #x = x.view(-1, C, V, T)
        
        #idx_jpt = self.idx_block(inputs).permute(0, 1, 3, 2).contiguous() # (N, 32, T, V)
        #x = self.block_input_jnt(inputs).permute(0, 1, 3, 2).contiguous() # (N, 32, T, V)
        #x = torch.cat([x, idx_jpt], 1) # (N, 64, T, V)
        
        #x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.l5(x)
        x = self.l6(x)
        x = self.l7(x)
        x = self.l8(x)
        x = self.l9(x)
        x = self.l10(x) # N*M, C, T, V

        return x


# bad model 20220511
@BACKBONES.register_module()
class GTCNDjntIdxDbone(nn.Module):
    def __init__(self,  
                 V=25,
                 in_channels=3,
                 adaptive=True,
                 clip_len=20,
                 idx_mode='jpt',
                 pretrained=None,
                 usebone=False):
        super(GTCNDjntIdxDbone, self).__init__()

        self.ntu_pairs = (
            (1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5),
            (7, 6), (8, 7), (9, 21), (10, 9), (11, 10), (12, 11),
            (13, 1), (14, 13), (15, 14), (16, 15), (17, 1), (18, 17),
            (19, 18), (20, 19), (22, 23), (21, 21), (23, 8), (24, 25),(25, 12)
        )

        self.usebone = usebone

        self.graph = GraphA()

        A = self.graph.A # 3,25,25

        self.num_point = V
        self.data_bn = nn.BatchNorm1d(in_channels * V)
        self.data_bn2 = nn.BatchNorm1d(in_channels * V)


        #self.block_input_jnt = block_joint_input(in_channels=in_channels, out_channels=32)
        self.idx_block = IdxBlock(T=clip_len, V=V, out_channels=32, mode=idx_mode)

        base_channel = 64
        #self.l1 = GCN_TCN(in_channels, base_channel, A, residual=False, adaptive=adaptive)

        self.l11 = GCN_TCN(in_channels, base_channel //2, A, residual=False, adaptive=adaptive)
        self.l12 = GCN_TCN(in_channels, base_channel //2, A, residual=False, adaptive=adaptive)

        if usebone:
            self.l11_bone = GCN_TCN(in_channels, base_channel //2, A, residual=False, adaptive=adaptive)
            self.l12_bone = GCN_TCN(in_channels, base_channel //2, A, residual=False, adaptive=adaptive)

        self.l2 = GCN_TCN(base_channel, base_channel, A, adaptive=adaptive)
        self.l3 = GCN_TCN(base_channel, base_channel, A, adaptive=adaptive)
        self.l4 = GCN_TCN(base_channel, base_channel, A, adaptive=adaptive)
        self.l5 = GCN_TCN(base_channel, base_channel*2, A, stride=2, adaptive=adaptive)
        self.l6 = GCN_TCN(base_channel*2, base_channel*2, A, adaptive=adaptive)
        self.l7 = GCN_TCN(base_channel*2, base_channel*2, A, adaptive=adaptive)
        self.l8 = GCN_TCN(base_channel*2, base_channel*4, A, stride=2, adaptive=adaptive)
        self.l9 = GCN_TCN(base_channel*4, base_channel*4, A, adaptive=adaptive)
        self.l10 = GCN_TCN(base_channel*4, base_channel*4, A, adaptive=adaptive)

        self.pretrained = pretrained


    def init_weights(self):
        """Initiate the parameters either from existing checkpoint or from
        scratch."""
        if isinstance(self.pretrained, str):
            logger = get_root_logger()
            logger.info(f'load model from: {self.pretrained}')

            load_checkpoint(self, self.pretrained, strict=False, logger=logger)

        elif self.pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
                elif isinstance(m, nn.Linear):
                    normal_init(m)
                elif isinstance(m, _BatchNorm):
                    constant_init(m, 1)
        else:
            raise TypeError('pretrained must be a str or None')


    def forward(self, inputs):
        if len(inputs.shape) == 5:
            N, C, T, V, _ = inputs.shape
            inputs = inputs.permute(0, 4, 1, 3, 2).contiguous().view(-1, C, V, T) # N*M C V T

            N2 = inputs.shape[0] # may be N*M

            # t_jpt
            t_jpt = inputs[:, :, :, 1:] - inputs[:, :, :, 0:-1] # (N, C, V, T-1)
            t_jpt = torch.cat([t_jpt.new(N2, C, V, 1).zero_(), t_jpt], dim=-1) # (N, C, V, T)

            # bone
            if self.usebone:
                bone = torch.zeros_like(inputs) # (N, C, V, T)
                for v1, v2 in self.ntu_pairs:
                    bone[:, :, v1-1, :] = inputs[:, :, v1-1, :] - inputs[:, :, v2-1, :] # (N, C, V, T)

                # t_bone
                t_bone = torch.zeros_like(bone)
                t_bone[:, :, :, 1:] = bone[:, :, :, 1:] - bone[:, :, :, :-1] # (N, C, V, T)

        idx_jpt = self.idx_block(inputs).permute(0, 1, 3, 2).contiguous() # (N, 32, T, V)

        # for input
        x = inputs.view(-1, C*V, T)
        x = self.data_bn(x)
        x = x.view(-1, C, V, T).permute(0, 1, 3, 2).contiguous() # N*M, C, T, V

        # for t_jnt
        t_x = t_jpt.view(-1, C*V, T)
        t_x = self.data_bn2(t_x)
        t_x = t_x.view(-1, C, V, T).permute(0, 1, 3, 2).contiguous() # N*M, C, T, V

        # for bone and t_bone
        if self.usebone:
            x_bone = bone.view(-1, C*V, T)
            x_bone = self.data_bn(x_bone)
            x_bone = x_bone.view(-1, C, V, T).permute(0, 1, 3, 2).contiguous() # N*M, C, T, V

            t_x_bone = t_bone.view(-1, C*V, T)
            t_x_bone = self.data_bn2(t_x_bone)
            t_x_bone = t_x_bone.view(-1, C, V, T).permute(0, 1, 3, 2).contiguous() # N*M, C, T, V

            x_bone = self.l11_bone(x_bone) + self.l12(t_x_bone)

        x = self.l11(x) + self.l12(t_x) # N*M C T V

        if self.usebone:
            x = torch.cat([x+x_bone, idx_jpt], 1)
        else:
            x = torch.cat([x, idx_jpt], 1) # (N, 64, T, V)
        
        #x = x.view(-1, C, V, T)
        
        #idx_jpt = self.idx_block(inputs).permute(0, 1, 3, 2).contiguous() # (N, 32, T, V)
        #x = self.block_input_jnt(inputs).permute(0, 1, 3, 2).contiguous() # (N, 32, T, V)
        #x = torch.cat([x, idx_jpt], 1) # (N, 64, T, V)
        
        #x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.l5(x)
        x = self.l6(x)
        x = self.l7(x)
        x = self.l8(x)
        x = self.l9(x)
        x = self.l10(x) # N*M, C, T, V

        return x

# 20220701 change from 'GTCNIdxNoAttDjnt', which as the current best performance (86.28). 

@BACKBONES.register_module()
class GTCNDjnt(nn.Module):
    def __init__(self,  
                 V=25,
                 in_channels=3,
                 adaptive=True,
                 clip_len=20,
                 idx_mode='jpt',
                 pretrained=None,):
        super(GTCNDjnt, self).__init__()

        self.graph = GraphA()

        #A = self.graph.A # 3,25,25
        A = self.graph.A_lr # 5,25,25
        #A = self.graph.A_inout_lr # 3, 25, 25

        self.num_point = V
        self.data_bn = nn.BatchNorm1d(in_channels * V)
        self.data_bn2 = nn.BatchNorm1d(in_channels * V)


        #self.block_input_jnt = block_joint_input(in_channels=in_channels, out_channels=32)
        #self.idx_block = IdxBlock(T=clip_len, V=V, out_channels=32, mode=idx_mode)

        base_channel = 64
        #self.l1 = GCN_TCN(in_channels, base_channel, A, residual=False, adaptive=adaptive)

        self.l11 = GCN_TCN(in_channels, base_channel, A, residual=False, adaptive=adaptive)
        self.l12 = GCN_TCN(in_channels, base_channel, A, residual=False, adaptive=adaptive)

        self.l2 = GCN_TCN(base_channel, base_channel, A, adaptive=adaptive)
        self.l3 = GCN_TCN(base_channel, base_channel, A, adaptive=adaptive)
        self.l4 = GCN_TCN(base_channel, base_channel, A, adaptive=adaptive)
        self.l5 = GCN_TCN(base_channel, base_channel*2, A, stride=2, adaptive=adaptive)
        self.l6 = GCN_TCN(base_channel*2, base_channel*2, A, adaptive=adaptive)
        self.l7 = GCN_TCN(base_channel*2, base_channel*2, A, adaptive=adaptive)
        self.l8 = GCN_TCN(base_channel*2, base_channel*4, A, stride=2, adaptive=adaptive)
        self.l9 = GCN_TCN(base_channel*4, base_channel*4, A, adaptive=adaptive)
        self.l10 = GCN_TCN(base_channel*4, base_channel*4, A, adaptive=adaptive)

        self.pretrained = pretrained


    def init_weights(self):
        """Initiate the parameters either from existing checkpoint or from
        scratch."""
        if isinstance(self.pretrained, str):
            logger = get_root_logger()
            logger.info(f'load model from: {self.pretrained}')

            load_checkpoint(self, self.pretrained, strict=False, logger=logger)

        elif self.pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
                elif isinstance(m, nn.Linear):
                    normal_init(m)
                elif isinstance(m, _BatchNorm):
                    constant_init(m, 1)
        else:
            raise TypeError('pretrained must be a str or None')



    def forward(self, inputs):
        if len(inputs.shape) == 5:
            N, C, T, V, _ = inputs.shape
            inputs = inputs.permute(0, 4, 1, 3, 2).contiguous().view(-1, C, V, T) # N*M C V T

            N2 = inputs.shape[0] # may be N*M

            # t_jpt
            t_jpt = inputs[:, :, :, 1:] - inputs[:, :, :, 0:-1] # (N, C, V, T-1)
            t_jpt = torch.cat([t_jpt.new(N2, C, V, 1).zero_(), t_jpt], dim=-1) # (N, C, V, T)

        # for input
        x = inputs.view(-1, C*V, T)
        x = self.data_bn(x)
        x = x.view(-1, C, V, T).permute(0, 1, 3, 2).contiguous() # N*M, C, T, V

        # for t_jnt
        t_x = t_jpt.view(-1, C*V, T)
        t_x = self.data_bn2(t_x)
        t_x = t_x.view(-1, C, V, T).permute(0, 1, 3, 2).contiguous() # N*M, C, T, V

        x = self.l11(x) + self.l12(t_x)
        
        #x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.l5(x)
        x = self.l6(x)
        x = self.l7(x)
        x = self.l8(x)
        x = self.l9(x)
        x = self.l10(x) # N*M, C, T, V

        return x


@BACKBONES.register_module()
class GTCNDjntInOutLR(nn.Module):
    def __init__(self,  
                 V=25,
                 in_channels=3,
                 adaptive=True,
                 clip_len=20,
                 idx_mode='jpt',
                 pretrained=None,):
        super(GTCNDjnt, self).__init__()

        self.graph = GraphA()

        #A = self.graph.A # 3,25,25
        #A = self.graph.A_lr # 5,25,25
        A = self.graph.A_inout_lr # 3, 25, 25

        self.num_point = V
        self.data_bn = nn.BatchNorm1d(in_channels * V)
        self.data_bn2 = nn.BatchNorm1d(in_channels * V)


        #self.block_input_jnt = block_joint_input(in_channels=in_channels, out_channels=32)
        #self.idx_block = IdxBlock(T=clip_len, V=V, out_channels=32, mode=idx_mode)

        base_channel = 64
        #self.l1 = GCN_TCN(in_channels, base_channel, A, residual=False, adaptive=adaptive)

        self.l11 = GCN_TCN(in_channels, base_channel, A, residual=False, adaptive=adaptive)
        self.l12 = GCN_TCN(in_channels, base_channel, A, residual=False, adaptive=adaptive)

        self.l2 = GCN_TCN(base_channel, base_channel, A, adaptive=adaptive)
        self.l3 = GCN_TCN(base_channel, base_channel, A, adaptive=adaptive)
        self.l4 = GCN_TCN(base_channel, base_channel, A, adaptive=adaptive)
        self.l5 = GCN_TCN(base_channel, base_channel*2, A, stride=2, adaptive=adaptive)
        self.l6 = GCN_TCN(base_channel*2, base_channel*2, A, adaptive=adaptive)
        self.l7 = GCN_TCN(base_channel*2, base_channel*2, A, adaptive=adaptive)
        self.l8 = GCN_TCN(base_channel*2, base_channel*4, A, stride=2, adaptive=adaptive)
        self.l9 = GCN_TCN(base_channel*4, base_channel*4, A, adaptive=adaptive)
        self.l10 = GCN_TCN(base_channel*4, base_channel*4, A, adaptive=adaptive)

        self.pretrained = pretrained


    def init_weights(self):
        """Initiate the parameters either from existing checkpoint or from
        scratch."""
        if isinstance(self.pretrained, str):
            logger = get_root_logger()
            logger.info(f'load model from: {self.pretrained}')

            load_checkpoint(self, self.pretrained, strict=False, logger=logger)

        elif self.pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
                elif isinstance(m, nn.Linear):
                    normal_init(m)
                elif isinstance(m, _BatchNorm):
                    constant_init(m, 1)
        else:
            raise TypeError('pretrained must be a str or None')



    def forward(self, inputs):
        if len(inputs.shape) == 5:
            N, C, T, V, _ = inputs.shape
            inputs = inputs.permute(0, 4, 1, 3, 2).contiguous().view(-1, C, V, T) # N*M C V T

            N2 = inputs.shape[0] # may be N*M

            # t_jpt
            t_jpt = inputs[:, :, :, 1:] - inputs[:, :, :, 0:-1] # (N, C, V, T-1)
            t_jpt = torch.cat([t_jpt.new(N2, C, V, 1).zero_(), t_jpt], dim=-1) # (N, C, V, T)

        # for input
        x = inputs.view(-1, C*V, T)
        x = self.data_bn(x)
        x = x.view(-1, C, V, T).permute(0, 1, 3, 2).contiguous() # N*M, C, T, V

        # for t_jnt
        t_x = t_jpt.view(-1, C*V, T)
        t_x = self.data_bn2(t_x)
        t_x = t_x.view(-1, C, V, T).permute(0, 1, 3, 2).contiguous() # N*M, C, T, V

        x = self.l11(x) + self.l12(t_x)
        
        #x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.l5(x)
        x = self.l6(x)
        x = self.l7(x)
        x = self.l8(x)
        x = self.l9(x)
        x = self.l10(x) # N*M, C, T, V

        return x



@BACKBONES.register_module()
class GTCNDjntLR(nn.Module):
    def __init__(self,  
                 V=25,
                 in_channels=3,
                 adaptive=True,
                 clip_len=20,
                 idx_mode='jpt',
                 pretrained=None,):
        super(GTCNDjntLR, self).__init__()

        self.graph = GraphA()

        #A = self.graph.A # 3,25,25
        A = self.graph.A_lr # 5,25,25
        #A = self.graph.A_inout_lr # 3, 25, 25

        self.num_point = V
        self.data_bn = nn.BatchNorm1d(in_channels * V)
        self.data_bn2 = nn.BatchNorm1d(in_channels * V)


        #self.block_input_jnt = block_joint_input(in_channels=in_channels, out_channels=32)
        #self.idx_block = IdxBlock(T=clip_len, V=V, out_channels=32, mode=idx_mode)

        base_channel = 64
        #self.l1 = GCN_TCN(in_channels, base_channel, A, residual=False, adaptive=adaptive)

        self.l11 = GCN_TCN(in_channels, base_channel, A, residual=False, adaptive=adaptive)
        self.l12 = GCN_TCN(in_channels, base_channel, A, residual=False, adaptive=adaptive)

        self.l2 = GCN_TCN(base_channel, base_channel, A, adaptive=adaptive)
        self.l3 = GCN_TCN(base_channel, base_channel, A, adaptive=adaptive)
        self.l4 = GCN_TCN(base_channel, base_channel, A, adaptive=adaptive)
        self.l5 = GCN_TCN(base_channel, base_channel*2, A, stride=2, adaptive=adaptive)
        self.l6 = GCN_TCN(base_channel*2, base_channel*2, A, adaptive=adaptive)
        self.l7 = GCN_TCN(base_channel*2, base_channel*2, A, adaptive=adaptive)
        self.l8 = GCN_TCN(base_channel*2, base_channel*4, A, stride=2, adaptive=adaptive)
        self.l9 = GCN_TCN(base_channel*4, base_channel*4, A, adaptive=adaptive)
        self.l10 = GCN_TCN(base_channel*4, base_channel*4, A, adaptive=adaptive)

        self.pretrained = pretrained


    def init_weights(self):
        """Initiate the parameters either from existing checkpoint or from
        scratch."""
        if isinstance(self.pretrained, str):
            logger = get_root_logger()
            logger.info(f'load model from: {self.pretrained}')

            load_checkpoint(self, self.pretrained, strict=False, logger=logger)

        elif self.pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
                elif isinstance(m, nn.Linear):
                    normal_init(m)
                elif isinstance(m, _BatchNorm):
                    constant_init(m, 1)
        else:
            raise TypeError('pretrained must be a str or None')



    def forward(self, inputs):
        if len(inputs.shape) == 5:
            N, C, T, V, _ = inputs.shape
            inputs = inputs.permute(0, 4, 1, 3, 2).contiguous().view(-1, C, V, T) # N*M C V T

            N2 = inputs.shape[0] # may be N*M

            # t_jpt
            t_jpt = inputs[:, :, :, 1:] - inputs[:, :, :, 0:-1] # (N, C, V, T-1)
            t_jpt = torch.cat([t_jpt.new(N2, C, V, 1).zero_(), t_jpt], dim=-1) # (N, C, V, T)

        # for input
        x = inputs.view(-1, C*V, T)
        x = self.data_bn(x)
        x = x.view(-1, C, V, T).permute(0, 1, 3, 2).contiguous() # N*M, C, T, V

        # for t_jnt
        t_x = t_jpt.view(-1, C*V, T)
        t_x = self.data_bn2(t_x)
        t_x = t_x.view(-1, C, V, T).permute(0, 1, 3, 2).contiguous() # N*M, C, T, V

        x = self.l11(x) + self.l12(t_x)
        
        #x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.l5(x)
        x = self.l6(x)
        x = self.l7(x)
        x = self.l8(x)
        x = self.l9(x)
        x = self.l10(x) # N*M, C, T, V

        return x


@BACKBONES.register_module()
class GTCNDjnt17(nn.Module):
    def __init__(self,  
                 V=25,
                 in_channels=3,
                 adaptive=True,
                 clip_len=20,
                 idx_mode='jpt',
                 pretrained=None,):
        super(GTCNDjnt17, self).__init__()

        self.graph = GraphA()

        #A = self.graph.A # 3,25,25
        #A = self.graph.A_lr # 5,25,25
        #A = self.graph.A_inout_lr # 3, 25, 25
        A = self.graph.A_17

        self.num_point = V
        self.data_bn = nn.BatchNorm1d(in_channels * V)
        self.data_bn2 = nn.BatchNorm1d(in_channels * V)


        #self.block_input_jnt = block_joint_input(in_channels=in_channels, out_channels=32)
        #self.idx_block = IdxBlock(T=clip_len, V=V, out_channels=32, mode=idx_mode)

        base_channel = 64
        #self.l1 = GCN_TCN(in_channels, base_channel, A, residual=False, adaptive=adaptive)

        self.l11 = GCN_TCN(in_channels, base_channel, A, residual=False, adaptive=adaptive)
        self.l12 = GCN_TCN(in_channels, base_channel, A, residual=False, adaptive=adaptive)

        self.l2 = GCN_TCN(base_channel, base_channel, A, adaptive=adaptive)
        self.l3 = GCN_TCN(base_channel, base_channel, A, adaptive=adaptive)
        self.l4 = GCN_TCN(base_channel, base_channel, A, adaptive=adaptive)
        self.l5 = GCN_TCN(base_channel, base_channel*2, A, stride=2, adaptive=adaptive)
        self.l6 = GCN_TCN(base_channel*2, base_channel*2, A, adaptive=adaptive)
        self.l7 = GCN_TCN(base_channel*2, base_channel*2, A, adaptive=adaptive)
        self.l8 = GCN_TCN(base_channel*2, base_channel*4, A, stride=2, adaptive=adaptive)
        self.l9 = GCN_TCN(base_channel*4, base_channel*4, A, adaptive=adaptive)
        self.l10 = GCN_TCN(base_channel*4, base_channel*4, A, adaptive=adaptive)

        self.pretrained = pretrained


    def init_weights(self):
        """Initiate the parameters either from existing checkpoint or from
        scratch."""
        if isinstance(self.pretrained, str):
            logger = get_root_logger()
            logger.info(f'load model from: {self.pretrained}')

            load_checkpoint(self, self.pretrained, strict=False, logger=logger)

        elif self.pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
                elif isinstance(m, nn.Linear):
                    normal_init(m)
                elif isinstance(m, _BatchNorm):
                    constant_init(m, 1)
        else:
            raise TypeError('pretrained must be a str or None')



    def forward(self, inputs):
        if len(inputs.shape) == 5:
            N, C, T, V, _ = inputs.shape
            inputs = inputs.permute(0, 4, 1, 3, 2).contiguous().view(-1, C, V, T) # N*M C V T

            N2 = inputs.shape[0] # may be N*M

            # t_jpt
            t_jpt = inputs[:, :, :, 1:] - inputs[:, :, :, 0:-1] # (N, C, V, T-1)
            t_jpt = torch.cat([t_jpt.new(N2, C, V, 1).zero_(), t_jpt], dim=-1) # (N, C, V, T)

        # for input
        x = inputs.view(-1, C*V, T)
        x = self.data_bn(x)
        x = x.view(-1, C, V, T).permute(0, 1, 3, 2).contiguous() # N*M, C, T, V

        # for t_jnt
        t_x = t_jpt.view(-1, C*V, T)
        t_x = self.data_bn2(t_x)
        t_x = t_x.view(-1, C, V, T).permute(0, 1, 3, 2).contiguous() # N*M, C, T, V

        x = self.l11(x) + self.l12(t_x)
        
        #x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.l5(x)
        x = self.l6(x)
        x = self.l7(x)
        x = self.l8(x)
        x = self.l9(x)
        x = self.l10(x) # N*M, C, T, V

        return x



@BACKBONES.register_module()
class GTCNDjnt17LR(nn.Module):
    def __init__(self,  
                 V=25,
                 in_channels=3,
                 adaptive=True,
                 clip_len=20,
                 idx_mode='jpt',
                 pretrained=None,):
        super(GTCNDjnt17LR, self).__init__()

        self.graph = GraphA()

        #A = self.graph.A # 3,25,25
        #A = self.graph.A_lr # 5,25,25
        #A = self.graph.A_inout_lr # 3, 25, 25
        #A = self.graph.A_17
        A = self.graph.A_17_lr

        self.num_point = V
        self.data_bn = nn.BatchNorm1d(in_channels * V)
        self.data_bn2 = nn.BatchNorm1d(in_channels * V)


        #self.block_input_jnt = block_joint_input(in_channels=in_channels, out_channels=32)
        #self.idx_block = IdxBlock(T=clip_len, V=V, out_channels=32, mode=idx_mode)

        base_channel = 64
        #self.l1 = GCN_TCN(in_channels, base_channel, A, residual=False, adaptive=adaptive)

        self.l11 = GCN_TCN(in_channels, base_channel, A, residual=False, adaptive=adaptive)
        self.l12 = GCN_TCN(in_channels, base_channel, A, residual=False, adaptive=adaptive)

        self.l2 = GCN_TCN(base_channel, base_channel, A, adaptive=adaptive)
        self.l3 = GCN_TCN(base_channel, base_channel, A, adaptive=adaptive)
        self.l4 = GCN_TCN(base_channel, base_channel, A, adaptive=adaptive)
        self.l5 = GCN_TCN(base_channel, base_channel*2, A, stride=2, adaptive=adaptive)
        self.l6 = GCN_TCN(base_channel*2, base_channel*2, A, adaptive=adaptive)
        self.l7 = GCN_TCN(base_channel*2, base_channel*2, A, adaptive=adaptive)
        self.l8 = GCN_TCN(base_channel*2, base_channel*4, A, stride=2, adaptive=adaptive)
        self.l9 = GCN_TCN(base_channel*4, base_channel*4, A, adaptive=adaptive)
        self.l10 = GCN_TCN(base_channel*4, base_channel*4, A, adaptive=adaptive)

        self.pretrained = pretrained


    def init_weights(self):
        """Initiate the parameters either from existing checkpoint or from
        scratch."""
        if isinstance(self.pretrained, str):
            logger = get_root_logger()
            logger.info(f'load model from: {self.pretrained}')

            load_checkpoint(self, self.pretrained, strict=False, logger=logger)

        elif self.pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
                elif isinstance(m, nn.Linear):
                    normal_init(m)
                elif isinstance(m, _BatchNorm):
                    constant_init(m, 1)
        else:
            raise TypeError('pretrained must be a str or None')



    def forward(self, inputs):
        if len(inputs.shape) == 5:
            N, C, T, V, _ = inputs.shape
            inputs = inputs.permute(0, 4, 1, 3, 2).contiguous().view(-1, C, V, T) # N*M C V T

            N2 = inputs.shape[0] # may be N*M

            # t_jpt
            t_jpt = inputs[:, :, :, 1:] - inputs[:, :, :, 0:-1] # (N, C, V, T-1)
            t_jpt = torch.cat([t_jpt.new(N2, C, V, 1).zero_(), t_jpt], dim=-1) # (N, C, V, T)

        # for input
        x = inputs.view(-1, C*V, T)
        x = self.data_bn(x)
        x = x.view(-1, C, V, T).permute(0, 1, 3, 2).contiguous() # N*M, C, T, V

        # for t_jnt
        t_x = t_jpt.view(-1, C*V, T)
        t_x = self.data_bn2(t_x)
        t_x = t_x.view(-1, C, V, T).permute(0, 1, 3, 2).contiguous() # N*M, C, T, V

        x = self.l11(x) + self.l12(t_x)
        
        #x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.l5(x)
        x = self.l6(x)
        x = self.l7(x)
        x = self.l8(x)
        x = self.l9(x)
        x = self.l10(x) # N*M, C, T, V

        return x



@BACKBONES.register_module()
class GTCNDjnt17UD(nn.Module):
    def __init__(self,  
                 V=25,
                 in_channels=3,
                 adaptive=True,
                 clip_len=20,
                 idx_mode='jpt',
                 pretrained=None,):
        super(GTCNDjnt17UD, self).__init__()

        self.graph = GraphA()

        #A = self.graph.A # 3,25,25
        #A = self.graph.A_lr # 5,25,25
        #A = self.graph.A_inout_lr # 3, 25, 25
        #A = self.graph.A_17
        #A = self.graph.A_17_lr
        A = self.graph.A_17_ud

        self.num_point = V
        self.data_bn = nn.BatchNorm1d(in_channels * V)
        self.data_bn2 = nn.BatchNorm1d(in_channels * V)


        #self.block_input_jnt = block_joint_input(in_channels=in_channels, out_channels=32)
        #self.idx_block = IdxBlock(T=clip_len, V=V, out_channels=32, mode=idx_mode)

        base_channel = 64
        #self.l1 = GCN_TCN(in_channels, base_channel, A, residual=False, adaptive=adaptive)

        self.l11 = GCN_TCN(in_channels, base_channel, A, residual=False, adaptive=adaptive)
        self.l12 = GCN_TCN(in_channels, base_channel, A, residual=False, adaptive=adaptive)

        self.l2 = GCN_TCN(base_channel, base_channel, A, adaptive=adaptive)
        self.l3 = GCN_TCN(base_channel, base_channel, A, adaptive=adaptive)
        self.l4 = GCN_TCN(base_channel, base_channel, A, adaptive=adaptive)
        self.l5 = GCN_TCN(base_channel, base_channel*2, A, stride=2, adaptive=adaptive)
        self.l6 = GCN_TCN(base_channel*2, base_channel*2, A, adaptive=adaptive)
        self.l7 = GCN_TCN(base_channel*2, base_channel*2, A, adaptive=adaptive)
        self.l8 = GCN_TCN(base_channel*2, base_channel*4, A, stride=2, adaptive=adaptive)
        self.l9 = GCN_TCN(base_channel*4, base_channel*4, A, adaptive=adaptive)
        self.l10 = GCN_TCN(base_channel*4, base_channel*4, A, adaptive=adaptive)

        self.pretrained = pretrained


    def init_weights(self):
        """Initiate the parameters either from existing checkpoint or from
        scratch."""
        if isinstance(self.pretrained, str):
            logger = get_root_logger()
            logger.info(f'load model from: {self.pretrained}')

            load_checkpoint(self, self.pretrained, strict=False, logger=logger)

        elif self.pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
                elif isinstance(m, nn.Linear):
                    normal_init(m)
                elif isinstance(m, _BatchNorm):
                    constant_init(m, 1)
        else:
            raise TypeError('pretrained must be a str or None')



    def forward(self, inputs):
        if len(inputs.shape) == 5:
            N, C, T, V, _ = inputs.shape
            inputs = inputs.permute(0, 4, 1, 3, 2).contiguous().view(-1, C, V, T) # N*M C V T

            N2 = inputs.shape[0] # may be N*M

            # t_jpt
            t_jpt = inputs[:, :, :, 1:] - inputs[:, :, :, 0:-1] # (N, C, V, T-1)
            t_jpt = torch.cat([t_jpt.new(N2, C, V, 1).zero_(), t_jpt], dim=-1) # (N, C, V, T)

        # for input
        x = inputs.view(-1, C*V, T)
        x = self.data_bn(x)
        x = x.view(-1, C, V, T).permute(0, 1, 3, 2).contiguous() # N*M, C, T, V

        # for t_jnt
        t_x = t_jpt.view(-1, C*V, T)
        t_x = self.data_bn2(t_x)
        t_x = t_x.view(-1, C, V, T).permute(0, 1, 3, 2).contiguous() # N*M, C, T, V

        x = self.l11(x) + self.l12(t_x)
        
        #x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.l5(x)
        x = self.l6(x)
        x = self.l7(x)
        x = self.l8(x)
        x = self.l9(x)
        x = self.l10(x) # N*M, C, T, V

        return x


@BACKBONES.register_module()
class GTCNDjnt17LRUD(nn.Module):
    def __init__(self,  
                 V=25,
                 in_channels=3,
                 adaptive=True,
                 clip_len=20,
                 idx_mode='jpt',
                 pretrained=None,):
        super(GTCNDjnt17LRUD, self).__init__()

        self.graph = GraphA()

        #A = self.graph.A # 3,25,25
        #A = self.graph.A_lr # 5,25,25
        #A = self.graph.A_inout_lr # 3, 25, 25
        #A = self.graph.A_17
        #A = self.graph.A_17_lr
        #A = self.graph.A_17_ud
        A = self.graph.A_17_lrud

        self.num_point = V
        self.data_bn = nn.BatchNorm1d(in_channels * V)
        self.data_bn2 = nn.BatchNorm1d(in_channels * V)


        #self.block_input_jnt = block_joint_input(in_channels=in_channels, out_channels=32)
        #self.idx_block = IdxBlock(T=clip_len, V=V, out_channels=32, mode=idx_mode)

        base_channel = 64
        #self.l1 = GCN_TCN(in_channels, base_channel, A, residual=False, adaptive=adaptive)

        self.l11 = GCN_TCN(in_channels, base_channel, A, residual=False, adaptive=adaptive)
        self.l12 = GCN_TCN(in_channels, base_channel, A, residual=False, adaptive=adaptive)

        self.l2 = GCN_TCN(base_channel, base_channel, A, adaptive=adaptive)
        self.l3 = GCN_TCN(base_channel, base_channel, A, adaptive=adaptive)
        self.l4 = GCN_TCN(base_channel, base_channel, A, adaptive=adaptive)
        self.l5 = GCN_TCN(base_channel, base_channel*2, A, stride=2, adaptive=adaptive)
        self.l6 = GCN_TCN(base_channel*2, base_channel*2, A, adaptive=adaptive)
        self.l7 = GCN_TCN(base_channel*2, base_channel*2, A, adaptive=adaptive)
        self.l8 = GCN_TCN(base_channel*2, base_channel*4, A, stride=2, adaptive=adaptive)
        self.l9 = GCN_TCN(base_channel*4, base_channel*4, A, adaptive=adaptive)
        self.l10 = GCN_TCN(base_channel*4, base_channel*4, A, adaptive=adaptive)

        self.pretrained = pretrained


    def init_weights(self):
        """Initiate the parameters either from existing checkpoint or from
        scratch."""
        if isinstance(self.pretrained, str):
            logger = get_root_logger()
            logger.info(f'load model from: {self.pretrained}')

            load_checkpoint(self, self.pretrained, strict=False, logger=logger)

        elif self.pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
                elif isinstance(m, nn.Linear):
                    normal_init(m)
                elif isinstance(m, _BatchNorm):
                    constant_init(m, 1)
        else:
            raise TypeError('pretrained must be a str or None')



    def forward(self, inputs):
        if len(inputs.shape) == 5:
            N, C, T, V, _ = inputs.shape
            inputs = inputs.permute(0, 4, 1, 3, 2).contiguous().view(-1, C, V, T) # N*M C V T

            N2 = inputs.shape[0] # may be N*M

            # t_jpt
            t_jpt = inputs[:, :, :, 1:] - inputs[:, :, :, 0:-1] # (N, C, V, T-1)
            t_jpt = torch.cat([t_jpt.new(N2, C, V, 1).zero_(), t_jpt], dim=-1) # (N, C, V, T)

        # for input
        x = inputs.view(-1, C*V, T)
        x = self.data_bn(x)
        x = x.view(-1, C, V, T).permute(0, 1, 3, 2).contiguous() # N*M, C, T, V

        # for t_jnt
        t_x = t_jpt.view(-1, C*V, T)
        t_x = self.data_bn2(t_x)
        t_x = t_x.view(-1, C, V, T).permute(0, 1, 3, 2).contiguous() # N*M, C, T, V

        x = self.l11(x) + self.l12(t_x)
        
        #x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.l5(x)
        x = self.l6(x)
        x = self.l7(x)
        x = self.l8(x)
        x = self.l9(x)
        x = self.l10(x) # N*M, C, T, V

        return x


@BACKBONES.register_module()
class GTCNDjnt17LRAbs(nn.Module):
    def __init__(self,  
                 V=25,
                 in_channels=3,
                 adaptive=True,
                 clip_len=20,
                 idx_mode='jpt',
                 pretrained=None,):
        super(GTCNDjnt17LRAbs, self).__init__()

        self.graph = GraphA()

        #A = self.graph.A # 3,25,25
        #A = self.graph.A_lr # 5,25,25
        #A = self.graph.A_inout_lr # 3, 25, 25
        #A = self.graph.A_17
        #A = self.graph.A_17_lr
        #A = self.graph.A_17_ud
        #A = self.graph.A_17_lrud
        A = self.graph.A_17_lrabs

        self.num_point = V
        self.data_bn = nn.BatchNorm1d(in_channels * V)
        self.data_bn2 = nn.BatchNorm1d(in_channels * V)


        #self.block_input_jnt = block_joint_input(in_channels=in_channels, out_channels=32)
        #self.idx_block = IdxBlock(T=clip_len, V=V, out_channels=32, mode=idx_mode)

        base_channel = 64
        #self.l1 = GCN_TCN(in_channels, base_channel, A, residual=False, adaptive=adaptive)

        self.l11 = GCN_TCN(in_channels, base_channel, A, residual=False, adaptive=adaptive)
        self.l12 = GCN_TCN(in_channels, base_channel, A, residual=False, adaptive=adaptive)

        self.l2 = GCN_TCN(base_channel, base_channel, A, adaptive=adaptive)
        self.l3 = GCN_TCN(base_channel, base_channel, A, adaptive=adaptive)
        self.l4 = GCN_TCN(base_channel, base_channel, A, adaptive=adaptive)
        self.l5 = GCN_TCN(base_channel, base_channel*2, A, stride=2, adaptive=adaptive)
        self.l6 = GCN_TCN(base_channel*2, base_channel*2, A, adaptive=adaptive)
        self.l7 = GCN_TCN(base_channel*2, base_channel*2, A, adaptive=adaptive)
        self.l8 = GCN_TCN(base_channel*2, base_channel*4, A, stride=2, adaptive=adaptive)
        self.l9 = GCN_TCN(base_channel*4, base_channel*4, A, adaptive=adaptive)
        self.l10 = GCN_TCN(base_channel*4, base_channel*4, A, adaptive=adaptive)

        self.pretrained = pretrained


    def init_weights(self):
        """Initiate the parameters either from existing checkpoint or from
        scratch."""
        if isinstance(self.pretrained, str):
            logger = get_root_logger()
            logger.info(f'load model from: {self.pretrained}')

            load_checkpoint(self, self.pretrained, strict=False, logger=logger)

        elif self.pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
                elif isinstance(m, nn.Linear):
                    normal_init(m)
                elif isinstance(m, _BatchNorm):
                    constant_init(m, 1)
        else:
            raise TypeError('pretrained must be a str or None')



    def forward(self, inputs):
        if len(inputs.shape) == 5:
            N, C, T, V, _ = inputs.shape
            inputs = inputs.permute(0, 4, 1, 3, 2).contiguous().view(-1, C, V, T) # N*M C V T

            N2 = inputs.shape[0] # may be N*M

            # t_jpt
            t_jpt = inputs[:, :, :, 1:] - inputs[:, :, :, 0:-1] # (N, C, V, T-1)
            t_jpt = torch.cat([t_jpt.new(N2, C, V, 1).zero_(), t_jpt], dim=-1) # (N, C, V, T)

        # for input
        x = inputs.view(-1, C*V, T)
        x = self.data_bn(x)
        x = x.view(-1, C, V, T).permute(0, 1, 3, 2).contiguous() # N*M, C, T, V

        # for t_jnt
        t_x = t_jpt.view(-1, C*V, T)
        t_x = self.data_bn2(t_x)
        t_x = t_x.view(-1, C, V, T).permute(0, 1, 3, 2).contiguous() # N*M, C, T, V

        x = self.l11(x) + self.l12(t_x)
        
        #x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.l5(x)
        x = self.l6(x)
        x = self.l7(x)
        x = self.l8(x)
        x = self.l9(x)
        x = self.l10(x) # N*M, C, T, V

        return x




@BACKBONES.register_module()
class Djnt17dual(nn.Module):
    def __init__(self,  
                 V=25,
                 in_channels=3,
                 adaptive=True,
                 clip_len=20,
                 idx_mode='jpt',
                 pretrained=None,):
        super(Djnt17dual, self).__init__()

        self.graph = GraphA()

        #A = self.graph.A # 3,25,25
        #A = self.graph.A_lr # 5,25,25
        #A = self.graph.A_inout_lr # 3, 25, 25
        #A = self.graph.A_17
        #A = self.graph.A_17_lr
        #A = self.graph.A_17_ud
        A = self.graph.A_17_lrud

        self.num_point = V
        self.data_bn = nn.BatchNorm1d(in_channels * V)
        self.data_bn2 = nn.BatchNorm1d(in_channels * V)


        #self.block_input_jnt = block_joint_input(in_channels=in_channels, out_channels=32)
        #self.idx_block = IdxBlock(T=clip_len, V=V, out_channels=32, mode=idx_mode)

        base_channel = 64
        #self.l1 = GCN_TCN(in_channels, base_channel, A, residual=False, adaptive=adaptive)

        self.l11 = GCN_TCN(in_channels, base_channel, A, residual=False, adaptive=adaptive)
        self.l12 = GCN_TCN(in_channels, base_channel, A, residual=False, adaptive=adaptive)

        self.l2 = GCN_TCN(base_channel, base_channel, A, adaptive=adaptive)
        self.l3 = GCN_TCN(base_channel, base_channel, A, adaptive=adaptive)
        self.l4 = GCN_TCN(base_channel, base_channel, A, adaptive=adaptive)
        self.l5 = GCN_TCN(base_channel, base_channel*2, A, stride=2, adaptive=adaptive)
        self.l6 = GCN_TCN(base_channel*2, base_channel*2, A, adaptive=adaptive)
        self.l7 = GCN_TCN(base_channel*2, base_channel*2, A, adaptive=adaptive)
        self.l8 = GCN_TCN(base_channel*2, base_channel*4, A, stride=2, adaptive=adaptive)
        self.l9 = GCN_TCN(base_channel*4, base_channel*4, A, adaptive=adaptive)
        self.l10 = GCN_TCN(base_channel*4, base_channel*4, A, adaptive=adaptive)

        self.pretrained = pretrained


    def init_weights(self):
        """Initiate the parameters either from existing checkpoint or from
        scratch."""
        if isinstance(self.pretrained, str):
            logger = get_root_logger()
            logger.info(f'load model from: {self.pretrained}')

            load_checkpoint(self, self.pretrained, strict=False, logger=logger)

        elif self.pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
                elif isinstance(m, nn.Linear):
                    normal_init(m)
                elif isinstance(m, _BatchNorm):
                    constant_init(m, 1)
        else:
            raise TypeError('pretrained must be a str or None')



    def forward(self, inputs):
        if len(inputs.shape) == 5:
            N, C, T, V, _ = inputs.shape # N C T V M
            inputs = inputs.permute(0, 4, 1, 3, 2).contiguous().view(-1, C, V, T) # N*M C V T

            N2 = inputs.shape[0] # may be N*M

            # t_jpt
            t_jpt = inputs[:, :, :, 1:] - inputs[:, :, :, 0:-1] # (N, C, V, T-1)
            t_jpt = torch.cat([t_jpt.new(N2, C, V, 1).zero_(), t_jpt], dim=-1) # (N, C, V, T)

        # for input
        x = inputs.view(-1, C*V, T)
        x = self.data_bn(x)
        x = x.view(-1, C, V, T).permute(0, 1, 3, 2).contiguous() # N*M, C, T, V

        # for t_jnt
        t_x = t_jpt.view(-1, C*V, T)
        t_x = self.data_bn2(t_x)
        t_x = t_x.view(-1, C, V, T).permute(0, 1, 3, 2).contiguous() # N*M, C, T, V

        x = self.l11(x) + self.l12(t_x)
        
        #x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.l5(x)
        x = self.l6(x)
        x = self.l7(x)
        x = self.l8(x)
        x = self.l9(x)
        x = self.l10(x) # N*M, C, T, V

        return x


@BACKBONES.register_module()
class Djnt17Coco(nn.Module):
    def __init__(self,  
                 in_channels=3,
                 adaptive=True,
                 clip_len=20,
                 graph_cfg=None,
                 pretrained=None,):
        super(Djnt17Coco, self).__init__()

        #self.graph = GraphA()
        self.graph = Graph(**graph_cfg)

        #A = self.graph.A # 3,25,25
        #A = self.graph.A_lr # 5,25,25
        #A = self.graph.A_inout_lr # 3, 25, 25
        #A = self.graph.A_17
        #A = self.graph.A_17_lr
        #A = self.graph.A_17_ud
        #A = self.graph.A_17_lrud

        #A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)
        A = self.graph.A

        V = 17 # A.size(1)

        self.num_point = V
        self.data_bn = nn.BatchNorm1d(in_channels * V)
        self.data_bn2 = nn.BatchNorm1d(in_channels * V)


        #self.block_input_jnt = block_joint_input(in_channels=in_channels, out_channels=32)
        #self.idx_block = IdxBlock(T=clip_len, V=V, out_channels=32, mode=idx_mode)

        base_channel = 64
        #self.l1 = GCN_TCN(in_channels, base_channel, A, residual=False, adaptive=adaptive)

        self.l11 = GCN_TCN(in_channels, base_channel, A, residual=False, adaptive=adaptive)
        self.l12 = GCN_TCN(in_channels, base_channel, A, residual=False, adaptive=adaptive)

        self.l2 = GCN_TCN(base_channel, base_channel, A, adaptive=adaptive)
        self.l3 = GCN_TCN(base_channel, base_channel, A, adaptive=adaptive)
        self.l4 = GCN_TCN(base_channel, base_channel, A, adaptive=adaptive)
        self.l5 = GCN_TCN(base_channel, base_channel*2, A, stride=2, adaptive=adaptive)
        self.l6 = GCN_TCN(base_channel*2, base_channel*2, A, adaptive=adaptive)
        self.l7 = GCN_TCN(base_channel*2, base_channel*2, A, adaptive=adaptive)
        self.l8 = GCN_TCN(base_channel*2, base_channel*4, A, stride=2, adaptive=adaptive)
        self.l9 = GCN_TCN(base_channel*4, base_channel*4, A, adaptive=adaptive)
        self.l10 = GCN_TCN(base_channel*4, base_channel*4, A, adaptive=adaptive)

        self.tcn = unit_tcn(base_channel*4, 512, seg=clip_len, mode='TV')

        self.pretrained = pretrained


    def init_weights(self):
        """Initiate the parameters either from existing checkpoint or from
        scratch."""
        if isinstance(self.pretrained, str):
            logger = get_root_logger()
            logger.info(f'load model from: {self.pretrained}')

            load_checkpoint(self, self.pretrained, strict=False, logger=logger)

        elif self.pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
                elif isinstance(m, nn.Linear):
                    normal_init(m)
                elif isinstance(m, _BatchNorm):
                    constant_init(m, 1)
        else:
            raise TypeError('pretrained must be a str or None')



    def forward(self, inputs):
        if len(inputs.size()) == 5:
            #N, C, T, V, _ = inputs.shape # N C T V M
            #inputs = inputs.permute(0, 4, 1, 3, 2).contiguous().view(-1, C, V, T) # N*M C V T

            # for ntu60-hrnet.pkl in pyskt
            N, M, T, V, C = inputs.size()
            inputs = inputs.permute(0, 1, 4, 3, 2).contiguous().view(-1, C, V, T) # N*M C V T
            
            N2 = inputs.shape[0] # may be N*M

            # t_jpt
            t_jpt = inputs[:, :, :, 1:] - inputs[:, :, :, 0:-1] # (N, C, V, T-1)
            t_jpt = torch.cat([t_jpt.new(N2, C, V, 1).zero_(), t_jpt], dim=-1) # (N, C, V, T)


        # for input
        x = inputs.view(-1, C*V, T)
        x = self.data_bn(x)
        x = x.view(-1, C, V, T).permute(0, 1, 3, 2).contiguous() # N*M, C, T, V

        # for t_jnt
        t_x = t_jpt.view(-1, C*V, T)
        t_x = self.data_bn2(t_x)
        t_x = t_x.view(-1, C, V, T).permute(0, 1, 3, 2).contiguous() # N*M, C, T, V
        

        x = self.l11(x) + self.l12(t_x)
        
        #x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.l5(x)
        x = self.l6(x)
        x = self.l7(x)
        x = self.l8(x)
        x = self.l9(x)
        x = self.l10(x) # N*M, C, T, V

        x = self.tcn(x) # (N*M, 512, 20, 1)

        # add on 20220720
        x = x.reshape((N, M) + x.shape[1:]) # N M C T V
        #x = x.view(N, M, C, T, V)

        return x
