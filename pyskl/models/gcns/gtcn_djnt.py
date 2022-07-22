import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

from .graph import Graph
#from ..tools import SelfAtt

from tools_cigcn import unit_tcn



def conv_init(conv):
    if conv.weight is not None:
        nn.init.kaiming_normal_(conv.weight, mode='fan_out')
    if conv.bias is not None:
        nn.init.constant_(conv.bias, 0)


def bn_init(bn, scale):
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)


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

        # Additional Max branch
        self.branches.append(
            nn.Sequential(
                CBRBlock(in_channels, branch_channels, mode=mode),
                nn.MaxPool2d(kernel_size=(3, 1), stride=(stride, 1), padding=(1, 0)),
                nn.BatchNorm2d(branch_channels),  # 为什么还要加bn
                #SelfAtt(branch_channels, branch_channels, mode='spatial')
            )
        )

        # Additional 1x1 branch
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


class ReshapeInput(nn.Module):
    def __init__(self, mode='VT', pos1=False):
        super().__init__()
        self.mode = mode
        self.pos1 = pos1

    def _computePos(self, inputs):
        _device = inputs.get_device()
        idx_center = torch.tensor([0]).cuda(_device) # pos 1 with index 0 (0-based)
        pos_1 = torch.index_select(inputs, 2, idx_center).cuda(_device) # M C 1 T
        inputs = inputs - pos_1

        return inputs


    def _getValidKpt17(self, inputs):
        N, T, V_C = inputs.shape
        V = 25 # revised on 20220712
        C = V_C // V
        inputs = inputs.view(N, T, V, C) # shape: (N, self.seg, 25, 3) # 3D position
        inputs = inputs.permute(0, 3, 2, 1).contiguous() # shape: (N, C, V, T)

        # by gzb: 20220713
        #delete_17 = [0, 2, 7, 11, 14, 18, 22, 24]
        idx_17 = [1, 3, 4, 5, 6, 8, 9, 10, 12, 13, 15, 16, 17, 19, 20, 21, 23]
        _device = inputs.get_device()
        inputs = torch.index_select(inputs, 2, torch.tensor(idx_17).cuda(_device)).cuda(_device) # N, C, V, T

        return inputs


    def forward(self, inputs):
        # change kpt25 to kpt17
        inputs = self._getValidKpt17(inputs) # N, C, V, T

        # relativePos1
        if self.pos1 ==  True:
            inputs = self._computePos(inputs)

        N, C, V, _ = inputs.shape

        ## motion joint info between frames
        t_jpt = inputs[:, :, :, 1:] - inputs[:, :, :, 0:-1] # (N, C, V, T-1)
        t_jpt = torch.cat([t_jpt.new(N, C, V, 1).zero_(), t_jpt], dim=-1) # (N, C, V, T)

        if self.mode == 'VT':
            return inputs, t_jpt
        elif self.mode == 'TV':
            inputs = inputs.permute(0, 1, 3, 2).contiguous() # N, C, T, V
            t_jpt = t_jpt.permute(0, 1, 3, 2).contiguous()
            return inputs, t_jpt


class GTCNDjnt17LR(nn.Module):
    def __init__(self,  
                 V=17,
                 in_channels=3,
                 adaptive=True,
                 clip_len=20,
                 spatial_type='max',
                 num_classes=60,
                 num_clips=1,):
        super(GTCNDjnt17LR, self).__init__()

        self.spatial_type = spatial_type
        self.num_classes = num_classes
        self.num_clips = num_clips

        #self.graph = GraphA()
        self.graph = Graph()

        #A = self.graph.A # 3,25,25
        #A = self.graph.A_lr # 5,25,25
        #A = self.graph.A_inout_lr # 3, 25, 25
        #A = self.graph.A_17
        A = self.graph.A_17_lr

        self.num_point = V
        self.reshape_input = ReshapeInput(mode='VT', pos1=True)
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

        #self.pretrained = pretrained

        # cls head
        self.tcn = unit_tcn(base_channel*4, 512, seg=clip_len, mode='TV')
        self.pool = None
        if self.spatial_type == 'avg':
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
        elif self.spatial_type == 'max':
            self.pool = nn.AdaptiveMaxPool2d((1, 1))
        else:
            raise NotImplementedError
            
        self.fc1 = nn.Linear(512, num_classes)


        # initial
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                '''
                n = m.out_channels
                k1 = m.kernel_size[0]
                k2 = m.kernel_size[1]
                n = n * k1 * k2
                m.weight.data.normal_(0, math.sqrt(2. / n))
                '''

                ''' from sgn
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                '''

                conv_init(m)
            if isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)



    def forward(self, inputs):
        if len(inputs.shape) == 5:
            N, C, T, V, _ = inputs.shape
            inputs = inputs.permute(0, 4, 1, 3, 2).contiguous().view(-1, C, V, T) # N*M C V T

            N2 = inputs.shape[0] # may be N*M

            # t_jpt
            t_jpt = inputs[:, :, :, 1:] - inputs[:, :, :, 0:-1] # (N, C, V, T-1)
            t_jpt = torch.cat([t_jpt.new(N2, C, V, 1).zero_(), t_jpt], dim=-1) # (N, C, V, T)
        
        elif len(inputs.shape) == 3:
            inputs, t_jpt = self.reshape_input(inputs) 

            N, C, V, T = inputs.shape


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


        # gtcn-head
        output = self.tcn(x) # (N*M, 512, 20, 1)
        #'''
        output = self.pool(output) # (N*M, 512, 1, 1)
        output = torch.flatten(output, start_dim=1, end_dim=-1) # (N*M, 512) 
        #'''
        output = self.fc1(output)

        #if self.num_clips != 1:
        #    output = output.contiguous().view(-1, self.num_clips, self.num_classes)
        #    output = torch.mean(output[:, :self.num_clips, :], dim=1)

        return output


