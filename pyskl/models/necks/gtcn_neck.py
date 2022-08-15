import torch
import torch.nn as nn

from ..builder import NECKS

#from mmcv.cnn import build_conv_layer, build_norm_layer
from ..gcns import GCNBlock, CNNBlock, unit_tcn2
from ...utils import Graph, GraphMulti


@NECKS.register_module()
class SeqGTCN(nn.Module):
    """GCN.

    Note that we use `view` to remove extra channel after pooling. We do not
    use `squeeze` as it will also remove the batch dimension when the tensor
    has a batch dimension of size 1, which can lead to unexpected errors.

    Args:
        dim (int): Dimensions of each sample channel, can be one of {1, 2, 3}.
            Default: 2

    20220328
    shape:
        input: (-1, 2560, 7, 7)
        output: (-1, -1)
    """
    def __init__(self,
                 in_channels=256,
                 out_channels=256,
                 graph_cfg=dict(layout='cocolr', mode='spatial_lr3a'),
                 **kwargs):
        super(SeqGTCN, self).__init__()
        

        self.graph = GraphMulti(**graph_cfg)
        A5 = self.graph.A5
        self.step_tcn = CNNBlock(in_channels=in_channels, out_channels=in_channels, dilations=[1,2])
        self.step_gcn = GCNBlock(in_channels=in_channels, out_channels=out_channels, A=A5, adaptive=False)
        self.uni_tcn = unit_tcn2(out_channels, out_channels, seg=25, mode='TV')

        '''
        if mode == 'GAP':
            self.pool = nn.AdaptiveAvgPool2d(1)
        elif mode == 'GMP':
            self.pool = nn.AdaptiveMaxPool2d(1)
        '''
        
    def init_weights(self):
        pass

    def forward(self, x):
        if isinstance(x, list):
            for item in x:
                assert len(item.shape) == 2
            x = [item.mean(dim=0) for item in x]
            x = torch.stack(x)


        N, M, C, T, V = x.shape
        x = x.reshape(N * M, C, T, V) # by gzb: N*M 256 100/4 17
        x = self.step_gcn(x) # -1 256 25 17
        x = self.step_tcn(x) # -1 256 25 17
        #x = torch.cat([x_tcn, x_gcn], dim=1) # -1 512 25 17
        x = self.uni_tcn(x) # -1 256 25 1

        x = x.reshape((N, M) + x.shape[1:]) # N M C 1 1

        #x = self.pool(x)  # by gzb: N*M 512 1 1
        #x = x.reshape(N, M, C)  # by gzb: N M 256

        #x = x.mean(dim=1) # by gzb: N 256

        return x

#@NECKS.register_module()
class UnitGCNGAP(nn.Module):
    """GCN.

    Note that we use `view` to remove extra channel after pooling. We do not
    use `squeeze` as it will also remove the batch dimension when the tensor
    has a batch dimension of size 1, which can lead to unexpected errors.

    Args:
        dim (int): Dimensions of each sample channel, can be one of {1, 2, 3}.
            Default: 2

    20220328
    shape:
        input: (-1, 2560, 7, 7)
        output: (-1, -1)
    """
    def __init__(self, dim=2):
        super(UnitGCNGAP, self).__init__()
        assert dim in [1, 2, 3], 'GlobalAveragePooling dim only support ' \
            f'{1, 2, 3}, get {dim} instead.'


        
        if dim == 1:
            self.gmp = nn.AdaptiveAvgPool1d(1)
        elif dim == 2:
            self.gmp = nn.AdaptiveAvgPool2d((1, 1))
        else:
            self.gmp = nn.AdaptiveAvgPool3d((1, 1, 1))
        
    def init_weights(self):
        pass

    def forward(self, inputs):
        inputs = tuple(inputs) # 20220629

        if isinstance(inputs, tuple):
            outs = tuple([self.gmp(self.gcn1(x, self.adjacent(x))) for x in inputs])
            outs = tuple([out.view(x.size(0), -1) for out, x in zip(outs, inputs)])
        elif isinstance(inputs, torch.Tensor):
            outs = self.gmp(self.gcn1(inputs, self.adjacent(inputs)))
            outs = outs.view(inputs.size(0), -1)
        else:
            raise TypeError('neck inputs should be tuple or torch.tensor')
        return outs

